import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer
from enum import IntEnum
import utils.molecular_augmentations as mol

class SamplingType(IntEnum):
    CosSimMean = 0,
    CosSimMax = 1,
    CosSimSent = 2,
    Random = 3

    @classmethod
    def strToEnum(cls, sampling_type: str):
        if sampling_type == "random":
            return SamplingType.Random
        elif sampling_type == "cos_sim_mean":
            return SamplingType.CosSimMean
        elif sampling_type == "cos_sim_max":
            return SamplingType.CosSimMax 
        elif sampling_type == "cos_sim_sent":
            return SamplingType.CosSimSent
        else:
            raise ValueError(f"Sampling type {sampling_type} is not supported. Supported values are: random, cos_sim_mean, cos_sim_max, cos_sim_sent")

class GINPretrainDataset(Dataset):
    def __init__(self, root, text_max_len, graph_aug1, graph_aug2, sampling_type, sampling_temp, sampling_eps):
        super(GINPretrainDataset, self).__init__(root)
        self.root = root
        self.graph_aug1 = graph_aug1
        self.graph_aug2 = graph_aug2
        self.text_max_len = text_max_len

        self.sampling_type = SamplingType.strToEnum(sampling_type)
        self.sampling_temp = sampling_temp
        self.sampling_eps = sampling_eps

        # Get sorted file names
        self.graph_name_list = os.listdir(root + 'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root + 'text/')
        self.text_name_list.sort()

        if sampling_type != SamplingType.Random:
            self.cos_sim_score_name_list = os.listdir(root + 'cosine_sim_score/')
            self.cos_sim_score_name_list.sort()


    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        sampling_temp = self.sampling_temp
        epsilon = self.sampling_eps

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        data_aug1 = self.augment(data_graph, self.graph_aug1)
        data_aug2 = self.augment(data_graph, self.graph_aug2)

        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line)
            if count >= 500:
                break
        # print(text_list)
        if len(text_list) < 2:
            two_text_list = [text_list[0], text_list[0][-self.text_max_len:]]
        else:
            if self.sampling_type == SamplingType.Random:
                two_text_list = random.sample(text_list, 2)
            else:
                # Load the cosine similarity scores.
                cos_sim_path = os.path.join(self.root, 'cosine_sim_score', self.cos_sim_score_name_list[index])
                cos_sim_scores = torch.load(cos_sim_path)
                cos_sim_scores = cos_sim_scores[self.sampling_type].cpu().numpy()  # This works b/c the cosine simliarity score enum values correspond to their index in the tensor
                
                # Apply epsilon sampling: https://arxiv.org/abs/2210.15191
                cos_sim_scores /= sampling_temp 
                cos_sim_scores =  np.where(cos_sim_scores < epsilon*(1/len(text_list)), 0, cos_sim_scores) 
                smax_scores = np.exp(cos_sim_scores) / np.sum(np.exp(cos_sim_scores)) 
                two_text_list = np.random.choice(text_list, 2, p=smax_scores) 
                
        text_list.clear()

        # # load and process text
        # text_path = os.path.join(self.root, 'text', text_name)
        # with open(text_path, 'r', encoding='utf-8') as f:
        #     text_list = f.readlines()
        # f.close()
        # # print(text_list)
        # if len(text_list) < 2:
        #     two_text_list = [text_list[0], text_list[0][-self.text_max_len:]]
        # else:
        #     two_text_list = random.sample(text_list, 2)
        # text_list.clear()

        # print(random.sample([1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,18],2))
        #if len(two_text_list[0]) > 256:
        #    two_text_list[0] = two_text_list[0][:256]
        #if len(two_text_list[1]) > 256:
        #    two_text_list[1] = two_text_list[1][:256]        
        text1, mask1 = self.tokenizer_text(two_text_list[0])
        text2, mask2 = self.tokenizer_text(two_text_list[1])

        # print(graph_name)
        # print(text_name)

        return data_aug1, data_aug2, text1.squeeze(0), mask1.squeeze(0), text2.squeeze(0), mask2.squeeze(0)

    def augment(self, data, graph_aug):
        # node_num = data.edge_index.max()
        # sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        # data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        return mol.methylation(deepcopy(data)) # 'TODO' hacky, pls fix

        if graph_aug == 'id':
            data_aug = deepcopy(data)
        elif graph_aug == 'chem':
            data_aug = mol.methylation(deepcopy(data)) 
        elif graph_aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif graph_aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif graph_aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif graph_aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif graph_aug == 'random2':  # choose one from two augmentations
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random3':  # choose one from three augmentations
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random4':  # choose one from four augmentations
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        # if graph_aug == 'dnodes' or graph_aug == 'subgraph' or graph_aug == 'random2' or graph_aug == 'random3' or graph_aug == 'random4':
        #     edge_idx = data_aug.edge_index.numpy()
        #     _, edge_num = edge_idx.shape
        #     idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
        #     node_num_aug = len(idx_not_missing)
        #     data_aug.x = data_aug.x[idx_not_missing]
        #     # data_aug.batch = data.batch[idx_not_missing]
        #     idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
        #     edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
        #     data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

        return data_aug

    def tokenizer_text(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids'][:512]  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask'][:512]  # [1,1,1,0,0,0]
        return input_ids, attention_mask


if __name__ == '__main__':
    # mydataset = GraphTextDataset()
    # train_loader = torch_geometric.loader.DataLoader(
    #     mydataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4
    # )
    # for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
    #     print(aug1.edge_index.shape)
    #     print(aug1.x.shape)
    #     print(aug1.ptr.size(0))
    #     print(aug2.edge_index.dtype)
    #     print(aug2.x.dtype)
    #     print(text1.shape)
    #     print(mask1.shape)
    #     print(text2.shape)
    #     print(mask2.shape)
    # mydataset = GraphormerPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    # from functools import partial
    # from data_provider.collator import collator_text
    # train_loader = torch.utils.data.DataLoader(
    #         mydataset,
    #         batch_size=8,
    #         num_workers=4,
    #         collate_fn=partial(collator_text,
    #                            max_node=128,
    #                            multi_hop_max_dist=5,
    #                            spatial_pos_max=1024),
    #     )
    # aug1, aug2, text1, mask1, text2, mask2 = mydataset[0]
    mydataset = GINPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    train_loader = torch_geometric.loader.DataLoader(
            mydataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            # persistent_workers = True
        )
    # aug1, aug2, text1, mask1, text2, mask2 = mydataset[0]
    # print(aug1)
    # print(aug1.x.shape)
    # print(aug2)
    # print(aug2.x.dtype)
    # print(text1.shape)
    # print(mask1.shape)
    # print(text2.shape)
    # print(mask2.shape)
    for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
        print(aug1)
        # print(aug1.x.shape)
        # print(aug2)
        # print(aug2.x.dtype)
        # print(text1.shape)
        # print(mask1.shape)
        # print(text2.shape)
        # print(mask2.shape)