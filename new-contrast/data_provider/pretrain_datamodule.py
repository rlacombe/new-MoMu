# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
from data_provider.pretrain_dataset import GINPretrainDataset
from utils.GraphAug import *


class GINPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        graph_aug1: str = 'dnodes',
        graph_aug2: str = 'subgraph',
        sampling_type: str = 'random',
        sampling_temp: float = .1,
        sampling_eps: float = .5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = GINPretrainDataset(root, text_max_len, graph_aug1, graph_aug2, sampling_type, sampling_temp, sampling_eps)

    def setup(self, stage: str = None):
        self.train_dataset = self.dataset

    # define the molecular augmentation of the 2nd graph item
    def collate_fn(self, batch):
        device = self.device

        data_aug1 = torch.stack([x[0] for x in batch]).to(device)
        data_aug2 = torch.stack([x[1] for x in batch]).to(device)
        text1 = torch.stack([x[2] for x in batch]).squeeze(1).to(device)
        mask1 = torch.stack([x[3] for x in batch]).squeeze(1).to(device)
        text2 = torch.stack([x[4] for x in batch]).squeeze(1).to(device)
        mask2 = torch.stack([x[5] for x in batch]).squeeze(1).to(device)
    
        # perform data_aug2 on the GPU
        data_aug2 = self.augment_data(data_aug2).to(device)
    
        return data_aug1, data_aug2, text1, mask1, text2, mask2
    
    # now define the augmentation
    
    def augment_data(batch):
        graphs = batch
        with torch.no_grad():
            for i in range(len(graphs)):
                graph = graphs[i]    
                graphs[i] = chemical_augmentation(graph)
   
        return graphs

    def train_dataloader(self):
        loader = torch_geometric.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn = self.collate_fn # set the collate_fn argument of the data loader to collate_fn
            # persistent_workers = True
        )
        print('len(train_dataloader)', len(loader))

        return loader
    
    
