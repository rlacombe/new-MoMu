import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from contrastive_gin import GINSimclr
from torch_geometric.data import LightningDataset
from data_provider.pretrain_datamodule import GINPretrainDataModule
from data_provider.pretrain_dataset import GINPretrainDataset

def get_ckpt_folder_name_from_args(args):
    # get the directory for the weights.
    # we want the weights to be stored in the correct directory according to the experiment being run.
    # TODO: Add type of graph and text encoder to args so that we don't have to hard code it in.
    # doing this will allow us to save checkpoints to folders based on what experiments we're running.
    graph_encoder = "gin"
    text_encoder = "bert"
    graph_augs = sorted([args.graph_aug1, args.graph_aug2])
    sub_dir_path = f"{graph_encoder}-{text_encoder}/{graph_augs[0]}-{graph_augs[1]}" 
    ckpt_folder_path = os.path.join("all_checkpoints/", sub_dir_path)
    if not os.path.exists(ckpt_folder_path): os.makedirs(ckpt_folder_path)
    return ckpt_folder_path

def main(args):
    pl.seed_everything(args.seed)

    # data
    # train_dataset = GINPretrainDataset(args.root, args.text_max_len, args.graph_aug1, args.graph_aug2)
    # dm = LightningDataset(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dm = GINPretrainDataModule.from_argparse_args(args)


    # model
    model = GINSimclr(
        temperature=args.temperature,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_num_layers=args.gin_num_layers,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.graph_pooling,
        graph_self=args.graph_self,
        bert_hidden_dim=args.bert_hidden_dim,
        bert_pretrain=args.bert_pretrain,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print('total params:', sum(p.numel() for p in model.parameters()))

    ckpt_dir_path = get_ckpt_folder_name_from_args(args)
    ckpt_callback = plc.ModelCheckpoint(dirpath=ckpt_dir_path, every_n_epochs=5)
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    trainer = Trainer.from_argparse_args(args, callbacks=[ckpt_callback], strategy=strategy)

    trainer.fit(model, datamodule=dm)
    
    print(f"\n\nBest model checkpoint: {ckpt_callback.best_model_path}\n\n")

    # Now, I'll actually delete the other checkpoints so we can save space.
    for ckpt in os.listdir(ckpt_dir_path):
        ckpt_path = os.path.join(os.path.abspath(ckpt_dir_path), ckpt)
        if ckpt_path == ckpt_callback.best_model_path: continue
        os.remove(ckpt_path)
    os.rename(ckpt_callback.best_model_path, os.path.join(os.path.abspath(ckpt_dir_path), 'best-ckpt.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--default_root_dir', type=str, default='./checkpoints/', help='location of model checkpoints')
    # parser.add_argument('--max_epochs', type=int, default=500)

    # GPU
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser = Trainer.add_argparse_args(parser)
    parser = GINSimclr.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args
    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    main(args)