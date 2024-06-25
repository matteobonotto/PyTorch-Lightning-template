import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from torch.utils.data import TensorDataset
import pandas as pd
from typing import Optional
from helper_functions.data import read_h5_numpy
import torchdata.datapipes as dp
import os


# import matplotlib.pyplot as plt

DTYPE = torch.float32



###
class BaseBulkDataModule(LightningDataModule):
    def __init__(
            self,
            path,
            data_opener_type : 'str', 
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0):
        super(BaseBulkDataModule,self).__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_opener_type = data_opener_type

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MNISTBulkDataset(path = self.path['train'])
        self.val_dataset = MNISTBulkDataset(path = self.path['val'])
        self.test_dataset = MNISTBulkDataset(path = self.path['test'])
           
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        

class MNISTBulkDataset(Dataset):
    def __init__(self,path : str):
        super(MNISTBulkDataset,self).__init__()
        self.path = path

        # load and prepare data
        data = read_h5_numpy(self.path)
        self.X = torch.tensor(data['X'] ,dtype=DTYPE).unsqueeze(1)
        self.y = torch.tensor(data['y'] .ravel(),dtype=torch.int64)
        del data

    def __getitem__(self, idx):
        return self.X[idx, ...]/255, self.y[idx, ...]
    
    def __len__(self):
        return self.y.shape[0]
    



###
class BaseStreamDataModule(LightningDataModule):
    def __init__(
            self,
            path,
            data_opener_type : 'str', 
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0):
        super(BaseStreamDataModule,self).__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_opener_type = data_opener_type

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MNISTStreamDatasetPdearena(path = self.path['train'])
        self.val_dataset = MNISTStreamDatasetPdearena(path = self.path['val'])
        self.test_dataset = MNISTStreamDatasetPdearena(path = self.path['test'])
           
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

class MNISTStreamDataset(Dataset):
    def __init__(self, path ):
        super(MNISTStreamDataset,self).__init__()
        self.data_path = os.path.join(self.path,  os.listdir(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pathfile = self.data_path[idx]
        data = read_h5_numpy(pathfile)
        X = torch.tensor(data['X'] ,dtype=DTYPE)/255
        y = torch.tensor(data['y'] .ravel(),dtype=torch.int64)
        return X, y

class MNISTStreamDatasetPdearena(dp.iter.IterDataPipe):
    def __init__(self, path : str):
        super(MNISTStreamDatasetPdearena,self).__init__()
        self.data_path = [os.path.join(path,  p) for p in os.listdir(path)]

    def __iter__(self):
        for path in self.data_path:
            data = read_h5_numpy(path)
            X = torch.tensor(data['X'] ,dtype=DTYPE)/255
            y = torch.tensor(data['y'] .ravel(),dtype=torch.int64)
            return X, y
    



DATA_MODULES = {
    'BulkDataOpener' : BaseBulkDataModule,
    'StreamDataOpener' : BaseStreamDataModule,
}



