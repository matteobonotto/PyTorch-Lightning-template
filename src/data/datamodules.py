import lightning as L

from typing import Optional, Callable
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
import torch
from lightning.pytorch.cli import instantiate_class 

from src.data.registry import DATAPIPE_REGISTRY

def collate_fun(batch):
    # Assuming pairs
    b1 = torch.cat([b[0] for b in batch], dim=0)
    b2 = torch.cat([b[1] for b in batch], dim=0)
    return b1, b2



class BaseDataModule(L.LightningDataModule):
    def __init__(
            self,
            path,
            datapipe : list[str], 
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0,
            pin_memory : bool = False,
            collate : bool = False,
            persistent_workers : bool = False
            ):
        super(BaseDataModule,self).__init__()
        self.path = path
        self.datapipe = datapipe
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.collate_fun = collate_fun if collate else None
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers

    def setup(self, stage: Optional[str] = None):
        dps = DATAPIPE_REGISTRY[self.datapipe[0]]
        self.train_dp = dps[self.datapipe[1]]["train"](data_path=self.path['train'])
        self.valid_dp = dps[self.datapipe[1]]["valid"](data_path=self.path['val'])
        self.test_dp = dps[self.datapipe[1]]["test"](data_path=self.path['test'])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dp,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fun,
            persistent_workers = self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dp,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fun,
            persistent_workers = self.persistent_workers
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dp,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fun,
            persistent_workers = self.persistent_workers
        )





















