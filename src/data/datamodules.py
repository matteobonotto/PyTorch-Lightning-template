import lightning as L

from typing import Optional, Callable
from torch.utils.data import DataLoader
import torchdata.datapipes as dp

from src.data.registry import DATAPIPE_REGISTRY

print('here')

class BaseDataModule(L.LightningDataModule):
    def __init__(
            self,
            path,
            datapipe_name : 'str', 
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0,
            pin_memory : bool = False
            ):
        super(BaseDataModule,self).__init__()
        self.path = path
        self.datapipe_name = datapipe_name
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        dps = DATAPIPE_REGISTRY[self.datapipe_name]
        self.train_dp = dps["train"](
            data_path=self.path['train'],
        )
        self.valid_dp = dps["valid"](
            data_path=self.path['val'],
        )
        self.test_dp = dps["test"](
            data_path=self.path['test'],
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dp,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            # collate_fn=collate_function,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dp,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=False,
            # collate_fn=collate_function,
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dp,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=False,
            # collate_fn=collate_function,
        )





















