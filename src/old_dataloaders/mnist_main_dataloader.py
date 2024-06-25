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

from src.old_dataloaders.mnist_dataloader_modules import (
    BaseBulkDataModule, BaseStreamDataModule)


DTYPE = torch.float32

DATA_MODULES = {
    'BulkDataOpener' : BaseBulkDataModule,
    'StreamDataOpener' : BaseStreamDataModule,
}

# class MainDataModule(LightningDataModule):
#     def __init__(
#             self,
#             path,
#             data_opener_type : 'str', 
#             batch_size : int = 32,
#             shuffle : bool = False,
#             num_workers : int = 0
#             ):
#         super(MainDataModule,self).__init__()
#         self = DATA_MODULES[data_opener_type](
#             batch_size,
#             shuffle,
#             num_workers
#         )








