import functools
from typing import Optional, Callable

import torch
import torchdata.datapipes as dp
import h5py
from torchdata.datapipes.iter import IterDataPipe
import numpy as np
# from helper_functions.helper_functions.data import read_h5_numpy
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms


transform = transforms.Compose([
    # transforms.RandomResizedCrop(125, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize(mean=(255/2),std=(255/2))
    ])

def read_h5_numpy(
        filename : str, 
        ):
    with h5py.File(filename, 'r') as hf:
        data = hdf5_to_dict(hf)
    return data
                 
def hdf5_to_dict(h5_file):
    def read_hdf5_file(h5_file):
        for key,val in h5_file.items():
            if type(val) == h5py._hl.dataset.Dataset:
                d[key] = np.array(val)
            else:
                d[key] = read_hdf5_file(val)
        return d
    
    d = dict()
    return read_hdf5_file(h5_file)

from lightning.pytorch import callbacks

class MnistDatasetOpener(IterDataPipe):
    def __init__(
            self,
            dp : IterDataPipe,
            transform = None 
            ):
        super(MnistDatasetOpener,self).__init__()
        self.dp = dp
        self.transform = transform

    def __len__(self):
        # NOTE: assuming all chunks with the same number of files inside
        sample_file = read_h5_numpy(next(iter(self.dp)))
        return sample_file['X'].shape[0]*len([p for p in self.dp])

    def __iter__(self):
        for path in self.dp:
            data = read_h5_numpy(path)

            for idx in range(data['y'].shape[0]):
                X = torch.tensor(data['X'][idx, ...], dtype=torch.float32).unsqueeze(0)
                if self.transform:
                    X = self.transform(X)            
                y = torch.tensor(data['y'][idx, ...],dtype=torch.int64)#.unsqueeze(0)
                yield X, y


def build_iter_datapipes(
    data_path,
    dataset_opener: Callable[..., IterDataPipe],
    lister: Callable[..., IterDataPipe],
    sharder: Callable[..., IterDataPipe],
    transform,
    mode: str,
):
    """Build datapipes for training and evaluation.

    Args:
        data_path (str): Path to the data.
        dataset_opener (Callable[..., dp.iter.IterDataPipe]): Dataset opener.
        lister (Callable[..., dp.iter.IterDataPipe]): List files.
        sharder (Callable[..., dp.iter.IterDataPipe]): Shard files.
        mode (str): Mode of the data. ["train", "valid", "test"]

    Returns:
        dpipe (IterDataPipe): IterDataPipe for training and evaluation.

    """
    dpipe = lister(
        data_path,
    )#.filter(filter_fn=filter_fn)

    if mode == "train":
        dpipe = dpipe.shuffle()

    dpipe = dataset_opener(
        sharder(dpipe),
        transform = transform
    )

    return dpipe


class MNISTBulkDataPipe(Dataset):
    def __init__(self, data_path, transform = None, **kwargs):
        super(MNISTBulkDataPipe,self).__init__()
        data = read_h5_numpy(data_path)
        self.X = data['X']
        self.y = data['y']
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx,...], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            X = self.transform(X)            
        y = torch.tensor(self.y[idx],dtype=torch.int64)
        return X, y



train_IterDataPipe_mnist = functools.partial(
    build_iter_datapipes,
    dataset_opener=MnistDatasetOpener,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    transform = transform,
    mode="train",
)

train_BulkDataPipe_mnist = functools.partial(
    MNISTBulkDataPipe,
    transform=transform)





















