import functools
from typing import Optional, Callable

import torch
import torchdata.datapipes as dp
import h5py
from torchdata.datapipes.iter import IterDataPipe
import numpy as np
# from helper_functions.helper_functions.data import read_h5_numpy



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



class MnistDatasetOpener(IterDataPipe):
    def __init__(
            self,
            dp : IterDataPipe):
        super(MnistDatasetOpener,self).__init__()
        self.dp = dp

    def __iter__(self):
        for path in self.dp:
            data = read_h5_numpy(path)

            for idx in range(data['y'].shape[0]):
                X = torch.tensor(data['X'][idx, ...])/255
                y = torch.tensor(data['y'][idx, ...])
                yield X, y




def build_datapipes(
    data_path,
    dataset_opener: Callable[..., IterDataPipe],
    lister: Callable[..., IterDataPipe],
    sharder: Callable[..., IterDataPipe],
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
    )

    return dpipe


train_datapipe_mnist = functools.partial(
    build_datapipes,
    dataset_opener=MnistDatasetOpener,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
)


























