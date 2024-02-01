import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

DTYPE = torch.float32


class FashionMnistDataset(Dataset):
    def __init__(self,path):
        super(FashionMnistDataset,self).__init__()
        self.path = path

        # load and prepare data
        data = pd.read_csv(self.path, low_memory=False).to_numpy()
        X,y = data[:,1:], data[:,0]
        dims = X.shape
        dims = [int(x) for x in [dims[0], np.sqrt(dims[1]), np.sqrt(dims[1])]]
        X = X.reshape(dims)/255
        # plt.imshow(X[10000,:,:])
        # plt.show()
        self.X = torch.tensor(X,dtype=DTYPE).unsqueeze(1)
        self.y = torch.tensor(y,dtype=torch.int64)
        # self.data = [X, y]


    def __getitem__(self, idx):
        return self.X[idx, ...], self.y[idx, ...]
    
    
    def __len__(self):
        return self.y.shape[0]


class FashionMnistDataLoader(DataLoader):
    def __init__(
            self,
            path : str,
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0):
        super(FashionMnistDataLoader,self).__init__(
            dataset = FashionMnistDataset(path=path),
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers=num_workers
        )

