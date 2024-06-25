
import pandas as pd
import time
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim

import lightning as L

from src.modules.custom_losses import UserCrossEntropyLoss
from torchmetrics import Accuracy


class Conv2dEncoderBlock(L.LightningModule):
    def __init__(
            self,
            in_channels : int = 1,
            out_channels : int = 1,
            kernel_size : int = 3):
        super(Conv2dEncoderBlock,self).__init__()   

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

    def forward(self,x: Tensor):
        return self.activation(self.maxpool2d(self.conv2d(x)))
    
 

class conv2d_classif(L.LightningModule):
    def __init__(
            self, 
            channel_in_list : list,
            channel_out_list : list,
            linear_in_features: int):
        super(conv2d_classif,self).__init__()
        # self.save_hyperparameters() #for model checkpointing

        module_list = []

        for in_channels, out_channels in zip(channel_in_list,channel_out_list):
            module_list.append(Conv2dEncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels
            ))

        module_list.extend([
            nn.Flatten(),
            nn.LazyLinear(out_features=linear_in_features),
            nn.GELU(),
            nn.Linear(in_features=linear_in_features, out_features=10),
            nn.GELU()
        ])
        self.layer_list = nn.ModuleList(module_list)

        self.loss = UserCrossEntropyLoss()

        self.val_metrics = nn.ModuleDict({
            'val_loss' : UserCrossEntropyLoss(),
            'val_acc' : Accuracy(task="multiclass", num_classes=10)
        })
        self.test_metrics = nn.ModuleDict({
            'test_loss' : UserCrossEntropyLoss(),
            'test_acc' : Accuracy(task="multiclass", num_classes=10)
        })


    def forward(self,x):
        for layer in self.layer_list:
            x = layer(x)
        return x
    
