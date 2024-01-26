import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import dataloader

# from typing import 


def Conv2dEncoderBlock(nn):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels,
            kernel_size):
        super(Conv2dEncoderBlock,self).__init__()   
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

    def forward(self,x):
        return self.activation(self.maxpool2d(self.conv2d))

def SimplePytorchModel(nn):
    def __init__(self):
        super(SimplePytorchModel,self).__init__()

        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.activation_1
    

def main():
    model = SimplePytorchModel()
    # datapipe = DataPipe()

    pl.Trainer(
        model=model,
        datapipe=datapipe)






if __name__ == "__main__":
    main()















