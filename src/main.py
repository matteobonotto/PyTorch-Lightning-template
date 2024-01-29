import numpy as np
import pytorch_lightning as pl
from torch import nn, Tensor
from torch.utils.data import dataloader

# from typing import 


class Conv2dEncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3):
        super(Conv2dEncoderBlock,self).__init__()   
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

    def forward(self,x: Tensor):
        return self.activation(self.maxpool2d(self.conv2d(x)))
    


class SimplePytorchModel(nn.Module):
    def __init__(
            self,
            image_sizes: list,
            channel_in_list:list = [3,8,16],
            channel_out_list: list = [8,16,32]):
        super(SimplePytorchModel,self).__init__()

        module_list = []

        for in_channels, out_channels in zip(channel_in_list,channel_out_list):
            module_list.append(self.Conv2dEncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels
            ))
        module_list.append([
            nn.Linear(in_features=1, out_features=100),
            nn.GELU(),
            nn.Linear(in_features=100, out_features=1),
            nn.GELU()
        ])
        self.layer_list = nn.ModuleList(module_list)



    

def main():
    model = SimplePytorchModel()
    # datapipe = DataPipe()

    pl.Trainer(
        model=model,
        datapipe=datapipe)






if __name__ == "__main__":
    main()















