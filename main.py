import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
from time import time
import pandas as pd
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
            channel_in_list:list = [3,8,16],
            channel_out_list: list = [8,16,32]):
        super(SimplePytorchModel,self).__init__()

        module_list = []

        for in_channels, out_channels in zip(channel_in_list,channel_out_list):
            module_list.append(Conv2dEncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels
            ))
        module_list.extend([
            nn.Flatten(),
            nn.LazyLinear(out_features=100),
            nn.GELU(),
            nn.Linear(in_features=100, out_features=1),
            nn.GELU()
        ])
        self.layer_list = nn.ModuleList(module_list)

    def forward(self,x):
        for layer in self.layer_list:
            x = layer(x)
        return x
    
     
    def dataloader(self,X,y,batch_size):
        return DataLoader(
            TensorDataset(
                torch.tensor(X,device=self.device,dtype=self.dtype),
                torch.tensor(y,device=self.device,dtype=self.dtype)),
            shuffle=True,
            batch_size=batch_size)
    

    def train_iteration(
        self,
        X,
        y
        ):
        """
        This method implements a single training iteration

        :param X: input data
        :param y: target data
        :return: loss and predictions

        """
        # Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        # Compute prediction and error
        pred = self(X)
        loss = self.loss_p(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss, pred

    
    def fit(self,X,y,epochs=50,batch_size=32,N_print=0):
        train_loader = self.dataloader(
            X,y,
            batch_size=batch_size)
        n_steps = len(train_loader)

        self.optimizer = optim.AdamW( # Adam with weight decay
            self.parameters(), 
            weight_decay=self.L2regularization)

        loss_values = []
        time_start = time.time()
        for epoch in range(epochs):
            # Loop over the dataset multiple times
            running_loss = 0.0
            time_epoch_start = time.time()
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                # Perform single training step
                loss, _ = self.train_iteration(X, y)
                running_loss += loss.item()
                # Print every N_print mini-batches
                if i % N_print == (N_print - 1):
                    print('epoch: {}/{} batch: {}/{} loss: {:4.3f}'.format(
                        epoch,
                        epochs,
                        i + 1,
                        n_steps,
                        running_loss / i))
            loss_values.append(running_loss / n_steps)

            time_epoch_end = time.time() - time_epoch_start
            print.info('elapsed time per epoch: {:4.3f}'.format(time_epoch_end))
        self.time_training_end = time.time() - time_start
        print('Finished Training')
        print('Training took {:4.3f}s'.format(self.time_training_end))

        return loss_values


def prepare_data(path):
    data = pd.read_csv(path, low_memory=False).to_numpy()
    X,y = data[:,1:], data[:,0]
    dims = X.shape
    dims = [int(x) for x in [dims[0], np.sqrt(dims[1]), np.sqrt(dims[1])]]
    return (X,y)


class FashionMnistPreproc():
    def __init__(self):
        pass



def main():
    path = r'./data/fashion-mnist/fashion-mnist_train.csv'
    X,y = prepare_data(path)
    model = SimplePytorchModel()
    model.fit(X,y)
    # datapipe = DataPipe()

    # pl.Trainer(
    #     model=model,
    #     datapipe=None)






if __name__ == "__main__":
    main()















