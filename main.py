import numpy as np
import pytorch_lightning as pl
import time
import pandas as pd
# import sys
# import os
# sys.path.append(os.getcwd())
# from src.models import SimplePytorchModel
from src.models.pytorch_model import SimplePytorchModel
from src.dataloaders.mnist_dataloader import FashionMnistDataloader
from src.utils import timer_func



@timer_func
def run_craft_implementation(path):
    loader = FashionMnistDataloader(path=path)
    model = SimplePytorchModel()
    model.fit(
        loader.dataloader(batch_size=32),
        epochs=5,
        N_print=900)


@timer_func
def run_pl_implementation(path):
    # datapipe = DataPipe()
    # pl.Trainer(
    #     model=model,
    #     datapipe=None)
    pass



def main():
    path = r'./data/fashion-mnist/fashion-mnist_train.csv'

    ### Craft implementation    
    run_craft_implementation(path)
    
    ### Pytorch-lightning implementation
    run_pl_implementation(path)






if __name__ == "__main__":
    main()















