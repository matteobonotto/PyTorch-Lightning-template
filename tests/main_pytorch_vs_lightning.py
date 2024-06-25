import numpy as np
# import pytorch_lightning as pl
import time
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
# from src.models import SimplePytorchModel

from src.models.pytorch_model import SimplePytorchModel
from src.old_dataloaders.mnist_main_dataloader import FashionMnistDataLoader, UserLightningDataModule
from src.models.pl_model import SimplePytorchLightningModel 

from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar

from src.utils import timer_func, CustomCli

import torch

import sys
import os
# from omegaconf import OmegaConf
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')

num_workers = os.cpu_count()-1
num_workers = 0

@timer_func
def run_craft_implementation(path):
    dataloader = FashionMnistDataLoader(
            path=path,
            num_workers=num_workers 
            )
    model = SimplePytorchModel(device = 'cuda')
    model.fit(
        dataloader,
        epochs=20,
        N_print=900)


@timer_func
def run_pl_implementation(path):

    # Initialize a trainer
    # trainer = Trainer(
    #     accelerator="cuda",
    #     max_epochs=20,
    #     # callbacks=[TQDMProgressBar(refresh_rate=250)],
    #     # enable_model_summary=False,
    #     barebones=True,
    #     # enable_checkpointing=False
    # )

    # trainer.fit(
    #     SimplePytorchLightningModel(), 
    #     FashionMnistDataLoader(
    #         path=path,
    #         num_workers=num_workers
    #         )
    #     )
    
    cli = CustomCli(
        model_class=SimplePytorchLightningModel,
        datamodule_class=UserLightningDataModule,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    cli.trainer.fit(
        cli.model, 
        datamodule=cli.datamodule
        )



def main():
    # args = OmegaConf.from_cli()
    cli = CustomCli(
        model_class=SimplePytorchLightningModel,
        datamodule_class=UserLightningDataModule,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    

    ### Pytorch-lightning implementation
    run_pl_implementation(
        path = r'./data/fashion-mnist/fashion-mnist_train.csv'
        )


    ### Craft implementation    
    run_craft_implementation(
        path = r'./data/fashion-mnist/fashion-mnist_train.csv'
        )
    




if __name__ == "__main__":
    main()















