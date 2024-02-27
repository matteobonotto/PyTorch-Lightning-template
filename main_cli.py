import os
import sys

import torch 

from src.dataloaders.mnist_dataloader import UserLightningDataModule
from src.models.pl_model import SimplePytorchLightningModel 
from src.utils import timer_func, CustomCli


###
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')

torch.set_float32_matmul_precision("high")

###
@timer_func
def main():
    cli = CustomCli(
        model_class=SimplePytorchLightningModel,
        datamodule_class=UserLightningDataModule,
        run=False
        )
    cli.trainer.fit(
        cli.model, 
        datamodule=cli.datamodule
        )
    
    

###
if __name__ == "__main__":
    main()















