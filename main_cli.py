import os
import sys

import torch 

from src.data.datamodules import BaseDataModule
from src.models.pl_model import BaseModel 
from src.utils import timer_func, CustomCli

from lightning.pytorch.loggers import WandbLogger

###
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')

torch.set_float32_matmul_precision("high")

###
@timer_func
def main():
    cli = CustomCli(
        model_class=BaseModel,
        datamodule_class=BaseDataModule,
        run=False,
        save_config_kwargs={"overwrite": True} # otherwise error from wandb (Aborting to avoid overwriting results of a previous run)
        )
    cli.trainer.fit(
        cli.model, 
        datamodule=cli.datamodule
        )
    
    

###
if __name__ == "__main__":
    main()















