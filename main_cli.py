import os
import sys

from src.dataloaders.mnist_dataloader import UserLightningDataModule
from src.models.pl_model import SimplePytorchLightningModel 


from src.utils import timer_func, CustomCli


###
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')


###
@timer_func
def main():
    cli = CustomCli(
        model_class=SimplePytorchLightningModel,
        datamodule_class=UserLightningDataModule,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    
    

###
if __name__ == "__main__":
    main()















