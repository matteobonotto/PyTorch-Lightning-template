import os
import sys
from lightning import Trainer
from src.utils import timer_func
from lightning.pytorch.callbacks import TQDMProgressBar
from src.models.pl_model import SimplePytorchLightningModel 
from src.old_dataloaders.mnist_main_dataloader import FashionMnistDataLoader
from lightning.pytorch.accelerators import find_usable_cuda_devices
from argparse import ArgumentParser
from omegaconf import OmegaConf
import yaml

###
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')

num_workers = os.cpu_count()-1
num_workers = 0


###
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--c', default=None, help='Configuration file')
    parser.add_argument('--prova_config_1', action='store_true')
    parser.add_argument('--prova_config_2', default=None)
    namespace, _ = parser.parse_known_args()
    return vars(namespace)


def get_configs(args):
    with open(args['c'],'r') as f:
        config_file = yaml.safe_load(f)
    return config_file


@timer_func
def main(args):
    # get configs
    config_file = get_configs(args)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1, 
        strategy="auto",
        max_epochs=20,
        # precision='bf16',
        # enable_model_summary=False,
        # barebones=True,
        # enable_checkpointing=False
        callbacks=[TQDMProgressBar(refresh_rate=250)],
    )

    trainer.fit(
        SimplePytorchLightningModel(
            channel_in_list = config_file['model']['channel_in_list'],
            channel_out_list = config_file['model']['channel_out_list'],
            linear_in_features = config_file['model']['linear_in_features'],
        ), 
        FashionMnistDataLoader(
            path=config_file['data']['path'],
            num_workers=num_workers
            )
        )
    

###
if __name__ == "__main__":
    args = get_arguments()
    main(args)















