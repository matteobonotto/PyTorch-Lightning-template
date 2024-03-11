
import pandas as pd
import time
import matplotlib.pyplot as plt
from lightning.pytorch.cli import instantiate_class 

from src.models.registry import MODEL_REGISTRY, LOSS_REGISTRY
from torchmetrics import Accuracy

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
from src.modules.mnist_modules import conv2d_classif

import lightning as L


def get_model(model_name):
    _model = MODEL_REGISTRY[model_name].copy()
    return instantiate_class(tuple(),_model)

def get_loss(loss_name):
    _loss = LOSS_REGISTRY[loss_name].copy()
    return instantiate_class(tuple(),_loss)





class BaseModel(L.LightningModule):
    def __init__(
            self,
            model_name : str,
            # class_path : str,
            loss_name : str
            ):
        super(BaseModel,self).__init__()

        self.model = get_model(model_name)
        self.loss = get_loss(loss_name)
        
        # TODO make val_criteria/test_criteria agnostic
        # self.val_criteria = { 
        #     "accuracy": Accuracy(task="multiclass", num_classes=10), 
        #     }
        # self.test_criteria = { 
        #     "accuracy": Accuracy(task="multiclass", num_classes=10), 
        #     }
        # self.metric1 = Accuracy(task="multiclass", num_classes=10)


    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch):
        x, y = batch
        return self.loss(self(x),y)
    
    def validation_step(self, batch):
        x, y = batch
        pred = self(x)
        return self.model.val_metric(pred,y)

    # def test_step(self, batch):
    #     x, y = batch
    #     pred = self(x)
    #     return {k:v(pred,y) for k,v in self.test_criteria.items()}


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())



































