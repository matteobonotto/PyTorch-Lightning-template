
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
        # self.val_metrics = self.model.val_metrics
        
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
        loss = self.loss(self(x),y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        pred = self(x)
        # val_metric = self.accuracy(pred,y)
        val_metrics = {}
        for k,v in self.model.val_metrics.items():
            val_metrics.update({
                k : v(pred,y)
            })
            self.log(k, val_metrics[k], prog_bar=True)
        return val_metrics

    def test_step(self, batch):
        x, y = batch
        pred = self(x)
        # val_metric = self.accuracy(pred,y)
        test_metrics = {}
        for k,v in self.model.test_metrics.items():
            test_metrics.update({
                k : v(pred,y)
            })
            self.log(k, test_metrics[k], prog_bar=True)
        return test_metrics


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())



































