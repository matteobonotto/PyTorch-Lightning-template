from torch import Tensor
from torch.nn import CrossEntropyLoss
import lightning as L


# Example of custom loss
class NavierStokesLoss(L.LightningModule):
    def __init__(self):
        super(NavierStokesLoss,self).__init__()
        pass

    def forward(self,x):
        return None
    



class UserCrossEntropyLoss(L.LightningModule):
    def __init__(self):
        super(UserCrossEntropyLoss,self).__init__()
        self.loss = CrossEntropyLoss()

    def forward(
            self,
            pred : Tensor,
            y: Tensor):
        return self.loss(pred,y)













