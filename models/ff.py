from torch import nn
from transformation_measure import ObservableLayersModel
from models import SequentialWithIntermediates
import torch.nn.functional as F


class FFNet(nn.Module,ObservableLayersModel):

    def use_bn(self): return False

    def __init__(self,input_shape,num_classes,h1=128,h2=64):
        self.bn=self.use_bn()
        super(FFNet, self).__init__()
        self.name = self.__class__.__name__


        h,w,channels=input_shape

        self.linear_size = h * w * channels

        layers=[
            nn.Linear(self.linear_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes),
        ]
        if self.bn:
            layers.insert(1,nn.BatchNorm1d(h1))
            layers.insert(4, nn.BatchNorm1d(h2))

        self.fc= SequentialWithIntermediates(*layers)

    def forward(self, x):
        x = x.view(-1, self.linear_size)
        x= self.fc(x)
        x=F.log_softmax(x,dim=1)
        return x

    def activation_names(self):
        names=["fc1", "fc1act", "fc2", "fc2act", "fc3", "logsoft"]
        if self.bn:
            names.insert(1, "bn1")
            names.insert(4, "bn2")
        return names

    def forward_intermediates(self,x)->(object,[]):
        x = x.view(-1, self.linear_size)
        x,intermediates = self.fc.forward_intermediates(x)
        x=F.log_softmax(x)
        return x,intermediates+[x]

class FFNetBN(FFNet):

    def use_bn(self):
        return True
    def __init__(self,input_shape,num_classes,h1=128,h2=64):
        super(FFNetBN, self).__init__(input_shape,num_classes,h1=h1,h2=h2)
