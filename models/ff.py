from torch import nn
from transformation_measure import ObservableLayersModule
from models import SequentialWithIntermediates
import torch.nn.functional as F
from models.util import Flatten

class FFNet( ObservableLayersModule):


    def __init__(self,input_shape,num_classes,h1=128,h2=64,bn=False):
        super(FFNet, self).__init__()
        self.bn = bn
        self.name = self.__class__.__name__
        if self.bn:
            self.name+="BN"


        h,w,channels=input_shape

        self.linear_size = h * w * channels

        layers=[
            Flatten(),
            nn.Linear(self.linear_size, h1),
            #bn
            nn.ELU(),
            nn.Linear(h1, h2),
            #bn
            nn.ELU(),
            nn.Linear(h2, num_classes),
            nn.LogSoftmax(dim=-1)
        ]
        if self.bn:
            layers.insert(2,nn.BatchNorm1d(h1))
            layers.insert(5, nn.BatchNorm1d(h2))

        self.fc= SequentialWithIntermediates(*layers)

    def forward(self, x):
        return self.fc(x)


    def activation_names(self)->[str]:
        return self.fc.activation_names()

    def forward_intermediates(self,x)->(object,[]):
        return self.fc.forward_intermediates(x)

class FFNetBN(FFNet):


    def __init__(self,input_shape,num_classes,h1=128,h2=64):
        super(FFNetBN, self).__init__(input_shape,num_classes,h1=h1,h2=h2,bn=True)
        self.name = self.__class__.__name__
