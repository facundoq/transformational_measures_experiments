import torch.nn as nn
import torch.nn.functional as F
from transformational_measures.pytorch import ObservableLayersModule
import numpy as np


from .util import Flatten, SequentialWithIntermediates

from ...tasks import Task

from enum import Enum

import transformational_measures as tm

class ActivationFunction(Enum):
    ELU="ELU"
    ReLU="ReLU"
    Tanh="TanH"
    Sigmoid= "Sigmoid"
    PReLU="PReLU"

    def get_activation_class(self):
        return activation_map[self]

activation_map={ ActivationFunction.ELU:nn.ELU
                , ActivationFunction.ReLU:nn.ReLU
                , ActivationFunction.Tanh:nn.Tanh
                , ActivationFunction.Sigmoid:nn.Sigmoid
                , ActivationFunction.PReLU : nn.PReLU
                }

from ...tasks.train import ModelConfig

class SimpleConvConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,task:Task,dataset:str,bn:bool=False,k=3, activation=ActivationFunction.ELU,max_pooling=True):

        conv = {"mnist": 128, "cifar10": 128, "fashion_mnist": 64,"lsa16":128,"rwth":128}
        fc = {"mnist": 128, "cifar10": 128, "fashion_mnist": 128,"lsa16":64,"rwth":128}
        return SimpleConvConfig(task,conv=conv[dataset], fc=fc[dataset],bn=bn,kernel_size=k,activation=activation,max_pooling=max_pooling)

    def epochs(self,dataset:str,task:Task,transformations:tm.TransformationSet):

        epochs = {'cifar10': 35, 'mnist': 45, 'fashion_mnist': 12, "lsa16": 25, "rwth": 25}
        return self.scale_by_transformations(epochs[dataset],transformations)

    def __init__(self, task:Task,
                 conv=32, fc=128, bn=False, kernel_size=3, activation=ActivationFunction.ELU,
                 max_pooling=True):
        self.task=task
        self.conv=conv
        self.fc=fc
        self.bn=bn
        self.kernel_size=kernel_size
        self.activation=activation
        self.max_pooling=max_pooling

    def id(self):
        return f"{self.__class__.__name__}(task={self.task.value},conv={self.conv},fc={self.fc},bn={self.bn},k={self.kernel_size},act={self.activation.value},mp={self.max_pooling})"


    def make(self,input_shape:np.ndarray, output_dim:int):
        return SimpleConv(input_shape,output_dim,self)


class SimpleConv(ObservableLayersModule):
    def __init__(self, input_shape:np.ndarray, output_dim:int, c:SimpleConvConfig):
        super(SimpleConv, self).__init__()
        self.name = self.__class__.__name__
        self.c=c
        h, w, channels = input_shape
        assert (c.kernel_size % 2) ==1
        same_padding = (c.kernel_size-1)//2
        conv_filters2=c.conv*2
        conv_filters4 = c.conv * 4
        activation_class= c.activation.get_activation_class()
        if c.max_pooling:
            mp_generator = lambda f: nn.MaxPool2d(stride=2, kernel_size=2)
        else:
            mp_generator = lambda f: nn.Conv2d(f,f,stride=2, kernel_size=3, padding=same_padding)

        conv_layers=[
        nn.Conv2d(channels, c.conv, c.kernel_size, padding=same_padding ),
        #bn
        activation_class(),
        nn.Conv2d(c.conv, c.conv, c.kernel_size, padding=same_padding ),
        # bn
        activation_class(),
        mp_generator(c.conv),
        nn.Conv2d(c.conv, conv_filters2, c.kernel_size, padding=same_padding ),
        # bn
        activation_class(),
        nn.Conv2d(conv_filters2, conv_filters2, c.kernel_size, padding=same_padding ),
        # bn
        activation_class(),
        mp_generator(conv_filters2),
        nn.Conv2d(conv_filters2, conv_filters4, 3, padding=1 ),
        # bn
        activation_class(),]

        if c.bn:
            conv_layers.insert(1,nn.BatchNorm2d(c.conv))
            conv_layers.insert(4, nn.BatchNorm2d(c.conv))
            conv_layers.insert(8, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(11, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(15, nn.BatchNorm2d(conv_filters4))


        conv = SequentialWithIntermediates(*conv_layers)

        self.linear_size = h * w * (conv_filters4) // 4 // 4

        fc_layers=[
            Flatten(),
            nn.Linear(self.linear_size, c.fc),
            # nn.BatchNorm1d(fc_filters),
            activation_class(),
            nn.Linear(c.fc, output_dim),
            ]

        if c.task == Task.Classification:
            fc_layers.append(nn.LogSoftmax(dim=-1))
        elif c.task == Task.TransformationRegression:
            fc_layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported task {c.task}")

        if c.bn:
            fc_layers.insert(2,nn.BatchNorm1d(c.fc))

        fc = SequentialWithIntermediates(*fc_layers)
        self.layers=SequentialWithIntermediates(conv,fc)

    def forward(self, x):
        return self.layers(x)
        # return self.fc(self.conv(x))

    def forward_intermediates(self, x)->(object,[]):
        return self.layers.forward_intermediates(x)
        # x,conv_intermediates = self.conv.forward_intermediates(x)
        # x,fc_intermediates = self.fc.forward_intermediates(x)
        # return x,conv_intermediates+fc_intermediates

    def conv_layers(self):
        return self.conv.layer_names()
    def fc_layers(self):
        return self.fc.layer_names()

    def activation_names(self):
        return self.layers.activation_names()

