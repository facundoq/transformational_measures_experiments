import torch.nn as nn
from torch.nn.functional import dropout
import numpy as np
from .util import Flatten,SequentialWithIntermediates,task_to_head
from tmeasures.pytorch import ActivationsModule
import tmeasures as tm

from ..tasks import Task
from ..tasks.train import ModelConfig

class VGG16DConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,task:Task,dataset:str,bn:bool=False):
        conv = {"mnist": 16, "cifar10": 64,"lsa16":64,"rwth":64, }
        fc = {"mnist": 64, "cifar10": 512,"lsa16":32,"rwth":512, }
        return VGG16DConfig(task, conv=conv[dataset], fc=fc[dataset],bn=bn)

    def epochs(self,dataset:str,task:Task,transformations:tm.TransformationSet):
        
        if task== Task.TransformationRegression:
            epochs_dataset = {'cifar10': 35, 'mnist': 45, 'fashion_mnist': 12, "lsa16": 25, "rwth": 25}
        elif task == Task.Classification:
            epochs_dataset = {'cifar10': 30, 'mnist': 15, 'fashion_mnist': 12, "lsa16": 25, "rwth": 10}
        else:
            raise ValueError(task)
        epochs = epochs_dataset[dataset]
        epochs = self.scale_by_transformations(epochs,transformations)
        return epochs

    def __init__(self, task:Task,conv=32, fc=128, bn=False):
        super().__init__(VGG16D)
        self.task=task
        self.conv=conv
        self.fc=fc
        self.bn=bn

    def id(self):
       return f"{self.name()}(task={self.task.value},conv={self.conv},fc={self.fc},bn={self.bn})"

    def make(self,input_shape:np.ndarray, output_dim:int):
        return VGG16D(input_shape,output_dim,self)





class ConvBNRelu(ActivationsModule):

    def __init__(self,input:int,output:int,bn:bool):
        super(ConvBNRelu, self).__init__()
        self.bn=bn
        if bn:
            self.name = "ConvBNElu"
            self.layers = SequentialWithIntermediates(
                nn.Conv2d(input, output, kernel_size=3, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(output),
            )
        else:
            self.name = "ConvElu"
            self.layers = SequentialWithIntermediates(
                nn.Conv2d(input, output, kernel_size=3, padding=1),
                nn.ELU(),
            )

    def activation_names(self):
        return self.layers.activation_names()

    def forward(self,x):
        return self.layers.forward(x)

    def forward_activations(self,x):
        return self.layers.forward_activations(x)

def block(filters_in:int,feature_maps:int,n_conv:int,bn:bool):
    '''
    A block of the VGG model, consists of :param n_conv convolutions followed by a 2x2 MaxPool

    :param filters_in:
    :param feature_maps:
    :param n_conv:
    :param bn:
    :return:
    '''
    layers=[]
    for i in range(n_conv):
        layers.append(ConvBNRelu(filters_in,feature_maps,bn))
        filters_in=feature_maps
    layers.append(nn.MaxPool2d(2, 2))
    return layers


class VGG16D(ActivationsModule):

    def __init__(self, input_shape, output_dim, c:VGG16DConfig):
        super().__init__()
        self.name = self.__class__.__name__
        self.bn = c.bn
        if self.bn:
            self.name+="BN"

        h, w, channels = input_shape

        # list of conv layers
        conv_layers=[]
        # Configuration, depends on conv_filters
        convs_per_block= [2,2,3,3]
        feature_maps = [c.conv,c.conv*2,c.conv*4,c.conv*8]
        # end config
        # Create blocks of layers
        input_feature_maps=channels
        for conv,f in zip(convs_per_block,feature_maps):
            conv_layers+=block(input_feature_maps,f,conv,c.bn)
            input_feature_maps=f

        # Calculate flattened output size
        max_pools=len(convs_per_block)
        hf, wf = h // (2 ** max_pools), w // (2 ** max_pools)
        flattened_output_size = hf * wf * input_feature_maps


        dense_layers=[
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened_output_size, c.fc),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(c.fc, c.fc),
            nn.ReLU(),
            nn.Linear(c.fc, output_dim),
            task_to_head(c.task)
        ]

        conv_layers = SequentialWithIntermediates(*conv_layers)
        dense_layers = SequentialWithIntermediates(*dense_layers)
        self.layers=SequentialWithIntermediates(conv_layers,dense_layers)

    def forward(self, x):
        return self.layers(x)

    def forward_activations(self,x)->list[object,list]:
        return self.layers.forward_activations(x)

    def activation_names(self):
        return self.layers.activation_names()
