import torch.nn as nn
from torch.nn.functional import dropout
import numpy as np
from .util import SequentialWithIntermediates, task_to_head
from transformational_measures.pytorch import ObservableLayersModule
import transformational_measures as tm

from ..tasks import Task
from ..tasks.train import ModelConfig



class AllConvolutionalConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,task:Task,dataset:str,bn:bool=False,dropout=False):        
        filters = {"mnist": 32, "cifar10": 96, "fashion_mnist": 96,"lsa16":32,"rwth":96}
        return AllConvolutionalConfig(task,filters=filters[dataset],bn=bn,dropout=dropout)

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
    
    def __init__(self, task:Task,filters:int=96,bn=False,
                 dropout=False):
        super().__init__(AllConvolutional)
        self.task=task
        self.filters=filters
        self.task=task
        self.dropout=dropout
        self.bn=bn        

    def id(self):
        return f"{self.__class__.__name__}(task={self.task.value},filters={self.filters},bn={self.bn},dropout={self.dropout})"


    def make(self,input_shape:np.ndarray, output_dim:int):
        return AllConvolutional(input_shape,output_dim,self)

    

class ConvBNAct(ObservableLayersModule):
    def __init__(self,in_filters,out_filters,kernel_size,stride=1,bn=False):
        super(ConvBNAct, self).__init__()
        if kernel_size==0:
            padding=0
        else:
            padding=1
        self.bn=bn

        c=nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        bn_layer=nn.BatchNorm2d(out_filters)
        r=nn.ELU()
        if bn:
            self.model = SequentialWithIntermediates(c,bn_layer,r)
        else:
            self.model = SequentialWithIntermediates(c, r)

    def forward(self,x):
        return self.model.forward(x)

    def forward_intermediates(self,x):
        return self.model.forward_intermediates(x)
    def activation_names(self)->[str]:
        return self.model.activation_names()


class PoolOut(nn.Module):

    def forward(self, x):
        return x.reshape(x.size(0), x.size(1), -1).mean(-1)


class AllConvolutional( ObservableLayersModule):
    def __init__(self, input_shape, output_dim,config:AllConvolutionalConfig):
        super(AllConvolutional, self).__init__()
        self.name = self.__class__.__name__
        self.config=config

        h,w,c=input_shape
        filters = config.filters
        filters2=filters*2
        bn=config.bn
        self.dropout=dropout
        self.layers=SequentialWithIntermediates(
              ConvBNAct(c, filters, 3, bn=bn)
             ,ConvBNAct(filters, filters, 3, bn=bn)
             ,ConvBNAct(filters, filters, 3, stride=2, bn=bn)
             ,ConvBNAct(filters, filters2, 3, bn=bn)
             ,ConvBNAct(filters2, filters2, 3, bn=bn)
             ,ConvBNAct(filters2, filters2, 3, stride=2, bn=bn)
             ,ConvBNAct(filters2, filters2, 3, bn=bn)
             ,ConvBNAct(filters2, filters2, 1, bn=bn)
             ,nn.Conv2d(filters2, output_dim, 1)
             ,PoolOut() 
             ,task_to_head(config.task)

        )
        

    def forward(self, x):
        return self.layers.forward(x)

    def forward_intermediates(self, x):
        return self.layers.forward_intermediates(x)

    def activation_names(self)->[str]:
        return self.layers.activation_names()



