
import torch.nn as nn
import numpy as np
from tmeasures.pytorch import ActivationsModule
from .util import SequentialWithIntermediates,Flatten,Add,GlobalAvgPool2d,task_to_head
import tmeasures as tm

from ..tasks import Task
from ..tasks.train import ModelConfig





class ResNetConfig(ModelConfig):
    # lr =5e-4

    def __init__(self, task:Task, bn=False,v=18):
        super().__init__(ResNet)
        self.bn=bn
        self.v=v
        self.task=task


    @classmethod
    def for_dataset(cls,task:Task,dataset:str,bn:bool=False,):
        if dataset == "cifar10":
            v = 32
        else:
            v = 18
        return ResNetConfig(task, bn=bn,v=v)

    def epochs(self,dataset:str,task:Task,transformations:tm.TransformationSet):
        
        if task== Task.TransformationRegression:
            epochs_dataset = {'cifar10': 40, 'mnist': 7, 'fashion_mnist': 12,"lsa16":20,"rwth":20}
        elif task == Task.Classification:
            epochs_dataset = {'cifar10': 40, 'mnist': 7, 'fashion_mnist': 12,"lsa16":20,"rwth":20}
        else:
            raise ValueError(task)
        epochs = epochs_dataset[dataset]
        epochs = self.scale_by_transformations(epochs,transformations)
        return epochs

   
    def id(self):
       return f"{self.name()}(task={self.task.value},v={self.v},bn={self.bn})"

    def make(self,input_shape:np.ndarray, output_dim:int):
        return ResNet(input_shape, output_dim, self)



class Block(ActivationsModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,bn=False):
        super(Block, self).__init__()
        layers=[nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                #bn1
                nn.ELU(),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                # bn2
                ]
        self.bn=bn
        if bn:
            layers.insert(1,nn.BatchNorm2d(planes))
            layers.insert(4, nn.BatchNorm2d(planes))

        main=SequentialWithIntermediates(*layers)

        shortcut_layers=[]
        self.use_shortcut=stride != 1 or in_planes != self.expansion*planes
        if self.use_shortcut:
            shortcut_layers=[nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),]
            if bn:
                shortcut_layers.append(nn.BatchNorm2d(self.expansion*planes))
        shortcut = SequentialWithIntermediates(*shortcut_layers)

        self.add=SequentialWithIntermediates(
            Add(main,shortcut)
            , nn.ELU()
            )


    def forward(self, x):
        return self.add(x)

    def forward_activations(self,x):
        return self.add.forward_activations(x)

    def activation_names(self)->list[str]:
        return self.add.activation_names()

class Bottleneck(ActivationsModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,bn=False):
        super(Bottleneck, self).__init__()
        self.bn=bn

        conv=[nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
              #bn
              ,nn.ELU()
              ,nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
              #bn
            , nn.ELU()
              ,nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
              ,]

        if self.bn:
            conv.insert(1, nn.BatchNorm2d(planes))
            conv.insert(4, nn.BatchNorm2d(planes))
            conv.insert(7, nn.BatchNorm2d(self.expansion*planes))
        conv = SequentialWithIntermediates(*conv)

        shortcut = []
        self.use_shortcut=stride != 1 or in_planes != self.expansion*planes
        if self.use_shortcut:
            shortcut.append(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            if self.bn:
                shortcut.append(nn.BatchNorm2d(self.expansion*planes))
        shortcut=SequentialWithIntermediates(*shortcut)
        self.add = SequentialWithIntermediates(
            Add(conv, shortcut)
            ,nn.ELU()
        )

    def forward(self, x):
        return self.add(x)

    def forward_activations(self,x):
        return self.add.forward_activations(x)

    def activation_names(self)->list[str]:
        return self.add.activation_names()



resnet_blocks = {18:[2, 2, 2, 2],
                  32:[3,4,6,3],
                  50:[3,4,6,3],
                  101:[3,4,23,3],
                  152:[3,8,36,3]
                 }

resnet_block_type =  {18:Block,
                  32:Block,
                  50:Bottleneck,
                  101:Bottleneck,
                  152:Bottleneck
                 }  

class ResNet( ActivationsModule):
    def __init__(self, input_shape,output_dim,config:ResNetConfig):        
        super(ResNet, self).__init__()
        self.name = self.__class__.__name__
        self.c=config
        bn=config.bn
        num_blocks = resnet_blocks[config.v]
        block = resnet_block_type[config.v]
        self.in_planes = 64
        h,w,c=input_shape
        layer0=[nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
                #bn
                ,nn.ELU()]
        if bn:
            layer0.insert(1,nn.BatchNorm2d(64))
        layer0=SequentialWithIntermediates(*layer0)

        layer1 = self._make_layer(block, 64, num_blocks[0], 1,bn)
        layer2 = self._make_layer(block, 128, num_blocks[1], 2,bn)
        layer3 = self._make_layer(block, 256, num_blocks[2], 2,bn)
        layer4 = self._make_layer(block, 512, num_blocks[3], 2,bn)
        conv = [layer0, layer1, layer2, layer3, layer4]

        self.conv=SequentialWithIntermediates(*conv)

        self.linear =SequentialWithIntermediates(
            GlobalAvgPool2d()
            ,Flatten()
            ,nn.Linear(512*block.expansion, output_dim)
            ,task_to_head(config.task)
            )

        self.layers = SequentialWithIntermediates(self.conv,self.linear)


    def _make_layer(self, block, planes, num_blocks, stride,bn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,bn))
            self.in_planes = planes * block.expansion
        return SequentialWithIntermediates(*layers)

    def forward(self, x):
        return self.layers.forward(x)

    def forward_activations(self,x):
        return self.layers.forward_activations(x)

    def activation_names(self)->list[str]:
        return self.conv.activation_names()+self.linear.activation_names()



