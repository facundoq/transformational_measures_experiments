# def convbnrelu(input,output):
#         return nn.Sequential(append(nn.Conv2d(input, output, kernel_size=3, padding=1))
#             ,self.conv_layers.append(nn.ELU())
#             ,self.conv_layers.append(nn.BatchNorm2d(output))
#         )

import torch.nn as nn
import torch.nn.functional as F
from models import Flatten
from models import SequentialWithIntermediates
from transformational_measures import  ObservableLayersModule

class ConvBNRelu(ObservableLayersModule):

    def __init__(self,input,output,bn):
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

    def forward_intermediates(self,x):
        return self.layers.forward_intermediates(x)

class VGGLike(ObservableLayersModule):
    def __init__(self, input_shape, num_classes,conv_filters,fc,bn=False):
        super(VGGLike, self).__init__()
        self.bn=bn
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        filters=conv_filters
        filters2=2*filters
        filters3=4*filters
        filters4=8*filters

        conv_layers = SequentialWithIntermediates(
            ConvBNRelu(channels, filters,bn),
            ConvBNRelu(filters, filters,bn),
            nn.MaxPool2d(2,2),
            ConvBNRelu(filters, filters2,bn),
            ConvBNRelu(filters2, filters2,bn),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(filters2, filters3,bn),
            ConvBNRelu(filters3, filters3,bn),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(filters3, filters4,bn),
            ConvBNRelu(filters4, filters4,bn),
            nn.MaxPool2d(2, 2),
        )
        max_pools=4
        hf, wf = h // (2 ** max_pools), w // (2 ** max_pools)
        flattened_output_size = hf * wf * filters4

        layers=[
            Flatten(),
            nn.Linear(flattened_output_size, fc),
            nn.ELU(),
            nn.Linear(fc,num_classes),
            nn.LogSoftmax(dim=-1),
        ]

        if self.bn:
            layers.insert(2,nn.BatchNorm1d(fc))

        dense_layers = SequentialWithIntermediates(*layers)
        self.layers=SequentialWithIntermediates(conv_layers,dense_layers)

    def forward(self, x):
        return self.layers(x)

    def forward_intermediates(self,x)->(object,[]):
        return self.layers.forward_intermediates(x)

    def activation_names(self):
        return self.layers.activation_names()



class VGGLikeBN(VGGLike):
    def __init__(self, input_shape, num_classes,conv_filters,fc):
        super(VGGLikeBN, self).__init__(input_shape, num_classes,conv_filters,fc,bn=True)
        self.name = self.__class__.__name__

