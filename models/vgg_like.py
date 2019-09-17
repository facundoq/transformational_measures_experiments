# def convbnrelu(input,output):
#         return nn.Sequential(append(nn.Conv2d(input, output, kernel_size=3, padding=1))
#             ,self.conv_layers.append(nn.ELU())
#             ,self.conv_layers.append(nn.BatchNorm2d(output))
#         )

import torch.nn as nn
import torch.nn.functional as F
from models import Flatten
from models import SequentialWithIntermediates
from transformation_measure import  ObservableLayersModel

class ConvBNRelu(nn.Module,ObservableLayersModel):

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
        if self.bn:
            return ["c","elu","bn"]
        else:
            return ["c", "elu"]

    def forward(self,x):
        return self.layers.forward(x)

    def forward_intermediates(self,x):
        return self.layers.forward_intermediates(x)

class VGGLike(nn.Module,ObservableLayersModel):
    def __init__(self, input_shape, num_classes,conv_filters,fc,bn=False):
        super(VGGLike, self).__init__()
        self.bn=bn
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        filters=conv_filters
        filters2=2*filters
        filters3=4*filters
        filters4=8*filters

        self.conv_layers = nn.Sequential(
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
            Flatten(),
        )
        max_pools=4
        hf, wf = h // (2 ** max_pools), w // (2 ** max_pools)
        flattened_output_size = hf * wf * filters4

        layers=[nn.Linear(flattened_output_size, fc),
            nn.ReLU(),
            nn.Linear(fc,num_classes)]

        if self.bn:
            layers.insert(1,nn.BatchNorm1d(fc))

        self.dense_layers = SequentialWithIntermediates(*layers)


    def forward(self, x):
        x=self.conv_layers.forward(x)
        x=self.dense_layers.forward(x)
        x=F.log_softmax(x, dim=1)
        return x

    def forward_intermediates(self,x)->(object,[]):
        outputs = []
        for i in range(4):
            x,intermediates = self.conv_layers[i*3].forward_intermediates(x)
            outputs.append(intermediates[0])
            outputs.append(intermediates[1])
            x,intermediates = self.conv_layers[i*3+1].forward_intermediates(x)
            outputs.append(intermediates[0])
            outputs.append(intermediates[1])
            x = self.conv_layers[i*3+ 2].forward(x)
            outputs.append(x)
        x=self.conv_layers[-1].forward(x)# flatten
        x,intermediates=self.dense_layers.forward_intermediates(x)
        outputs.append(intermediates[0])
        outputs.append(intermediates[2])
        outputs.append(intermediates[3])
        x = F.log_softmax(x, dim=1)
        outputs.append(x)
        return x, outputs

    def activation_names(self):
        names=[]
        for i in range(4):
            names.append(f"c{i}_0")
            names.append(f"c{i}_0act")
            if self.bn:
                names.append(f"c{i}_0bn")
            names.append(f"c{i}_1")
            names.append(f"c{i}_1act")
            if self.bn:
                names.append(f"c{i}_1bn")
            names.append(f"mp{i}")

        fc_names=["fc1","fc1act","fc2","fc2act"]
        if self.bn:
            fc_names.insert(1,"bn")

        return names+fc_names



class VGGLikeBN(VGGLike):
    def __init__(self, input_shape, num_classes,conv_filters,fc):
        super(VGGLikeBN, self).__init__(input_shape, num_classes,conv_filters,fc,bn=True)
        self.name = self.__class__.__name__

