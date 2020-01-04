import torch.nn as nn
import torch.nn.functional as F
from transformation_measure import ObservableLayersModule

from models import SequentialWithIntermediates

from models.util import Flatten
from enum import Enum


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

class SimpleConv(ObservableLayersModule):


    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128, bn=False, kernel_size=3, activation=ActivationFunction.ELU,max_pooling=True):
        super(SimpleConv, self).__init__()
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        self.bn=bn
        self.kernel_size=kernel_size
        self.max_pooling=max_pooling
        assert (kernel_size % 2) ==1
        same_padding = (kernel_size-1)//2
        conv_filters2=conv_filters*2
        conv_filters4 = conv_filters * 4
        self.activation=activation
        activation_class= activation.get_activation_class()
        if max_pooling:
            mp_generator = lambda f: nn.MaxPool2d(stride=2, kernel_size=2)
        else:
            mp_generator = lambda f: nn.Conv2d(f,f,stride=2, kernel_size=3,padding=same_padding)

        conv_layers=[nn.Conv2d(channels, conv_filters, kernel_size, padding=same_padding ),
        #bn
        activation_class(),
        nn.Conv2d(conv_filters, conv_filters, kernel_size, padding=same_padding ),
        # bn
        activation_class(),
        mp_generator(conv_filters),
        nn.Conv2d(conv_filters, conv_filters2, kernel_size, padding=same_padding ),
        # bn
        activation_class(),
        nn.Conv2d(conv_filters2, conv_filters2, kernel_size, padding=same_padding ),
        # bn
        activation_class(),
        mp_generator(conv_filters2),
        nn.Conv2d(conv_filters2, conv_filters4, 3, padding=same_padding ),
        # bn
        activation_class(),]

        if self.bn:
            conv_layers.insert(1,nn.BatchNorm2d(conv_filters))
            conv_layers.insert(4, nn.BatchNorm2d(conv_filters))
            conv_layers.insert(8, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(11, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(15, nn.BatchNorm2d(conv_filters4))


        conv = SequentialWithIntermediates(*conv_layers)

        self.linear_size = h * w * (conv_filters * 4) // 4 // 4

        fc_layers=[
            Flatten(),
            nn.Linear(self.linear_size, fc_filters),
            # nn.BatchNorm1d(fc_filters),
            activation_class(),
            nn.Linear(fc_filters, num_classes),
            nn.LogSoftmax(dim=-1),
            ]
        if self.bn:
            fc_layers.insert(2,nn.BatchNorm1d(fc_filters))
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
        return self.conv.activation_names()
    def fc_layers(self):
        return self.fc.activation_names()

    def activation_names(self):
        return self.layers.activation_names()

