import torch.nn as nn
import torch.nn.functional as F
from transformational_measures import ObservableLayersModule

from models import SequentialWithIntermediates

from models.util import Flatten

class SimplestConv(ObservableLayersModule):
    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128,bn=False):
        super(SimplestConv, self).__init__()
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        self.bn=bn
        if self.bn:
            self.name+="BN"
        conv_filters2=conv_filters*2
        conv_layers=[nn.Conv2d(channels, conv_filters, 3, padding=1),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters, conv_filters2, 3, padding=1),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters2, conv_filters2, 3, padding=1),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        ]

        if self.bn:
            conv_layers.insert(1,nn.BatchNorm2d(conv_filters))
            conv_layers.insert(5, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(9, nn.BatchNorm2d(conv_filters2))


        conv = SequentialWithIntermediates(*conv_layers)

        self.linear_size = (h * w // 4 // 4 //4) * conv_filters2

        fc_layers=[
            Flatten(),
            nn.Linear(self.linear_size, fc_filters),
            # nn.BatchNorm1d(fc_filters),
            nn.ELU(),
            nn.Linear(fc_filters, num_classes),
            nn.LogSoftmax(dim=-1),
            ]
        if self.bn:
            fc_layers.insert(2,nn.BatchNorm1d(fc_filters))
        fc = SequentialWithIntermediates(*fc_layers)
        self.layers=SequentialWithIntermediates(conv,fc)

    def forward(self, x):
        return self.layers(x)

    def forward_intermediates(self, x)->(object,[]):
        return self.layers.forward_intermediates(x)

    def activation_names(self):
        return self.layers.activation_names()

class SimplestConvBN(SimplestConv):
    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128):
        super(SimplestConvBN, self).__init__(input_shape, num_classes, conv_filters=conv_filters, fc_filters=fc_filters,bn=True)
        self.name = self.__class__.__name__