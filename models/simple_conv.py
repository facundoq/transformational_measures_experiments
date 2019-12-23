import torch.nn as nn
import torch.nn.functional as F
from transformation_measure import ObservableLayersModule

from models import SequentialWithIntermediates

from models.util import Flatten

class SimpleConv(ObservableLayersModule):


    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128,bn=False,kernel_size=3):
        super(SimpleConv, self).__init__()
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        self.bn=bn
        self.kernel_size=kernel_size
        assert (kernel_size % 2) ==1
        same_padding = (kernel_size-1)//2
        conv_filters2=conv_filters*2
        conv_filters4 = conv_filters * 4
        conv_layers=[nn.Conv2d(channels, conv_filters, kernel_size, padding=same_padding ),
        #bn
        nn.ELU(),
        nn.Conv2d(conv_filters, conv_filters, kernel_size, padding=same_padding ),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters, conv_filters2, kernel_size, padding=same_padding ),
        # bn
        nn.ELU(),
        nn.Conv2d(conv_filters2, conv_filters2, kernel_size, padding=same_padding ),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters2, conv_filters4, kernel_size, padding=same_padding ),
        # bn
        nn.ELU(),]

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

