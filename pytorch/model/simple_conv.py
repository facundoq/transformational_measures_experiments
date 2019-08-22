import torch.nn as nn
import torch.nn.functional as F
import torch
from transformation_measure import ObservableLayersModel

from pytorch.model.util import SequentialWithIntermediates


class SimpleConv(nn.Module, ObservableLayersModel):

    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128):
        super(SimpleConv, self).__init__()
        self.name = self.__class__.__name__
        h, w, channels = input_shape

        # self.conv=nn.Sequential(
        #     nn.Conv2d(channels, conv_filters, 3,padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters, conv_filters, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters, conv_filters*2, 3, padding=1,stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters*2, conv_filters*2, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_filters*2, conv_filters*4, 3, padding=1,stride=2),
        #     nn.ReLU(),
        #     )

        self.conv = SequentialWithIntermediates(
            nn.Conv2d(channels, conv_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(conv_filters, conv_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters * 2, conv_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(conv_filters * 2, conv_filters * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.linear_size = h * w * (conv_filters * 4) // 4 // 4

        self.fc = SequentialWithIntermediates(
            nn.Linear(self.linear_size, fc_filters),
            # nn.BatchNorm1d(fc_filters),
            nn.ReLU(),
            nn.Linear(fc_filters, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.linear_size)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward_intermediates(self, x)->(object,[]):
        x1, convs = self.conv.forward_intermediates(x)

        x2 = x1.view(-1, self.linear_size)
        x3, fcs = self.fc.forward_intermediates(x2)
        x4 = F.log_softmax(x3, dim=-1)
        return x4, convs + fcs + [x4]

    def activation_names(self):
        conv_layer_names = ["c1", "c1act", "c2", "c2act", "mp1",
                            "c3", "c3act", "c4", "c4act", "mp2",
                            "c5", "c5act"]
        fc_layer_names = ["fc1", "fc1act", "fc2", "fc2act"]

        return conv_layer_names + fc_layer_names
