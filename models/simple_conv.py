import torch.nn as nn
import torch.nn.functional as F
from transformation_measure import ObservableLayersModel

from models import SequentialWithIntermediates


class SimpleConv(nn.Module, ObservableLayersModel):
    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128,bn=False):
        super(SimpleConv, self).__init__()
        self.name = self.__class__.__name__
        h, w, channels = input_shape
        self.bn=bn
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
        conv_filters2=conv_filters*2
        conv_filters4 = conv_filters * 4
        conv_layers=[nn.Conv2d(channels, conv_filters, 3, padding=1),
        #bn
        nn.ReLU(),
        nn.Conv2d(conv_filters, conv_filters, 3, padding=1),
        # bn
        nn.ReLU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters, conv_filters2, 3, padding=1),
        # bn
        nn.ReLU(),
        nn.Conv2d(conv_filters2, conv_filters2, 3, padding=1),
        # bn
        nn.ReLU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters2, conv_filters4, 3, padding=1),
        # bn
        nn.ReLU(),]

        if self.bn:
            conv_layers.insert(1,nn.BatchNorm2d(conv_filters))
            conv_layers.insert(4, nn.BatchNorm2d(conv_filters))
            conv_layers.insert(8, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(11, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(15, nn.BatchNorm2d(conv_filters4))


        self.conv = SequentialWithIntermediates(*conv_layers)

        self.linear_size = h * w * (conv_filters * 4) // 4 // 4

        fc_layers=[nn.Linear(self.linear_size, fc_filters),
            # nn.BatchNorm1d(fc_filters),
            nn.ReLU(),
            nn.Linear(fc_filters, num_classes)]
        if self.bn:
            fc_layers.insert(1,nn.BatchNorm1d(fc_filters))
        self.fc = SequentialWithIntermediates(*fc_layers)

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
        if self.bn:
            conv_layer_names.insert(1,  "bn1")
            conv_layer_names.insert(4,  "bn2")
            conv_layer_names.insert(8,  "bn3")
            conv_layer_names.insert(11, "bn4")
            conv_layer_names.insert(15, "bn5")

        fc_layer_names = ["fc1", "fc1act", "fc2", "softmax"]

        if self.bn:
            fc_layer_names.insert(1, "bn6")

        return conv_layer_names + fc_layer_names


class SimpleConvBN(SimpleConv):
    def __init__(self, input_shape, num_classes, conv_filters=32, fc_filters=128):
        super(SimpleConvBN, self).__init__(input_shape, num_classes, conv_filters=conv_filters, fc_filters=fc_filters,bn=True)
        self.name = self.__class__.__name__