import torch.nn as nn
import torch.nn.functional as F
from transformation_measure import ObservableLayersModule
from transformation_measure import TransformationSet,Transformation
import torch
from models import SequentialWithIntermediates

from models.util import Flatten

class TIPoolingSimpleConvConfig():
    def __init__(self,transformations:TransformationSet,conv_filters=32,fc_filters=128,bn=False):
        self.transformations=transformations
        self.conv_filters=conv_filters
        self.fc_filters=fc_filters
        self.bn=False

class TIPoolingSimpleConv(ObservableLayersModule):
    def __init__(self, input_shape, num_classes, transformations:TransformationSet,conv_filters=32, fc_filters=128,bn=False):
        super().__init__()
        self.name = self.__class__.__name__
        self.bn = bn
        h, w, channels = input_shape

        self.transformations=transformations
        self.transformations.set_input_shape(input_shape)
        self.transformations.set_pytorch(True)
        self.transformations.set_cuda(torch.cuda.is_available())

        conv_filters2=conv_filters*2
        conv_filters4 = conv_filters * 4
        conv_layers=[nn.Conv2d(channels, conv_filters, 3, padding=1),
        #bn
        nn.ELU(),
        nn.Conv2d(conv_filters, conv_filters, 3, padding=1),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters, conv_filters2, 3, padding=1),
        # bn
        nn.ELU(),
        nn.Conv2d(conv_filters2, conv_filters2, 3, padding=1),
        # bn
        nn.ELU(),
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(conv_filters2, conv_filters4, 3, padding=1),
        # bn
        nn.ELU(),
        ]

        if self.bn:
            conv_layers.insert(1,nn.BatchNorm2d(conv_filters))
            conv_layers.insert(4, nn.BatchNorm2d(conv_filters))
            conv_layers.insert(8, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(11, nn.BatchNorm2d(conv_filters2))
            conv_layers.insert(15, nn.BatchNorm2d(conv_filters4))


        self.conv = SequentialWithIntermediates(*conv_layers)

        self.linear_size = h * w * (conv_filters4) // 4 // 4

        fc_layers=[
            nn.Linear(self.linear_size, fc_filters),
            # nn.BatchNorm1d(fc_filters),
            nn.ELU(),
            nn.Linear(fc_filters, num_classes),
            nn.LogSoftmax(dim=-1),
            ]
        if self.bn:
            fc_layers.insert(2,nn.BatchNorm1d(fc_filters))
        self.fc = SequentialWithIntermediates(*fc_layers)


    def forward(self, x):
        results=[]
        for t in self.transformations:
            transformed_x = t(x)
            feature_maps = self.conv(transformed_x)
            flattened_feature_maps = feature_maps.view(feature_maps.shape[0], -1)
            results.append(flattened_feature_maps)
        x=torch.stack(results,dim=1)
        x, _ =x.max(dim=1)
        x = self.fc(x)
        return x

    def forward_intermediates(self, x)->(object,[]):
        results = []
        conv_intermediates = []

        for t in self.transformations:
            transformed_x = t(x)
            feature_maps,intermediates = self.conv.forward_intermediates(transformed_x)
            conv_intermediates.extend(intermediates)
            flattened_feature_maps = feature_maps.view(feature_maps.shape[0], -1)
            results.append(flattened_feature_maps)


        x = torch.stack(results, dim=1)

        pooled, _ = x.max(dim=1)

        x, fc_activations = self.fc.forward_intermediates(pooled)
        return x, conv_intermediates+[pooled]+fc_activations

    def layer_before_pooling_each_transformation(self)->int:
        return len(self.original_conv_names())

    def original_conv_names(self)->[str]:
        return self.conv.activation_names()

    def fc_names(self)->[str]:
        return self.fc.activation_names()

    def activation_names(self):
        conv_names=[]
        original_conv_names=self.original_conv_names()
        for i in range(len(self.transformations)):
            t_names = [f"t{i:03}_{n}" for n in original_conv_names]
            conv_names.extend(t_names)

        return conv_names+["Pool+Flatten"]+self.fc.activation_names()
