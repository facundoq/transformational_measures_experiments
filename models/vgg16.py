import torch.nn as nn
from models import Flatten
from models import SequentialWithIntermediates
from transformation_measure import  ObservableLayersModule

class ConvBNRelu(ObservableLayersModule):

    def __init__(self,input:int,output:int,bn:bool):
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

def block(filters_in:int,feature_maps:int,n_conv:int,bn:bool):
    '''
    A block of the VGG model, consists of :param n_conv convolutions followed by a 2x2 MaxPool

    :param filters_in:
    :param feature_maps:
    :param n_conv:
    :param bn:
    :return:
    '''
    layers=[]
    for i in range(n_conv):
        layers.append(ConvBNRelu(filters_in,feature_maps,bn))
        filters_in=feature_maps
    layers.append(nn.MaxPool2d(2, 2))
    return layers


class VGG16D(ObservableLayersModule):

    def __init__(self, input_shape, num_classes, conv_filters=64, fc_filters=4096, bn=False):
        super().__init__()
        self.name = self.__class__.__name__
        self.bn = bn
        if self.bn:
            self.name+="BN"

        h, w, channels = input_shape

        # list of conv layers
        conv_layers=[]
        # Configuration, depends on conv_filters
        convs_per_block= [2,2,3,3]
        feature_maps = [conv_filters,conv_filters*2,conv_filters*4,conv_filters*8]
        # end config
        # Create blocks of layers
        input_feature_maps=channels
        for c,f in zip(convs_per_block,feature_maps):
            conv_layers+=block(input_feature_maps,f,c,bn)
            input_feature_maps=f

        # Calculate flattened output size
        max_pools=len(convs_per_block)
        hf, wf = h // (2 ** max_pools), w // (2 ** max_pools)
        flattened_output_size = hf * wf * input_feature_maps


        dense_layers=[
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened_output_size, fc_filters),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_filters, fc_filters),
            nn.ReLU(),
            nn.Linear(fc_filters, num_classes),
            nn.LogSoftmax(dim=-1),
        ]

        conv_layers = SequentialWithIntermediates(*conv_layers)
        dense_layers = SequentialWithIntermediates(*dense_layers)
        self.layers=SequentialWithIntermediates(conv_layers,dense_layers)

    def forward(self, x):
        return self.layers(x)

    def forward_intermediates(self,x)->(object,[]):
        return self.layers.forward_intermediates(x)

    def activation_names(self):
        return self.layers.activation_names()



class VGG16DBN(VGG16D):
    def __init__(self, input_shape, num_classes,conv_filters,fc):
        super().__init__(input_shape, num_classes,conv_filters,fc,bn=True)
        self.name = self.__class__.__name__