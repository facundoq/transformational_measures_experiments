import torch.nn as nn

from models.util import SequentialWithIntermediates
from transformational_measures.pytorch import ObservableLayersModule


class ConvBNAct(ObservableLayersModule):
    def __init__(self,in_filters,out_filters,kernel_size,stride=1,bn=False):
        super(ConvBNAct, self).__init__()
        if kernel_size==0:
            padding=0
        else:
            padding=1
        self.bn=bn

        c=nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        bn_layer=nn.BatchNorm2d(out_filters)
        r=nn.ELU()
        if bn:
            self.model = SequentialWithIntermediates(c,bn_layer,r)
        else:
            self.model = SequentialWithIntermediates(c, r)

    def forward(self,x):
        return self.model.forward(x)

    def forward_intermediates(self,x):
        return self.model.forward_intermediates(x)
    def activation_names(self)->[str]:
        return self.model.activation_names()




class AllConvolutional( ObservableLayersModule):
    def __init__(self, input_shape, num_classes,filters=96,dropout=False,bn=False):
        super(AllConvolutional, self).__init__()
        self.name = self.__class__.__name__
        self.bn = bn

        h,w,c=input_shape
        filters2=filters*2
        self.dropout=dropout
        self.convs=SequentialWithIntermediates(
              ConvBNAct(c, filters, 3, bn=bn)
             ,ConvBNAct(filters, filters, 3, bn=bn)
             ,ConvBNAct(filters, filters, 3, stride=2, bn=bn)
             ,ConvBNAct(filters, filters2, 3, bn=bn)
             ,ConvBNAct(filters2, filters2, 3, bn=bn)
             ,ConvBNAct(filters2, filters2, 3, stride=2, bn=bn)
             ,ConvBNAct(filters2, filters2, 3, bn=bn)
             ,ConvBNAct(filters2, filters2, 1, bn=bn)
             ,nn.Conv2d(filters2, num_classes, 1)
             ,PoolOut()
             ,nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.convs(x)

    def forward_intermediates(self, x):
        return self.convs.forward_intermediates(x)

    def activation_names(self)->[str]:
        return self.convs.activation_names()

class PoolOut(nn.Module):

    def forward(self, x):
        return x.reshape(x.size(0), x.size(1), -1).mean(-1)

class AllConvolutionalBN(AllConvolutional):
    def __init__(self, input_shape, num_classes,filters=96,dropout=False):
        super(AllConvolutionalBN, self).__init__(input_shape, num_classes,filters=filters,dropout=dropout,              bn=True)
        self.name = self.__class__.__name__
