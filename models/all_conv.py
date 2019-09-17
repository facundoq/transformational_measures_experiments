import torch.nn as nn
import torch.nn.functional as F

from transformation_measure import ObservableLayersModel

from models.util import SequentialWithIntermediates

class ConvBNAct(nn.Module, ObservableLayersModel):
    def __init__(self,in_filters,out_filters,kernel_size,stride=1,bn=False):
        super(ConvBNAct, self).__init__()
        if kernel_size==0:
            padding=0
        else:
            padding=1
        self.bn=bn

        c=nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        bn_layer=nn.BatchNorm2d(out_filters)
        r=nn.ReLU()
        if bn:
            self.model=SequentialWithIntermediates(c,bn_layer,r)
        else:
            self.model = SequentialWithIntermediates(c, r)

    def forward(self,x):
        return self.model.forward(x)

    def forward_intermediates(self,x):
        return self.model.forward_intermediates(x)
    def activation_names(self):
        if self.bn:
            return ["c","bn","relu"]
        else:
            return ["c", "relu"]



class AllConvolutional(nn.Module, ObservableLayersModel):
    def __init__(self, input_shape, num_classes,filters=96,dropout=False,bn=False):
        super(AllConvolutional, self).__init__()
        self.name = self.__class__.__name__
        self.bn=bn
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
        )

        self.class_conv = nn.Conv2d(filters2, num_classes, 1)


    def forward(self, x):
        x = self.convs.forward(x)
        x = self.class_conv.forward(x)
        pool_out = x.reshape(x.size(0), x.size(1), -1).mean(-1)
        return F.log_softmax(pool_out,dim=1)

    def forward_intermediates(self, x):
        outputs = []
        for module in self.convs._modules.values():
            x,intermediates = module.forward_intermediates(x)
            outputs+=intermediates
        x= self.class_conv(x)
        outputs.append(x)

        pool_out = x.reshape(x.size(0), x.size(1), -1).mean(-1)
        outputs.append(pool_out)

        log_probabilities=F.log_softmax(pool_out,dim=1)
        outputs.append(log_probabilities)

        return log_probabilities,outputs

    def layer_names(self):
        return [f"c{i}" for i in range(8)]+["cc"]


    def activation_names(self):
        names=[]
        for i,module in enumerate(self.convs._modules.values()):

            module_names=module.activation_names()
            module_names=[f"{name}_{i}" for name in module_names]
            names+=module_names
        names+=["class_conv","avgp","smax"]
        return names

class AllConvolutionalBN(AllConvolutional):
    def __init__(self, input_shape, num_classes,filters=96,dropout=False):
        super(AllConvolutionalBN, self).__init__(input_shape, num_classes,filters=filters,dropout=dropout,              bn=True)
        self.name = self.__class__.__name__
