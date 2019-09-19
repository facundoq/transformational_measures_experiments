import torch
import torch.nn as nn
import torch.nn.functional as F

from transformation_measure import ObservableLayersModule
from models.util import SequentialWithIntermediates,Flatten,Add,GlobalAvgPool2d

class Block(ObservableLayersModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,bn=False):
        super(Block, self).__init__()
        layers=[nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                #bn1
                nn.ELU(),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                # bn2
                ]
        self.bn=bn
        if bn:
            layers.insert(1,nn.BatchNorm2d(planes))
            layers.insert(4, nn.BatchNorm2d(planes))

        main=SequentialWithIntermediates(*layers)

        shortcut_layers=[]
        self.use_shortcut=stride != 1 or in_planes != self.expansion*planes
        if self.use_shortcut:
            shortcut_layers=[nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),]
            if bn:
                shortcut_layers.append(nn.BatchNorm2d(self.expansion*planes))
        shortcut = SequentialWithIntermediates(*shortcut_layers)

        self.add=SequentialWithIntermediates(
            Add(main,shortcut)
            , nn.ELU()
            )


    def forward(self, x):
        return self.add(x)

    def forward_intermediates(self,x):
        return self.add.forward_intermediates(x)

    def activation_names(self)->[str]:
        return self.add.activation_names()

class Bottleneck(ObservableLayersModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,bn=False):
        super(Bottleneck, self).__init__()
        self.bn=bn

        conv=[nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
              #bn
              ,nn.ELU()
              ,nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
              #bn
            , nn.ELU()
              ,nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
              ,]
        conv=SequentialWithIntermediates(*conv)
        if self.bn:
            conv.insert(1, nn.BatchNorm2d(planes))
            conv.insert(4, nn.BatchNorm2d(planes))
            conv.insert(7, nn.BatchNorm2d(self.expansion*planes))

        shortcut = []
        self.use_shortcut=stride != 1 or in_planes != self.expansion*planes
        if self.use_shortcut:
            shortcut.append(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            if self.bn:
                shortcut.append(nn.BatchNorm2d(self.expansion*planes))
        shortcut=SequentialWithIntermediates(*shortcut)
        self.add = SequentialWithIntermediates(
            Add(conv, shortcut)
            , nn.ELU()
        )

    def forward(self, x):
        out=self.add(x)
        return out

    def forward_intermediates(self,x):
        out,outputs=self.add.forward_intermediates(x)
        return out,outputs

    def activation_names(self)->[str]:
        return self.add.activation_names()
        # conv_names = ["c0", "c0act", "c1", "c1act","c2"]
        # if self.bn:
        #     conv_names.insert(1, "bn0")
        #     conv_names.insert(4, "bn1")
        #     conv_names.insert(7, "bn2")
        # short_names=[]
        # if self.use_shortcut:
        #     short_names.append( "short_c0")
        #     if self.bn:
        #         short_names.append("short_bn1")
        # last_names=["c1+short", "act"]
        # return conv_names+short_names+last_names


class ResNet( ObservableLayersModule):
    def __init__(self, block, num_blocks,input_shape,num_classes,bn=False):
        super(ResNet, self).__init__()
        self.name = self.__class__.__name__
        self.bn=bn
        self.in_planes = 64
        h,w,c=input_shape
        layer0=[nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
                #bn
                ,nn.ELU()]
        if self.bn:
            layer0.insert(1,nn.BatchNorm2d(64))
        layer0=SequentialWithIntermediates(*layer0)

        layer1 = self._make_layer(block, 64, num_blocks[0], 1,bn)
        layer2 = self._make_layer(block, 128, num_blocks[1], 2,bn)
        layer3 = self._make_layer(block, 256, num_blocks[2], 2,bn)
        layer4 = self._make_layer(block, 512, num_blocks[3], 2,bn)
        conv = [layer0, layer1, layer2, layer3, layer4]

        self.conv=SequentialWithIntermediates(*conv)

        self.linear =SequentialWithIntermediates(
            GlobalAvgPool2d()
            ,Flatten()
            ,nn.Linear(512*block.expansion, num_classes)
            ,nn.LogSoftmax(dim=-1)
            )




    def _make_layer(self, block, planes, num_blocks, stride,bn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,bn))
            self.in_planes = planes * block.expansion
        return SequentialWithIntermediates(*layers)

    def forward(self, x):
        x=self.conv(x)
        # for layer in self.layers:
        #     x = layer(x)
        out = self.linear(x)
        return out

    def forward_intermediates(self,x):
        x,outputs=self.conv.forward_intermediates(x)
        # outputs=[]
        # for layer in self.layers:
        #     x,intermediates = layer.forward_intermediates(x)
        #     outputs=outputs+intermediates
        #     # print(x.shape)
        x,fc_outputs=self.linear.forward_intermediates(x)
        outputs+=fc_outputs
        return x,outputs

    def activation_names(self)->[str]:
        return self.conv.activation_names()+self.linear.activation_names()
        # names=[]
        # for i,l in enumerate(self.layers):
        #     for j,block in enumerate(l):
        #         l_names=[f"l{i}_b{j}_{n}" for n in block.activation_names()]
        #         names.extend(l_names)
        #
        # names+=["avgp","flatten","fc0","sm"]
        # return names





def ResNet18(input_shape,num_classes):
    return ResNet(Block, [2, 2, 2, 2], input_shape, num_classes)

def ResNet34(input_shape,num_classes):
    return ResNet(Block, [3, 4, 6, 3], input_shape, num_classes)

def ResNet50(input_shape,num_classes):
    return ResNet(Bottleneck, [3,4,6,3],input_shape,num_classes)

def ResNet101(input_shape,num_classes):
    return ResNet(Bottleneck, [3,4,23,3],input_shape,num_classes)

def ResNet152(input_shape,num_classes):
    return ResNet(Bottleneck, [3,8,36,3],input_shape,num_classes)

def ResNet18BN(input_shape,num_classes):
    return ResNetBN(Block, [2, 2, 2, 2], input_shape, num_classes)

def ResNet34BN(input_shape,num_classes):
    return ResNetBN(Block, [3, 4, 6, 3], input_shape, num_classes)

def ResNet50BN(input_shape,num_classes):
    return ResNetBN(Bottleneck, [3,4,6,3],input_shape,num_classes)

def ResNet101BN(input_shape,num_classes):
    return ResNetBN(Bottleneck, [3,4,23,3],input_shape,num_classes)

def ResNet152BN(input_shape,num_classes):
    return ResNetBN(Bottleneck, [3,8,36,3],input_shape,num_classes)


def test():
    net = ResNet18((32,32,3),10)
    y = net(torch.randn(1,3,32,32))
    print(y.size())


class ResNetBN(ResNet):

    def __init__(self, block, num_blocks, input_shape, num_classes):
        super(ResNetBN, self).__init__(block, num_blocks, input_shape, num_classes,bn=True)