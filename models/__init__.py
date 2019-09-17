from models.util import *
from models.simple_conv import *
from models.vgg_like import *
from models.all_conv import *
from models.resnet import *
from models.ff import *

names= [SimpleConv.__name__, VGGLike.__name__, ResNet.__name__, AllConvolutional.__name__, FFNet.__name__,
        SimpleConvBN.__name__, VGGLikeBN.__name__, ResNet.__name__, AllConvolutionalBN.__name__, FFNetBN.__name__
        ]

