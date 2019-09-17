from pytorch import training
import models

from torch import optim,nn
from torch.optim.optimizer import Optimizer
from typing import Tuple
import datasets

class ExperimentModel:
    def __init__(self, model, parameters, optimizer):
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer

def get_model_names():
    return models.names


def get_model(name:str,dataset:datasets.ClassificationDataset,use_cuda:bool)->Tuple[nn.Module,Optimizer]:

    def setup_model(model,lr,wd)->Optimizer:
        if use_cuda:
            model = model.cuda()
        parameters = training.add_weight_decay(model.named_parameters(), wd)
        optimizer = optim.AdamW(parameters, lr=lr)
        #rp = optim.lr_scheduler.ReduceLROnPlateau(optimizer , patience=2, cooldown=0)
        return optimizer

    def ffnet(bn=False)->Tuple[nn.Module,Optimizer]:
        fc1 = {"mnist": 64, "cifar10": 256, "fashion_mnist": 128}
        fc2= {"mnist": 32, "cifar10": 128, "fashion_mnist": 64}
        if bn:
            klass = models.FFNetBN
        else:
            klass = models.FFNet
        model = klass(dataset.input_shape, dataset.num_classes,
                                  h1=fc1[dataset.name], h2=fc2[dataset.name])
        optimizer=setup_model(model,0.001,1e-9)
        return model, optimizer


    def simple_conv(bn=False)->Tuple[nn.Module,Optimizer]:
        conv_filters = {"mnist": 32, "cifar10": 64, "fashion_mnist": 64}
        fc_filters = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128}
        if bn:
            klass = models.SimpleConvBN
        else:
            klass = models.SimpleConv
        model = klass(dataset.input_shape, dataset.num_classes,
                                  conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
        optimizer=setup_model(model,0.001,1e-9)

        return model, optimizer

    def all_convolutional(bn=False)->Tuple[nn.Module,Optimizer]:
        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32}

        if bn:
            klass = models.AllConvolutionalBN
        else:
            klass = models.AllConvolutional

        model = klass(dataset.input_shape, dataset.num_classes,
                                        filters=filters[dataset.name])
        optimizer=setup_model(model,1e-3,1e-13)
        return model, optimizer

    def vgglike(bn=False)->Tuple[nn.Module,Optimizer]:
        filters = {"mnist": 16, "cifar10": 64,}
        fc= {"mnist": 64,  "cifar10": 128, }
        if bn:
            klass = models.VGGLikeBN
        else:
            klass = models.VGGLike
        model = klass(dataset.input_shape, dataset.num_classes,
                               conv_filters=filters[dataset.name], fc=fc[dataset.name])

        optimizer=setup_model(model,0.00001,1e-13)
        return model, optimizer

    def resnet(bn=False)->Tuple[nn.Module,Optimizer]:
        resnet_version = {"mnist": models.ResNet18,
                          "cifar10": models.ResNet50,
                          "fashion_mnist": models.ResNet34,
                          }
        model = resnet_version[dataset.name](dataset.input_shape, dataset.num_classes)
        optimizer=setup_model(model,0.00001,1e-13)
        return model, optimizer

    all_models = {models.SimpleConv.__name__: simple_conv,
                  models.AllConvolutional.__name__: all_convolutional,
                  models.VGGLike.__name__: vgglike,
                  models.ResNet.__name__: resnet,
                  models.FFNet.__name__:ffnet,

                  models.FFNetBN.__name__: lambda :ffnet(bn=True),
                  models.AllConvolutionalBN.__name__: lambda: all_convolutional(bn=True),
                  models.VGGLikeBN.__name__: lambda: vgglike(bn=True),
                  models.SimpleConvBN.__name__: lambda: simple_conv(bn=True),

                  }
    if name not in all_models :
        raise ValueError(f"Model \"{name}\" does not exist. Choices: {', '.join(all_models .keys())}")
    return all_models [name]()

import transformation_measure as tm
import numpy as np


