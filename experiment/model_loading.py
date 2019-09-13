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

    def ffnet()->Tuple[nn.Module,Optimizer]:
        fc1 = {"mnist": 64, "cifar10": 256, "fashion_mnist": 128}
        fc2= {"mnist": 32, "cifar10": 128, "fashion_mnist": 64}
        model = models.FFNet(dataset.input_shape, dataset.num_classes,
                                  h1=fc1[dataset.name], h2=fc2[dataset.name])
        optimizer=setup_model(model,0.001,1e-9)
        return model, optimizer


    def simple_conv()->Tuple[nn.Module,Optimizer]:
        conv_filters = {"mnist": 32, "cifar10": 64, "fashion_mnist": 64}
        fc_filters = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128}
        model = models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                  conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
        optimizer=setup_model(model,0.001,1e-9)

        return model, optimizer

    def all_convolutional()->Tuple[nn.Module,Optimizer]:
        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32}
        model = models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                        filters=filters[dataset.name])
        optimizer=setup_model(model,1e-3,1e-13)
        return model, optimizer

    def vgglike()->Tuple[nn.Module,Optimizer]:
        filters = {"mnist": 16, "cifar10": 64,}
        fc= {"mnist": 64,  "cifar10": 128, }
        model = models.VGGLike(dataset.input_shape, dataset.num_classes,
                               conv_filters=filters[dataset.name], fc=fc[dataset.name])
        optimizer=setup_model(model,0.00001,1e-13)
        return model, optimizer

    def resnet()->Tuple[nn.Module,Optimizer]:
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
                  }
    if name not in all_models :
        raise ValueError(f"Model \"{name}\" does not exist. Choices: {', '.join(all_models .keys())}")
    return all_models [name]()

import transformation_measure as tm
import numpy as np

def get_epochs(model:str,dataset:str, t:tm.TransformationSet)-> int:

    if model== models.SimpleConv.__name__:
        epochs={'cifar10':70,'mnist':5,'fashion_mnist':12}
    elif model== models.AllConvolutional.__name__:
        epochs={'cifar10':32,'mnist':15,'fashion_mnist':12}
    elif model== models.VGGLike.__name__:
        epochs={'cifar10':70,'mnist':15,'fashion_mnist':12,}
    elif model== models.ResNet.__name__:
        epochs={'cifar10':70,'mnist':15,'fashion_mnist':12}
    elif model == models.FFNet.__name__:
        epochs = {'cifar10': 10, 'mnist': 5, 'fashion_mnist': 8}
    else:
        raise ValueError(f"Model \"{model}\" does not exist. Choices: {', '.join(get_model_names())}")

    n=len(t)
    if n>np.e:
        factor=1.3*np.log(n)
    else:
        factor=1

    return int(epochs[dataset]*factor)
