from pytorch import training
import models

from torch import optim,nn
from torch.optim.optimizer import Optimizer
from typing import Tuple,Dict,Callable

import datasets
from transformation_measure import ObservableLayersModule

class ExperimentModel:
    def __init__(self, model, parameters, optimizer):
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer


def model_maker(model_function,**kwargs)->Callable[[datasets.ClassificationDataset,bool],Callable]:
    def make_model(dataset:datasets.ClassificationDataset,use_cuda:bool):
        return model_function(dataset,use_cuda,**kwargs)
    return make_model

def get_all_models()->Dict[str,Callable]:

    def setup_model(model:ObservableLayersModule,use_cuda:bool,lr:float,wd:float)->Optimizer:
        if use_cuda:
            model = model.cuda()
        parameters = training.add_weight_decay(model.named_parameters(), wd)
        optimizer = optim.AdamW(parameters, lr=lr)
        #rp = optim.lr_scheduler.ReduceLROnPlateau(optimizer , patience=2, cooldown=0)
        return optimizer

    def ffnet(dataset:datasets.ClassificationDataset,use_cuda:bool,bn=False)->(ObservableLayersModule,Optimizer):
        fc1 = {"mnist": 64, "cifar10": 256, "fashion_mnist": 128}
        fc2= {"mnist": 32, "cifar10": 128, "fashion_mnist": 64}
        if bn:
            klass = models.FFNetBN
        else:
            klass = models.FFNet
        model = klass(dataset.input_shape, dataset.num_classes,
                                  h1=fc1[dataset.name], h2=fc2[dataset.name])
        optimizer=setup_model(model,use_cuda,0.001,1e-9)
        return model, optimizer


    def simple_conv(dataset:datasets.ClassificationDataset,use_cuda:bool,bn=False)->(ObservableLayersModule,Optimizer):
        conv_filters = {"mnist": 32, "cifar10": 64, "fashion_mnist": 64}
        fc_filters = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128}
        if bn:
            klass = models.SimpleConvLargeKernel
        else:
            klass = models.SimpleConv
        model = klass(dataset.input_shape, dataset.num_classes,
                                  conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
        optimizer=setup_model(model,use_cuda,0.001,1e-9)

        return model, optimizer

    def simple_conv_large_kernel(dataset:datasets.ClassificationDataset,use_cuda:bool,kernel_size:int)->(ObservableLayersModule,Optimizer):
        conv_filters = {"mnist": 32, "cifar10": 64, "fashion_mnist": 64}
        fc_filters = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128}
        model = models.SimpleConvLargeKernel(dataset.input_shape, dataset.num_classes,
                                             conv_filters=conv_filters[dataset.name],
                                             fc_filters=fc_filters[dataset.name],kernel_size=kernel_size)
        optimizer=setup_model(model,use_cuda,0.001,1e-9)

        return model, optimizer

    def all_convolutional(dataset:datasets.ClassificationDataset,use_cuda:bool,bn=False)->(ObservableLayersModule,Optimizer):
        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32}

        if bn:
            klass = models.AllConvolutionalBN
        else:
            klass = models.AllConvolutional

        model :ObservableLayersModule = klass(dataset.input_shape, dataset.num_classes,
                                        filters=filters[dataset.name])
        optimizer=setup_model(model,use_cuda,1e-3,1e-13)
        return model, optimizer

    def vgglike(dataset:datasets.ClassificationDataset,use_cuda:bool,bn=False)->(ObservableLayersModule,Optimizer):
        filters = {"mnist": 16, "cifar10": 64,}
        fc= {"mnist": 64,  "cifar10": 128, }
        if bn:
            klass = models.VGGLikeBN
        else:
            klass = models.VGGLike
        model = klass(dataset.input_shape, dataset.num_classes,
                               conv_filters=filters[dataset.name], fc=fc[dataset.name])

        optimizer=setup_model(model,use_cuda,0.00001,1e-13)
        return model, optimizer

    def resnet(dataset:datasets.ClassificationDataset,use_cuda:bool,bn=False)->(ObservableLayersModule,Optimizer):
        if bn:
            resnet_version = {"mnist": models.ResNet18BN,
                          "cifar10": models.ResNet50BN,
                          "fashion_mnist": models.ResNet34BN,
                          }
        else:
            resnet_version = {"mnist": models.ResNet18,
                              "cifar10": models.ResNet50,
                              "fashion_mnist": models.ResNet34,
                              }

        model = resnet_version[dataset.name](dataset.input_shape, dataset.num_classes)
        optimizer=setup_model(model,use_cuda,0.0001,1e-13)
        return model, optimizer

    all_models = {models.SimpleConv.__name__: model_maker(simple_conv),
                  models.AllConvolutional.__name__: all_convolutional,
                  models.VGGLike.__name__: vgglike,
                  models.ResNet.__name__: resnet,
                  models.FFNet.__name__:ffnet,

                  models.FFNetBN.__name__: model_maker(ffnet,bn=True),
                  models.AllConvolutionalBN.__name__: model_maker(all_convolutional,bn=True),
                  models.VGGLikeBN.__name__: model_maker(vgglike,bn=True),
                  models.SimpleConvBN.__name__: model_maker(simple_conv,bn=True),
                  models.ResNetBN.__name__: model_maker(resnet,bn=True),
                  }

    for i in [3,5,7,9]:
        model_id=f"{models.SimpleConvLargeKernel.__name__}(k={i})"
        all_models[model_id]=  model_maker(simple_conv_large_kernel,kernel_size=i)

    return all_models

all_models = get_all_models()
model_names =all_models.keys()


def get_model(name:str,dataset:datasets.ClassificationDataset,use_cuda:bool)->(ObservableLayersModule,Optimizer):

    all_models = get_all_models()
    if name not in all_models.keys():
        raise ValueError(f"Model \"{name}\" does not exist. Choices: {', '.join(all_models .keys())}")
    return all_models[name](dataset,use_cuda)

