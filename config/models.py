import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Tuple,Dict,Callable
from torch import optim,nn
import datasets
import models
import transformation_measure as tm
import itertools
import config




import abc
class ModelConfig(abc.ABC):

    @abc.abstractmethod
    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool)->(tm.ObservableLayersModule, Optimizer):
        pass

    @abc.abstractmethod
    def id(self) -> str:
        pass

    def __repr__(self)->str:
        return self.id()

    def default_optimizer(self,model:nn.Module,cuda:bool,lr=0.001,wd=1e-9):
        if cuda:
            model = model.cuda()
        # parameters = training.add_weight_decay(model.named_parameters(), wd)
        parameters = model.parameters()
        optimizer = optim.AdamW(parameters, lr=lr)
        # rp = optim.lr_scheduler.ReduceLROnPlateau(optimizer , patience=2, cooldown=0)
        return model, optimizer

    @property
    def name(self)->str:
        idstr = self.id()
        # return id
        try:
            end_of_name_index=idstr.index("(")
            return idstr[:end_of_name_index]
        except ValueError:
            return idstr

class TIPoolingSimpleConvConfig(ModelConfig):

    @classmethod
    def for_dataset(cls, dataset: str, bn: bool,t:tm.TransformationSet):
        # TODO different from SimpleConv, half the filters
        conv = {"mnist": 16, "cifar10": 64, "fashion_mnist": 32,"lsa16":16,"rwth":32}
        fc = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128,"lsa16":64,"rwth":128}

        return TIPoolingSimpleConvConfig(t, conv[dataset], fc[dataset], bn)

    def __init__(self, transformations:tm.TransformationSet, conv=32, fc=128, bn=False):
        self.transformations=transformations
        self.conv=conv
        self.fc=fc
        self.bn=bn

    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool):
        model = models.TIPoolingSimpleConv(input_shape, num_classes, self.transformations, self.conv, self.fc, self.bn)
        model, optimizer = self.default_optimizer(model,cuda)
        return model,optimizer

    def id(self):
        return f"{models.TIPoolingSimpleConv.__name__}(t={self.transformations.id()},conv={self.conv},fc={self.fc},bn={self.bn})"


class SimpleConvConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,dataset:str,bn:bool=False,k=3, activation=models.ActivationFunction.ELU,max_pooling=True):
        conv = {"mnist": 32, "cifar10": 128, "fashion_mnist": 64,"lsa16":128,"rwth":128}
        fc = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128,"lsa16":64,"rwth":256}

        return SimpleConvConfig(conv=conv[dataset], fc=fc[dataset],bn=bn,k=k,activation=activation,max_pooling=max_pooling)

    def __init__(self, conv=32, fc=128, bn=False, k=3, activation=models.ActivationFunction.ELU,max_pooling=True):
        self.conv=conv
        self.fc=fc
        self.bn=bn
        self.activation_function=activation
        self.k=k
        self.max_pooling=max_pooling

    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool):
        model = models.SimpleConv(input_shape, num_classes, self.conv, self.fc, self.bn, self.k, self.activation_function,self.max_pooling)
        model, optimizer = self.default_optimizer(model,cuda,lr=5e-4)
        return model,optimizer

    def id(self):
        params=[]
        if self.bn:
            params.append(f"bn={self.bn}")
        if self.k!=3:
            params.append(f"k={self.k}")
        if self.activation_function != models.ActivationFunction.ELU:
            params.append(f"act={self.activation_function.value}")
        if not self.max_pooling:
            params.append(f"mp={self.max_pooling}")
        params = ",".join(params)
        if len(params)>0:
            params=f"({params})"

        return f"{models.SimpleConv.__name__}{params}"
        # TODO rerun all
        #return f"SimpleConv(conv={self.conv},fc={self.fc},bn={self.bn},k={self.k},act={self.activation_function.value},mp={self.mp})"

class AllConvolutionalConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,dataset:str,bn:bool=False):
        conv = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32,"lsa16":64,"rwth":64}
        return AllConvolutionalConfig(conv=conv[dataset], dropout=False, bn=bn)

    def __init__(self, conv=32, dropout=False, bn=False):
        self.conv=conv
        self.bn=bn

    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool):
        model = models.AllConvolutional(input_shape, num_classes,self.conv, self.bn)
        model, optimizer = self.default_optimizer(model,cuda,lr=0.01)
        return model,optimizer

    def id(self):
        return f"{models.AllConvolutional.__name__}(conv={self.conv},bn={self.bn})"

class FFNetConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,dataset:str,bn:bool=False):
        h1 = {"mnist": 64, "cifar10": 256, "fashion_mnist": 128,"lsa16":64,"rwth":256}
        h2 = {"mnist": 32, "cifar10": 128, "fashion_mnist": 64,"lsa16":32,"rwth":128}
        return FFNetConfig(h1=h1[dataset], h2=h2[dataset],bn=bn)

    def __init__(self, h1:int,h2:int, bn=False):
        self.h1=h1
        self.h2=h2
        self.bn=bn

    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool):
        model = models.FFNet(input_shape, num_classes,self.h1,self.h2, self.bn)
        model, optimizer = self.default_optimizer(model,cuda)
        return model,optimizer

    def id(self):
        return f"{models.FFNet.__name__}(h1={self.h1},h2={self.h2},bn={self.bn})"

class VGG16DConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,dataset:str,bn:bool=False):
        conv = {"mnist": 16, "cifar10": 64,"lsa16":64,"rwth":64, }
        fc = {"mnist": 64, "cifar10": 512,"lsa16":32,"rwth":512, }
        return VGG16DConfig(conv=conv[dataset], fc=fc[dataset],bn=bn)

    def __init__(self, conv=32, fc=128, bn=False):
        self.conv=conv
        self.fc=fc
        self.bn=bn

    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool):
        model = models.VGG16D(input_shape, num_classes,self.conv, self.fc, self.bn)
        model, optimizer = self.default_optimizer(model,cuda,lr=0.0001)
        return model,optimizer

    def id(self):
       return f"{models.VGG16D.__name__}(conv={self.conv},fc={self.fc},bn={self.bn})"

class ResNetConfig(ModelConfig):

    @classmethod
    def for_dataset(cls,dataset:str,bn:bool=False,v="18"):
        return ResNetConfig(bn=bn,v=v)

    def __init__(self,  bn=False,v="18"):
        self.bn=bn
        self.v=v

    def make_model(self, input_shape:[int, int, int], num_classes:int, cuda:bool):
        if self.v == "18":
            model = models.ResNet18(input_shape, num_classes, self.bn)
            model, optimizer = self.default_optimizer(model,cuda,lr=5e-4)
            return model,optimizer
        else:
            raise ValueError(f"Invalid ResNet version {self.v}")

    def id(self):
        return f"{models.ResNet.__name__}(v={self.v},bn={self.bn})"

from models.simple_conv import ActivationFunction
def all_models()->[ModelConfig]:
    kernel_sizes=[3,5,7]
    mps=[False,True]
    bns=[False,True]
    activation_functions=list(ActivationFunction)
    combinations = itertools.product(datasets.names,bns)
    simple_conv_combinations=itertools.product(mps,kernel_sizes,activation_functions)
    models_configs:[ModelConfig]=[]


    for dataset,bn in combinations:
        for mp,k,af in simple_conv_combinations:
            models_configs.append(SimpleConvConfig.for_dataset(dataset,bn,k,af,mp))
        models_configs.append(AllConvolutionalConfig.for_dataset(dataset, bn))
        models_configs.append(VGG16DConfig.for_dataset(dataset, bn))
        models_configs.append(ResNetConfig.for_dataset(dataset, bn))
        models_configs.append(FFNetConfig.for_dataset(dataset, bn))

        for t in config.all_transformations():
            models_configs.append(TIPoolingSimpleConvConfig.for_dataset(dataset, bn, t))

    models_configs = { m.id():m for m in models_configs}
    return models_configs




def get_epochs(model_config: ModelConfig, dataset: str, t: tm.TransformationSet) -> int:
    model = model_config.name
    if model.startswith('SimpleConv'):
        epochs = {'cifar10': 25, 'mnist': 5, 'fashion_mnist': 12,"lsa16":25,"rwth":25}
    elif model.startswith('SimpleConvLargeKernel'):
        epochs = {'cifar10': 25, 'mnist': 5, 'fashion_mnist': 12,"lsa16":25,"rwth":25}
    elif model == models.SimplestConv.__name__:
        epochs = {'cifar10': 40, 'mnist': 5, 'fashion_mnist': 12,"lsa16":25,"rwth":25}
    elif model == models.TIPoolingSimpleConv.__name__ :
        epochs = {'cifar10': 20, 'mnist': 5, 'fashion_mnist': 12,"lsa16":25,"rwth":25}
    elif model == models.AllConvolutional.__name__ :
        epochs = {'cifar10': 40, 'mnist': 30, 'fashion_mnist': 12,"lsa16":25,"rwth":25}
    # elif model == models.VGGLike.__name__ or model == models.VGGLikeBN.__name__:
    #     epochs = {'cifar10': 50, 'mnist': 40, 'fashion_mnist': 12, }
    elif model == models.VGG16D.__name__ :
        epochs = {'cifar10': 30, 'mnist': 10, 'fashion_mnist': 12,"lsa16":25,"rwth":25, }
    elif model == models.ResNet.__name__:
        epochs = {'cifar10': 40, 'mnist': 10, 'fashion_mnist': 12,"lsa16":25,"rwth":25}
    elif model == models.FFNet.__name__:
        epochs = {'cifar10': 30, 'mnist': 10, 'fashion_mnist': 8,"lsa16":25,"rwth":15}
    else:
        raise ValueError(f"Model \"{model}\" does not exist.")

    # scale by log(#transformations)
    m = len(t)
    if m > np.e:
        factor = 1.1 * np.log(m)
    else:
        factor = 1

    # if not model_config.bn:
    #     factor *= 1.5

    return int(epochs[dataset] * factor)

def min_accuracy(model: str, dataset: str) -> float:
    min_accuracies = {"mnist": .90, "cifar10": .5,"lsa16":.6,"rwth":.45}
    min_accuracy = min_accuracies[dataset]

    if model == models.FFNet.__name__:
        if dataset == "mnist" :
            min_accuracy = 0.85
        elif dataset == "cifar10":
            min_accuracy = 0.45
        else:
            min_accuracy=min_accuracy*0.8

    return min_accuracy
