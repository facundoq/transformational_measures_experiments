from typing import List,Tuple,Sized,Iterable,Iterator
import numpy as np
import torch
import abc
class Transformation:

    @abc.abstractmethod
    def __call__(self, x:np.ndarray)->np.ndarray:
        pass


    def apply_pytorch(self,x:torch.FloatTensor):
        raise NotImplementedError(f"Transformation {self.__class__.__name__} does not have a PyTorch implementation")


class TransformationSet(Sized, Iterable[Transformation]):

    @abc.abstractmethod
    def __iter__(self)->Iterator[Transformation]:
        pass

    def __len__(self):
        return len(list(self.__iter__()))

    @abc.abstractmethod
    def id(self):
        pass


class IdentityTransformation(Transformation):

    def __call__(self, x:np.ndarray)->np.ndarray:
        return x

    def apply_pytorch(self,x:torch.FloatTensor):
        return x
