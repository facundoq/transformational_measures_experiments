from typing import List,Tuple,Sized,Iterable,Iterator
import numpy as np

import abc
class Transformation:

    @abc.abstractmethod
    def __call__(self, x:np.ndarray)->np.ndarray:
        pass

class TransformationSet(Sized, Iterable[Transformation]):

    @abc.abstractmethod
    def __iter__(self)->Iterator[Transformation]:
        pass

    def __len__(self):
        return len(list(self.__iter__()))

    @abc.abstractmethod
    def id(self):
        pass

