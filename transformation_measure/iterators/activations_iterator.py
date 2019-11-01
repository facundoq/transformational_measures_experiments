import abc
from typing import List

import transformation_measure as tm

class ActivationsIterator(abc.ABC):
    """
       Iterate over the activations of a network, varying the samples and transformations
       In both orders
    """

    @abc.abstractmethod
    def get_transformations(self)->tm.TransformationSet:
        pass

    @abc.abstractmethod
    def transformations_first(self):
        pass

    @abc.abstractmethod
    def samples_first(self):
        pass

    @abc.abstractmethod
    def activation_names(self) -> List[str]:
        pass





