import abc
from transformation_measure.iterators.activations_iterator import ActivationsIterator

from .stratified import StratifiedMeasure

import numpy as np
from typing import Dict, List, Tuple
from .utils import RunningMean
from .layer_transformation import ConvAggregation,apply_aggregation_function

class   MeasureResult:
    def __init__(self,layers:List[np.ndarray],source:str):
        self.layers=layers
        self.source=source

    def __repr__(self):
        return f"MeasureResult {self.source}"

    def all_1d(self):
        return np.any([ len(l.shape)==1 for l in self.layers])

    def average_per_layer(self) -> np.ndarray:

        result = []
        for layer in self.layers:
            layer=layer[:]
            layer_average=layer[np.isfinite(layer)].mean()
            result.append(layer_average)
        return np.array(result)

    def weighted_global_average(self):
        return self.average_per_layer().mean()

    def global_average(self)-> float:
        rm = RunningMean()
        for layer in self.layers:
                for act in layer[:]:
                    if np.isfinite(act):
                        rm.update(act)
        return rm.mean()

    def collapse_convolutions(self,conv_aggregation_function:ConvAggregation):
        results=[]
        for layer in self.layers:
            # if conv average out spatial dims
            if len(layer.shape) == 4:
                flat_activations=apply_aggregation_function(layer,conv_aggregation_function)
                assert (len(flat_activations.shape) == 2)
            else:
                flat_activations = layer.copy()
            results.append(flat_activations)

        return MeasureResult(results,f"{self.source}_conv_agg_{conv_aggregation_function}")


from enum import Enum
class MeasureFunction(Enum):
    var = "var"
    std = "std"
    meanabs = "meanabs"
    mean = "mean"


class Measure:
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abc.abstractmethod
    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        '''

        '''
        pass