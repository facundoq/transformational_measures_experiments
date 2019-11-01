import numpy as np
from enum import Enum

class ConvAggregation(Enum):
    mean = "mean"
    max = "max"
    min = "min"
    sum = "sum"
    none = "none"
    def functions(self):
        return {ConvAggregation.mean: np.nanmean
        , ConvAggregation.sum: np.nansum
        , ConvAggregation.min: np.nanmin
        , ConvAggregation.max: np.nanmax
        }

    def apply(self, layer:np.ndarray) -> np.ndarray:
        '''

        :param layer:  a 4D np array
        :param conv_aggregation_function:
        :return:
        '''


        if self == ConvAggregation.none:
            return layer

        if not self in list(ConvAggregation):
            raise ValueError(
                f"Invalid aggregation function: {self}. Options: {list(ConvAggregation)}")

        function = self.functions()[self]
        n, c, h, w = layer.shape
        flat_activations = np.zeros((n, c))
        for i in range(n):
            flat_activations[i, :] = function(layer[i, :, :, :], axis=(1,2))

        return flat_activations


    def apply3D(self,layer:np.ndarray,) -> np.ndarray:
        '''

        :param layer:  a 3D np array
        :param conv_aggregation_function:
        :return:
        '''
        if self == ConvAggregation.none:
            return layer

        if not self in list(ConvAggregation):
            raise ValueError(
                f"Invalid aggregation function: {self}. Options: {list(ConvAggregation)}")
        function = self.functions()[self]
        flat_activations = function(layer,axis=(1,2))

        return flat_activations