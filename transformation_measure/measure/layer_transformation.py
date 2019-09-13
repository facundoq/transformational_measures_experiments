import numpy as np
from enum import Enum

class ConvAggregation(Enum):
    mean = "mean"
    max = "max"
    min = "min"
    sum = "sum"
    none = "none"



functions={ConvAggregation.mean : np.nanmean
               ,ConvAggregation.sum : np.nansum
               ,ConvAggregation.min : np.nanmin
               ,ConvAggregation.max : np.nanmax
               }

def apply_aggregation_function(layer:np.ndarray,conv_aggregation_function:ConvAggregation) -> np.ndarray:
    '''

    :param layer:  a 4D np array
    :param conv_aggregation_function:
    :return:
    '''
    if conv_aggregation_function == ConvAggregation.none:
        return layer

    if not conv_aggregation_function in functions.keys():
        raise ValueError(
            f"Invalid aggregation function: {conv_aggregation_function}. Options: {list(ConvAggregation)}")
    function = functions[conv_aggregation_function]

    n, c, h, w = layer.shape
    flat_activations = np.zeros((n, c))

    #TODO flat_activations =  function(layer, axis=(2,3))

    for i in range(n):
        flat_activations[i, :] = function(layer[i, :, :, :], axis=(1,2))

    return flat_activations


def apply_aggregation_function3D(layer:np.ndarray,conv_aggregation_function:ConvAggregation) -> np.ndarray:
    '''

    :param layer:  a 4D np array
    :param conv_aggregation_function:
    :return:
    '''
    if conv_aggregation_function == ConvAggregation.none:
        return layer

    if not conv_aggregation_function in functions.keys():
        raise ValueError(
            f"Invalid aggregation function: {conv_aggregation_function}. Options: {list(ConvAggregation)}")
    function = functions[conv_aggregation_function]

    # c, h, w = layer.shape
    # flat_activations = np.zeros(c)
    #
    # for i in range(c):
    #     flat_activations[i] = function(layer[i, :, :])

    flat_activations = function(layer,axis=(1,2))

    return flat_activations