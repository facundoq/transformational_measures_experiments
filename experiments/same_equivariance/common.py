from .base import SameEquivarianceExperiment
from ..common import *


def get_ylim_normalized(measure: tm.numpy.NumpyMeasure):
    # TODO dict
    if measure.__class__ == tm.numpy.NormalizedDistanceSameEquivariance:
        return 8
    elif measure.__class__ == tm.numpy.NormalizedVarianceSameEquivariance:
        return 8
    elif measure.__class__ == tm.numpy.NormalizedVarianceInvariance:
        return 1.4
    elif measure.__class__ == tm.numpy.NormalizedDistanceInvariance:
        return 1.4
    else:
        raise ValueError(measure)


def simple_conv_sameequivariance_activation_filter(m:ObservableLayersModule,name:str): return m.activation_names().index(name) < 6