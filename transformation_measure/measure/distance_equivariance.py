from .base import Measure,MeasureFunction,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.running_stats import RunningMeanAndVariance,RunningMean
from typing import List
from enum import Enum

from sklearn.metrics.pairwise import euclidean_distances
import sklearn

from .distance import DistanceAggregation

import transformation_measure as tm

class DistanceSameEquivarianceMeasure(Measure):
    def __init__(self, measure_function:MeasureFunction, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.measure_function=measure_function
        self.distance_aggregation=distance_aggregation

    def __repr__(self):
        return f"DSEM(f={self.measure_function.value},da={self.distance_aggregation.value})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.activation_names()
        n_intermediates = len(layer_names)
        transformations= activations_iterator.get_transformations()
        mean_variances_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in activations_iterator.samples_first():
            # activations has the activations for all the transformations

            for j, layer_activations in enumerate(activations):
                self.inverse_trasform_feature_maps(layer_activations,transformations)
                layer_measure= self.measure_function.apply(layer_activations)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)

    def inverse_trasform_feature_maps(self,layer_activations:[np.ndarray],transformations:tm.TransformationSet)->[np.ndarray]:
        transformations = list(transformations)
        inverses = [t.inverse() for t in transformations]
        for layer in layer_activations:
            if len(layer.shape)==4:
                for i,inverse in enumerate(inverses):
                    layer[i,:]= inverse(layer[i,:])

