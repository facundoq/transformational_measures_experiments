from .base import Measure,MeasureFunction,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.running_stats import RunningMeanAndVariance,RunningMean
from typing import List
from enum import Enum

from sklearn.metrics.pairwise import euclidean_distances
import sklearn



class DistanceAggregation(Enum):
    max = "max"
    mean = "mean"

    def apply(self,x:np.ndarray):
        l = len(x.shape)
        if l == 4:
            return self.apply_feature_maps(x)
        elif l == 2:
            return self.apply_features(x)
        else:
            raise ValueError(f"Activation shape not supported {x.shape}")

    def apply_feature_maps(self,x:np.ndarray):
        n,c,h,w=x.shape
        x = x.reshape((n,c,h*w))
        result=np.zeros(c)
        for i in range(c):
            d = euclidean_distances(x[:,i,:])
            result[i] = self.aggregate(d)
        return result

    def apply_features(self,x:np.ndarray):
        n,c=x.shape
        result=np.zeros(c)
        for i in range(c):
            d = euclidean_distances(x[:,i])
            result[i] = self.aggregate(d)
        return result

    def aggregate(self,d:np.ndarray):
        if self == DistanceAggregation.max:
            return d.max()
        elif self == DistanceAggregation.mean:
            return d.mean()
        else:
            raise ValueError(f"Unknown distance aggregation {self}")




class DistanceTransformationMeasure(Measure):
    def __init__(self, measure_function:MeasureFunction, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.measure_function=measure_function
        self.distance_aggregation=distance_aggregation

    def __repr__(self):
        return f"DTM(f={self.measure_function.value},da={self.distance_aggregation.value})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.activation_names()
        n_intermediates = len(layer_names)
        mean_variances_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_measure= self.measure_function.apply(layer_activations)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)



class DistanceSampleMeasure(Measure):
    def __init__(self, measure_function: MeasureFunction, distance_aggregation: DistanceAggregation):
        super().__init__()
        self.measure_function = measure_function
        self.distance_aggregation = distance_aggregation

    def __repr__(self):
        return f"DSM(f={self.measure_function.value},da={self.distance_aggregation.value})"

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names = activations_iterator.activation_names()
        n_layers = len(layer_names)
        mean_variances_running = [RunningMean() for i in range(n_layers)]

        for transformation, transformation_activations in activations_iterator.transformations_first():
            # calculate the variance of all samples for this transformation
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    for i in range(layer_activations.shape[0]):
                        layer_measure = self.measure_function.apply(layer_activations)
                        mean_variances_running [j].update(layer_measure )

        # calculate the final mean over all transformations (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)


from transformation_measure import QuotientMeasure
class DistanceMeasure(QuotientMeasure):
    def __init__(self, measure_function: MeasureFunction, distance_aggregation: DistanceAggregation):
        self.measure_function = measure_function
        self.distance_aggregation = distance_aggregation

        super().__init__(DistanceTransformationMeasure(measure_function,distance_aggregation),
                         DistanceSampleMeasure(measure_function,distance_aggregation))



    def __repr__(self):
        return f"DM(f={self.measure_function.value},da={self.distance_aggregation.value})"


