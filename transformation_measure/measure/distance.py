from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.stats_running import RunningMeanAndVariance,RunningMean
from typing import List
from enum import Enum

from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import scipy.spatial.distance



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
            sample = x[:,i,:]
            d = scipy.spatial.distance.pdist(sample, 'euclidean')
            #d = euclidean_distances(sample)
            result[i] = self.aggregate(d)
        return result

    def apply_features(self,x:np.ndarray):
        n, c = x.shape
        x = x[:,:,np.newaxis]
        result = np.zeros(c)
        for i in range(c):
            sample = x[:, i,:]
            d = scipy.spatial.distance.pdist(sample, 'euclidean')
            #d = euclidean_distances(sample)
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
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation

    def __repr__(self):
        return f"DTM(da={self.distance_aggregation.value})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.activation_names()
        n_intermediates = len(layer_names)
        mean_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_measure = self.distance_aggregation.apply(layer_activations)
                # update the mean over all transformation
                mean_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances,layer_names,self)



class DistanceSampleMeasure(Measure):
    def __init__(self, distance_aggregation: DistanceAggregation):
        super().__init__()
        self.distance_aggregation = distance_aggregation

    def __repr__(self):
        return f"DSM(da={self.distance_aggregation.value})"

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names = activations_iterator.activation_names()
        n_layers = len(layer_names)
        mean_running = [RunningMean() for i in range(n_layers)]

        for transformation, transformation_activations in activations_iterator.transformations_first():
            # calculate the variance of all samples for this transformation
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    layer_measure = self.distance_aggregation.apply(layer_activations)
                    mean_running [j].update(layer_measure)

        # calculate the final mean over all transformations (and layers)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances,layer_names,self)


from transformation_measure import QuotientMeasure
class DistanceMeasure(QuotientMeasure):
    def __init__(self, distance_aggregation: DistanceAggregation):

        self.distance_aggregation = distance_aggregation

        super().__init__(DistanceTransformationMeasure(distance_aggregation),
                         DistanceSampleMeasure(distance_aggregation))



    def __repr__(self):
        return f"DM(da={self.distance_aggregation.value})"


