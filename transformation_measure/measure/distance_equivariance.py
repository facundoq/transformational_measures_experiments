from .base import Measure,MeasureFunction,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.running_stats import RunningMeanAndVariance,RunningMean
from typing import List
from enum import Enum

from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import matplotlib.pyplot as plt
from .distance import DistanceAggregation
from pathlib import Path
import transformation_measure as tm
from time import gmtime, strftime

class DistanceSameEquivarianceMeasure(Measure):
    def __init__(self, measure_function:MeasureFunction, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.measure_function=measure_function
        self.distance_aggregation=distance_aggregation

    def __repr__(self):
        return f"DSEM(f={self.measure_function.value},da={self.distance_aggregation.value})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.activation_names()
        n_layers = len(layer_names)
        transformations = activations_iterator.get_transformations()
        mean_variances_running = [RunningMean() for i in range(n_layers)]
        for activations, x_transformed in activations_iterator.samples_first():
            # activations has the activations for all the transformations
            self.inverse_trasform_feature_maps(activations, transformations)
            for j, layer_activations in enumerate(activations):
                layer_measure= self.measure_function.apply(layer_activations)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)

    def inverse_trasform_feature_maps(self,activations:[np.ndarray],transformations:tm.TransformationSet)->[np.ndarray]:
        transformations = list(transformations)
        inverses = [t.inverse() for t in transformations]

        for j,layer in enumerate(activations):
            if len(layer.shape)==4:
                # n,c,h,w=layer.shape
                # f,ax=plt.subplots(2,n)
                for i,inverse in enumerate(inverses):
                    # ax[0, i].imshow(layer[i,0,:,:],cmap="gray")
                    # ax[0, i].axis("off")
                    layer[i,:] = inverse(layer[i,:])
                    # ax[1, i].imshow(layer[i,0,:,:],cmap="gray")
                    # ax[1, i].axis("off")


                # path = Path("testing/dsem/")
                # path.mkdir(parents=True,exist_ok=True)
                # now=strftime("%Y-%m-%d %H:%M:%S", gmtime())
                # filepath=path / (f"{now}_{j}.png")
                # plt.savefig(filepath)
                # plt.close(f)



