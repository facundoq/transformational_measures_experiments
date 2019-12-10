from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.stats_running import RunningMeanAndVariance,RunningMean
from typing import List
from enum import Enum

from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import matplotlib.pyplot as plt
from .distance import DistanceAggregation
from pathlib import Path
import transformation_measure as tm
from time import gmtime, strftime

def list_get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]

class DistanceSameEquivarianceMeasure(Measure):
    def __init__(self, distance_aggregation:DistanceAggregation,normalized=True):
        super().__init__()
        self.distance_aggregation=distance_aggregation
        self.normalized = normalized

    def __repr__(self):
        return f"DSE(da={self.distance_aggregation.value},n={self.normalized})"


    def get_valid_layers(self, activations: [np.ndarray], layer_names: [str], x_shape: np.ndarray):
        # get indices of layers with the same shape as x
        # (criteria to declare valid for transformation, we assume if the shapes are the same
        # then the transformation applies)
        indices=[i for i,a in enumerate(activations) if a.shape == x_shape]
        # keep layers and generate mean_variances_running only for these layers
        layer_names = list_get_all(layer_names,indices)
        n_layers = len(layer_names)
        mean_variances_running = [RunningMean() for i in range(n_layers)]
        return layer_names,mean_variances_running,indices

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        transformations = list(activations_iterator.get_transformations())
        first_iteration = True
        mean_variances_running= None
        layer_names = None
        indices = None

        for x, transformation_activations_iterator in activations_iterator.samples_first():
            # transformation_activations_iterator can iterate over all transforms
            t_start=0
            for x_transformed, activations in transformation_activations_iterator:
                if first_iteration:
                    # find out which layers can be transformed (ones with same dims as x)
                    layer_names,mean_variances_running,indices = self.get_valid_layers(activations,activations_iterator.activation_names(),x_transformed.shape)
                    first_iteration = False
                # keep only those activations valid for the transformation
                activations = list_get_all(activations,indices)
                n = x_transformed.shape[0]
                t_end=t_start+n

                self.inverse_trasform_feature_maps(activations, transformations[t_start:t_end])
                t_start=t_end
                for j, layer_activations in enumerate(activations):
                    # TODO change normalize to True
                    layer_measure= self.distance_aggregation.apply(layer_activations)
                    # update the mean over all transformation
                    mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)

    def inverse_trasform_feature_maps(self,activations:[np.ndarray],transformations:tm.TransformationSet)->[np.ndarray]:
        transformations = list(transformations)
        inverses = [t.inverse() for t in transformations]

        for j,layer in enumerate(activations):

            for i,inverse in enumerate(inverses):
                # ax[0, i].imshow(layer[i,0,:,:],cmap="gray")
                # ax[0, i].axis("off")
                layer[i,:] = inverse(layer[i,:])

            if self.normalized:
                layer-=layer.min(axis=1,keepdims=True)
                max_values=layer.max(axis=1,keepdims=True)
                max_values[max_values==0]=1
                layer/=max_values

            # if len(layer.shape)==4:
                # n,c,h,w=layer.shape
                # f,ax=plt.subplots(2,n)
                # ax[1, i].imshow(layer[i,0,:,:],cmap="gray")
                # ax[1, i].axis("off")
                # path = Path("testing/dsem/")
                # path.mkdir(parents=True,exist_ok=True)
                # now=strftime("%Y-%m-%d %H:%M:%S", gmtime())
                # filepath=path / (f"{now}_{j}.png")
                # plt.savefig(filepath)
                # plt.close(f)



