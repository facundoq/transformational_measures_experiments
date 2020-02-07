from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
from transformation_measure.measure.stats_running import RunningMeanAndVarianceWelford,RunningMeanWelford
from .distance import DistanceAggregation
import scipy.spatial.distance as distance_functions
import numpy as np

def list_get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]

class DistanceFunction:
    def __init__(self,keep_shape:bool):
        self.keep_shape=keep_shape

    def distance(self,batch:np.ndarray,batch_inverted:np.ndarray,mean_running:RunningMeanWelford):
        n_shape=len(batch.shape)
        assert n_shape>=2
        n,f=batch.shape[0],batch.shape[1]

        if not self.keep_shape:
            batch = batch.reshape(n,-1)

        # ssd of all values
        distances = (batch-batch_inverted)**2

        if n_shape>2:
            # aggregate extra dims to keep only the feature dim
            distances= distances.sum(axis=tuple(range(2,n)))
        # now we only have a 2d (samples,features) array
        #assert len(distances.shape)==2

        mean_running.update_all(distances)


class DistanceSameEquivarianceSimpleMeasure(Measure):
    def __init__(self, distance_function:DistanceFunction):
        super().__init__()
        self.distance_function=distance_function

    def __repr__(self):
        return f"DSES(df={self.distance_function})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        activations_iterator = activations_iterator.get_both_iterator()
        mean_running=None

        for x, transformation_activations_iterator in activations_iterator.samples_first():
            # transformation_activations_iterator can iterate over all transforms
            for x_transformed, activations,inverted_activations in transformation_activations_iterator:
                if mean_running is None:
                    mean_running = [RunningMeanWelford() for i in range(len(activations))]
                for j, (layer_activations,inverted_layer_activations) in enumerate(zip(activations,inverted_activations)):
                    self.distance_function.distance(layer_activations,inverted_layer_activations,mean_running[j])
        # calculate the final mean over all samples (and layers)
        means = [b.mean() for b in mean_running]
        return MeasureResult(means,activations_iterator.layer_names(),self)


    def name(self)->str:
        return "Distance Same-Equivariance Simple"
    def abbreviation(self):
        return "DSES"