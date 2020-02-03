from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.stats_running import RunningMeanAndVarianceWellford,RunningMean
from typing import List
from enum import Enum
from .quotient import divide_activations

from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import matplotlib.pyplot as plt
from .distance import DistanceAggregation
from pathlib import Path
import transformation_measure as tm
from time import gmtime, strftime

def list_get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]

class BaseDistanceSameEquivarianceMeasure(Measure):
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation
    def get_valid_layers(self, activations: [np.ndarray], layer_names: [str], x_shape: np.ndarray):
        # get indices of layers with the same number of dims as x
        # (criteria to declare valid for transformation, we assume if the shapes are the same
        # then the transformation applies)
        indices=[i for i,a in enumerate(activations) if len(a.shape) == len(x_shape)]
        # keep layers and generate mean_running only for these layers
        layer_names = list_get_all(layer_names,indices)
        n_layers = len(layer_names)
        mean_running = [RunningMean() for i in range(n_layers)]
        return layer_names,mean_running,indices

    def get_inverse_transformations(self, activations:[np.ndarray], indices:[int], transformation_set:tm.TransformationSet):
        valid_activations=[activations[i] for i in indices]
        shapes = [a.shape for a in valid_activations]
        inverse_transformation_sets=[]

        for s in shapes:
            n,c,h,w=s
            layer_transformation_set:tm.TransformationSet = transformation_set.copy()
            layer_transformation_set.set_pytorch(False)
            layer_transformation_set.set_input_shape((h, w, c))
            layer_transformation_set_list = [l.inverse() for l in layer_transformation_set]
            inverse_transformation_sets.append(layer_transformation_set_list)
        return inverse_transformation_sets
    def inverse_trasform_feature_maps(self,activations:[np.ndarray],transformations:[tm.TransformationSet],t_start:int,t_end:int)->[np.ndarray]:



        for layer,layer_transformations in zip(activations,transformations):
            for i,inverse in enumerate(layer_transformations[t_start:t_end]):
                # ax[0, i].imshow(layer[i,0,:,:],cmap="gray")
                # ax[0, i].axis("off")
                #print(inverse.__class__,layer.__class__)
                fm=layer[i:i+1,:].transpose(0,2,3,1)
                inverse_fm=inverse(fm)
                inverse_fm=inverse_fm.transpose(0,3,1,2)
                # print(fm.shape, inverse_fm.shape)
                layer[i,:] = inverse_fm[0,:]


            # if self.normalized:
            #     layer-=layer.min(axis=1,keepdims=True)
            #     max_values=layer.max(axis=1,keepdims=True)
            #     max_values[max_values==0]=1
            #     layer/=max_values

            # self.plot_debug(layer, j)

    def plot_debug(self,layer,j):

        if len(layer.shape)==4:
            n, c, h, w = layer.shape
            for i in  range(n):
                f,ax=plt.subplots(2,n)
                ax[1, i].imshow(layer[i,0,:,:],cmap="gray")
                ax[1, i].axis("off")
                path = Path("testing/dsem/")
                path.mkdir(parents=True,exist_ok=True)
                now=strftime("%Y-%m-%d %H:%M:%S", gmtime())
                filepath=path / (f"{now}_{j}.png")
                plt.savefig(filepath)
                plt.close(f)



class TransformationDistanceSameEquivarianceMeasure(BaseDistanceSameEquivarianceMeasure):


    def __repr__(self):
        return f"TDSE(da={self.distance_aggregation.name})"

    def name(self)->str:
        return "Transformation Distance Same-Equivariance"
    def abbreviation(self):
        return "TDSE"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        transformations = activations_iterator.get_transformations()
        first_iteration = True
        mean_running= None
        layer_names = None
        indices = None
        inverse_transformation_sets = None

        for x, transformation_activations_iterator in activations_iterator.samples_first():
            # transformation_activations_iterator can iterate over all transforms
            t_start=0
            for x_transformed, activations in transformation_activations_iterator:
                if first_iteration:
                    # find out which layers can be transformed (ones with same dims as x)
                    layer_names,mean_running,indices = self.get_valid_layers(activations, activations_iterator.layer_names(), x_transformed.shape)
                    inverse_transformation_sets = self.get_inverse_transformations(activations,indices,transformations)
                    first_iteration = False
                # keep only those activations valid for the transformation
                activations = list_get_all(activations,indices)
                n = x_transformed.shape[0]
                t_end=t_start+n

                self.inverse_trasform_feature_maps(activations,inverse_transformation_sets,t_start,t_end)
                t_start=t_end
                for j, layer_activations in enumerate(activations):
                    layer_measure= self.distance_aggregation.apply(layer_activations)
                    # update the mean over all transformation
                    mean_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        means = [b.mean() for b in mean_running]
        return MeasureResult(means,layer_names,self)







class SampleDistanceSameEquivarianceMeasure(BaseDistanceSameEquivarianceMeasure):


    def __repr__(self):
        return f"SDSE(da={self.distance_aggregation.name})"

    def name(self)->str:
        return "Sample Distance Same-Equivariance"
    def abbreviation(self):
        return "SDSE"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        transformations = activations_iterator.get_transformations()
        first_iteration = True
        mean_running= None
        layer_names = None
        indices = None
        inverse_transformation_sets = None
        for transformation, samples_activations_iterator in activations_iterator.transformations_first():

            # transformation_activations_iterator can iterate over all transforms
            t_start=0
            for x,activations in samples_activations_iterator:
                if first_iteration:
                    # find out which layers can be transformed (ones with same dims as x)
                    layer_names,mean_running,indices = self.get_valid_layers(activations, activations_iterator.layer_names(), x.shape)
                    inverse_transformation_sets = self.get_inverse_transformations(activations,indices,transformations)
                    first_iteration = False
                # keep only those activations valid for the transformation
                activations = list_get_all(activations,indices)
                n = x.shape[0]
                t_end=t_start+n

                self.inverse_trasform_feature_maps(activations,inverse_transformation_sets,t_start,t_end)
                t_start=t_end
                for j, layer_activations in enumerate(activations):
                    layer_measure= self.distance_aggregation.apply(layer_activations)
                    # update the mean over all transformation
                    mean_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        means = [b.mean() for b in mean_running]
        return MeasureResult(means,layer_names,self)


class NormalizedDistanceSameEquivarianceMeasure(Measure):
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation
        self.transformation_measure=TransformationDistanceSameEquivarianceMeasure(distance_aggregation)
        self.sample_measure=SampleDistanceSameEquivarianceMeasure(distance_aggregation)


    def __repr__(self):
        return f"NDSE(da={self.distance_aggregation.name})"

    def name(self)->str:
        return "NDistance Same-Equivariance"
    def abbreviation(self):
        return "NDSE"
    transformation_key=TransformationDistanceSameEquivarianceMeasure.__name__
    sample_key=SampleDistanceSameEquivarianceMeasure.__name__

    def eval(self,activations_iterator:ActivationsIterator) ->MeasureResult:

        transformation_result = self.transformation_measure.eval(activations_iterator)
        sample_result = self.sample_measure.eval(activations_iterator)
        result=divide_activations(transformation_result.layers,sample_result.layers)

        extra_values={ self.transformation_key:transformation_result,
                       self.sample_key:sample_result,
                       }
        return MeasureResult(result, activations_iterator.layer_names(),self,extra_values=extra_values)
