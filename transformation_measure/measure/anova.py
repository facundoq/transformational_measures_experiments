from .base import Measure,MeasureFunction,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.running_stats import RunningMeanAndVariance,RunningMean
from .layer_transformation import ConvAggregation,apply_aggregation_function
from typing import List
from .transformations import TransformationMeasure
class AnovaMeasure(Measure):
    def __init__(self, measure_function: MeasureFunction, conv_aggregation: ConvAggregation):
        super().__init__()
        self.measure_function = measure_function
        self.conv_aggregation = conv_aggregation

    def __repr__(self):
        return f"AnovaMeasure(f={self.measure_function},ca={self.conv_aggregation.value})"

    def eval(self,activations_iterator:ActivationsIterator,layer_names:List[str])->MeasureResult:

        means_per_layer_and_transformation,samples_per_transformation=self.eval_means_per_transformation(activations_iterator)
        n_layers=len(activations_iterator.activation_names())
        global_means=self.eval_global_means(means_per_layer_and_transformation,n_layers)
        ssdb_per_layer=self.eval_between_transformations_ssd(means_per_layer_and_transformation,global_means)
        ssdb_per_layer

        return MeasureResult(mean_variances,layer_names,self)
    def eval_between_transformations_ssd(self,means_per_layer_and_transformation:[[np.ndarray]],global_means:[np.ndarray],samples_per_transformation:[int])->[np.ndarray]:
        '''

        :param means_per_transformation: has len(transformations), each item has len(layers)
        :param global_means: has len(layers)
        :return:
        '''
        n_layers=len(global_means)
        n_transformations=len(means_per_layer_and_transformation)
        ssdb_per_layer=[0 for l in n_layers]
        for  i,means_per_layer in enumerate(means_per_layer_and_transformation):
            n=samples_per_transformation[i]
            for j in range(n_layers):
                ssdb_per_layer[j] += n* ((means_per_layer[i]-global_means[j])**2)

        for j in range(n_layers):
            ssdb_per_layer[j]/=(n_transformations-1)
        return ssdb_per_layer

    def eval_global_means(self, means_per_layer_and_transformation:[[np.ndarray]], n_layers:int)->[np.ndarray]:
        n_transformations=len(means_per_layer_and_transformation)
        global_means_running = [RunningMean() for i in range(n_layers)]
        for means_per_layer in means_per_layer_and_transformation:
            for i,means in enumerate(means_per_layer):
                global_means_running[i].update(means)
        return [rm.mean()/n_transformations for rm in global_means_running]




    def eval_means_per_transformation(self,activations_iterator:ActivationsIterator)->([[np.ndarray]],[int]):
        n_layers = len(activations_iterator.activation_names())
        means_per_transformation = []
        samples_per_transformation=[]
        for transformation, transformation_activations in activations_iterator.transformations_first():
            samples_variances_running = [RunningMean() for i in range(n_layers)]
            # calculate the variance of all samples for this transformation
            n_samples=0
            for x, batch_activations in transformation_activations:
                n_samples+=x.shape[0]
                for j, layer_activations in enumerate(batch_activations):
                    for i in range(layer_activations.shape[0]):
                        layer_activations = self.preprocess_activations(layer_activations)
                        samples_variances_running[j].update(layer_activations[i,])
            samples_per_transformation.append(n_samples)
            means_per_transformation.append(samples_variances_running.mean())
        return means_per_transformation,samples_per_transformation

    def samples_variance(self,samples_variance_running):
        if self.measure_function == MeasureFunction.var:
            return samples_variance_running.var()
        elif self.measure_function == MeasureFunction.std:
            return samples_variance_running.std()
        elif self.measure_function == MeasureFunction.mean:
            return samples_variance_running.mean()
        else:
            raise ValueError(f"Unsupported measure function {self.measure_function}")

    def preprocess_activations(self, layer_activations):
        if len(layer_activations.shape) == 4:
            return apply_aggregation_function(layer_activations, self.conv_aggregation)
        else:
            return layer_activations

