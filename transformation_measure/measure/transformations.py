from .base import Measure,MeasureFunction,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.running_stats import RunningMeanAndVariance,RunningMean
from .layer_transformation import ConvAggregation
from typing import List


class TransformationMeasure(Measure):
    def __init__(self, measure_function:MeasureFunction,conv_aggregation:ConvAggregation):
        super().__init__()
        self.measure_function=measure_function
        self.conv_aggregation=conv_aggregation

    def __repr__(self):
        return f"TM(f={self.measure_function.value},ca={self.conv_aggregation.value})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.activation_names()
        n_intermediates = len(layer_names)
        mean_variances_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_activations = self.preprocess_activations(layer_activations)
                # calculate the measure for all transformations of this sample
                layer_measure = self.measure_function.apply(layer_activations)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)



    def preprocess_activations(self, layer_activations:np.ndarray):
        if len(layer_activations.shape) == 4:
            return self.conv_aggregation.apply(layer_activations)
        else:
            return layer_activations


