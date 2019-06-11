from .base import Measure,MeasureFunction,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.utils import RunningMeanAndVariance,RunningMean
from .layer_transformation import ConvAggregation,apply_aggregation_function


class SampleMeasure(Measure):
    def __init__(self, measure_function: MeasureFunction, conv_aggregation: ConvAggregation):
        super().__init__()
        self.measure_function = measure_function
        self.conv_aggregation = conv_aggregation

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def eval_v_samples(self,activations_iterator:ActivationsIterator)->MeasureResult:
        n_layers = len(activations_iterator.activation_names())
        mean_variances_running = [RunningMean() for i in range(n_layers)]

        for transformation, transformation_activations in activations_iterator.transformations_first():
            samples_variances_running = [RunningMeanAndVariance() for i in range(n_layers)]
            # calculate the variance of all samples for this transformation
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    for i in range(layer_activations.shape[0]):
                        layer_activations=self.preprocess_activations(layer_activations)
                        samples_variances_running[j].update(layer_activations[i,])
            # update the mean over all transformation (and layers)
            for layer_mean_variances_running, layer_samples_variance_running in zip(mean_variances_running,samples_variances_running):
                samples_variance=self.samples_variance(layer_samples_variance_running)
                layer_mean_variances_running.update(samples_variance)
        # calculate the final mean over all transformations (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,self)



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

