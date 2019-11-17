from transformation_measure import ConvAggregation
from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
from transformation_measure.measure.stats_running import RunningMeanAndVariance,RunningMean
from transformation_measure import MeasureFunction,QuotientMeasure
import numpy as np

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
        mean_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                # apply function to conv layers
                layer_activations = self.conv_aggregation.apply(layer_activations)
                # directly compute the function
                layer_measure = self.measure_function.apply(layer_activations)
                # update the mean over all transformation
                mean_running[j].update(layer_measure)

        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances,layer_names,self)

class SampleMeasure(Measure):
    def __init__(self, measure_function: MeasureFunction, conv_aggregation: ConvAggregation):
        super().__init__()
        self.measure_function = measure_function
        self.conv_aggregation = conv_aggregation

    def __repr__(self):
        return f"SM(f={self.measure_function.value},ca={self.conv_aggregation.value})"

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names = activations_iterator.activation_names()
        n_layers = len(layer_names)
        mean_variances_running = [RunningMean() for i in range(n_layers)]

        for transformation, transformation_activations in activations_iterator.transformations_first():
            samples_variances_running = [RunningMeanAndVariance() for i in range(n_layers)]
            # calculate the variance of all samples for this transformation
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    layer_activations = self.conv_aggregation.apply(layer_activations)
                    samples_variances_running[j].update_all(layer_activations)
            # update the mean over all transformation (and layers)
            for layer_mean_variances_running, layer_samples_variance_running in zip(mean_variances_running,samples_variances_running):
                samples_variance=self.measure_function.apply_running(layer_samples_variance_running)
                layer_mean_variances_running.update(samples_variance)

        # calculate the final mean over all transformations (and layers)

        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,layer_names,self)


class NormalizedMeasure(QuotientMeasure):
    def __init__(self,measure_function:MeasureFunction,conv_aggregation:ConvAggregation):
        sm = SampleMeasure(measure_function,conv_aggregation)
        ttm = TransformationMeasure(measure_function,conv_aggregation)
        super().__init__(ttm,sm)
        self.numerator_measure = ttm
        self.denominator_measure = sm
        self.measure_function=measure_function
        self.conv_aggregation=conv_aggregation

    def __repr__(self):
        return f"NM(f={self.measure_function.value},ca={self.conv_aggregation.value})"