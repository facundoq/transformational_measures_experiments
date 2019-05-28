import abc
from variance_measure.measures.utils import RunningMeanAndVariance,RunningMean
import logging
from variance_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from typing import Dict, List, Tuple
from .result import MeasureResult
from .layer_transformation import ConvAggregation,apply_aggregation_function

from .stratified import StratifiedMeasure


class Measure:
    def __init__(self,activations_iterator:ActivationsIterator):
        self.activations_iterator=activations_iterator

    def __repr__(self):
        return f"{self.__class__.__name__}"



    @abc.abstractmethod
    def eval(self):
        '''

        :return: A VarianceMeasureResult object containing the variance of each activation
        '''
        pass


class MeanNormalizedMeasure(Measure):
    def __init__(self, activations_iterator:ActivationsIterator,options):
        super().__init__(activations_iterator)

        self.var_or_std=options.get("var_or_std","var")
        self.conv_aggregation_function=options.get("conv_aggregation_function",None)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def eval(self)->MeasureResult:
        logging.debug("Evaluating v_samples")
        v_samples=self.eval_mean_transformations()

        logging.debug("Evaluating v_transformations")
        v_transformations=self.eval_v_transformations()

        logging.debug("Evaluating v_normalized")
        v=self.eval_v_normalized(v_transformations.layers,v_samples.layers)
        return MeasureResult(v,f"v,{self.var_or_std}")

    def eval_mean_transformations(self)->MeasureResult:

        n_intermediates = len(self.activations_iterator.activation_names())
        mean_variances_running = [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in self.activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_activations = self.preprocess_activations(layer_activations)
                # calculate the measure for all transformations of this sample

                layer_measure = np.abs(layer_activations)
                layer_measure = np.nanmean(layer_measure,axis=0)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]

        return MeasureResult(mean_variances, f"u_trasformation")

    def eval_v_normalized(self,v_transformations,v_samples):
        eps = 0
        measures = []  # coefficient of variations

        for layer_v_transformations,layer_v_samples in zip(v_transformations,v_samples):
            # print(layer_baseline.shape, layer_measure.shape)
            normalized_measure = layer_v_transformations.copy()
            normalized_measure[layer_v_samples  > eps] /= layer_v_samples [layer_v_samples  > eps]
            both_below_eps = np.logical_and(layer_v_samples  <= eps,
                                            layer_v_transformations <= eps)
            normalized_measure[both_below_eps] = 1
            only_baseline_below_eps = np.logical_and(
                layer_v_samples  <= eps,
                layer_v_transformations > eps)
            normalized_measure[only_baseline_below_eps] = np.inf
            measures.append(normalized_measure)
        return measures

    def eval_v_transformations(self,)->MeasureResult:
        n_intermediates = len(self.activations_iterator.activation_names())
        mean_variances_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in self.activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_activations = self.preprocess_activations(layer_activations)
                # calculate the measure for all transformations of this sample
                layer_measure = self.layer_measure(layer_activations)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,f"v_trasformation,{self.var_or_std}")

    def layer_measure(self,layer_activations):
        if self.var_or_std == "var":
            return layer_activations.var(axis=0)
        else:
            return layer_activations.std(axis=0)
    def samples_variance(self,samples_variance_running):
        if self.var_or_std == "var":
            return samples_variance_running.var()
        elif self.var_or_std=="std":
            return samples_variance_running.std()
        else:
            raise ValueError

    def preprocess_activations(self, layer_activations):

        if len(layer_activations.shape) == 4:
            return apply_aggregation_function(layer_activations, self.conv_aggregation_function)
        else:
            return layer_activations


class NormalizedMeasure(Measure):
    def __init__(self, activations_iterator,options):
        super().__init__(activations_iterator)

        self.var_or_std=options.get("var_or_std","var")
        self.conv_aggregation_function=options.get("conv_aggregation_function",None)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def eval(self):
        logging.debug("Evaluating v_samples")
        v_samples=self.eval_v_samples()


        logging.debug("Evaluating v_transformations")
        v_transformations=self.eval_v_transformations()

        logging.debug("Evaluating v_normalized")
        v=self.eval_v_normalized(v_transformations.layers,v_samples.layers)
        return MeasureResult(v,f"v,{self.var_or_std}")

    def eval_v_normalized(self,v_transformations,v_samples):
        eps = 0
        measures = []  # coefficient of variations

        for layer_v_transformations,layer_v_samples in zip(v_transformations,v_samples):
            # print(layer_baseline.shape, layer_measure.shape)
            normalized_measure = layer_v_transformations.copy()
            normalized_measure[layer_v_samples  > eps] /= layer_v_samples [layer_v_samples  > eps]
            both_below_eps = np.logical_and(layer_v_samples  <= eps,
                                            layer_v_transformations <= eps)
            normalized_measure[both_below_eps] = 1
            only_baseline_below_eps = np.logical_and(
                layer_v_samples  <= eps,
                layer_v_transformations > eps)
            normalized_measure[only_baseline_below_eps] = np.inf
            measures.append(normalized_measure)
        return measures

    def eval_v_samples(self)->MeasureResult:
        n_layers = len(self.activations_iterator.activation_names())
        mean_variances_running = [RunningMean() for i in range(n_layers)]

        for transformation, transformation_activations in self.activations_iterator.transformations_first():
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
        return MeasureResult(mean_variances,f"v_samples,{self.var_or_std}")

    def eval_v_transformations(self,)->MeasureResult:
        n_intermediates = len(self.activations_iterator.activation_names())
        mean_variances_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in self.activations_iterator.samples_first():
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_activations = self.preprocess_activations(layer_activations)
                # calculate the measure for all transformations of this sample
                layer_measure = self.layer_measure(layer_activations)
                # update the mean over all transformation
                mean_variances_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances,f"v_trasformation,{self.var_or_std}")

    def layer_measure(self,layer_activations):
        if self.var_or_std == "var":
            return layer_activations.var(axis=0)
        else:
            return layer_activations.std(axis=0)
    def samples_variance(self,samples_variance_running):
        if self.var_or_std == "var":
            return samples_variance_running.var()
        elif self.var_or_std=="std":
            return samples_variance_running.std()
        else:
            raise ValueError

    def preprocess_activations(self, layer_activations):
        if not self.conv_aggregation_function is None and len(layer_activations.shape) == 4:
            return apply_aggregation_function(layer_activations, self.conv_aggregation_function)
        else:
            return layer_activations

