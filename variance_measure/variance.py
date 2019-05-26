import abc
from .utils import RunningMeanAndVariance,RunningMean
import logging

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import os
from pytorch.classification_dataset import ImageDataset

class StratifiedMeasure:
    def __init__(self,classes_iterators,variance_measure):
        '''

        :param classes_iterators:  A list of iterators, one for each class or subset
        :param variance_measure: A function that takes an iterator and returns a MeasureResult object
        '''
        self.classes_iterators=classes_iterators
        self.variance_measure=variance_measure


    def eval(self):
        '''
        Calculate the `variance_measure` for each class separately
        Also calculate the average stratified `variance_measure` over all classes
        '''
        variance_per_class = [self.variance_measure(iterator)[0] for iterator in self.classes_iterators]
        variance_stratified = self.mean_variance_over_classes(variance_per_class)
        return variance_stratified,variance_per_class


    def mean_variance_over_classes(self,class_variance_result):
        # calculate the mean activation of each unit in each layer over the set of classes
        class_variance_layers=[r.layers for r in class_variance_result]
        # class_variance_layers is a list (classes) of list layers)
        # turn it into a list (layers) of lists (classes)
        layer_class_vars=[list(i) for i in zip(*class_variance_layers)]
        # compute average variance of each layer over classses
        layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
        return MeasureResult(layer_vars,"v_stratified")


from enum import Enum

class ConvAggregation(Enum):
    mean = "mean"
    max = "max"
    min = "min"
    sum = "sum"
    none = "none"




class Measure:
    def __init__(self,activations_iterator):
        self.activations_iterator=activations_iterator

    @abc.abstractmethod
    def eval(self):
        '''

        :return: A VarianceMeasureResult object containing the variance of each activation
        '''
        pass

class MeanNormalizedMeasure(Measure):
    def __init__(self, activations_iterator,options):
        super().__init__(activations_iterator)

        self.var_or_std=options.get("var_or_std","var")
        self.conv_aggregation_function=options.get("conv_aggregation_function",None)

    def eval(self):
        logging.debug("Evaluating v_samples")
        v_samples=self.eval_mean_samples()

        logging.debug("Evaluating v_transformations")
        v_transformations=self.eval_v_transformations()

        logging.debug("Evaluating v_normalized")
        v=self.eval_v_normalized(v_transformations.layers,v_samples.layers)
        return MeasureResult(v,f"v,{self.var_or_std}"),v_transformations,v_samples

    def eval_mean_samples(self):

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

    def eval_v_transformations(self,):
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


class NormalizedMeasure(Measure):
    def __init__(self, activations_iterator,options):
        super().__init__(activations_iterator)

        self.var_or_std=options.get("var_or_std","var")
        self.conv_aggregation_function=options.get("conv_aggregation_function",None)


    def eval(self):
        logging.debug("Evaluating v_samples")
        v_samples=self.eval_v_samples()


        logging.debug("Evaluating v_transformations")
        v_transformations=self.eval_v_transformations()

        logging.debug("Evaluating v_normalized")
        v=self.eval_v_normalized(v_transformations.layers,v_samples.layers)
        return MeasureResult(v,f"v,{self.var_or_std}"),v_transformations,v_samples

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

    def eval_v_samples(self):
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

    def eval_v_transformations(self,):
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

class MeasureResult:
    def __init__(self,layers,source):
        self.layers=layers
        self.source=source
    def __repr__(self):
        return f"MeasureResult {self.source}"

    def all_1d(self):
        return np.any([ len(l.shape)==1 for l in self.layers])

    def average_per_layer(self):

        result = []
        for layer in self.layers:
            layer=layer[:]
            layer_average=layer[np.isfinite(layer)].mean()
            result.append(layer_average)
        return np.array(result)

    def weighted_global_average(self):
        return self.average_per_layer().mean()

    def global_average(self):
        rm = RunningMean()
        for layer in self.layers:
                for act in layer[:]:
                    if np.isfinite(act):
                        rm.update(act)
        return rm.mean()

    def collapse_convolutions(self,conv_aggregation_function):
        results=[]
        for layer in self.layers:
            # if conv average out spatial dims
            if len(layer.shape) == 4:
                flat_activations=apply_aggregation_function(layer,conv_aggregation_function)
                assert (len(flat_activations.shape) == 2)
            else:
                flat_activations = layer.copy()
            results.append(flat_activations)

        return MeasureResult(results,f"{self.source}_conv_agg_{conv_aggregation_function}")

functions={ConvAggregation.mean : np.nanmean
               ,ConvAggregation.sum : np.nansum
               ,ConvAggregation.min : np.nanmin
               ,ConvAggregation.max : np.nanmax
               }

def apply_aggregation_function(layer,conv_aggregation_function):

    if conv_aggregation_function == ConvAggregation.none:
        return layer
    if not conv_aggregation_function in functions.keys():
        raise ValueError(
            f"Invalid aggregation function: {conv_aggregation_function}. Options: {list(ConvAggregation)}")

    n, c, w, h = layer.shape
    flat_activations = np.zeros((n, c))
    function=functions[conv_aggregation_function]

    for i in range(n):
        flat_activations[i, :] = function(layer[i, :, :, :], axis=(1, 2))

    return flat_activations

