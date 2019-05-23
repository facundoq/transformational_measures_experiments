
from .utils import RunningMeanAndVariance,RunningMean
import logging

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import os
from pytorch.classification_dataset import ImageDataset

class StratifiedVarianceMeasure:
    def __init__(self,classes_iterators):
        self.classes_iterators=classes_iterators

    def eval(self):
        class_variance = [NormalizedVarianceMeasure(iterator) for iterator in self.classes_iterators]
        class_variance_result = [m.eval() for m in class_variance ]
        stratified_variance = self.mean_variance_over_classes(class_variance_result)
        return stratified_variance,class_variance_result

    def mean_variance_over_classes(self,class_variance_result):
        # calculate the mean activation of each unit in each layer over the set of classes
        layer_class_vars=[list(i) for i in zip(*class_variance_result)]
        layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
        return [layer_vars]


class Measure:
    def __init__(self,activations_iterator):
        self.activations_iterator=activations_iterator

class NormalizedVarianceMeasure(Measure):
    def __init__(self, activations_iterator):
        super().__init__(activations_iterator)

    def eval(self):
        logging.debug("Evaluating v_samples")
        v_samples=self.eval_v_samples()


        logging.debug("Evaluating v_transformations")
        v_transformations=self.eval_v_transformations()

        logging.debug("Evaluating v_normalized")
        v=self.eval_v_normalized(v_transformations,v_samples)
        return v

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
        n_layers = len(self.activations_iterator.layer_names())
        mean_variances_running = [RunningMean() for i in range(n_layers)]

        for transformation, transformation_activations in self.activations_iterator.transformations_first():
            samples_variances_running = [RunningMeanAndVariance() for i in range(n_layers)]
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    for i in range(layer_activations.shape[0]):
                        samples_variances_running[j].update(layer_activations[i,])
            for j, m in enumerate(mean_variances_running):
                mean_variances_running[j].update(samples_variances_running[j].var())
        mean_variances = [b.mean() for b in mean_variances_running]
        return mean_variances

    def eval_v_transformations(self,):
        n_intermediates = len(self.activations_iterator.layer_names())
        mean_variances_running= [RunningMean() for i in range(n_intermediates)]
        for activations, x_transformed in self.activations_iterator.samples_first():
            for j, layer_activations in enumerate(activations):
                mean_variances_running[j].update(layer_activations.var(axis=0))
        mean_variances = [b.mean() for b in mean_variances_running]
        return mean_variances


class VarianceMeasureResult:
    def __init__(self,layers):
        self.layers=layers

    def average_per_layer(self):
        result=[]
        for layer in self.layers:
            for act in layer[:]:
                result.append(act[np.isfinite(act)].mean())
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
                flat_activations=self.apply_aggregation_function(layer,conv_aggregation_function)
                assert (len(flat_activations.shape) == 2)
            else:
                flat_activations = layer.copy()
            results.append(flat_activations)

        return VarianceMeasureResult(results)

    def apply_aggregation_function(self,layer,conv_aggregation_function):
        n, c, w, h = layer.shape
        flat_activations = np.zeros((n, c))
        for i in range(n):
            if conv_aggregation_function == "mean":
                flat_activations[i, :] = np.nanmean(layer[i, :, :, :], axis=(1, 2))
            elif conv_aggregation_function == "max":
                flat_activations[i, :] = np.nanmax(layer[i, :, :, :], axis=(1, 2))
            elif conv_aggregation_function == "sum":
                flat_activations[i, :] = np.nansum(layer[i, :, :, :], axis=(1, 2))
            else:
                raise ValueError(
                    f"Invalid aggregation function: {conv_aggregation_function}. Options: mean, max, sum")
            return flat_activations

