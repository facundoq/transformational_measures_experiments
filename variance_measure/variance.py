
from .utils import RunningMeanAndVariance,RunningMean

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import os
from pytorch.classification_dataset import ImageDataset

class StratifiedVariance:

    def eval(self,model, dataset, config, rotations, conv_aggregation_function, batch_size=256):
        x = dataset.x_test
        y=dataset.y_test
        y_ids=y.argmax(axis=1)
        classes=np.unique(y_ids)
        classes.sort()

        per_class_variance=[]
        # calculate the var measure for each class
        for i, c in enumerate(classes):
            # logging.debug(f"Evaluating vars for class {c}...")
            ids=np.where(y_ids==c)
            ids=ids[0]
            x_class,y_class=x[ids,:],y[ids]
            n=x_class.shape[0]
            class_dataset=ImageDataset(x_class,y_class,rotation=True)
            effective_batch_size=min(n,batch_size)
            layer_vars=eval_var(class_dataset, model, config, rotations, effective_batch_size,conv_aggregation_function)
            per_class_variance.append(layer_vars)
        variance= self.mean_variance_over_classes(per_class_variance)

        return per_class_variance,variance,classes


    def mean_variance_over_classes(class_layer_vars):
        # calculate the mean activation of each unit in each layer over the set of classes

        layer_class_vars=[list(i) for i in zip(*class_layer_vars)]
        layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
        return [layer_vars]


class Measure:
    def __init__(self,activations_iterator):
        self.activations_iterator=activations_iterator

class NormalizedMeasure(Measure):
    def __init__(self, activations_iterator):
        super().__init__(activations_iterator)

    def eval(self):
        print("Evaluating v_samples")
        v_samples=self.eval_v_samples()
        for layer in v_samples:
            print(layer.shape)
        print("layers",len(v_samples))
        print("Evaluating v_transformations")
        v_transformations=self.eval_v_transformations()
        print("layers", len(v_transformations))
        print("Evaluating v_normalized")
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
        n_intermediates = len(self.activations_iterator.layer_names())
        mean_variances_running = [RunningMean() for i in range(n_intermediates)]

        for transformation, batch_activations in self.activations_iterator.transformations_first():
            samples_variances_running = [RunningMeanAndVariance() for i in range(n_intermediates)]
            for x, batch_activation in batch_activations:
                for j, m in enumerate(batch_activation):
                    samples_variances_running[j].update(m)
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



class BatchvarMeasure:
    def __init__(self,batch_size,n_intermediates):
        self.batch_size=batch_size
        self.n_intermediates=n_intermediates
        self.batch_stats = [[RunningMeanAndVariance() for i in range(batch_size)] for j in range(n_intermediates)]
        self.batch_stats = np.array(self.batch_stats)

    def update(self,batch_activations):
        for i, layer_activations in enumerate(batch_activations):
            for j in range(layer_activations.shape[0]):
                self.batch_stats[i, j].update(layer_activations[j, :])

    def update_global_measures(self,dataset_stats):
        for i in range(self.n_intermediates):
            mean_var=dataset_stats[i]
            for j in range(self.batch_size):
                mean_var.update(self.batch_stats[i, j].std())


def transform_activations(activations_gpu,conv_aggregation_function):
    activations = activations_gpu.detach().cpu().numpy()

    # if conv average out spatial dims
    if len(activations.shape) == 4:
        n, c, w, h = activations.shape
        flat_activations = np.zeros((n, c))
        for i in range(n):
            if conv_aggregation_function=="mean":
                flat_activations[i, :] = np.nanmean(activations[i, :, :, :],axis=(1, 2))
            elif conv_aggregation_function=="max":
                flat_activations[i, :] = np.nanmax(activations[i, :, :, :],axis=(1, 2))
            elif conv_aggregation_function=="sum":
                flat_activations[i, :] = np.nansum(activations[i, :, :, :],axis=(1, 2))
            else:
                raise ValueError(f"Invalid aggregation function: {conv_aggregation_function}. Options: mean, max, sum")
        assert (len(flat_activations.shape) == 2)
    else:
        flat_activations = activations
    return flat_activations

def global_average_variance(result):
    rm=RunningMean()
    for layers in result:
        for layer in layers:
            for act in layer[:]:
                if np.isfinite(act):
                    rm.update(act)
    return rm.mean()




def run_models(models,dataset,version, config, n_rotations,conv_aggregation_function,batch_size=256):
    rotations = np.linspace(-180, 180, n_rotations, endpoint=False)
    results={}
    for model in models:
        if version=="stratified":
            variance, rotated_stratified_layer_vars, classes = variance_stratified(model, dataset, config, rotations, conv_aggregation_function, batch_size=batch_size, )
            results[model.name] = (variance,classes,rotated_stratified_layer_vars)
        else:

            variance, classes = eval(model, dataset, config,
                                                    rotations, conv_aggregation_function, batch_size=batch_size)
            results[model.name] = (variance, classes)




# def run_models(model, rotated_model, dataset, config, n_rotations, conv_aggregation_function, batch_size=256):
#     rotations = np.linspace(-180, 180, n_rotations, endpoint=False)
#
#     print("Calculating variance for all samples by class...")
#     rotated_var, rotated_stratified_layer_vars, classes = variance_stratified(rotated_model, dataset, config, rotations, conv_aggregation_function, batch_size=batch_size, )
#
#     var, stratified_layer_vars, classes = variance_stratified(model, dataset, config, rotations, conv_aggregation_function, batch_size=batch_size)
#
#     # Plot variance for all
#     print("Calculating variance for all samples...")
#     rotated_var_all_dataset, classes = eval(rotated_model, dataset, config,
#                                             rotations, conv_aggregation_function, batch_size=batch_size)
#     var_all_dataset, classes = eval(model, dataset, config, rotations, conv_aggregation_function, batch_size=batch_size)
#
#     return var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset
