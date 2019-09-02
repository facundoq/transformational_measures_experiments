from .base import Measure,MeasureFunction,MeasureResult,ActivationsByLayer
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from transformation_measure.measure.running_stats import RunningMeanAndVariance,RunningMean
from .layer_transformation import ConvAggregation,apply_aggregation_function
from typing import List
from .transformations import TransformationMeasure
import scipy.stats

class AnovaMeasure(Measure):
    def __init__(self, conv_aggregation: ConvAggregation,alpha:float):
        super().__init__()
        self.anova_f_measure=AnovaFMeasure(conv_aggregation)
        assert(alpha>0)
        assert (alpha <1)
        self.alpha=alpha


    def __repr__(self):
        return f"AnovaMeasure(ca={self.anova_f_measure.conv_aggregation.value})"

    def eval(self, activations_iterator: ActivationsIterator, layer_names: List[str]) -> MeasureResult:
        f_result=self.anova_f_measure.eval(activations_iterator,layer_names)
        d_b=f_result.extra_values["d_b"]
        d_w = f_result.extra_values["d_w"]
        critical_value = scipy.stats.f.ppf(self.alpha,dfn=d_b,dfd=d_w)
        f_result.layers=[ f>critical_value for f in f_result.layers]
        return f_result

class AnovaFMeasure(Measure):
    def __init__(self, conv_aggregation: ConvAggregation):
        super().__init__()
        self.conv_aggregation = conv_aggregation


    def __repr__(self):
        return f"AnovaFMeasure(ca={self.conv_aggregation.value})"

    def eval(self,activations_iterator:ActivationsIterator,layer_names:List[str])->MeasureResult:

        means_per_layer_and_transformation,samples_per_transformation=self.eval_means_per_transformation(activations_iterator)
        n_layers=len(activations_iterator.activation_names())
        global_means=self.eval_global_means(means_per_layer_and_transformation,n_layers)
        ssdb_per_layer,d_b=self.eval_between_transformations_ssd(means_per_layer_and_transformation,global_means,samples_per_transformation)
        ssdw_per_layer, d_w=self.eval_within_transformations_ssd(activations_iterator, means_per_layer_and_transformation)

        f_score=self.divide_per_layer(ssdb_per_layer,ssdw_per_layer)

        return MeasureResult(f_score,layer_names,self,extra_values={"d_b":d_b,"d_w":d_w})

    def divide_per_layer(self,a_per_layer:ActivationsByLayer,b_per_layer:ActivationsByLayer)->(ActivationsByLayer,int):
        return [a/b for (a,b) in zip(a_per_layer,b_per_layer)]

    def eval_within_transformations_ssd(self,activations_iterator:ActivationsIterator,means_per_layer_and_transformation:[ActivationsByLayer],)->ActivationsByLayer:
            n_layers = len(activations_iterator.activation_names())

            ssdw_per_layer = [0] * n_layers
            samples_per_transformation = []
            for means_per_layer,(transformation, transformation_activations) in zip(means_per_layer_and_transformation, activations_iterator.transformations_first()):

                # calculate the variance of all samples for this transformation
                n_samples = 0
                for x, batch_activations in transformation_activations:
                    n_samples += x.shape[0]
                    for j, layer_activations in enumerate(batch_activations):
                        for i in range(layer_activations.shape[0]):
                            layer_activations = self.preprocess_activations(layer_activations)
                            d=(layer_activations[i,]-means_per_layer[j])**2
                            ssdw_per_layer[j]=ssdw_per_layer[j]+d
                samples_per_transformation.append(n_samples)
            # divide by degrees of freedom
            degrees_of_freedom = (samples_per_transformation[0] - 1) * len(samples_per_transformation)
            ssdw_per_layer = [s / degrees_of_freedom for s in ssdw_per_layer]
            return ssdw_per_layer,degrees_of_freedom

    def eval_between_transformations_ssd(self,means_per_layer_and_transformation:[ActivationsByLayer],global_means:ActivationsByLayer,samples_per_transformation:[int])->(ActivationsByLayer,int):
        '''

        :param means_per_transformation: has len(transformations), each item has len(layers)
        :param global_means: has len(layers)
        :return:
        '''
        n_layers=len(global_means)
        n_transformations=len(means_per_layer_and_transformation)
        ssdb_per_layer=[0]*n_layers
        for  i,means_per_layer in enumerate(means_per_layer_and_transformation):
            n=samples_per_transformation[i]
            for j in range(n_layers):
                ssdb_per_layer[j] += n* ((means_per_layer[j]-global_means[j])**2)
        degrees_of_freedom=(n_transformations-1)
        for j in range(n_layers):
            ssdb_per_layer[j]/=degrees_of_freedom
        return ssdb_per_layer,degrees_of_freedom

    def eval_global_means(self, means_per_layer_and_transformation:[[np.ndarray]], n_layers:int)->ActivationsByLayer:
        n_transformations=len(means_per_layer_and_transformation)
        global_means_running = [RunningMean() for i in range(n_layers)]
        for means_per_layer in means_per_layer_and_transformation:
            for i,means in enumerate(means_per_layer):
                global_means_running[i].update(means)
        return [rm.mean()/n_transformations for rm in global_means_running]




    def eval_means_per_transformation(self,activations_iterator:ActivationsIterator)->([ActivationsByLayer],[int]):
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
            means_per_transformation.append([rm.mean() for rm in samples_variances_running])
        return means_per_transformation,samples_per_transformation

    def preprocess_activations(self, layer_activations):
        if len(layer_activations.shape) == 4:
            return apply_aggregation_function(layer_activations, self.conv_aggregation)
        else:
            return layer_activations

