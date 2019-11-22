
import abc
from transformation_measure.iterators.activations_iterator import ActivationsIterator
import numpy as np
from typing import  List
from .stats_running import RunningMean
from .layer_transformation import ConvAggregation

ActivationsByLayer = [np.ndarray]

class MeasureResult:
    def __init__(self,layers:ActivationsByLayer,layer_names:List[str],measure:'Measure',extra_values=dict()):
        self.layers=layers
        self.layer_names=layer_names
        self.measure=measure
        self.extra_values=extra_values

    def __repr__(self):
        return f"MeasureResult {self.measure}"

    def all_1d(self):
        return np.any([ len(l.shape)==1 for l in self.layers])

    def per_layer_average(self) -> np.ndarray:

        result = []
        for layer in self.layers:
            layer=layer[:]
            layer_average=layer[np.isfinite(layer)].mean()
            result.append(layer_average)
        return np.array(result)

    def weighted_global_average(self):
        return self.per_layer_average().mean()

    def global_average(self)-> float:
        rm = RunningMean()
        for layer in self.layers:
                for act in layer[:]:
                    if np.isfinite(act):
                        rm.update(act)
        return rm.mean()

    def collapse_convolutions(self,conv_aggregation_function:ConvAggregation):
        results=[]
        for layer in self.layers:
            # if conv average out spatial dims
            if len(layer.shape) == 3:
                flat_activations=conv_aggregation_function.apply3D(layer)
                assert len(flat_activations.shape) == 1,f"After collapsing, the activation shape should have only one dimension. Found vector with shape {flat_activations.shape} instead."
            else:
                flat_activations = layer.copy()
            results.append(flat_activations)

        return MeasureResult(results,self.layer_names,self.measure)


class StratifiedMeasureResult(MeasureResult):
    def __init__(self,layers:ActivationsByLayer,layer_names:List[str],measure:'Measure'
                 ,class_measures:List[MeasureResult],class_labels:List[str]):
        super().__init__(layers,layer_names,measure)
        self.class_measures=class_measures
        self.class_labels=class_labels

    def __repr__(self):
        return f"StratifiedMeasureResult {self.measure}"


# TODO use abc.ABC as base class
class Measure():

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def id(self):
        return str(self)

    @abc.abstractmethod
    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        '''

        '''
        pass


    def eval_stratified(self,classes_iterators:[ActivationsIterator],class_labels:[str])-> StratifiedMeasureResult:
        '''
        Calculate the `variance_measure` for each class separately
        Also calculate the average stratified `variance_measure` over all classes
        '''
        variance_per_class = [self.eval(iterator) for iterator in classes_iterators]
        stratified_measure_layers = self.mean_variance_over_classes(variance_per_class)
        layer_names=classes_iterators[0].activation_names()

        return StratifiedMeasureResult(stratified_measure_layers,layer_names,self,variance_per_class,class_labels)


    def mean_variance_over_classes(self,class_variance_result:List[MeasureResult]) -> List[np.ndarray]:
        # calculate the mean activation of each unit in each layer over the set of classes
        class_variance_layers=[r.layers for r in class_variance_result]
        # class_variance_layers is a list (classes) of list layers)
        # turn it into a list (layers) of lists (classes)
        layer_class_vars=[list(i) for i in zip(*class_variance_layers)]
        # compute average variance of each layer over classses
        layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
        return layer_vars

