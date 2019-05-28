import numpy as np
from typing import Dict, List, Tuple
from .result import MeasureResult
from variance_measure.iterators.activations_iterator import ActivationsIterator

class StratifiedMeasure:
    def __init__(self,classes_iterators:List[ActivationsIterator],variance_measure):
        '''

        :param classes_iterators:  A list of iterators, one for each class or subset
        :param variance_measure: A function that takes an iterator and returns a MeasureResult object
        '''
        self.classes_iterators=classes_iterators
        self.variance_measure=variance_measure


    def eval(self)-> (MeasureResult,MeasureResult):
        '''
        Calculate the `variance_measure` for each class separately
        Also calculate the average stratified `variance_measure` over all classes
        '''
        variance_per_class = [self.variance_measure(iterator)[0] for iterator in self.classes_iterators]
        source=str(variance_per_class[0])
        for i,v in enumerate(variance_per_class):
            v.source+=f"_class{i}"
        variance_stratified = self.mean_variance_over_classes(variance_per_class,source)
        return variance_stratified,variance_per_class


    def mean_variance_over_classes(self,class_variance_result:List[MeasureResult],source:str) -> MeasureResult:
        # calculate the mean activation of each unit in each layer over the set of classes
        class_variance_layers=[r.layers for r in class_variance_result]
        # class_variance_layers is a list (classes) of list layers)
        # turn it into a list (layers) of lists (classes)
        layer_class_vars=[list(i) for i in zip(*class_variance_layers)]
        # compute average variance of each layer over classses
        layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
        return MeasureResult(layer_vars,f"{source}_stratified")

