from __future__ import annotations
from enum import Enum
import datasets
from pathlib import Path
import transformational_measures as tm
import os

import transformational_measures.measure

import abc
class DatasetSize(abc.ABC):
    @abc.abstractmethod
    def get_size(self, original_size: int):
        pass

class DatasetSizePercentage(DatasetSize):
    def __init__(self,percentage:float):
        assert (percentage > 0)
        assert (percentage <= 1)
        self.percentage = percentage
    def get_size(self,original_size:int):
        return int(original_size*self.percentage)
    def __repr__(self):
        return f"Percentage({self.percentage:.2})"

class DatasetSizeFixed(DatasetSize):
    def __init__(self,size:int):
        self.size=size

    def get_size(self,original_size:int):
        return self.size
    def __repr__(self):
        return f"Fixed({self.size})"

class DatasetParameters:
    def __init__(self,name:str,subset: datasets.DatasetSubset,size:DatasetSize):
        assert(name in datasets.names)
        self.name=name
        self.subset=subset
        self.size=size

    def __repr__(self):
        return f"{self.name}({self.subset.value},size={self.size})"
    def id(self):
        return str(self)


class Parameters:
    def __init__(self, model_id:str, dataset:DatasetParameters, transformations:tm.pytorch.PyTorchTransformationSet, measure:tm.numpy.NumpyMeasure, stratified:bool=False,suffix=None):
        self.model_id=model_id
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations
        self.stratified=stratified
        self.suffix=suffix

    def id(self):
        measure=self.measure.id()

        if self.stratified:
            measure=f"Stratified({measure})"
        suffix = "" if self.suffix is None else f"_{self.suffix}"
        return f"{self.model_id}/{self.dataset}_{self.transformations.id()}_{measure}{suffix}"

    def __repr__(self):
        measure = self.measure.id()
        if self.stratified:
            measure = f"Stratified({measure})"
        suffix = "" if self.suffix is None else f", {self.suffix}"
        return f"Parameters(model={self.model_id}, dataset={self.dataset}, transformations={self.transformations}, measure={measure}{suffix})"

class Options:
    def __init__(self,verbose:bool,batch_size:int,num_workers:int,adapt_dataset:bool):
        self.verbose=verbose
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.adapt_dataset=adapt_dataset
    def __repr__(self):
        return f"Options(verbose={self.verbose},batch_size={self.batch_size},num_workers={self.num_workers},adapt_dataset={self.adapt_dataset})"

class MeasureExperimentResult:
    def __init__(self, parameters:Parameters, measure_result: transformational_measures.measure.MeasureResult):
        self.parameters=parameters
        self.measure_result=measure_result

    def __repr__(self):
        return f"{MeasureExperimentResult.__name__}({self.parameters})"







def non_filter(model:tm.pytorch.ObservableLayersModule,name:str): return True

class PyTorchParameters:
    def __init__(self, model_id:str, dataset:DatasetParameters, transformations:tm.pytorch.PyTorchTransformationSet,
                 measure:tm.pytorch.PyTorchMeasure,options:tm.pytorch.PyTorchMeasureOptions,
                 adapt_dataset=False,
                 stratified:bool=False,suffix=None,model_filter:tm.pytorch.model.ActivationFilter=non_filter):
        self.model_id=model_id
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations
        self.stratified=stratified
        self.suffix=suffix
        self.options=options
        self.adapt_dataset=adapt_dataset
        self.model_filter=model_filter

    def id(self):
        measure=self.measure.id()

        if self.stratified:
            measure=f"Stratified({measure})"
        suffix = "" if self.suffix is None else f"_{self.suffix}"
        return f"{self.model_id}/{self.dataset}_{self.transformations.id()}_{measure}{suffix}"

    def __repr__(self):
        measure = self.measure.id()
        if self.stratified:
            measure = f"Stratified({measure})"
        suffix = "" if self.suffix is None else f", {self.suffix}"
        return f"Parameters(model={self.model_id}, dataset={self.dataset}, transformations={self.transformations}, measure={measure}{suffix})"


class PyTorchMeasureExperimentResult:
    def __init__(self, parameters:PyTorchParameters, measure_result: tm.pytorch.PyTorchMeasureResult):
        self.parameters=parameters
        self.measure_result=measure_result

    def __repr__(self):
        return f"{PyTorchMeasureExperimentResult.__name__}({self.parameters})"
