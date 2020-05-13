from enum import Enum
import datasets
from pathlib import Path
import transformation_measure as tm
import os


class DatasetParameters:
    def __init__(self,name:str,subset: datasets.DatasetSubset,percentage:float):
        assert(percentage>0)
        assert(percentage<=1)
        assert(name in datasets.names)
        self.name=name
        self.subset=subset
        self.percentage=percentage

    def __repr__(self):
        return f"{self.name}({self.subset.value},p={self.percentage:.2})"
    def id(self):
        return str(self)

class Parameters:
    def __init__(self, model_path:Path, dataset:DatasetParameters, transformations:tm.TransformationSet, measure:tm.Measure,stratified:bool=False):
        self.model_path=model_path
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations
        self.stratified=stratified

    def model_name(self):
        base,filename_ext=os.path.split(self.model_path)
        filename,ext=os.path.splitext(filename_ext)
        return filename

    def id(self):
        measure=self.measure.id()

        if self.stratified:
            measure=f"Stratified({measure})"
        return f"{self.model_name()}_{self.dataset}_{self.transformations.id()}_{measure}"

    def __repr__(self):
        measure = self.measure.id()
        if self.stratified:
            measure = f"Stratified({measure})"
        return f"VarianceExperiment(model={self.model_name()}, dataset={self.dataset}, transformations={self.transformations}, measure={measure})"

class Options:
    def __init__(self,verbose:bool,batch_size:int,num_workers:int,adapt_dataset:bool):
        self.verbose=verbose
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.adapt_dataset=adapt_dataset

    def __repr__(self):
        return f"Options(verbose={self.verbose},batch_size={self.batch_size},num_workers={self.num_workers},adapt_dataset={self.adapt_dataset})"
class MeasureExperimentResult:
    def __init__(self, parameters:Parameters, measure_result:tm.MeasureResult):
        self.parameters=parameters
        self.measure_result=measure_result

    def __repr__(self):
        description = f"VarianceExperimentResult, params: {self.parameters}"
        return description

    def id(self):
        return f"{self.parameters.id()}"

