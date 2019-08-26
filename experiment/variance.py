import os
import typing

import transformation_measure as tm
from enum import Enum

class DatasetSubset(Enum):
    train="train"
    test="test"
    values=[train,test]

class DatasetParameters:
    def __init__(self,name:str,subset: DatasetSubset,percentage:float):
        assert(percentage>0)
        assert(percentage<=1)
        self.name=name
        self.subset=subset
        self.percentage=percentage
    def __repr__(self):
        return f"{self.name}({self.subset.value},p={self.percentage:.2})"
    def id(self):
        return str(self)

class Parameters:
    def __init__(self, model_path:str, dataset:DatasetParameters, transformations:tm.TransformationSet, measure:tm.Measure):
        self.model_path=model_path
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations
    def model_name(self):
        base,filename_ext=os.path.split(self.model_path)
        filename,ext=os.path.splitext(filename_ext)
        return filename
    def id(self):

        return f"{self.model_name()}_{self.dataset}_{self.transformations.id()}_{self.measure.id()}"

    def __repr__(self):
        return f"VarianceExperiment parameters: models={self.model_name()}, dataset={self.dataset} transformations={self.transformations}, measure={self.measure}"

class Options:
    def __init__(self,verbose:bool,batch_size:int):
        self.verbose=verbose
        self.batch_size=batch_size


dataset_names=["mnist","cifar10"]

from experiment import training


def possible_experiment_parameters()->[]:
    transformations = tm.common_transformations()
    measures= tm.common_measures()

    dataset_percentages = [.1, .5, 1.0]
    dataset_subsets=[DatasetSubset.train,DatasetSubset.test]
    datasets=[]
    for dataset in dataset_names:
        for dataset_subset in dataset_subsets:
            for dataset_percentage in dataset_percentages:
                datasets.append(DatasetParameters(dataset,dataset_subset,dataset_percentage))

    parameters=[datasets, measures, transformations]
    def list2dict(list):
        return {x.id(): x for x in list}
    parameters=[ list2dict(p) for p in parameters ]

    return parameters

import argcomplete, argparse


def parse_parameters()->typing.Tuple[Parameters,Options]:
    bool_parser = lambda x: (str(x).lower() in ['true', '1', 'yes'])

    def is_valid_file(filepath):
        if not os.path.exists(filepath):
            raise argparse.ArgumentTypeError("The model file %s does not exist!" % filepath)
        else:
            return filepath

    datasets, measures, transformations=possible_experiment_parameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", metavar="mo",type=is_valid_file,required=True)
    parser.add_argument("-dataset", metavar="d", choices=datasets.keys(),required=True)
    parser.add_argument("-measure", metavar="me", choices=measures.keys(),required=True)
    parser.add_argument("-transformation", metavar="t", choices=transformations.keys(),required=True)
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/models/transformations')
    parser.add_argument('-batchsize', metavar='b'
                        , help=f'batchsize to use during training'
                        , type=int
                        , default=256)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()


    p = Parameters(args.model,
                   datasets[args.dataset],
                   transformations[args.transformation],
                   measures[args.measure])
    o = Options(args.verbose,args.batchsize)
    return p,o

class VarianceExperimentResult:
    def __init__(self, parameters:Parameters, measure_result:tm.MeasureResult,stratified_measure_result:tm.StratifiedMeasureResult):
        self.parameters=parameters
        self.measure_result=measure_result
        self.stratified_measure_result=stratified_measure_result

    def __repr__(self):
        description = f"VarianceExperimentResult, params: {self.parameters}"
        return description

    def id(self):
        return f"{self.parameters.id()}"

