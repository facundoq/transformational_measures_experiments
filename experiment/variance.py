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
    def __init__(self, model_path:str, dataset:DatasetParameters, transformations:tm.TransformationSet, measure:tm.Measure,stratified:bool=False):
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
        return f"VarianceExperiment parameters: models={self.model_name()}, dataset={self.dataset} transformations={self.transformations}, measure={measure}"

class Options:
    def __init__(self,verbose:bool,batch_size:int,num_workers:int):
        self.verbose=verbose
        self.batch_size=batch_size
        self.num_workers=num_workers

class VarianceExperimentResult:
    def __init__(self, parameters:Parameters, measure_result:tm.MeasureResult):
        self.parameters=parameters
        self.measure_result=measure_result

    def __repr__(self):
        description = f"VarianceExperimentResult, params: {self.parameters}"
        return description

    def id(self):
        return f"{self.parameters.id()}"


dataset_names=["mnist","cifar10"]

import config

def possible_experiment_parameters()->[]:
    transformations = config.all_transformations(10)
    measures= config.common_measures()

    dataset_percentages = config.common_dataset_sizes()
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
    parser.add_argument("-stratified", metavar="stra",type=bool_parser,default=False)
    parser.add_argument("-transformation", metavar="t", choices=transformations.keys(),required=True)
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/models/transformations')

    parser.add_argument('-num_workers', metavar='nw'
                        , help=f'num_workersto use during training'
                        , type=int
                        , default=2)
    parser.add_argument('-batchsize', metavar='b'
                        , help=f'batchsize to use during eval'
                        , type=int
                        , default=256)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()


    p = Parameters(args.model,
                   datasets[args.dataset],
                   transformations[args.transformation],
                   measures[args.measure],stratified=args.stratified)
    o = Options(args.verbose,args.batchsize,args.num_workers)
    return p,o


