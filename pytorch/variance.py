import os
import pickle
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
    def __init__(self,model:str,dataset:DatasetParameters,transformations:typing.Iterable[typing.Callable],measure:tm.Measure):
        self.model=model
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations

    def id(self):
        return f"{self.model}_{self.dataset}_{self.transformations}{self.measure}"

    def __repr__(self):
        return f"VarianceExperiment parameters: model={self.model}, dataset={self.dataset} transformations={self.transformations}, measure={self.measure}"

class Options:
    def __init__(self,verbose:bool,batch_size:int):
        self.verbose=verbose
        self.batch_size=batch_size


dataset_names=["mnist","cifar10"]

from pytorch.experiment import training
def possible_experiment_parameters():
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
    return parameters+[training.get_models()]

import argcomplete, argparse


def parse_parameters()->typing.Tuple[Parameters,Options]:
    bool_parser = lambda x: (str(x).lower() in ['true', '1', 'yes'])

    datasets, measures, transformations, models,=possible_experiment_parameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", metavar="mo", choices=models,required=True)
    parser.add_argument("-dataset", metavar="d", choices=datasets.keys(),required=True)
    parser.add_argument("-measure", metavar="me", choices=measures.keys(),required=True)
    parser.add_argument("-transformation", metavar="t", choices=transformations.keys(),required=True)
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/model/transformations')
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
    def __init__(self, parameters:Parameters, measure_result,stratified_measure_result):
        self.parameters=parameters
        self.measure_result=measure_result
        self.stratified_measure_result=stratified_measure_result

    def __repr__(self):
        description = f"VarianceExperimentResult for {self.parameters}"
        return description

    def id(self):
        return f"{self.parameters.id()}"

def base_folder()->str: return os.path.expanduser("~/variance/")


def default_results_folder()->str:
    return os.path.join(base_folder(),"results")

def save_results(r:VarianceExperimentResult,results_folder=default_results_folder()):
    path = os.path.join(results_folder, f"{r.id()}.pickle")
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(r,open(path,"wb"))

def load_results(path)->VarianceExperimentResult:
    return pickle.load(open(path, "rb"))

def plots_base_folder():
    return os.path.join(base_folder(),"plots")

def plots_folder(r:VarianceExperimentResult):
    folderpath = os.path.join(plots_base_folder(), f"{r.id()}")
    if not os.path.exists(folderpath):
        os.makedirs(folderpath,exist_ok=True)
    return folderpath
