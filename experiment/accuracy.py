import os
import typing
from experiment import training
import datasets, torch
import pytorch
from torch import nn
import transformation_measure as tm
from enum import Enum
from pathlib import Path
from transformation_measure.iterators.pytorch_image_dataset import TransformationStrategy

class DatasetParameters:
    def __init__(self,name:str,subset: datasets.DatasetSubset,percentage:float):
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
    def __init__(self, model_path:Path, dataset:DatasetParameters, transformations:tm.TransformationSet):
        self.model_path=model_path
        self.dataset=dataset
        self.transformations=transformations

    def model_name(self):
        return self.model_path.stem

    def id(self):
        return f"{self.model_name()}_{self.dataset}_{self.transformations.id()}"

    def __repr__(self):

        return f"AccuracyExperiment parameters: models={self.model_name()}, dataset={self.dataset} transformations={self.transformations}"

class Options:
    def __init__(self,verbose:bool,batch_size:int,num_workers:int,use_cuda:bool):
        self.verbose=verbose
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda

    def get_eval_options(self):
        return training.EvalOptions(self.use_cuda,self.batch_size,self.num_workers)

class AccuracyExperimentResult:
    #TODO change name to MeasureExperimentResult
    def __init__(self, parameters:Parameters,accuracy:float):
        self.parameters=parameters
        self.accuracy=accuracy

    def __repr__(self):
        description = f"{AccuracyExperimentResult.__name__}, params: {self.parameters}, accuracy:{self.accuracy}"
        return description

    def id(self):
        return f"{self.parameters.id()}"




import config

def possible_experiment_parameters()->[]:
    transformations = config.all_transformations()

    dataset_percentages = config.common_dataset_sizes()
    dataset_subsets=[datasets.DatasetSubset.train,datasets.DatasetSubset.test]

    dataset_parameters=[]
    for dataset in datasets.names:
        for dataset_subset in dataset_subsets:
            for dataset_percentage in dataset_percentages:
                dataset_parameters.append(DatasetParameters(dataset,dataset_subset,dataset_percentage))

    parameters=[dataset_parameters, transformations]
    def list2dict(list):
        return {x.id(): x for x in list}
    parameters=[ list2dict(p) for p in parameters ]

    return parameters

import argcomplete, argparse


def experiment(p: Parameters, o: Options):
    assert(len(p.transformations)>0)
    use_cuda = torch.cuda.is_available()

    model, training_parameters, training_options, scores = training.load_model(p.model_path, use_cuda)

    if o.verbose:
        print("### ", model)
        print("### Scores obtained:")
        training.print_scores(scores)

    dataset = datasets.get(p.dataset.name)
    dataset = dataset.reduce_size_stratified(p.dataset.percentage)
    if o.verbose:
        print(dataset.summary())

    p.transformations.set_input_shape(dataset.input_shape)
    p.transformations.set_pytorch(False)
    #p.transformations.set_cuda(use_cuda)

    if o.verbose:
        print(f"Measuring accuracy with transformations {p.transformations} on dataset {p.dataset} of size {dataset.size(p.dataset.subset)}...")

    result:float=measure(model,dataset,p.transformations,o,p.dataset.subset)

    return AccuracyExperimentResult(p, result)

def measure(model:nn.Module,dataset:datasets.ClassificationDataset,transformations:tm.TransformationSet,o:Options,subset:datasets.DatasetSubset)-> float:

    scores = training.eval_scores(model,dataset,transformations,TransformationStrategy.iterate_all,o.get_eval_options(),subsets=subset.value)
    loss, accuracy = scores[subset.value]
    return accuracy

def parse_parameters()->typing.Tuple[Parameters,Options]:
    bool_parser = lambda x: (str(x).lower() in ['true', '1', 'yes'])

    def is_valid_file(filepath):
        filepath=Path(filepath)
        if not filepath.exists():
            raise argparse.ArgumentTypeError("The model file %s does not exist!" % filepath)
        else:
            return filepath

    datasets,  transformations=possible_experiment_parameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", metavar="mo",type=is_valid_file,required=True)
    parser.add_argument("-dataset", metavar="d", choices=datasets.keys(),required=True)
    parser.add_argument("-transformation", metavar="t", choices=transformations.keys(),required=True)
    parser.add_argument('-verbose', metavar='v',type=bool_parser, default=True,
                        help=f'Print info about dataset/models/transformations')
    parser.add_argument('-num_workers', metavar='nw'
                        , help=f'num_workers to use during evaluation'
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
                   transformations[args.transformation])
    o = Options(args.verbose,args.batchsize,args.num_workers,torch.cuda.is_available())
    return p,o
