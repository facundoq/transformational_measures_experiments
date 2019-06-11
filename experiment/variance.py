import os
import pickle
import typing



import transformation_measure as tm
from enum import Enum
class DatasetSubset(Enum):
    train="train"
    test="test"

class DatasetParameters:
    def __init__(self,name:str,subset: DatasetSubset,percentage:float):
        assert(percentage>0)
        assert(percentage<=1)
        self.name=name
        self.subset=subset
        self.percentage=percentage
    def __repr__(self):
        return f"{self.name}_{self.subset.value}_p{self.percentage:.2}"

class Parameters:
    def __init__(self,model:str,dataset:DatasetParameters,transformations:typing.Iterable[typing.Callable],measure:tm.Measure):
        self.model=model
        self.dataset=dataset
        self.measure=measure
        self.transformations=transformations

    def id(self):
        return f"{self.model}_{self.dataset}_{self.transformations}{self.measure}"

    def __repr__(self):
        return self.id()


from pytorch.models import SimpleConv,AllConvolutional,ResNet

datasets=["mnist","cifar10"]
models=[SimpleConv.__name__
        ,AllConvolutional.__name__
        ,ResNet.__name__]


def possible_experiment_parameters():


    transformations=[tm.SimpleAffineTransformationGenerator(n_rotations=16)]
    measures=[tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum)
              ,tm.NormalizedMeasure(tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum),tm.SampleMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum))
              ,
              ]
    dataset_percentages = [.1, .5, 1.0]
    dataset_subsets=[DatasetSubset.train,DatasetSubset.test]
    dataset_parameters=[]
    for dataset in datasets:
        for dataset_subset in dataset_subsets:
            for dataset_percentage in dataset_percentages:
                dataset_parameters.append(DatasetParameters(dataset,dataset_subset,dataset_percentage))
    parameters=[datasets, transformations, measures, models]

    def list2dict(list):
        return {str(x): x for x in list}
    parameters=[ list2dict(p) for p in parameters ]
    return parameters

import argcomplete, argparse

def parse_parameters()->Parameters:
    datasets, transformations, measures, models=possible_experiment_parameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", choices=models.keys(),required=True)
    parser.add_argument("-dataset", choices=datasets.keys(),required=True)
    parser.add_argument("-measure", choices=measures.keys(),required=True)
    parser.add_argument("-transformation", choices=transformations.keys(),required=True)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    print(args)

    return Parameters(models[args.model],
                      datasets[args.dataset],
                      transformations[args.transformation],
                      measures[args.measure])


class VarianceExperimentResult:
    def __init__(self, parameters:Parameters, activation_names, class_names, transformations, options, rotated_measures, unrotated_measures):

        self.parameters=parameters

        self.activation_names = activation_names
        self.class_names = class_names
        self.rotated_measures=rotated_measures
        self.unrotated_measures=unrotated_measures
    def description(self):
        description = "-".join([str(v) for v in self.parameters.values()])
        return description


results_folder=os.path.expanduser("~/variance_results/values")

def get_path(model_name,dataset_name,description):
    return os.path.join(results_folder, f"{model_name}_{dataset_name}_{description}.pickle")


def save_results(r:VarianceExperimentResult):
    path=get_path(r.parameters.model,r.parameters.dataset,r.description())
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(r,open(path,"wb"))

def load_results(path)->VarianceExperimentResult:
    return pickle.load(open(path, "rb"))

def plots_base_folder():
    return os.path.expanduser("~/variance_results/plots/")

def plots_folder(r:VarianceExperimentResult):
    folderpath = os.path.join(plots_base_folder(), f"{r.parameters.model}_{r.parameters.dataset}_{r.description()}")

    if not os.path.exists(folderpath):
        os.makedirs(folderpath,exist_ok=True)
    return folderpath
