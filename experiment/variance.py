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
        return f"{self.name}_{self.subset.value}_p{self.percentage:0.2}"

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
models=[SimpleConv.__class__.__name__
        ,AllConvolutional.__class__.__name__
        ,ResNet.__class__.__name__]


import numpy as np
n_rotations=16
rotations = np.linspace(-np.pi, np.pi, n_rotations, endpoint=False)
dataset_percentages=[.1,.5,1]
transformations_parameters={"rotation":rotations,"scale":[(1, 1)],"translation":[(0,0)]}

transformations_parameters_combinations=tm.generate_transformation_parameter_combinations(transformations_parameters)

transformations=tm.generate_transformations(transformations_parameters_combinations)

measures=[]

dataset_subsets=[DatasetSubset.train,DatasetSubset.test]

experiment_parameters=[]
for model in models:
    for dataset in datasets:
        for dataset_subset in dataset_subsets:
            for dataset_percentage in dataset_percentages:
                dataset_parameters=DatasetParameters(dataset,dataset_subset,dataset_percentage)
                for transformation in transformations:
                    for measure in measures:
                        p=Parameters(model,dataset_parameters,transformation,measure)
                        experiment_parameters.append(p)


experiment_parameters={p.id():p for p in experiment_parameters}

import argcomplete, argparse

def parse_parameters()->Parameters:
    print("init")
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", choices=('http', 'https', 'ssh', 'rsync', 'wss'))
    argcomplete.autocomplete(parser)
    print("parseargs")
    args = parser.parse_args()
    print("after")

    return experiment_parameters[args.parameters]


class VarianceExperimentResult:
    def __init__(self, parameters:Parameters, activation_names, class_names, transformations, options, rotated_measures, unrotated_measures):

        self.parameters=parameters

        self.activation_names = activation_names
        self.class_names = class_names
        self.rotated_measures=rotated_measures
        self.unrotated_measures=unrotated_measures
    def description(self):
        description = "-".join([str(v) for v in self.options.values()])
        return description


results_folder=os.path.expanduser("~/variance_results/values")

def get_path(model_name,dataset_name,description):
    return os.path.join(results_folder, f"{model_name}_{dataset_name}_{description}.pickle")


def save_results(r:VarianceExperimentResult):
    path=get_path(r.model_name,r.dataset_name,r.description())
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(r,open(path,"wb"))

def load_results(path)->VarianceExperimentResult:
    return pickle.load(open(path, "rb"))

def plots_base_folder():
    return os.path.expanduser("~/variance_results/plots/")

def plots_folder(r:VarianceExperimentResult):
    folderpath = os.path.join(plots_base_folder(), f"{r.model_name}_{r.dataset_name}_{r.description()}")

    if not os.path.exists(folderpath):
        os.makedirs(folderpath,exist_ok=True)
    return folderpath
