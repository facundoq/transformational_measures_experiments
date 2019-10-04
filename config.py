import os,pickle
from experiment import variance


def base_path():
    return os.path.expanduser("~/variance/")


from experiment import training


def models_folder():
    model_folderpath = os.path.join(base_path(), "models")
    os.makedirs(model_folderpath, exist_ok=True)
    return model_folderpath

def model_path(p: training.Parameters,savepoint=None,model_folderpath= models_folder()):

    filename=f"{p.id(savepoint=savepoint)}.pt"
    filepath=os.path.join(model_folderpath,filename)
    return filepath



def get_models_filenames():
    files=os.listdir(models_folder())
    model_filenames=[f for f in files if f.endswith(".pt")]
    return model_filenames
def get_models_filepaths():
    model_folderpath = models_folder()
    return [os.path.join(model_folderpath,f) for f in get_models_filenames()]

def training_plots_path():
    plots_folderpath = "training_plots"
    plots_folderpath = os.path.join(base_path(), plots_folderpath)
    os.makedirs(plots_folderpath, exist_ok=True)
    return plots_folderpath




def variance_results_folder()->str:
    return os.path.join(base_path(), "results")



def results_paths(ps:[variance.Parameters], results_folder=variance_results_folder()):
    variance_paths= [f'{results_path(p)}' for p in ps]
    return variance_paths

def results_path(p:variance.Parameters, results_folder=variance_results_folder()):
    return  os.path.join(results_folder, f"{p.id()}.pickle")

def save_results(r:variance.VarianceExperimentResult, results_folder=variance_results_folder()):
    path = results_path(r.parameters, results_folder)
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(r,open(path,"wb"))

def load_result(path)->variance.VarianceExperimentResult:
    return pickle.load(open(path, "rb"))


def load_results(filepaths:[str])-> [variance.VarianceExperimentResult]:
    results = []
    for filepath in filepaths:
        result = load_result(filepath)
        results.append(result)
    return results

def load_all_results(folderpath:str)-> [variance.VarianceExperimentResult]:
    filepaths=[os.path.join(folderpath, filename) for filename in os.listdir(folderpath)]
    filepaths= [ f for f in filepaths if os.path.isfile(f)]
    return load_results(filepaths)


def plots_base_folder():
    return os.path.join(base_path(), "plots")

# def plots_folder(r:VarianceExperimentResult):
#     folderpath = os.path.join(plots_base_folder(), f"{r.id()}")
#     if not os.path.exists(folderpath):
#         os.makedirs(folderpath,exist_ok=True)
#     return folderpath




from transformation_measure import *


def all_measures()-> [Measure]:
    cas=[ConvAggregation.max,ConvAggregation.sum,ConvAggregation.mean,ConvAggregation.min,ConvAggregation.none]
    measures=[]
    for ca in cas:
        measures.append(SampleMeasure(MeasureFunction.std,ca))
        measures.append(TransformationMeasure(MeasureFunction.std, ca))
        measures.append(NormalizedMeasure(MeasureFunction.std, ca))

    measures.append(AnovaFMeasure(ConvAggregation.none))
    alphas=[0.90, 0.95, 0.99, 0.999]
    for alpha in alphas:
        measures.append(AnovaMeasure(ConvAggregation.none, alpha=alpha, bonferroni=True ))
        measures.append(AnovaMeasure(ConvAggregation.none, alpha=alpha, bonferroni=False))
    return measures


def common_measures()-> [Measure]:
    measures=[ SampleMeasure(MeasureFunction.std, ConvAggregation.sum)
             ,TransformationMeasure(MeasureFunction.std, ConvAggregation.sum)
     ,NormalizedMeasure(MeasureFunction.std, ConvAggregation.sum)
        ,AnovaFMeasure(ConvAggregation.none)
        ,AnovaMeasure(ConvAggregation.none,alpha=0.95)
        ,AnovaMeasure(ConvAggregation.none, alpha=0.95,bonferroni=True)

    ]
    return measures

def common_transformations() -> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator()]
    return transformations+common_transformations_without_identity()

def common_transformations_without_identity()-> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator(n_rotations=16)
        , SimpleAffineTransformationGenerator(n_translations=2)
        , SimpleAffineTransformationGenerator(n_scales=2)]
    return transformations

def rotation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(n_rotations=2**r) for r in range(1,n+1)]

def scale_transformations(n:int):
    return [SimpleAffineTransformationGenerator(n_scales=2**r) for r in range(n)]

def translation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(n_translations=r) for r in range(1,n+1)]

def all_transformations(n:int):
    return common_transformations()+rotation_transformations(n)+scale_transformations(n)+translation_transformations(n)



def common_dataset_sizes()->[float]:
    return [0.01,0.02,0.05,0.1,0.5,1.0]



import models
import transformation_measure as tm
import numpy as np


model_names=models.names

def get_epochs(model: str, dataset: str, t: tm.TransformationSet) -> int:
    if model == models.SimpleConv.__name__ or model == models.SimpleConvBN.__name__:
        epochs = {'cifar10': 40, 'mnist': 10, 'fashion_mnist': 12}
    elif model == models.AllConvolutional.__name__ or model == models.AllConvolutionalBN.__name__:
        epochs = {'cifar10': 50, 'mnist': 40, 'fashion_mnist': 12}
    elif model == models.VGGLike.__name__ or model == models.VGGLikeBN.__name__:
        epochs = {'cifar10': 50, 'mnist': 40, 'fashion_mnist': 12, }
    elif model == models.ResNet.__name__:
        epochs = {'cifar10': 60, 'mnist': 40, 'fashion_mnist': 12}
    elif model == models.FFNet.__name__ or model == models.FFNetBN.__name__:
        epochs = {'cifar10': 20, 'mnist': 15, 'fashion_mnist': 8}
    else:
        raise ValueError(f"Model \"{model}\" does not exist. Choices: {', '.join(models.names)}")


    n = len(t)
    if n > np.e:
        factor = 1.1 * np.log(n)
    else:
        factor = 1

    if not model.endswith("BN"):
        factor *= 2

    return int(epochs[dataset] * factor)

import models
def min_accuracy(model:str,dataset:str)-> float:
    min_accuracies = {"mnist": .95, "cifar10": .5}
    min_accuracy = min_accuracies[dataset]

    if dataset == "mnist" and model == models.FFNet.__name__:
        min_accuracy = 0.85
    if dataset == "mnist" and model == models.FFNetBN.__name__:
        min_accuracy = 0.85

    if dataset == "cifar10" and model == models.FFNet.__name__:
        min_accuracy = 0.45
    if dataset == "cifar10" and model == models.FFNetBN.__name__:
        min_accuracy = 0.45


    return min_accuracy