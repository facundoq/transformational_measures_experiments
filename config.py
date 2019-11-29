import os,pickle
from experiment import variance,training,model_loading
import itertools
from pathlib import Path
import models

def base_path():
    return Path(os.path.expanduser("~/variance/"))

def testing_folder():
    return base_path() / "testing"

def models_folder():
    model_folderpath = base_path() / "models"
    model_folderpath.mkdir(parents=True, exist_ok=True)
    return model_folderpath

def model_path(p: training.Parameters,savepoint=None,model_folderpath= models_folder())->Path:
    filename=f"{p.id(savepoint=savepoint)}.pt"
    filepath=model_folderpath / filename
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



def heatmaps_folder()->Path:
    return base_path() / "heatmaps"

def results_folder()->Path:
    return base_path() / "results"



def results_paths(ps:[variance.Parameters], results_folder=results_folder())->[Path]:
    variance_paths= [results_path(p,results_folder) for p in ps]
    return variance_paths

def results_path(p:variance.Parameters, results_folder=results_folder())-> Path:
    return  results_folder / f"{p.id()}.pickle"

def save_results(r:variance.VarianceExperimentResult, results_folder=results_folder()):
    path = results_path(r.parameters, results_folder)
    basename:Path = path.parent
    basename.mkdir(exist_ok=True,parents=True)
    pickle.dump(r,path.open(mode="wb"))

def load_result(path:Path)->variance.VarianceExperimentResult:
    return pickle.load(path.open(mode="rb"))


def load_results(filepaths:[Path])-> [variance.VarianceExperimentResult]:
    results = []
    for filepath in filepaths:
        result = load_result(filepath)
        results.append(result)
    return results

def load_all_results(folderpath:Path)-> [variance.VarianceExperimentResult]:
    filepaths=[f for f in folderpath.iterdir() if f.is_file()]
    return load_results(filepaths)


def results_filepaths_for_model(training_parameters)->[variance.VarianceExperimentResult]:
    model_id = training_parameters.id()
    results_folderpath = results_folder()
    all_results_filepaths = results_folderpath.iterdir()
    results_filepaths = [f for f in all_results_filepaths if f.name.startswith(model_id)]
    return results_filepaths


def plots_base_folder():
    return base_path() /"plots"

# def plots_folder(r:VarianceExperimentResult):
#     folderpath = os.path.join(plots_base_folder(), f"{r.id()}")
#     if not os.path.exists(folderpath):
#         os.makedirs(folderpath,exist_ok=True)
#     return folderpath




from transformation_measure import *


def all_measures()-> [Measure]:
    cas=[ConvAggregation.max,ConvAggregation.sum,ConvAggregation.mean,ConvAggregation.none]
    das = [tm.DistanceAggregation.mean,tm.DistanceAggregation.max]
    measure_functions = [MeasureFunction.std]
    measures=[]

    for (ca,mf) in itertools.product(cas,measure_functions):
        measures.append(SampleMeasure(mf,ca))
        measures.append(TransformationMeasure(mf, ca))
        measures.append(NormalizedMeasure(mf, ca))
        measures.append(SampleVarianceMeasure(mf, ca))
        measures.append(TransformationVarianceMeasure(mf, ca))
        measures.append(NormalizedVarianceMeasure(mf, ca))

    for (da,mf) in itertools.product(das,measure_functions):
        measures.append(DistanceSampleMeasure(da))
        measures.append(DistanceTransformationMeasure(da))
        measures.append(DistanceMeasure( da))
        measures.append(DistanceSameEquivarianceMeasure(da))

    for percentage in [0.01,0.001,0.1]:
        measures.append(GoodfellowMeasure(activations_percentage=percentage))
        measures.append(GoodfellowNormalMeasure(alpha=1-percentage))

    measures.append(AnovaFMeasure(ConvAggregation.none))
    alphas=[0.90, 0.95, 0.99, 0.999]
    for (alpha,ca) in itertools.product(alphas,cas):
            measures.append(AnovaMeasure(ca, alpha=alpha, bonferroni=True ))
            measures.append(AnovaMeasure(ca, alpha=alpha, bonferroni=False))
    return measures


def common_measures()-> [Measure]:
    dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
    mf, ca_sum, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.sum, tm.ConvAggregation.mean
    ca_none = ConvAggregation.none
    measures=[
        SampleMeasure(mf,ca_none)
        ,TransformationMeasure(mf,ca_none)
        ,NormalizedMeasure(mf,ca_none)
        ,NormalizedMeasure(mf, ca_mean)
        ,AnovaFMeasure(ca_none)
        ,AnovaMeasure(ca_none, alpha=0.99,bonferroni=True)
        ,tm.DistanceMeasure(dmean)
        ,tm.DistanceSameEquivarianceMeasure(dmean)

    ]
    return measures

def common_transformations() -> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator()]
    return transformations+common_transformations_without_identity()

def common_transformations_without_identity()-> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator(r=360),
                       SimpleAffineTransformationGenerator(t=5),
                       SimpleAffineTransformationGenerator(s=5),
                       # SimpleAffineTransformationGenerator(r=360, s=4, t=3),
                       ]
    return transformations

def rotation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(r=i * 360 // n) for i in range(0, n+1)]

def scale_transformations(n:int):
    return [SimpleAffineTransformationGenerator(s=i) for i in range(n)]

def translation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(t=i) for i in range(n)]

def combined_transformations():
    ts=[]
    for i in range(17):
        r = i * 360 //16
        for s in range(6):
            for t in range(6):
                ts.append(SimpleAffineTransformationGenerator(r=r, s=s, t=t))

    return ts

def all_transformations():
    return combined_transformations() #common_transformations()+rotation_transformations(16)+scale_transformations(6)+translation_transformations(6) +


def common_dataset_sizes()->[float]:
    return [0.01,0.02,0.05,0.1,0.5,1.0]



import transformation_measure as tm
import numpy as np

model_names=model_loading.model_names

def get_epochs(model: str, dataset: str, t: tm.TransformationSet) -> int:

    if model.startswith('SimpleConv'):
        epochs = {'cifar10': 25, 'mnist': 5, 'fashion_mnist': 12}
    elif model.startswith('SimpleConvLargeKernel'):
        epochs = {'cifar10': 25, 'mnist': 5, 'fashion_mnist': 12}
    elif model == models.SimplestConv.__name__ or model == models.SimplestConvBN.__name__:
        epochs = {'cifar10': 40, 'mnist': 5, 'fashion_mnist': 12}
    elif model == models.AllConvolutional.__name__ or model == models.AllConvolutionalBN.__name__:
        epochs = {'cifar10': 40, 'mnist': 30, 'fashion_mnist': 12}
    elif model == models.VGGLike.__name__ or model == models.VGGLikeBN.__name__:
        epochs = {'cifar10': 50, 'mnist': 40, 'fashion_mnist': 12, }
    elif model == models.ResNet.__name__ or model == models.ResNetBN.__name__:
        epochs = {'cifar10': 60, 'mnist': 25, 'fashion_mnist': 12}
    elif model == models.FFNet.__name__ or model == models.FFNetBN.__name__:
        epochs = {'cifar10': 30, 'mnist': 10, 'fashion_mnist': 8}
    else:
        raise ValueError(f"Model \"{model}\" does not exist. Choices: {', '.join(model_names)}")


    n = len(t)
    if n > np.e:
        factor = 1.1 * np.log(n)
    else:
        factor = 1

    if not model.endswith("BN"):
        factor *= 1.5

    return int(epochs[dataset] * factor)


def min_accuracy(model:str,dataset:str)-> float:
    min_accuracies = {"mnist": .90, "cifar10": .5}
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
