import os
import pickle
from pathlib import Path

import transformational_measures.measure
from .models import *
from .datasets import *
from .measures import *
from transformational_measures.transformations import *

from experiment import measure, training,accuracy
import torch



def base_path():
    return Path(os.path.expanduser("~/variance/"))

def testing_path():
    return base_path() / "testing"

def models_folder():
    model_folderpath = base_path() / "models"
    model_folderpath.mkdir(parents=True, exist_ok=True)
    return model_folderpath

def model_path(p: training.Parameters,savepoint=None,model_folderpath= models_folder())->Path:
    filename=f"{p.id(savepoint=savepoint)}.pt"
    filepath=model_folderpath / filename
    return filepath

def model_path_from_id(id:str,model_folderpath=models_folder())->Path:
    filename=f"{id}.pt"
    filepath=model_folderpath / filename
    return filepath

def model_path_from_filename(filename:str,model_folderpath= models_folder())->Path:
    return model_folderpath / filename

def load_model(p: training.Parameters,savepoint=None,model_folderpath= models_folder(),use_cuda:bool=torch.cuda.is_available(),load_state=True):
    path = model_path(p,savepoint,model_folderpath)
    return training.load_model(path,use_cuda,load_state)

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

def accuracies_folder()->Path:
    return base_path() / "accuracies"

########## TRANSFORMATIONAL MEASURES EXPERIMENTS #######################

def results_paths(ps:[measure.Parameters], results_folder=results_folder())->[Path]:
    variance_paths= [results_path(p,results_folder) for p in ps]
    return variance_paths

def results_path(p:measure.Parameters, results_folder=results_folder())-> Path:
    return  results_folder / f"{p.id()}.pickle"

def save_experiment_results(r:measure.MeasureExperimentResult, results_folder=results_folder()):
    path = results_path(r.parameters, results_folder)
    basename:Path = path.parent
    basename.mkdir(exist_ok=True,parents=True)
    pickle.dump(r,path.open(mode="wb"))

def load_experiment_result(path:Path)->measure.MeasureExperimentResult:
    r:measure.MeasureExperimentResult=pickle.load(path.open(mode="rb"))
    return r

def load_measure_result(path:Path)-> transformational_measures.measure.MeasureResult:
    return load_experiment_result(path).measure_result

def load_results(filepaths:[Path])-> [measure.MeasureExperimentResult]:
    results = []
    for filepath in filepaths:
        result = load_experiment_result(filepath)
        results.append(result)
    return results

def load_measure_results(filepaths:[Path])-> [transformational_measures.measure.MeasureResult]:
    results = load_results(filepaths)
    results = [r.measure_result for r in results]
    return results

def load_all_results(folderpath:Path)-> [measure.MeasureExperimentResult]:
    filepaths=[f for f in folderpath.iterdir() if f.is_file()]
    return load_results(filepaths)


def results_filepaths_for_model(training_parameters)->[measure.MeasureExperimentResult]:
    model_id = training_parameters.id()
    results_folderpath = results_folder()
    all_results_filepaths = results_folderpath.iterdir()
    results_filepaths = [f for f in all_results_filepaths if f.name.startswith(model_id)]
    return results_filepaths

########## PLOTS EXPERIMENTS #######################

def plots_base_folder():
    return base_path() /"plots"

########## ACCURACY EXPERIMENTS #######################

def accuracy_path(p:accuracy.Parameters, accuracies_folder=accuracies_folder())-> Path:
    return  accuracies_folder / f"{p.id()}.pickle"

def save_accuracy(r:accuracy.AccuracyExperimentResult, accuracy_folder=accuracies_folder()):
    path = accuracy_path(r.parameters, accuracy_folder)
    basename: Path = path.parent
    basename.mkdir(exist_ok=True, parents=True)
    pickle.dump(r, path.open(mode="wb"))

def load_accuracy(path:Path)->accuracy.AccuracyExperimentResult:
    r:accuracy.AccuracyExperimentResult=pickle.load(path.open(mode="rb"))
    return r

def accuracies_paths(ps:[accuracy.Parameters], accuracy_folder=accuracies_folder())->[Path]:
    variance_paths= [accuracy_path(p, accuracy_folder) for p in ps]
    return variance_paths

def load_accuracies(filepaths:[Path])-> [accuracy.AccuracyExperimentResult]:
    results = []
    for filepath in filepaths:
        result = load_accuracy(filepath)
        results.append(result)
    return results