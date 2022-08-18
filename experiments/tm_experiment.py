
from re import A
from experiment.measure.parameters import DatasetParameters, PyTorchParameters
from experiments.language import Spanish,English
import pickle
import tmeasures as tm
from experiment import measure
import torch
import os
from pathlib import Path
from .tasks.train import TrainParameters,Task,ModelConfig
from .tasks import train
from typing import Union,Type
import matplotlib.pyplot as plt

from .base import Experiment


import config
import datasets




class TMExperiment(Experiment):


    def base_path(self,):
        return self.base_folderpath
    def commons_folder(self,):
        return self.base_folderpath / ".common"
    def models_folder(self,):
        model_folderpath = self.commons_folder() / "models"
        model_folderpath.mkdir(parents=True, exist_ok=True)
        return model_folderpath

    def model_path_from_id(self,id: str) -> Path:
        filename = f"{id}.pt"
        filepath = self.models_folder() / filename
        return filepath

    def model_path_from_filename(self,filename: str) -> Path:
        return self.models_folder() / filename


    def get_models_filenames(self,):
        files = os.listdir(self.models_folder())
        model_filenames = [f for f in files if f.endswith(".pt")]
        return model_filenames

    def get_models_filepaths(self,):
        model_folderpath = self.models_folder()
        return [os.path.join(model_folderpath, f) for f in self.get_models_filenames()]

    def training_plots_path(self,):
        plots_folderpath = self.commons_folder() /"training_plots"
        os.makedirs(plots_folderpath, exist_ok=True)
        return plots_folderpath

    def heatmaps_folder(self,) -> Path:
        return self.commons_folder() / "heatmaps"

    def results_folder(self,) -> Path:
        return self.commons_folder() / "results"


    ########## TRANSFORMATIONAL MEASURES EXPERIMENTS #######################

    def results_paths(self,ps: list[measure.Parameters], custom_results_folder=None) -> list[Path]:
        custom_results_folder = self.results_folder() if custom_results_folder is None else custom_results_folder
        variance_paths = [self.results_path(p,custom_results_folder) for p in ps]
        return variance_paths

    def results_path(self,p: measure.Parameters, custom_results_folder = None) -> Path:
        custom_results_folder = self.results_folder() if custom_results_folder is None else custom_results_folder
        return custom_results_folder / f"{p.id()}.pickle"

    def save_experiment_results(self,r: measure.MeasureExperimentResult, custom_results_folder=None):
        custom_results_folder = self.results_folder() if custom_results_folder is None else custom_results_folder
        path = self.results_path(r.parameters, custom_results_folder)
        basename: Path = path.parent
        basename.mkdir(exist_ok=True, parents=True)
        pickle.dump(r, path.open(mode="wb"))

    def load_experiment_result(self,path: Path) -> measure.MeasureExperimentResult:
        r: measure.MeasureExperimentResult = pickle.load(path.open(mode="rb"))
        return r

    def load_measure_result(self,path: Path) -> tm.measure.MeasureResult:
        return self.load_experiment_result(path).measure_result

    def load_results(self,filepaths: list[Path]) -> list[measure.MeasureExperimentResult]:
        results = []
        for filepath in filepaths:
            result = self.load_experiment_result(filepath)
            results.append(result)
        return results

    def load_measure_results_p(self, ps:list[measure.Parameters], custom_results_folder=None) -> list[tm.measure.MeasureResult]:
        return self.load_measure_results(self.results_paths(ps,custom_results_folder=custom_results_folder))

    def load_measure_results(self,filepaths: list[Path]) -> list[tm.measure.MeasureResult]:
        results = self.load_results(filepaths)
        results = [r.measure_result for r in results]
        return results

    def load_measure_results(self,filepaths: list[Path]) -> list[tm.measure.MeasureResult]:
        results = self.load_results(filepaths)
        results = [r.measure_result for r in results]
        return results

    def load_all_results(self,folderpath: Path) -> list[measure.MeasureExperimentResult]:
        filepaths = [f for f in folderpath.iterdir() if f.is_file()]
        return self.load_results(filepaths)

    def results_filepaths_for_model(self,training_parameters) -> list[measure.MeasureExperimentResult]:
        model_id = training_parameters.id()
        results_folderpath = self.results_folder()
        all_results_filepaths = results_folderpath.iterdir()
        results_filepaths = [f for f in all_results_filepaths if f.name.startswith(model_id)]
        return results_filepaths

    ########## PLOTS EXPERIMENTS #######################

    def plots_base_folder(self,):
        return self.base_path() / "plots"

    ################ NEW STUFF

    def model_trained(self, p: TrainParameters)->bool:
        # custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filepaths = [self.model_path_new(p)] + [self.model_path_new(p, s) for s in p.tc.savepoints]
        exist = [p.exists() for p in filepaths]
        return all(exist)

    def model_path_new(self, p: TrainParameters, savepoint=None, custom_models_folderpath=None)->Path:
        custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filename = f"{p.id(savepoint)}.pt"
        folder = p.mc.__class__.__name__
        filepath = custom_models_folderpath / folder / filename
        return filepath

    def train_measure(self,p:TrainParameters,mp:measure.PyTorchParameters,verbose=False):
        self.train(p)
        model_path = self.model_path_new(p)
        return self.measure(model_path,mp,verbose=verbose)


    def measure(self,model_path:str,p:measure.PyTorchParameters,verbose=False)->tm.pytorch.PyTorchMeasureResult:
        
        results_path = self.results_path(p)
        if results_path.exists():
            return self.load_measure_result(results_path)

        message = f"Measuring:\n{p}\n{p.options}"
        self.print_date(message)
        measure_experiment_result = measure.main_pytorch(p,model_path,verbose=verbose)
        self.save_experiment_results(measure_experiment_result)
        return measure_experiment_result.measure_result

    def train(self,p:TrainParameters):
        if not self.model_trained(p):
            print(f"Training model {p.id()} for {p.tc.epochs} epochs ({p.tc.convergence_criteria}), savepoints at epochs: {p.tc.savepoints})...")
            train.train(p, self)
        else:
            print(f"Model {p.id()} already trained.")
    
    def savefig(self,path:Path):
        plt.savefig(path,bbox_inches='tight')
        plt.close()

    def train_default(self,task:Task,dataset:str,transformations:tm.pytorch.PyTorchTransformationSet,mc:Union[ModelConfig,type[ModelConfig]]):
        if not isinstance(mc, ModelConfig):
            mc: train.ModelConfig = mc.for_dataset(task,dataset)
        tc,metric = self.get_train_config(mc,dataset,task,transformations)
        p = train.TrainParameters(mc, tc, dataset, transformations, task)
        self.train(p)
        model_path = self.model_path_new(p)
        return mc,tc,p,model_path

    def measure_default(self,dataset:str,model_id:str,model_path:Path,transformation:tm.pytorch.PyTorchTransformationSet,measure:tm.pytorch.PyTorchMeasure,measure_options:tm.pytorch.PyTorchMeasureOptions,dataset_percentage:float,subset = datasets.DatasetSubset.test,adapt_dataset=False):
        p_dataset = DatasetParameters(dataset,subset, dataset_percentage)
        mp = PyTorchParameters(model_id, p_dataset, transformation, measure, measure_options,adapt_dataset=adapt_dataset)
        return self.measure(model_path, mp, verbose=False).numpy()