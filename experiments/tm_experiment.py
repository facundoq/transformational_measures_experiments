from experiments.language import Spanish,English
import pickle
import transformational_measures as tm
from experiment import measure, training,accuracy
import torch
import os

from pathlib import Path
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

    def accuracies_folder(self,) -> Path:
        return self.commons_folder() / "accuracies"

    ########## TRANSFORMATIONAL MEASURES EXPERIMENTS #######################

    def results_paths(self,ps: [measure.Parameters], custom_results_folder=None) -> [Path]:
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

    def load_results(self,filepaths: [Path]) -> [measure.MeasureExperimentResult]:
        results = []
        for filepath in filepaths:
            result = self.load_experiment_result(filepath)
            results.append(result)
        return results

    def load_measure_results_p(self, ps:[measure.Parameters], custom_results_folder=None) -> [tm.measure.MeasureResult]:
        return self.load_measure_results(self.results_paths(ps,custom_results_folder=custom_results_folder))

    def load_measure_results(self,filepaths: [Path]) -> [tm.measure.MeasureResult]:
        results = self.load_results(filepaths)
        results = [r.measure_result for r in results]
        return results

    def load_measure_results(self,filepaths: [Path]) -> [tm.measure.MeasureResult]:
        results = self.load_results(filepaths)
        results = [r.measure_result for r in results]
        return results

    def load_all_results(self,folderpath: Path) -> [measure.MeasureExperimentResult]:
        filepaths = [f for f in folderpath.iterdir() if f.is_file()]
        return self.load_results(filepaths)

    def results_filepaths_for_model(self,training_parameters) -> [measure.MeasureExperimentResult]:
        model_id = training_parameters.id()
        results_folderpath = self.results_folder()
        all_results_filepaths = results_folderpath.iterdir()
        results_filepaths = [f for f in all_results_filepaths if f.name.startswith(model_id)]
        return results_filepaths

    ########## PLOTS EXPERIMENTS #######################

    def plots_base_folder(self,):
        return self.base_path() / "plots"

    ########## ACCURACY EXPERIMENTS #######################

    def accuracy_path(self,p: accuracy.Parameters, custom_accuracies_folder=None) -> Path:
        custom_accuracies_folder  = self.accuracies_folder() if custom_accuracies_folder is None else custom_accuracies_folder
        return custom_accuracies_folder / f"{p.id()}.pickle"

    def save_accuracy(self,r: accuracy.AccuracyExperimentResult,custom_accuracies_folder=None):
        custom_accuracies_folder = self.accuracies_folder() if custom_accuracies_folder is None else custom_accuracies_folder
        path = self.accuracy_path(r.parameters, custom_accuracies_folder)
        basename: Path = path.parent
        basename.mkdir(exist_ok=True, parents=True)
        pickle.dump(r, path.open(mode="wb"))

    def load_accuracy(self,path: Path) -> accuracy.AccuracyExperimentResult:
        r: accuracy.AccuracyExperimentResult = pickle.load(path.open(mode="rb"))
        return r

    def accuracies_paths(self,ps: [accuracy.Parameters], custom_accuracies_folder=None) -> [Path]:
        custom_accuracies_folder = self.accuracies_folder() if custom_accuracies_folder is None else custom_accuracies_folder
        variance_paths = [self.accuracy_path(p, custom_accuracies_folder) for p in ps]
        return variance_paths

    def load_accuracies(self,filepaths: [Path]) -> [accuracy.AccuracyExperimentResult]:
        results = []
        for filepath in filepaths:
            result = self.load_accuracy(filepath)
            results.append(result)
        return results


