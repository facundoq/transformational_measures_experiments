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
from .tasks.train import TrainParameters,Task
from .tasks import train

class TMExperiment(Experiment):


    def base_path(self,):
        return self.base_folderpath
    def commons_folder(self,):
        return self.base_folderpath / ".common"
    def models_folder(self,):
        model_folderpath = self.commons_folder() / "models"
        model_folderpath.mkdir(parents=True, exist_ok=True)
        return model_folderpath

    def model_trained(self,p: TrainParameters):
        # custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filepaths = [self.model_path_new(p)]+[self.model_path_new(p,s) for s in p.tc.savepoints]
        exist = [p.exists() for p in filepaths]
        return all(exist)

    def model_path_new(self,p:TrainParameters,savepoint=None, custom_models_folderpath=None):
        custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filename = f"{p.id(savepoint)}.pt"
        folder = p.mc.__class__.__name__
        filepath = custom_models_folderpath / folder / filename
        return filepath

    def model_path(self, p: training.Parameters, savepoint=None, custom_models_folderpath=None) -> Path:
        custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filename = f"{p.id(savepoint=savepoint)}.pt"
        filepath = custom_models_folderpath / filename
        return filepath

    def model_path_from_id(self,id: str) -> Path:
        filename = f"{id}.pt"
        filepath = self.models_folder() / filename
        return filepath

    def model_path_from_filename(self,filename: str) -> Path:
        return self.models_folder() / filename

    def load_model(self,p: training.Parameters, savepoint=None,
                   use_cuda: bool = torch.cuda.is_available(), load_state=True):
        path = self.model_path(p, savepoint)
        return training.load_model(path, use_cuda, load_state)

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


    def experiment_finished(self, p: training.Parameters):
        model_path = self.model_path(p)
        if model_path.exists():
            if p.savepoints == []:
                return True
            else:
                savepoint_missing = [sp for sp in p.savepoints if not self.model_path(p, sp).exists()]
                return savepoint_missing == []
        else:
            return False

    def train(self,p:train.TrainParameters):
        if not self.model_trained(p):
            print(f"Training model {p.id()} for {p.tc.epochs} epochs ({p.tc.convergence_criteria})...")
            train.train(p, self)
        else:
            print(f"Model {p.id()} already trained.")

    def experiment_training(self, p: training.Parameters, min_accuracy=None, num_workers=0, batch_size=256):


        # import train here to avoid circular dependency
        from . import train
        if min_accuracy is None:
            min_accuracy = config.min_accuracy(p.model, p.dataset)
        if self.experiment_finished(p):
            return

        o = training.Options(verbose_general=False, verbose_train=True, verbose_batch=False, save_model=True, batch_size=batch_size,
                             num_workers=num_workers, use_cuda=torch.cuda.is_available(), plots=True, max_restarts=5)
        message = f"Training with {p}\n{o}"
        self.print_date(message)
        train.main(self,p, o, min_accuracy)



    def experiment_measure(self, p: measure.Parameters, batch_size: int = 64, num_workers: int = 0,adapt_dataset=False,model_path:Path=None):
        if model_path==None:
            model_path=self.model_path_from_id(p.model_id)
        results_path = self.results_path(p)
        if results_path.exists():
            return self.load_measure_result(results_path)
        o = measure.Options(verbose=False, batch_size=batch_size, num_workers=num_workers, adapt_dataset=adapt_dataset)

        message = f"Measuring:\n{p}\n{o}"
        self.print_date(message)
        measure_experiment_result=measure.main(p, o,model_path)
        return measure_experiment_result.measure_result

    default_accuracy_options = accuracy.Options(False, 64, 0, torch.cuda.is_available())

    def experiment_accuracy(self, p: accuracy.Parameters, o=default_accuracy_options):
        results_path = self.accuracy_path(p)
        if results_path.exists():
            return
        accuracy.main(p, o)

    def train_measure(self, model_config: config.ModelConfig, dataset: str, transformation: tm.TransformationSet,
                      m: tm.NumpyMeasure, task:Task, p=None):

        epochs = config.get_epochs(model_config, dataset, transformation)
        p_training = training.Parameters(model_config, dataset, transformation, epochs,)
        self.experiment_training(p_training)
        if p is None:
            p = config.dataset_size_for_measure(m)
        p_dataset = measure.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
        p_variance = measure.Parameters(p_training.id(), p_dataset, transformation, m)
        self.experiment_measure(p_variance)
        return p_training, p_variance, p_dataset
