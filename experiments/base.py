import os
from pathlib import Path
from datetime import datetime
from experiment import measure, training, accuracy
import config
import transformation_measure as tm
import abc
import torch
from .language import Spanish,English

import train

import sys
import torch
from utils.profiler import  Profiler

class Experiment(abc.ABC):

    def __init__(self,language=Spanish()):
        self.plot_folderpath = config.plots_base_folder() / self.id()
        self.plot_folderpath.mkdir(exist_ok=True, parents=True)
        with open(self.plot_folderpath / "description.txt", "w") as f:
            f.write(self.description())
        self.venv = Path(".")
        self.l=language

    def id(self):
        return self.__class__.__name__

    def set_venv(self, venv: Path):
        self.venv = venv

    def __call__(self, force=False, venv=".", *args, **kwargs):
        stars = "*" * 15
        strf_format = "%Y/%m/%d %H:%M:%S"
        dt_started = datetime.now()
        dt_started_string = dt_started.strftime(strf_format)
        if not self.has_finished() or force:
            self.mark_as_unfinished()
            print(f"[{dt_started_string}] {stars} Running experiment {self.id()}  {stars}")
            self.run()

            # time elapsed and finished
            dt_finished= datetime.now()
            dt_finished_string =dt_finished.strftime(strf_format)
            elapsed = dt_finished - dt_started
            print(f"[{dt_finished_string }] {stars} Finished experiment {self.id()}  ({elapsed} elapsed) {stars}")
            self.mark_as_finished()
        else:
            print(f"[{dt_started_string}] {stars}Experiment {self.id()} already finished, skipping. {stars}")

    def has_finished(self):
        return self.finished_filepath().exists()

    def finished_filepath(self):
        return self.plot_folderpath / "finished"

    def mark_as_finished(self):
        self.finished_filepath().touch(exist_ok=True)

    def mark_as_unfinished(self):
        f = self.finished_filepath()
        if f.exists():
            f.unlink()

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    def experiment_finished(self, p: training.Parameters):
        model_path = config.model_path(p)
        if model_path.exists():
            if p.savepoints == []:
                return True
            else:
                savepoint_missing = [sp for sp in p.savepoints if not config.model_path(p, sp).exists()]
                return savepoint_missing == []
        else:
            return False

    def print_date(self,message):
        strf_format = "%Y/%m/%d %H:%M:%S"
        dt = datetime.now()
        dt_string = dt.strftime(strf_format)
        message=f"[{dt_string}] *** {message}"
        print(message)

    def experiment_fork(self,message,function):
        self.print_date(message)

        new_pid=os.fork()
        if new_pid == 0:
            function()
            os._exit(0)
        else:
            pid,status = os.waitpid(0, 0)
            if status !=0:
                self.print_date(f" Error in: {message}")
                sys.exit(status)

    def experiment_training(self, p: training.Parameters, min_accuracy=None,num_workers=0,batch_size=256):
        if min_accuracy is None:
            min_accuracy = config.min_accuracy(p.model, p.dataset)
        if self.experiment_finished(p):
            return

        o=training.Options(verbose=False,train_verbose=True,save_model=True,batch_size=batch_size,num_workers=num_workers,use_cuda=torch.cuda.is_available(),plots=True,max_restarts=5)
        message=f"Training with {p}\n{o}"
        self.print_date(message)
        train.main(p,o,min_accuracy)


    def experiment_measure(self, p: measure.Parameters, model_path: Path, batch_size: int = 64, num_workers: int = 0, adapt_dataset=False):

        results_path = config.results_path(p)
        if results_path.exists():
            return
        o=measure.Options(verbose=False, batch_size=batch_size, num_workers=num_workers, adapt_dataset=adapt_dataset)

        message=f"Measuring:\n{p}\n{o}"
        self.print_date(message)

        measure.main(p,o)

    default_accuracy_options=accuracy.Options(False,64,0,torch.cuda.is_available())


    def experiment_accuracy(self, p: accuracy.Parameters, o=default_accuracy_options):
        results_path = config.accuracy_path(p)
        if results_path.exists():
            return
        accuracy.main(p,o)




    def train_measure(self, model_config:config.ModelConfig, dataset:str, transformation:tm.TransformationSet, m:tm.Measure, p=None):

        epochs = config.get_epochs(model_config, dataset, transformation)
        p_training = training.Parameters(model_config, dataset, transformation, epochs)
        self.experiment_training(p_training)
        if p is None:
            p = config.dataset_size_for_measure(m)
        p_dataset = measure.DatasetParameters(dataset, measure.DatasetSubset.test, p)
        p_variance = measure.Parameters(p_training.id(), p_dataset, transformation, m)
        model_path = config.model_path(p_training)
        self.experiment_measure(p_variance, model_path)
        return p_training,p_variance,p_dataset

