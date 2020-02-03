import os
from pathlib import Path
from datetime import datetime
from experiment import variance, training, accuracy,utils_runner
import config
import transformation_measure as trans_measure
import abc
import torch
from .language import Spanish,English

import train,measure,measure_accuracy
import sys

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

    def experiment_training_fork(self,p:training.Parameters,min_accuracy=None,num_workers=0,batch_size=256):


        o=training.Options(verbose=False,train_verbose=True,save_model=True,batch_size=batch_size,num_workers=num_workers,use_cuda=torch.cuda.is_available(),plots=True,max_restarts=5)
        message=f"Training with {p}\n{o}"
        self.print_date(message)
        train.main(p,o,min_accuracy)
        #self.experiment_fork(message,lambda: train.main(p,o,min_accuracy))



    def experiment_training(self, p: training.Parameters, min_accuracy=None,num_workers=0,use_fork=True,batch_size=256):
        if not min_accuracy:
            min_accuracy = config.min_accuracy(p.model, p.dataset)
        if self.experiment_finished(p):
            return
        if use_fork:
            self.experiment_training_fork(p,min_accuracy,num_workers,batch_size)
        else:
            if len(p.suffix) > 0:
                suffix = f'-suffix "{p.suffix}"'
            else:
                suffix = ""

            savepoints = ",".join([str(sp) for sp in p.savepoints])
            python_command = f'train.py -model "{p.model}" -dataset "{p.dataset}" -transformation "{p.transformations.id()}" -epochs {p.epochs}  -num_workers {num_workers} -min_accuracy {min_accuracy} -max_restarts 5 -batchsize {batch_size} -savepoints "{savepoints}" {suffix}'
            utils_runner.run_python(self.venv, python_command)

    def experiment_variance(self, p: variance.Parameters, model_path: Path, batch_size: int = 64, num_workers: int = 0,
                            adapt_dataset=False):

        results_path = config.results_path(p)
        if os.path.exists(results_path):
            return
        if p.stratified:
            stratified = "-stratified"
        else:
            stratified = ""

        python_command = f'measure.py -mo "{model_path}" -me "{p.measure.id()}" -d "{p.dataset.id()}" -t "{p.transformations.id()}" -verbose False -batchsize {batch_size} -num_workers {num_workers} {stratified}'

        if adapt_dataset:
            python_command = f"{python_command} -adapt_dataset True"

        utils_runner.run_python(self.venv, python_command)
    import torch

    default_accuracy_options=accuracy.Options(False,64,0,torch.cuda.is_available())
    def experiment_accuracy(self, p: accuracy.Parameters, o=default_accuracy_options):
        results_path = config.accuracy_path(p)
        if results_path.exists():
            return
        python_command = f'measure_accuracy.py -mo "{p.model_path}" -d "{p.dataset.id()}" -transformation "{p.transformations.id()}" -transformation_strategy "{p.transformation_strategy.value}" -verbose {o.verbose} -batchsize {o.batch_size} -num_workers {o.num_workers}'

        utils_runner.run_python(self.venv, python_command)


    def train_measure(self, model_config:config.ModelConfig, dataset:str, transformation:trans_measure.TransformationSet, measure:trans_measure.Measure, p=None):

        epochs = config.get_epochs(model_config, dataset, transformation)
        p_training = training.Parameters(model_config, dataset, transformation, epochs)
        self.experiment_training(p_training)
        if p is None:
            p = config.dataset_size_for_measure(measure)
        p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
        p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
        model_path = config.model_path(p_training)
        self.experiment_variance(p_variance, model_path)
        return p_training,p_variance,p_dataset

