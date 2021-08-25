from .. import TMExperiment

import config

from ..language import English

from ..tasks.train import TrainParameters,Task
from ..tasks import train
import transformational_measures as tm
from experiment import measure

from .models import ModelConfig

class SameEquivarianceExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        base_folderpath = config.base_path() / "same_equivariance"
        super().__init__(base_folderpath)


    def model_trained(self, p: TrainParameters):
        # custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filepaths = [self.model_path_new(p)] + [self.model_path_new(p, s) for s in p.tc.savepoints]


        exist = [p.exists() for p in filepaths]
        return all(exist)

    def model_path_new(self, p: TrainParameters, savepoint=None, custom_models_folderpath=None):
        custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filename = f"{p.id(savepoint)}.pt"
        folder = p.mc.__class__.__name__
        filepath = custom_models_folderpath / folder / filename
        return filepath



    def train(self,p:TrainParameters):
        if not self.model_trained(p):
            print(f"Training model {p.id()} for {p.tc.epochs} epochs ({p.tc.convergence_criteria})...")
            train.train(p, self)
        else:
            print(f"Model {p.id()} already trained.")

    def get_regression_trainconfig(self,mc: ModelConfig, dataset: str, task: Task,
                                   transformations: tm.pytorch.PyTorchTransformationSet,savepoints=True):
        epochs = mc.epochs(dataset, task, transformations)
        if savepoints:
            savepoints_percentages = [0, 1, 2, 5, 10, 25, 50, 75, 100]
            savepoints_epochs = []#[0,1,2,3]
            savepoints = savepoints_epochs + [sp * epochs // 100 for sp in savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
        else:
            savepoints=[]

        metric = "rae"
        cc = train.MaxMetricConvergence(mc.max_rae(dataset, task, transformations), metric)
        optimizer = dict(optim="adam", lr=0.0001)
        tc = train.TrainConfig(epochs, cc, optimizer=optimizer, savepoints=savepoints, verbose=False, num_workers=4)
        return tc, metric

    def train_measure(self,p:TrainParameters,mp:measure.PyTorchParameters,verbose=False):
        self.train(p)
        model_path = self.model_path_new(p)
        return self.measure(model_path,mp,verbose=verbose)


    def measure(self,model_path:str,p:measure.PyTorchParameters,verbose=False)->tm.pytorch.PyTorchMeasureResult:

        # TODO: esto es un quilombo. Cambiar todo por un PyTorchParameter (y llamarlo directamente Parameter) y listo
        results_path = self.results_path(p)
        if results_path.exists():
            return self.load_measure_result(results_path)

        message = f"Measuring:\n{p}\n{p.options}"
        self.print_date(message)
        measure_experiment_result = measure.main_pytorch(p,model_path,verbose=verbose)
        self.save_experiment_results(measure_experiment_result)
        return measure_experiment_result.measure_result


