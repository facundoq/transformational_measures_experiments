
from .. import TMExperiment, train
from pathlib import Path
from experiment import measure, training, accuracy
import transformational_measures as tm
import datasets
import config
import torch
from ..language import English

from ..tasks  import Task, train


class InvarianceExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        base_folderpath = config.base_path() / "invariance"
        super().__init__(base_folderpath)


    def get_train_config(self,mc: train.ModelConfig, dataset: str, task: train.Task,
                                   transformations: tm.pytorch.PyTorchTransformationSet,savepoints=True,verbose=False,batch_size=64):
        epochs = mc.epochs(dataset, task, transformations)
        if savepoints:
            savepoints_percentages = [0, 1, 2, 5, 10, 25, 50, 75, 100]
            savepoints_epochs = list(range(min(epochs,3))) # add the first 3 epochs
            savepoints = savepoints_epochs + [sp * epochs // 100 for sp in savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
        else:
            savepoints=[]

        lr_task = {Task.Classification:0.0001,Task.TransformationRegression:0.0001}
        cc = train.MinAccuracyConvergence(mc.min_accuracy(dataset, task, transformations))
        optimizer = dict(optim="adam", lr=lr_task[task])
        tc = train.TrainConfig(epochs, cc, optimizer=optimizer, savepoints=savepoints, verbose=verbose, num_workers=4,batch_size=batch_size)
        return tc, cc.metric


    # def experiment_finished(self, p: training.Parameters):
    #     model_path = self.model_path(p)
    #     if model_path.exists():
    #         if p.savepoints == []:
    #             return True
    #         else:
    #             savepoint_missing = [sp for sp in p.savepoints if not self.model_path(p, sp).exists()]
    #             return savepoint_missing == []
    #     else:
    #         return False


    # def experiment_training(self, p: training.Parameters, min_accuracy=None, num_workers=0, batch_size=256):
    #     # import train here to avoid circular dependency

    #     if min_accuracy is None:
    #         min_accuracy = config.min_accuracy(p.model, p.dataset)
    #     if self.experiment_finished(p):
    #         return

    #     o = training.Options(verbose_general=False, verbose_train=True, verbose_batch=False, save_model=True, batch_size=batch_size,
    #                          num_workers=num_workers, use_cuda=torch.cuda.is_available(), plots=True, max_restarts=5)
    #     message = f"Training with {p}\n{o}"
    #     self.print_date(message)
    #     train.main(self,p, o, min_accuracy)



    # def experiment_measure(self, p: measure.Parameters, batch_size: int = 64, num_workers: int = 0,adapt_dataset=False,model_path:Path=None)->tm.MeasureResult:
    #     if model_path is None:
    #         model_path=self.model_path_from_id(p.model_id)
    #     results_path = self.results_path(p)
    #     if results_path.exists():
    #         return self.load_measure_result(results_path)
    #     o = measure.Options(verbose=False, batch_size=batch_size, num_workers=num_workers, adapt_dataset=adapt_dataset)

    #     message = f"Measuring:\n{p}\n{o}"
    #     self.print_date(message)
    #     measure_experiment_result=measure.main(p, o,model_path)
    #     return measure_experiment_result.measure_result
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_accuracy_options = accuracy.Options(False, 64, 0, device=device)

    def experiment_accuracy(self, p: accuracy.Parameters, o=default_accuracy_options):
        results_path = self.accuracy_path(p)
        if results_path.exists():
            return self.load_accuracy(results_path)
        return accuracy.main(p, o)

    # def train_measure(self, model_config: config.ModelConfig, dataset: str, transformation: tm.TransformationSet,
    #                   m: tm.numpy.NumpyMeasure, p=None):

    #     epochs = config.get_epochs(model_config, dataset, transformation)
    #     p_training = training.Parameters(model_config, dataset, transformation, epochs,)
    #     self.experiment_training(p_training)
    #     if p is None:
    #         p = config.dataset_size_for_measure(m)
    #     p_dataset = measure.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
    #     p_variance = measure.Parameters(p_training.id(), p_dataset, transformation, m)
    #     self.experiment_measure(p_variance)
    #     return p_training, p_variance, p_dataset

    # def model_trained(self, p: training.Parameters):
    #     # custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
    #     filepaths = [self.model_path(p)] + [self.model_path(p, s) for s in p.tc.savepoints]
    #     exist = [p.exists() for p in filepaths]
    #     return all(exist)


    # def model_path(self, p: training.Parameters, savepoint=None, custom_models_folderpath=None) -> Path:
    #     custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
    #     filename = f"{p.id(savepoint=savepoint)}.pt"
    #     filepath = custom_models_folderpath / filename
    #     return filepath

    # def load_model(self,p: training.Parameters, savepoint=None,
    #                use_cuda: bool = torch.cuda.is_available(), load_state=True):
    #     path = self.model_path(p, savepoint)
    #     return training.load_model(path, use_cuda, load_state)