from ..base import Experiment
from pathlib import Path
from experiment import measure, training, accuracy
import transformational_measures as tm
import datasets
import train
import config
import torch


class SameEquivarianceExperiment(Experiment):


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

    def experiment_training(self, p: training.Parameters, min_accuracy=None, num_workers=0, batch_size=256):
        if min_accuracy is None:
            min_accuracy = config.min_accuracy(p.model, p.dataset)
        if self.experiment_finished(p):
            return

        o = training.Options(verbose_general=False, verbose_train=True, verbose_batch=False, save_model=True, batch_size=batch_size,
                             num_workers=num_workers, use_cuda=torch.cuda.is_available(), plots=True, max_restarts=5)
        message = f"Training with {p}\n{o}"
        self.print_date(message)
        train.main(p, o, min_accuracy)

    def experiment_measure(self, p: measure.Parameters, batch_size: int = 64, num_workers: int = 0,adapt_dataset=False,model_path:Path=None):
        if model_path==None:
            model_path=config.model_path_from_id(p.model_id)
        results_path = config.results_path(p)
        if results_path.exists():
            return config.load_measure_result(results_path)
        o = measure.Options(verbose=False, batch_size=batch_size, num_workers=num_workers, adapt_dataset=adapt_dataset)

        message = f"Measuring:\n{p}\n{o}"
        self.print_date(message)
        measure_experiment_result=measure.main(p, o,model_path)
        return measure_experiment_result.measure_result

    default_accuracy_options = accuracy.Options(False, 64, 0, torch.cuda.is_available())

    def experiment_accuracy(self, p: accuracy.Parameters, o=default_accuracy_options):
        results_path = config.accuracy_path(p)
        if results_path.exists():
            return
        accuracy.main(p, o)

    def train_measure(self, model_config: config.ModelConfig, dataset: str, transformation: tm.TransformationSet,
                      m: tm.NumpyMeasure, p=None):

        epochs = config.get_epochs(model_config, dataset, transformation)
        p_training = training.Parameters(model_config, dataset, transformation, epochs)
        self.experiment_training(p_training)
        if p is None:
            p = config.dataset_size_for_measure(m)
        p_dataset = measure.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
        p_variance = measure.Parameters(p_training.id(), p_dataset, transformation, m)
        model_path = config.model_path(p_training)
        self.experiment_measure(p_variance)
        return p_training, p_variance, p_dataset


