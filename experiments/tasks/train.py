import matplotlib.pyplot as plt
import pytorch.metrics
import tmeasures as tm
from . import Task
import torch
import numpy as np
import datasets

from pathlib import Path

from utils.poutyne import TotalProgressCallback
from poutyne import Model, Callback,EpochProgressionCallback

from pytorch.pytorch_image_dataset import ImageClassificationDataset, TransformationStrategy, \
    ImageTransformRegressionNormalizedDataset

from pytorch.numpy_dataset import NumpyDataset
from abc import ABC, abstractmethod


def default_device(): return "cuda" if torch.cuda.is_available() else "cpu"


class ModelConfig(ABC):

    def __init__(self,klass) -> None:
        super().__init__()
        self.klass =klass
    def name(self,):
        return self.klass.__name__

    @abstractmethod
    def make(self, input_shape: np.ndarray, output_dim: int) -> tm.pytorch.ActivationsModule:
        pass

    @abstractmethod
    def id(self) -> str:
        pass
    
    @abstractmethod
    def epochs(self, dataset: str, task: Task, transformations: tm.TransformationSet):
        pass

    def min_accuracy(self, dataset: str, task: Task, transformations: tm.TransformationSet):

        min_accuracies = {"mnist": .90, "cifar10": .5,"lsa16":0.85, "rwth":0.7}

        return min_accuracies[dataset]

    def max_smape(self, dataset: str, task: Task, transformations: tm.TransformationSet):

        mi, ma = transformations.parameter_range()
        n_parameters = len(mi)
        coefficient = {"mnist": 0.20, "cifar10": 0.20}
        val = n_parameters * coefficient[dataset]
        return val

    def max_mae(self, dataset: str, task: Task, transformations: tm.TransformationSet):
        mi, ma = transformations.parameter_range()
        n_parameters = len(mi)
        coefficient = {"mnist": 0.20, "cifar10": 0.20}
        val = n_parameters * coefficient[dataset]
        return val

    def max_rae(self, dataset: str, task: Task, transformations: tm.TransformationSet):
        mi, ma = transformations.parameter_range()
        n_parameters = len(mi)
        coefficient = {"mnist": 0.30, "cifar10": 0.30}
        val = coefficient[dataset]
        return val

    def max_rmse(self, dataset: str, task: Task, transformations: tm.TransformationSet):

        mi, ma = transformations.parameter_range()
        n_parameters = len(mi)
        coefficient = {"mnist": 0.20, "cifar10": 0.20}
        max_rmse = n_parameters * coefficient[dataset]
        return max_rmse

    def scale_by_transformations(self, epochs: int, transformations: tm.TransformationSet):
        m = len(transformations)
        if m > np.e:
            factor = 1.1 * np.log(m)
        else:
            factor = 1

        return int(epochs * factor)


class ConvergenceCriteria(ABC):
    @abstractmethod
    def converged(self, metrics: dict[str, float]):
        pass

    @abstractmethod
    def metrics(self):
        pass


class MinMetricConvergence(ConvergenceCriteria):
    def __init__(self, minimum_value: float, metric: str):
        self.minimum_value = minimum_value
        self.metric = metric

    def converged(self, metrics: dict[str, float]):
        return metrics[f"test_{self.metric}"] > self.minimum_value

    def metrics(self):
        return [self.metric]

    def __repr__(self):
        return f"MinValue(v={self.minimum_value},m={self.metric})"


class MaxMetricConvergence(ConvergenceCriteria):
    def __init__(self, maximum_value: float, metric: str):
        self.maximum_value = maximum_value
        self.metric = metric

    def converged(self, metrics: dict[str, float]):
        return metrics[f"test_{self.metric}"] < self.maximum_value

    def metrics(self):
        return [self.metric]

    def __repr__(self):
        return f"MaxValue(v={self.maximum_value},m={self.metric})"


class MinAccuracyConvergence(MinMetricConvergence):
    def __init__(self, min_accuracy: float):
        super().__init__(min_accuracy, "acc")


class MaxRMSEConvergence(MaxMetricConvergence):
    def __init__(self, max_mse: float):
        super(MaxRMSEConvergence, self).__init__(max_mse, "mse")


class TrainConfig:
    def __init__(self, epochs: int, cc: ConvergenceCriteria, optimizer="adam", save_model=True, max_restarts: int = 5,
                 savepoints: list[int] = None,
                 device=default_device(), suffix="",
                 verbose=False, num_workers=2, batch_size=64, plots=True):
        self.epochs = epochs
        self.optimizer = optimizer
        self.suffix = suffix
        self.device = device
        self.savepoints = savepoints
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.save_model = save_model
        self.plots = plots
        self.max_restarts = max_restarts
        self.convergence_criteria = cc


class TrainParameters:
    def __init__(self, mc: ModelConfig, tc: TrainConfig, dataset: str, transformations: tm.TransformationSet,
                 task: Task):
        self.mc = mc
        self.tc = tc
        self.dataset_name = dataset
        self.transformations = transformations
        self.task = task

    def id(self, savepoint: float = None)->str:
        result = f"{self.mc.id()}_{self.dataset_name}_{self.transformations.id()}"

        suffix = self.tc.suffix
        if len(suffix) > 0:
            result += f"_{suffix}"

        if not savepoint is None:
            assert (savepoint <= self.tc.epochs)
            assert (savepoint >= 0)
            if not savepoint in self.tc.savepoints:
                raise ValueError(
                    f"Invalid savepoint {savepoint}. Options: {', '.join(self.tc.savepoints)}")
            result += f"/savepoint={savepoint:03}"

        return result

    

def prepare_dataset(transformations:tm.TransformationSet, dataset_name:str, task:Task):

    strategy = TransformationStrategy.random_sample
    if task == Task.TransformationRegression:
        dataset = datasets.get_regression(dataset_name)
        dim_output = len(transformations[0].parameters())
        dataset.normalize_features()
        train_dataset = ImageTransformRegressionNormalizedDataset(
            NumpyDataset(dataset.x_train), transformations, strategy)
        test_dataset = ImageTransformRegressionNormalizedDataset(
            NumpyDataset(dataset.x_test), transformations, strategy)
    elif task == Task.Classification:
        dataset = datasets.get_classification(dataset_name)
        dim_output = dataset.num_classes
        dataset.normalize_features()
        
        train_dataset = ImageClassificationDataset(NumpyDataset(dataset.x_train, dataset.y_train), transformations, strategy)
        test_dataset = ImageClassificationDataset(NumpyDataset(dataset.x_test, dataset.y_test), transformations, strategy)
    else:
        raise ValueError(task)

    return train_dataset, test_dataset, dataset.input_shape, dim_output


# keep import so that new metrics are registered


def prepare_model(p: TrainParameters, input_shape, dim_output):
    task = p.task
    if task == Task.Classification:
        loss_function = "cross_entropy"
        batch_metrics = ['accuracy']
    elif task == Task.TransformationRegression:
        loss_function = "mse"
        batch_metrics = ["mae", "smape", "rae"] + \
            p.tc.convergence_criteria.metrics()
    else:
        raise ValueError(task)
    batch_metrics = sorted(list(set(batch_metrics)))

    model = p.mc.make(input_shape, dim_output)

    poutyne_model = Model(model,
                          optimizer=p.tc.optimizer,
                          loss_function=loss_function,
                          batch_metrics=batch_metrics,
                          device=p.tc.device)
    return model, poutyne_model


class SavepointCallback(Callback):

    def __init__(self, p: TrainParameters, model: Model, test_set, pc):
        self.model = model
        self.p = p
        self.test_set = test_set
        self.pc = pc
        super().__init__()

    def on_train_begin(self, logs: dict):
        if 0 in self.p.tc.savepoints:
            self.save_model_with_scores(0)

    def on_epoch_end(self, epoch_number: int, logs: dict):
        if epoch_number in self.p.tc.savepoints:
            self.save_model_with_scores(epoch_number)

    def save_model_with_scores(self, epoch_number):
        tc = self.p.tc
        scores = self.model.evaluate_dataset(
            self.test_set, num_workers=tc.num_workers, batch_size=tc.batch_size, verbose=False, return_dict_format=True)
        # if self.p.tc.verbose:
        #     print(f"Saving model {self.model.network.name} at epoch {epoch_number}/{tc.epochs}.")
        save_model(self.p, self.model.network, scores,
                   self.pc.model_path_new(self.p, epoch_number))



class ConvergenceError(Exception):
    def __init__(self,  metrics, convergence: ConvergenceCriteria):
        self.metrics = metrics
        self.convergence = convergence
        self.message = f"Convergence Criteria {convergence} not reached (metrics={metrics})"
        super().__init__(self.message)

def replace_in_keys(d:dict,a:str,b:str):
    n = len(a)
    keys = d.copy()
    for k in keys:
            if k.startswith(a):
                new_k = b+k[n:]
                d[new_k] = d[k]
                del d[k]

def train(p: TrainParameters, path_config):
    train_dataset, test_dataset, input_shape, dim_output = prepare_dataset(p.transformations, p.dataset_name, p.task)
    if p.tc.epochs == 0:
        print("Warning: epochs chosen = 0, saving model without training..")
        model, poutyne_model = prepare_model(p, input_shape, dim_output)

        metrics = poutyne_model.evaluate_dataset(test_dataset, batch_size=p.tc.batch_size, num_workers=p.tc.num_workers,return_dict_format=True, verbose=False, dataloader_kwargs={"pin_memory": True})
        save_model(p, model, metrics, path_config.model_path_new(p))

        return model,metrics

    restarts = 0
    cc = p.tc.convergence_criteria
    converged = False
    # train until convergence or p.tc.max_restarts
    model, loss, metrics = None, None, None
    
    progress = TotalProgressCallback()
    while restarts < p.tc.max_restarts and not converged:
        model, poutyne_model = prepare_model(p, input_shape, dim_output)

        savepoint_callback = SavepointCallback(
            p, poutyne_model, test_dataset, path_config)

        history = poutyne_model.fit_dataset(train_dataset, test_dataset, batch_size=p.tc.batch_size, epochs=p.tc.epochs, callbacks=[
                                                savepoint_callback,progress], verbose=False,
                                            num_workers=p.tc.num_workers,
                                            dataloader_kwargs={"shuffle": True, "pin_memory": True,"drop_last":True})

        metrics = poutyne_model.evaluate_dataset(test_dataset, batch_size=p.tc.batch_size, num_workers=p.tc.num_workers,
                                                 return_dict_format=True, verbose=False, dataloader_kwargs={"pin_memory": True})
        train_metrics = poutyne_model.evaluate_dataset(train_dataset, batch_size=p.tc.batch_size, num_workers=p.tc.num_workers,
                                                       return_dict_format=True, verbose=False, dataloader_kwargs={"pin_memory": True})

        replace_in_keys(train_metrics,"test","train")

        plot_history(history, p, path_config.training_plots_path())
        if cc.converged(metrics):
            converged = True
        else:
            print(
                f"{restarts}/{p.tc.max_restarts}: Convergence Criteria {cc} not reached, metrics: {metrics}")
            restarts += 1

    if not converged:
        raise ConvergenceError(metrics, cc)

    save_model(p, model, metrics, path_config.model_path_new(p))
    return model, metrics, train_metrics


def save_model(p: TrainParameters, model: torch.nn.Module, scores:dict, filepath: Path):
    filepath.parent.mkdir(exist_ok=True, parents=True)
    torch.save({"parameters": p,
                "models": model,
                "model_state": model.state_dict(),
                "scores": scores,
                }, filepath)


def load_model(model_filepath: Path, device: str, load_state=True):
    data = torch.load(model_filepath, map_location=device)
    model_state = data["model_state"]
    model = data["models"]
    p: TrainParameters = data["parameters"]
    scores = data["scores"]

    if load_state:
        model.load_state_dict(model_state)
        model.eval()
    return model, p, scores


def plot_history(history, p: TrainParameters, folderpath: Path):
    if p.task == Task.TransformationRegression:
        metrics = ["loss"]+p.tc.convergence_criteria.metrics()
    elif p.task == Task.Classification:
        metrics = ["loss"]+p.tc.convergence_criteria.metrics()
    else:
        raise ValueError(p.task)
    f, ax_metrics = plt.subplots(1, len(metrics))
    folderpath = folderpath / f"{p.id()}.png"

    for i, m_train in enumerate(metrics):
        ax = ax_metrics[i]
        y_m_train = np.array([e[m_train] for e in history])
        m_val = f"val_{m_train}"
        y_m_val = np.array([e[m_val] for e in history])
        ax.plot(y_m_train)
        ax.plot(y_m_val)
        ax.set_ylabel(f"{m_train}/{m_val}")
        ax.set_xlabel('epoch')
        max_value = max(y_m_val.max(), y_m_train.max())
        ax.set_ylim(0, max_value*1.1)
        ax.legend(['train', 'val'], loc='lower right')
    # f.suptitle(f"({p.id()})")
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(folderpath)
    plt.close(f)
