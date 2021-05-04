import transformational_measures as tm
from . import Task
import torch
import numpy as np
import datasets
from typing import Dict
from pathlib import Path

from poutyne import Model,Callback

from pytorch.pytorch_image_dataset import ImageClassificationDataset,TransformationStrategy,ImageTransformRegressionDataset
from pytorch.numpy_dataset import NumpyDataset
from abc import ABC,abstractmethod

def default_device(): return "cuda " if torch.cuda.is_available() else "cpu"

from ..tm_experiment import TMExperiment

class ModelConfig(ABC):
    @abstractmethod
    def make(self,input_shape:np.ndarray, output_dim:int)->tm.ObservableLayersModule:
        pass

    @abstractmethod
    def id(self)->str:
        pass

    @abstractmethod
    def epochs(self,dataset:str,task:Task,transformations:tm.TransformationSet):
        pass

    def min_accuracy(self,dataset:str,task:Task,transformations:tm.TransformationSet):
        min_accuracies = {"mnist": .90, "cifar10": .5, "lsa16": .6, "rwth": .45}
        return min_accuracies[dataset]

    def scale_by_transformations(self,epochs:int,transformations:tm.TransformationSet):
        m = len(transformations)
        if m > np.e:
            factor = 1.1 * np.log(m)
        else:
            factor = 1
        return int(epochs * factor)

class ConvergenceCriteria(ABC):
    @abstractmethod
    def converged(self,loss,metrics):
        pass

class MinAccuracyConvergence(ConvergenceCriteria):
    def __init__(self,loss,min_accuracy:float):
        self.min_accuracy=min_accuracy
    def converged(self,metrics):
        return metrics["accuracy"]>self.min_accuracy
    def __repr__(self):
        return f"MinAccuracy({self.min_accuracy})"

class MaxMSEConvergence(ConvergenceCriteria):
    def __init__(self,loss,max_mse:float):
        self.max_mse=max_mse
    def converged(self,metrics):
        return metrics["mse"]<self.max_mse
    def __repr__(self):
        return f"MaxMSE({self.max_mse})"

class TrainConfig:
    def __init__(self, epochs: int, cc:ConvergenceCriteria, save_model=True, max_restarts: int = 5, savepoints: [int] = None,
                 device=default_device(), suffix="",
                 verbose=False, num_workers=0, batch_size=32, plots=True):
        self.epochs = epochs
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
    def __init__(self,mc:ModelConfig,tc:TrainConfig,dataset:str,transformations:tm.TransformationSet,task:Task):
        self.mc=mc
        self.tc=tc
        self.dataset_name=dataset
        self.transformations=transformations
        self.task=task

    def id(self,savepoint:float=None):
        result = f"{self.mc.id()}_{self.dataset_name}_{self.transformations.id()}_{self.task.value}"

        if not savepoint is None:
            assert (savepoint <= self.tc.epochs)
            assert (savepoint >= 0)
            if not savepoint in self.tc.savepoints:
                raise ValueError(f"Invalid savepoint {savepoint}. Options: {', '.join(self.tc.savepoints)}")
            result += f"_savepoint={savepoint}:03"

        suffix = self.tc.suffix
        if len(suffix)>0:
            result += f"_{suffix}"
        return result


def prepare_dataset(p:TrainParameters):
    tc, transformations, dataset_name, task = p.tc, p.transformations, p.dataset_name, p.task
    strategy = TransformationStrategy.random_sample
    if p.task == Task.Regression:

        dataset = datasets.get_regression(dataset_name)
        dim_output = len(transformations)
        dataset.normalize_features()
        train_dataset = ImageTransformRegressionDataset(NumpyDataset(dataset.x_train), p.transformations, strategy)
        test_dataset = ImageTransformRegressionDataset(NumpyDataset(dataset.x_test), p.transformations, strategy)

    elif task == Task.Classification:
        dataset = datasets.get_classification(dataset_name)
        dim_output = dataset.num_classes
        dataset.normalize_features()
        train_dataset = ImageClassificationDataset(NumpyDataset(dataset.x_train,dataset.y_train), p.transformations, strategy)
        test_dataset = ImageClassificationDataset(NumpyDataset(dataset.x_test,dataset.y_test), p.transformations, strategy)

    else:
        raise ValueError(task)

    return train_dataset,test_dataset,dataset.input_shape,dim_output


def prepare_model(p:TrainParameters, input_shape, dim_output):
    task = p.task
    if task == Task.Classification:
        loss_function = "cross_entropy"
        batch_metrics = ['accuracy']
    elif task == Task.Regression:
        loss_function = "mse"
        batch_metrics = ['error']
    else:
        ValueError(task)

    model = p.mc.make(input_shape, dim_output)
    poutyne_model = Model(model,
                          optimizer='adam',
                          loss_function=loss_function,
                          batch_metrics=batch_metrics,
                          device=p.tc.device)
    return model,poutyne_model


class SavepointCallback(Callback):

    def __init__(self,p:TrainParameters,model:Model,test_set,pc:TMExperiment):
        self.model=model
        self.p=p
        self.test_set = test_set
        self.pc=pc

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        tc = self.p.tc
        if epoch_number in tc.savepoints:
            # scores = training.eval_scores(model, dataset, p.transformations, TransformationStrategy.random_sample,
            #                               o.get_eval_options())

            scores = self.model.evaluate_dataset(self.test_set,num_workers=tc.num_workers,batch_size=tc.batch_size)

            if self.tc.verbose:
                print(f"Saving model {self.model.network.name} at epoch {epoch_number}/{tc.epochs}.")

            save_model(self.p.mc,tc, self.model.network, scores, self.pc.model_path(self.p, epoch_number))

class ConvergenceError(Exception):
    def __init__(self, loss,metrics,convergence:ConvergenceCriteria):
        self.metrics=metrics
        self.loss=loss
        self.convergence=convergence
        self.message = f"Convergence Criteria {convergence} not reached (l={loss},metrics={metrics})"
        super().__init__(self.message)

def train(p:TrainParameters,path_config:TMExperiment):
    train_dataset, test_dataset, input_shape, dim_output = prepare_dataset(p)
    restarts = 0
    converged = False
    # train until convergence or p.tc.max_restarts
    while restarts<p.tc.max_restarts and not converged:
        model, poutyne_model = prepare_model(p, input_shape, dim_output)
        savepoint_callback = SavepointCallback(p, poutyne_model, test_dataset, path_config)
        history = poutyne_model.fit_dataset(train_dataset, test_dataset,batch_size=p.tc.batch_size,
                                            epochs=p.tc.epochs, callbacks=[savepoint_callback])
        loss, metrics = poutyne_model.evaluate_generator(test_dataset)
        plot_history(history, p, path_config.training_plots_path())
        if p.tc.convergence_criteria.converged(loss,metrics):
            converged=True
        else:
            restarts+=1

    if not converged:
        raise ConvergenceError(loss,metrics,p.tc.convergence_criteria)
    scores = (loss,metrics)
    save_model(p.mc,p.tc,model,scores,path_config.model_path(p))
    return model,loss,metrics


def save_model(mc:ModelConfig,tc:TrainConfig,model:torch.nn.Module,scores,filepath:Path):

    torch.save({"model_config":mc,
                "models":model,
                "model_state": model.state_dict(),
                "scores":scores,
                "train_config":tc,
    }, filepath)


def load_model(model_filepath:Path,device:str,load_state=True):
    #print("load_model",model_filepath)

    data = torch.load(model_filepath,map_location=device)
    model_state=data["model_state"]
    model=data["models"]
    mc:ModelConfig=data["model_config"]
    tc:TrainConfig = data["train_config"]
    scores=data["scores"]

    if load_state:
        model.load_state_dict(model_state)
        model.eval()
    return model,mc,tc,scores

def print_scores(scores):
    for k, v in scores.items():
        print('%s score: loss=%f, accuracy=%f' % (k, v[0], v[1]))

import matplotlib.pyplot as plt

def plot_history(history, p:TrainParameters, folderpath:Path):
    f, (a1,a2) = plt.subplots(1,2)
    folderpath=folderpath / f"{p.id()}.png"
    # accuracy
    a1.plot(history['acc'])
    a1.plot(history['acc_val'])
    a1.set_title(p.id())
    a1.set_ylabel('accuracy')
    a1.set_xlabel('epoch')
    a1.set_ylim(0,1.1)
    a1.legend(['train', 'test'], loc='lower right')
    # loss
    a2.plot(history['loss'])
    a2.plot(history['loss_val'])
    #a2.set_title('Loss')
    a2.set_ylabel('loss')
    a2.set_xlabel('epoch')
    a2.legend(['train', 'test'], loc='upper right')
    # f.suptitle(f"{p.model} trained with {p.dataset} and {p.transformations} ({p.id()})")
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(folderpath)
    plt.close(f)