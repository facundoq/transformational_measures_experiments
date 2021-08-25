import matplotlib.pyplot as plt
from pytorch.training import train,test
import numpy as np
import torch
from pathlib import Path
import typing
import logging
from torch import nn
from torch.optim.optimizer import Optimizer

import  transformational_measures as tm

import datasets
from typing import *

from torch.utils.data import DataLoader
from pytorch.pytorch_image_dataset import ImageClassificationDataset,TransformationStrategy
from pytorch.numpy_dataset import NumpyDataset
import config

class Parameters:
    def __init__(self,model:config.ModelConfig,dataset:str
                 ,transformations:tm.TransformationSet
                 ,epochs:int
                 ,notransform_epochs:int=0,savepoints:[int]=None,suffix=""):

        self.model=model
        self.dataset=dataset
        self.transformations=transformations
        self.epochs=epochs
        self.notransform_epochs=notransform_epochs
        self.suffix=suffix

        if savepoints is None:
            savepoints=[]
        self.savepoints=savepoints

    def __repr__(self):
        if self.notransform_epochs>0:
            notransform_message=f", Notransform_epochs={self.notransform_epochs}"
        else:
            notransform_message=""
        return f"Model: {self.model.id()}, Dataset={self.dataset}, Transformations=({self.transformations}), Epochs={self.epochs}{notransform_message}"

    def id(self,savepoint:float=None):
        result = f"{self.model.id()}_{self.dataset}_{self.transformations.id()}"

        if self.notransform_epochs > 0:
            notransform_message = "_notransform_epochs={self.notransform_epochs}"
            result+=notransform_message

        if not savepoint is None:
            assert (savepoint <= self.epochs+self.notransform_epochs)
            assert (savepoint >= 0)
            if not self.savepoints.__contains__(savepoint):
                raise ValueError(f"Invalid savepoint {savepoint}. Options: {', '.join(self.savepoints)}")
            result += f"_savepoint={savepoint}:03d"

        suffix = self.suffix
        if len(suffix)>0:
            result += f"_{suffix}"
        return result

class Options:
    def __init__(self, save_model:bool, batch_size:int, num_workers:int, use_cuda:bool, plots:bool, max_restarts:int, verbose_general:bool=False, verbose_train:bool=False, verbose_batch:bool=False):
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.verbose_general=verbose_general
        self.verbose_train = verbose_train
        self.verbose_batch = verbose_batch
        self.save_model=save_model
        self.plots=plots
        self.use_cuda=use_cuda
        self.max_restarts=max_restarts

    def get_eval_options(self):
        return EvalOptions(self.use_cuda,self.batch_size,self.num_workers)

    def __repr__(self):
        return f"batch_size={self.batch_size}, num_workers={self.num_workers}, verbose_dataset={self.verbose_general}, verbose_train={self.verbose_train}," \
            f" save_model={self.save_model}, plots={self.plots}, use_cuda={self.use_cuda}, max_restarts={self.max_restarts}"





def do_run(model:nn.Module,dataset:datasets.ClassificationDataset,t:tm.TransformationSet,o:Options, optimizer:Optimizer,epochs:int,loss_function:torch.nn.Module,epochs_callbacks:{int:Callable}):

    train_dataset = get_data_generator(dataset.x_train, dataset.y_train, t, o.batch_size, o.num_workers,TransformationStrategy.random_sample)
    test_dataset = get_data_generator(dataset.x_test, dataset.y_test, t, o.batch_size, o.num_workers,TransformationStrategy.random_sample)

    history = train(model, epochs, optimizer, o.use_cuda, train_dataset, test_dataset, loss_function, verbose=o.verbose_train, epochs_callbacks=epochs_callbacks,batch_verbose=o.verbose_batch)
    return history


def run(p:Parameters,o:Options,model:nn.Module,optimizer:Optimizer,
        dataset:datasets.ClassificationDataset,loss_function=torch.nn.NLLLoss(),epochs_callbacks:{int:Callable}={}):

    if p.notransform_epochs > 0:
        if o.verbose_train:
            print(f"### Pretraining models |{p.model}| with untransformed dataset |{dataset.name}|for {p.notransform_epochs} epochs...",flush=True)
        t=config.identity_transformation
        pre_history =do_run(model,dataset,t,o,optimizer,p.epochs,loss_function,epochs_callbacks)

    history =do_run(model,dataset,p.transformations,o,optimizer,p.epochs,loss_function,epochs_callbacks)
    if o.verbose_batch:
        print("### Testing models on dataset...",flush=True)
    scores=eval_scores(model, dataset, p.transformations, TransformationStrategy.random_sample, o.get_eval_options(), loss_function)

    return scores,history

class EvalOptions:
    def __init__(self,use_cuda:torch.cuda.is_available(),batch_size=32,num_workers=0):
        self.use_cuda=use_cuda
        self.batch_size=batch_size
        self.num_workers=num_workers


def eval_scores(m:nn.Module, dataset:datasets.ClassificationDataset, transformations:tm.TransformationSet, transformation_strategy:TransformationStrategy, o:EvalOptions, loss_function=torch.nn.NLLLoss(), subsets=["train", "test"].copy()):
    train_dataset = get_data_generator(dataset.x_train, dataset.y_train, transformations, o.batch_size, o.num_workers,transformation_strategy)
    test_dataset = get_data_generator(dataset.x_test, dataset.y_test, transformations, o.batch_size, o.num_workers,transformation_strategy)

    datasets = {}
    if "train" in subsets:
        datasets["train"]= train_dataset
    if "test" in subsets:
        datasets["test"] = test_dataset

    scores = {}
    for k, dataset in datasets.items():
        loss, accuracy, correct, n = test(m, dataset, o.use_cuda, loss_function)
        scores[k] = (loss, accuracy)

    return scores




def get_data_generator(x:np.ndarray, y:np.ndarray,
                       transformation:tm.TransformationSet, batch_size:int, num_workers:int, transformation_strategy:TransformationStrategy)->DataLoader:

    dataset=NumpyDataset(x,y[:,np.newaxis])
    # TODO verify this
    image_dataset=ImageClassificationDataset(dataset, transformation, transformation_strategy)
    dataloader=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True,pin_memory=True,)

    return dataloader




def plot_history(history, p:Parameters, folderpath:str):
    from time import gmtime, strftime
    t=strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    import os
    f, (a1,a2) = plt.subplots(1,2)
    folderpath=os.path.join(folderpath, f"{p.id()}.png")
    # accuracy
    a1.plot(history['acc'])
    a1.plot(history['acc_val'])
    #a1.set_title('Accuracy')
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



def save_model(p:Parameters,o:Options,model:nn.Module,scores,filepath:Path):

    torch.save({"parameters":p,
                "models":model,
                "model_state": model.state_dict(),
                "scores":scores,
                "options":o,
    }, filepath)


def load_model(model_filepath:Path,device:torch.device,load_state=True)->(nn.Module,Parameters,Options,typing.Dict):
    #print("load_model",model_filepath)
    logging.info(f"Loading models from {model_filepath}...")
    data = torch.load(model_filepath,map_location=device)
    model_state=data["model_state"]
    model=data["models"]
    p:Parameters=data["parameters"]
    # o:Options = data["options"]
    scores=data["scores"]
    # dataset=datasets.get(p.dataset)
    #models, optimizer = model_loading.get_model(p.models,dataset,use_cuda)
    if load_state:
        model.load_state_dict(model_state)
        model.eval()
    return model,p,scores

def print_scores(scores):
    for k, v in scores.items():
        print('%s score: loss=%f, accuracy=%f' % (k, v[0], v[1]))
