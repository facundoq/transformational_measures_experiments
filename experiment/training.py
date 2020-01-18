import matplotlib.pyplot as plt
from typing import Callable
from pytorch.training import train,test
import numpy as np
import os
import torch
from pathlib import Path
import typing
import logging
from torch import nn
from torch.optim.optimizer import Optimizer

import  transformation_measure as tm
import config
import datasets
from typing import *

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
            result += f"_savepoint={savepoint}"

        #TODO simply use self.suffix when retraining all models
        suffix = getattr(self, 'suffix', '')
        if len(suffix)>0:
            result += f"_{suffix}"
        return result

class Options:
    def __init__(self, verbose:bool,train_verbose:bool, save_model:bool, batch_size:int, num_workers:int, use_cuda:bool, plots:bool,max_restarts:int):
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.verbose=verbose
        self.train_verbose = train_verbose
        self.save_model=save_model
        self.plots=plots
        self.use_cuda=use_cuda
        self.max_restarts=max_restarts

    def __repr__(self):
        return f"batch_size={self.batch_size}, num_workers={self.num_workers}, verbose={self.verbose}, train_verbose={self.train_verbose}," \
            f" save_model={self.save_model}, plots={self.plots}, use_cuda={self.use_cuda}, max_restarts={self.max_restarts}"





def do_run(model:nn.Module,dataset:datasets.ClassificationDataset,t:tm.TransformationSet,o:Options, optimizer:Optimizer,epochs:int,loss_function:torch.nn.Module,epochs_callbacks:{int:Callable}):

    train_dataset = get_data_generator(dataset.x_train, dataset.y_train, t, o.batch_size, o.num_workers)
    test_dataset = get_data_generator(dataset.x_test, dataset.y_test, t, o.batch_size, o.num_workers)

    history = train(model, epochs, optimizer, o.use_cuda, train_dataset, test_dataset, loss_function,verbose=o.train_verbose,epochs_callbacks=epochs_callbacks)
    return history


def run(p:Parameters,o:Options,model:nn.Module,optimizer:Optimizer,
        dataset:datasets.ClassificationDataset,loss_function=torch.nn.NLLLoss(),epochs_callbacks:{int:Callable}={}):

    if p.notransform_epochs == 0:
        if o.train_verbose:
            print(f"### Skipping pretraining model |{p.model}| with dataset |{dataset.name}| and no transformation")
    else:
        if o.train_verbose:
            print(f"### Pretraining models |{p.model}| with untransformed dataset |{dataset.name}|for {p.notransform_epochs} epochs...",flush=True)
        t=tm.SimpleAffineTransformationGenerator()
        pre_history =do_run(model,dataset,t,o,optimizer,p.epochs,loss_function,epochs_callbacks)

    history =do_run(model,dataset,p.transformations,o,optimizer,p.epochs,loss_function,epochs_callbacks)
    if o.train_verbose:
        print("### Testing models on dataset...",flush=True)
    scores=eval_scores(model,dataset,p,o,loss_function)

    return scores,history


def eval_scores(m:nn.Module,dataset:datasets.ClassificationDataset,p:Parameters,o:Options,loss_function=torch.nn.NLLLoss()):

    train_dataset = get_data_generator(dataset.x_train, dataset.y_train, p.transformations, o.batch_size, o.num_workers)
    test_dataset = get_data_generator(dataset.x_test, dataset.y_test, p.transformations, o.batch_size, o.num_workers)

    datasets = {"train": train_dataset, "test": test_dataset}
    scores = {}
    for k, dataset in datasets.items():
        loss, accuracy, correct, n = test(m, dataset, o.use_cuda, loss_function)
        scores[k] = (loss, accuracy)

    return scores


from torch.utils.data import DataLoader
from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
from pytorch.numpy_dataset import NumpyDataset

def get_data_generator(x:np.ndarray, y:np.ndarray,
                       transformation:tm.TransformationSet, batch_size:int,num_workers:int)->DataLoader:

    dataset=NumpyDataset(x,y)
    image_dataset=ImageDataset(dataset,transformation)
    dataloader=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True,pin_memory=False,)

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
    f.suptitle(f"{p.model} trained with {p.dataset} and {p.transformations} ({p.id()})")
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


def load_model(model_filepath:Path,use_cuda:bool,load_state=True)->(nn.Module,Parameters,Options,typing.Dict):
    logging.info(f"Loading models from {model_filepath}...")
    if use_cuda:
        data = torch.load(model_filepath)
    else:
        data = torch.load(model_filepath,map_location='cpu')
    model_state=data["model_state"]
    model=data["models"]
    p:Parameters=data["parameters"]
    o:Options = data["options"]
    scores=data["scores"]
    # dataset=datasets.get(p.dataset)
    #models, optimizer = model_loading.get_model(p.models,dataset,use_cuda)
    if load_state:
        model.load_state_dict(model_state)
        model.eval()
    return model,p,o,scores

def print_scores(scores):
    for k, v in scores.items():
        print('%s score: loss=%f, accuracy=%f' % (k, v[0], v[1]))
