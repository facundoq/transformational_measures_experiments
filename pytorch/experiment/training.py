import matplotlib.pyplot as plt
from collections import namedtuple

from pytorch.training import train,test
import numpy as np
import os
import torch

import logging
from pytorch.utils import autolabel
from torch import nn
from torch.optim.optimizer import Optimizer

import  transformation_measure as tm

import datasets

class Parameters:
    def __init__(self,model:str,dataset:str
                 ,transformations:tm.TransformationSet
                 ,epochs:int
                 ,notransform_epochs:int):
        self.model=model
        self.dataset=dataset
        self.transformations=transformations
        self.epochs=epochs
        self.notransform_epochs=notransform_epochs

    def __repr__(self):
        if self.notransform_epochs>0:
            notransform_message=f", Notransform_epochs={self.notransform_epochs}"
        else:
            notransform_message=""
        return f"Model: {self.model}, Dataset={self.dataset}, Transformations=({self.transformations}), Epochs={self.epochs}{notransform_message}"

    def id(self):
        if self.notransform_epochs>0:
            notransform_message="_notransform_epochs={self.notransform_epochs}"
        else:
            notransform_message=""

        return f"{self.model}_{self.dataset}_{self.transformations}_epochs={self.epochs}{notransform_message}"


class Options:
    def __init__(self, verbose:bool, save_model:bool, batch_size:int, use_cuda:bool, plots:bool):
        self.batch_size=batch_size
        self.verbose=verbose
        self.save_model=save_model
        self.plots=plots
        self.use_cuda=use_cuda
    def __repr__(self):
        return f"batch_size={self.batch_size}, verbose={self.verbose}," \
            f" save_model={self.save_model}, plots={self.plots}, use_cuda={self.use_cuda}"




def run(p:Parameters,o:Options,model:nn.Module,optimizer:Optimizer,
        dataset:datasets.ClassificationDataset,loss_function=torch.nn.NLLLoss()):


    if p.notransform_epochs == 0:
        print(f"### Skipping pretraining rotated model |{model.name}| with dataset |{dataset.name}|")
    else:
        print(f"### Pretraining rotated model |{model.name}| with unrotated dataset |{dataset.name}|for {p.notransform_epochs} epochs...",flush=True)
        t=tm.SimpleAffineTransformationGenerator()
        pre_train_dataset = get_data_generator(dataset.x_train, dataset.y_train,t, o.batch_size)
        pre_test_dataset = get_data_generator(dataset.x_test, dataset.y_test,t, o.batch_size)

        pre_history = train(model,p.epochs,optimizer,o.use_cuda,pre_train_dataset,pre_test_dataset,loss_function)



    train_dataset = get_data_generator(dataset.x_train, dataset.y_train,p.transformations, o.batch_size)
    test_dataset = get_data_generator(dataset.x_test, dataset.y_test,p.transformations, o.batch_size)
    history = train(model,p.epochs,optimizer,o.use_cuda,train_dataset,test_dataset,loss_function)

    print("### Testing models on dataset...",flush=True)

    datasets={"train":train_dataset,"test":test_dataset}
    scores={}
    for k,dataset in datasets.items():
        loss, accuracy, correct, n = test(model, dataset, o.use_cuda, loss_function)
        scores[k]=(loss,accuracy)

    # models = {"rotated_model": rotated_model, "model": model}
    # datasets = {"test_dataset": test_dataset, "rotated_test_dataset": rotated_test_dataset,
    #              "train_dataset": train_dataset, "rotated_train_dataset": rotated_train_dataset}
    # scores=eval_scores(models,datasets,config,loss_function)
    # train_test_path=train_test_accuracy_barchart2(scores,model.name,dataset.name,save_plots)
    # experiment_plot = os.path.join("plots",f"{model.name}_{dataset.name}_train_rotated.png")
    #
    # os.system(f"convert {accuracy_plot_path} {rotated_accuracy_plot_path} {train_test_path} +append {experiment_plot}")
    # logging.info("training info saved to {experiment_plot}")

    return scores,history



from torch.utils.data import DataLoader
from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
from pytorch.numpy_dataset import NumpyDataset

def get_data_generator(x:np.ndarray, y:np.ndarray,
                       transformation:tm.TransformationSet, batch_size:int)->DataLoader:

    dataset=NumpyDataset(x,y)
    image_dataset=ImageDataset(dataset,transformation)
    dataloader=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=8,drop_last=True)

    return dataloader




def plot_history(history,p:Parameters):
    from time import gmtime, strftime
    t=strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    import os
    f, (a1,a2) = plt.subplots(1,2)
    path= experiment_plot_path()
    path=os.path.join(path,f"{p.id()}.png")
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
    plt.savefig(path)
    plt.close(f)

def base_path():
    return os.path.expanduser("~/variance/")

def experiment_plot_path():
    base=base_path()
    plots_folderpath = "plots"
    plots_folderpath = os.path.join(base, plots_folderpath)
    os.makedirs(plots_folderpath, exist_ok=True)
    return plots_folderpath

def experiment_model_path():
    base = base_path()
    model_folderpath="models"
    model_folderpath = os.path.join(base, model_folderpath)
    os.makedirs(model_folderpath, exist_ok=True)
    return model_folderpath


def save_model(p:Parameters,o:Options,model:nn.Module,scores):
    model_folderpath = experiment_model_path()
    filename=f"{p.id()}.pt"
    filepath=os.path.join(model_folderpath,filename)
    torch.save({"parameters":p,
                "model":model,
                "model_state": model.state_dict(),
                "scores":scores,
                "options":o,
    }, filepath)

from pytorch.experiment import model_loading

def get_models():
    model_folderpath = experiment_model_path()
    files=os.listdir(model_folderpath)
    model_files=[f for f in files if f.endswith(".pt")]
    return model_files

def load_model(filename:str,use_cuda:bool):
    model_folderpath = experiment_model_path()
    model_filepath=os.path.join(model_folderpath,filename)
    logging.info(f"Loading model from {model_filepath}...")
    if use_cuda:
        data = torch.load(model_filepath)
    else:
        data = torch.load(model_filepath,map_location='cpu')
    model_state=data["model_state"]
    model=data["model"]
    p:Parameters=data["parameters"]
    o:Options = data["options"]
    # dataset=datasets.get(p.dataset)
    #model, optimizer = model_loading.get_model(p.model,dataset,use_cuda)

    model.load_state_dict(model_state)
    model.eval()
    return model,p,o,data["scores"]

def print_scores(scores):
    for k, v in scores.items():
        print('%s score: loss=%f, accuracy=%f' % (k, v[0], v[1]))


def write_experiment(p:Parameters,o:Options,scores):
    # with open(output_file, "a+") as f:
    #     f.write(general_message)
    #     print(general_message)
    #     for k, v in scores.items():
    #         message = '%s score: loss=%f, accuracy=%f\n' % (k, v[0], v[1])
    #         print(message)
    #         f.write(message)
    #     if config:
    #         config_message="Config: "+str(config)
    #         print(config_message)
    #         f.write(config_message)
    #         f.write("\n\n")
    pass