import matplotlib.pyplot as plt
from collections import namedtuple

from pytorch.training import train,eval_scores
import numpy as np
import os

from pytorch.classification_dataset import get_data_generator
import torch

import logging

from pytorch.utils import autolabel
from torch import nn
from torch.optim.optimizer import Optimizer

TrainRotatedConfig = namedtuple('TrainRotatedConfig', 'batch_size epochs pre_rotated_epochs rotated_epochs optimizer rotated_optimizer use_cuda')
from experiment_training import Parameters,Options
from  datasets import ClassificationDataset

def run(p:Parameters,o:Options,model:nn.Module,optimizer:Optimizer,
        dataset:ClassificationDataset,loss_function=torch.nn.NLLLoss()):

    os.makedirs(experiment_plot_path(model.name, dataset.name),exist_ok=True)
    train_dataset =get_data_generator(dataset.x_train,dataset.y_train, o.batch_size)
    test_dataset = get_data_generator(dataset.x_test, dataset.y_test, o.batch_size)

    history = train(model,p.epochs,optimizer,o.use_cuda,train_dataset,test_dataset,loss_function)

    if o.save_plots:
        accuracy_plot_path =plot_history(history,"unrotated",model.name,dataset.name)

    # ROTATED MODEL, UNROTATED DATASET
    if config.pre_rotated_epochs == 0:
        print(f"### Skipping pretraining rotated model |{model.name}| with unrotated dataset |{dataset.name}|")
    else:
        print(f"### Pretraining rotated model |{model.name}| with unrotated dataset |{dataset.name}|for {config.pre_rotated_epochs} epochs...",flush=True)
        pre_rotated_history = train(rotated_model, config.rotated_epochs, config.rotated_optimizer, config.use_cuda,
                                train_dataset,test_dataset,loss_function)
        if plot_accuracy:
            plot_history(pre_rotated_history,"pre_rotated",model.name,dataset.name,save_plots)

    print("### Testing both models on both datasets...",flush=True)

    models = {"rotated_model": rotated_model, "model": model}
    datasets = {"test_dataset": test_dataset, "rotated_test_dataset": rotated_test_dataset,
                 "train_dataset": train_dataset, "rotated_train_dataset": rotated_train_dataset}
    scores=eval_scores(models,datasets,config,loss_function)
    train_test_path=train_test_accuracy_barchart2(scores,model.name,dataset.name,save_plots)
    experiment_plot = os.path.join("plots",f"{model.name}_{dataset.name}_train_rotated.png")

    os.system(f"convert {accuracy_plot_path} {rotated_accuracy_plot_path} {train_test_path} +append {experiment_plot}")
    logging.info("training info saved to {experiment_plot}")

    return scores


def write_scores(scores,output_file,general_message,config=None):
    with open(output_file, "a+") as f:
        f.write(general_message)
        print(general_message)
        for k, v in scores.items():
            message = '%s score: loss=%f, accuracy=%f\n' % (k, v[0], v[1])
            print(message)
            f.write(message)
        if config:
            config_message="Config: "+str(config)
            print(config_message)
            f.write(config_message)
        f.write("\n\n")


def train_test_accuracy_barchart2(scores,model_name,dataset_name,savefig):
    test_dataset_scores = [scores["model_test_dataset"][1], scores["rotated_model_test_dataset"][1]]
    rotated_test_dataset_scores = [scores["model_rotated_test_dataset"][1], scores["rotated_model_rotated_test_dataset"][1]]
    accuracies = np.array([test_dataset_scores, rotated_test_dataset_scores])
    return train_test_accuracy_barchart(model_name, dataset_name, accuracies,savefig)

def experiment_plot_path(model_name,dataset_name):
    return f"plots/rotated_model/{model_name}_{dataset_name}"

def experiment_model_path(model_name,dataset_name):
    return f"models/{model_name}_{dataset_name}"

def train_test_accuracy_barchart(model_name, dataset_name, accuracies,savefig):
    import os
    # Accuracies:    |   Train unrotated   |   Train rotated
    # Test unrotated |                     |
    # Test rotated   |                     |
    #
    assert (accuracies.shape == (2, 2))

    fig, ax = plt.subplots()

    index = np.arange(2)
    bar_width = 0.3

    opacity = 0.4

    rects1 = ax.bar(index, accuracies[0, :], bar_width,
                    alpha=opacity, color='b',
                    label="Test unrotated")

    rects2 = ax.bar(index + bar_width, accuracies[1, :], bar_width,
                    alpha=opacity, color='r',
                    label="Test rotated")

    ax.set_ylim(0, 1.19)
    # ax.set_xlabel('Training scheme')
    ax.set_ylabel('Test accuracy')
    ax.set_title(f'Final accuracy on test sets.')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(("Train unrotated", "Train rotated"))
    ax.legend(loc="upper center")
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    path=os.path.join(experiment_plot_path(model_name,dataset_name), f"train_test.png")
    if savefig:
        plt.savefig(path)
    plt.show()
    return path


def plot_history(history,name,model_name,dataset_name):
    from time import gmtime, strftime
    t=strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    import os
    f, (a1,a2) = plt.subplots(1,2)
    path= experiment_plot_path(model_name, dataset_name)
    path=os.path.join(path,f"{name}.png")
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
    f.suptitle(f"{model_name} trained with {name} {dataset_name}")
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(path)
    plt.close(f)
    return path


MODEL_SAVE_FILENAME="models.pt"

def save_models(dataset,model,rotated_model,scores,config):
    model_folderpath = experiment_model_path(model.name, dataset.name)
    homedir = os.path.expanduser("~")
    model_folderpath = os.path.join(homedir,model_folderpath)
    if not os.path.exists(model_folderpath):
        os.makedirs(model_folderpath)
    filepath=os.path.join(model_folderpath,MODEL_SAVE_FILENAME)
    torch.save({"rotated":rotated_model.state_dict(),
                "unrotated": model.state_dict(),
                "scores":scores,
                "config":config,
    }, filepath)
from pytorch.experiment import model_loading


def load_models(dataset,model_name,use_cuda):
    models_state=load_models_state(dataset.name,model_name,use_cuda)
    model,optimizer,rotated_model,rotated_optimizer=model_loading.get_model(model_name, dataset, use_cuda)
    model.load_state_dict(models_state["unrotated"])
    rotated_model.load_state_dict(models_state["rotated"])
    model.eval()
    rotated_model.eval()
    return model,rotated_model,models_state["scores"],models_state["config"]

def load_models_state(dataset_name,model_name,use_cuda):
    model_folderpath = experiment_model_path(model_name, dataset_name)
    homedir = os.path.expanduser("~")
    model_filepath=os.path.join(homedir,model_folderpath,MODEL_SAVE_FILENAME)
    if not os.path.exists(model_filepath):
        message=f"The model |{model_name}| was not trained on dataset " \
                f"|{dataset_name}| ({model_filepath} does not exist)." \
                f"Run experiment_rotation.py to generate the model"
        raise ValueError(message)
    else:
        logging.info(f"Loading model from {model_filepath}...")
    if use_cuda:
        models = torch.load(model_filepath)
    else:
        models=torch.load(model_filepath,map_location='cpu')
    return models


def print_scores(scores):
    for k, v in scores.items():
        print('%s score: loss=%f, accuracy=%f' % (k, v[0], v[1]))

from torch.utils.data import DataLoader

from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
from pytorch.numpy_dataset import NumpyDataset

def get_data_generator(x,y,batch_size):

    dataset=NumpyDataset(x,y)

    image_dataset=ImageDataset(dataset)
    dataloader=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

    rotated_dataset = NumpyDataset(x, y)
    image_rotated_dataset = ImageDataset(rotated_dataset, rotation=(-180,180))
    rotated_dataloader= DataLoader(image_rotated_dataset , batch_size=batch_size, shuffle=True, num_workers=1)

    return dataloader,rotated_dataloader
