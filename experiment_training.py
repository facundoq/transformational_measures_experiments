#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
logging.getLogger().setLevel(logging.DEBUG)
import torch


# DATASET
import datasets
from pytorch import models

from pytorch.experiment import model_loading,train_transformation

import pytorch.experiment.utils as utils
import argparse,argcomplete

from pytorch.model import
from pytorch.experiment import model_loading
import transformation_measure as tm
from typing import Tuple

class Parameters:
    def __init__(self,model:str,dataset:str
                 ,transformations:tm.SimpleAffineTransformationGenerator
                 ,epochs:int
                 ,notransform_epochs:int):
        self.model=model
        self.dataset=dataset
        self.transformations=transformations
        self.epochs=epochs
        self.notransform_epochs=notransform_epochs

    def __repr__(self):
        if self.notransform_epochs>0:
            notransform_message="_notransform_epochs={self.notransform_epochs}"
        else:
            notransform_message=""

        return f"{self.model}_{self.dataset}_{self.transformations}_epochs={self.epochs}{notransform_message}"

class Options:
    def __init__(self,verbose:bool,save_model:bool,batch_size:int,use_cuda:bool,save_plots:bool):
        self.batch_size=batch_size
        self.verbose=verbose
        self.save_model=save_model
        self.save_plots=save_plots
        self.use_cuda=use_cuda


def parse_args()->Tuple[Parameters,Options]:
    transformations = [tm.SimpleAffineTransformationGenerator(),tm.SimpleAffineTransformationGenerator(n_rotations=16)]

    parser = argparse.ArgumentParser(description="Script to train a model with a dataset and transformations")

    parser.add_argument('-verbose', metavar='v'
                        ,help=f'Print info about dataset/model/transformations'
                        ,type=bool
                        , default=True)
    parser.add_argument('-batchsize', metavar='b'
                        , help=f'batchsize to use during training'
                        , type=int
                        , default=256)

    parser.add_argument('-notransform_epochs', metavar='b'
                        , help=f'Train with no transformations for notransform_epochs epochs'
                        , type=int
                        , default=0)


    parser.add_argument('-savemodel', metavar='b'
                        , help=f'Save model after training'
                        , type=bool
                        , default=True)
    parser.add_argument('-usecuda', metavar='b'
                        , help=f'Use cuda'
                        , type=bool
                        , default=torch.cuda.is_available())

    parser.add_argument('-model', metavar='m',
                        help=f'Model to train/use. Allowed values: {", ".join(model_loading.get_model_names())}'
                        ,choices=model_loading.get_model_names()
                        ,required=True)
    parser.add_argument('-dataset', metavar='d',
                        help=f'Dataset to train/eval model. Allowed values: {", ".join(datasets.names)}'
                        ,choices=datasets.names
                        ,required=True)

    parser.add_argument('-transformation', metavar='t',
                        help=f'Transformations to apply to the dataset to train a model. Allowed values: {", ".join(transformations.names)}'
                        , choices=[str(t) for t in transformations]
                        ,default=tm.SimpleAffineTransformationGenerator())

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    epochs = model_loading.get_epochs(args.model,args.dataset,args.transformation)

    p=Parameters(args.model,args.dataset,args.transformation,epochs,args.notransform_epochs)
    o=Options(args.verbose,args.savemodel,args.batchsize,args.usecuda)
    return p,o

p,o = parse_args()
dataset = datasets.get(p.dataset)
model, optimizer = model_loading.get_model(p.model, dataset, o.use_cuda)

if o.verbose:
    print(f"Experimenting with dataset {p.dataset}.")
    print(dataset.summary())
    print(f"Training with model {p.model}.")
    print(model)

# TRAINING

scores=train_transformation.run(model, dataset,p,o)
train_transformation.print_scores(scores)

# SAVING

if o.save_model:
    train_transformation.save_models(dataset, model, scores)

