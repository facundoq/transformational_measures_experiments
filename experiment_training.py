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
from pytorch.experiment import training

import argparse,argcomplete


from pytorch.experiment import model_loading
import transformation_measure as tm
from typing import Tuple



def parse_args()->Tuple[training.Parameters,training.Options]:

    bool_parser=lambda x: (str(x).lower() in ['true','1', 'yes'])
    transformations = [tm.SimpleAffineTransformationGenerator(),tm.SimpleAffineTransformationGenerator(n_rotations=16)]

    parser = argparse.ArgumentParser(description="Script to train a model with a dataset and transformations")

    parser.add_argument('-verbose', metavar='v'
                        ,help=f'Print info about dataset/model/transformations'
                        ,type=bool_parser
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
                        , type=bool_parser
                        , default=True)
    parser.add_argument('-plots', metavar='b'
                        , help=f'Generate plots for training epochs'
                        , type=bool_parser
                        , default=True)

    parser.add_argument('-usecuda', metavar='b'
                        , help=f'Use cuda'
                        , type=bool_parser
                        , default=torch.cuda.is_available())

    parser.add_argument('-model', metavar='m',
                        help=f'Model to train/use. Allowed values: {", ".join(model_loading.get_model_names())}'
                        ,choices=model_loading.get_model_names()
                        ,required=True)

    parser.add_argument('-dataset', metavar='d',
                        help=f'Dataset to train/eval model. Allowed values: {", ".join(datasets.names)}'
                        ,choices=datasets.names
                        ,required=True)

    transformation_names=[t.id() for t in transformations]
    parser.add_argument('-transformation', metavar='t',
                        help=f'Transformations to apply to the dataset to train a model. Allowed values: {", ".join(transformation_names)}'
                        , choices=transformation_names
                        ,default=tm.SimpleAffineTransformationGenerator())

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    epochs = model_loading.get_epochs(args.model,args.dataset,args.transformation)

    p=training.Parameters(args.model,args.dataset,args.transformation,epochs,args.notransform_epochs)
    o=training.Options(args.verbose,args.savemodel,args.batchsize,args.usecuda,args.plots)
    return p,o

p,o = parse_args()
dataset = datasets.get(p.dataset)
model, optimizer = model_loading.get_model(p.model, dataset, o.use_cuda)

print("Parameters: ",p)
print("Options: ",o)
if o.verbose:
    print(f"Experimenting with dataset {p.dataset}.")
    print(dataset.summary())
    print(f"Training with model {p.model}.")
    print(model)


# TRAINING

scores,history=training.run(p,o,model,optimizer, dataset)
training.print_scores(scores)
training.plot_history(history,p)

# SAVING
if o.save_model:
    training.save_model(p, o, model, scores)

