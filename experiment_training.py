#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import torch
import datasets
from pytorch.experiment import training,model_loading
import argparse,argcomplete
import transformation_measure as tm
from typing import Tuple


def parse_args()->Tuple[training.Parameters,training.Options]:

    bool_parser=lambda x: (str(x).lower() in ['true','1', 'yes'])
    transformations=tm.common_transformations()
    transformations={t.id():t for t in transformations}

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

    parser.add_argument('-transformation', metavar='t',
                        help=f'Transformations to apply to the dataset to train a model. Allowed values: {", ".join(transformations.keys())}'
                        , choices=transformations.keys()
                        ,default=tm.SimpleAffineTransformationGenerator())

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    transformation=transformations[args.transformation]
    epochs = model_loading.get_epochs(args.model,args.dataset,transformation)

    p=training.Parameters(args.model,args.dataset,transformation,epochs,args.notransform_epochs)
    o=training.Options(args.verbose,args.savemodel,args.batchsize,args.usecuda,args.plots)
    return p,o

if __name__ == "__main__":
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
    import time

    t=time.perf_counter()
    scores,history=training.run(p,o,model,optimizer, dataset)
    print(f"Elapsed {time.perf_counter()-t} seconds.")

    training.print_scores(scores)
    training.plot_history(history,p)

    # SAVING
    if o.save_model:
        training.save_model(p, o, model, scores)

