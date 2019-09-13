#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import config
import torch
import datasets
from experiment import model_loading, training
import argparse,argcomplete
import transformation_measure as tm
from typing import Tuple


def parse_args()->Tuple[training.Parameters, training.Options]:

    bool_parser=lambda x: (str(x).lower() in ['true','1', 'yes'])
    transformations=config.all_transformations(10)
    transformations={t.id():t for t in transformations}

    parser = argparse.ArgumentParser(description="Script to train a models with a dataset and transformations")

    parser.add_argument('-verbose', metavar='v'
                        ,help=f'Print info about dataset/models/transformations'
                        ,type=bool_parser
                        , default=True)

    parser.add_argument('-train_verbose', metavar='tvb'
                        , help=f'Print details about the training'
                        , type=bool_parser
                        , default=True)

    parser.add_argument('-batchsize', metavar='b'
                        , help=f'batchsize to use during training'
                        , type=int
                        , default=256)

    parser.add_argument('-num_workers', metavar='nw'
                        , help=f'num_workersto use during training'
                        , type=int
                        , default=2)

    parser.add_argument('-notransform_epochs', metavar='n'
                        , help=f'Train with no transformations for notransform_epochs epochs'
                        , type=int
                        , default=0)

    parser.add_argument('-savemodel', metavar='b'
                        , help=f'Save models after training'
                        , type=bool_parser
                        , default=True)
    parser.add_argument('-plots', metavar='p'
                        , help=f'Generate plots for training epochs'
                        , type=bool_parser
                        , default=True)

    parser.add_argument('-usecuda', metavar='c'
                        , help=f'Use cuda'
                        , type=bool_parser
                        , default=torch.cuda.is_available())

    parser.add_argument('-model', metavar='m',
                        help=f'Model to train/use. Allowed values: {", ".join(model_loading.get_model_names())}'
                        ,choices=model_loading.get_model_names()
                        ,required=True)

    parser.add_argument('-dataset', metavar='d',
                        help=f'Dataset to train/eval models. Allowed values: {", ".join(datasets.names)}'
                        ,choices=datasets.names
                        ,required=True)

    parser.add_argument('-transformation', metavar='t',
                        help=f'Transformations to apply to the dataset to train a models. Allowed values: {", ".join(transformations.keys())}'
                        , choices=transformations.keys()
                        ,default=tm.SimpleAffineTransformationGenerator())

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    transformation=transformations[args.transformation]
    epochs = model_loading.get_epochs(args.model, args.dataset, transformation)


    p= training.Parameters(args.model, args.dataset, transformation, epochs, args.notransform_epochs)
    o= training.Options(args.verbose, args.train_verbose, args.savemodel, args.batchsize, args.num_workers, args.usecuda, args.plots)
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
        print(f"Training with models {p.model}.")
        print(model)


    # TRAINING
    import time

    t=time.perf_counter()
    scores,history= training.run(p, o, model, optimizer, dataset)
    print(f"Elapsed {time.perf_counter()-t} seconds.")

    training.print_scores(scores)

    training.plot_history(history, p, config.training_plots_path())

    min_accuracies={"mnist":.95,"cifar10":.5}
    min_accuracy=min_accuracies[p.dataset]
    # SAVING
    if o.save_model:
        test_accuracy=scores["test"][1]
        if test_accuracy>=min_accuracy:
            path=config.model_path(p)
            training.save_model(p, o, model, scores, path)
            print(f"Model saved to {path}")
        else:
            print(f"Model was not saved since it did not reach minimum accuracy. Accuracy={test_accuracy}<{min_accuracies}.")

