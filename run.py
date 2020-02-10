#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse

import os
from pathlib import Path

import argcomplete

import texttable


from experiments import *

import datasets
import torch


class CompareAnovaMeasures(Experiment):
    def description(self):
        return """Determine which Anova measure is more appropriate"""

    def run(self):
        pass




class Options:
    def __init__(self, venv: Path, show_list: bool, force: bool):
        self.venv = venv
        self.show_list = show_list
        self.force = force


def parse_args(experiments: [Experiment]) -> ([Experiment], Options):
    parser = argparse.ArgumentParser(description="Run experiments with transformation measures.")

    experiment_names = [e.id() for e in experiments]
    experiment_dict = dict(zip(experiment_names, experiments))

    parser.add_argument('-experiment',
                        help=f'Choose an experiment to run',
                        type=str,
                        default=None,
                        required=False,choices=experiment_names, )
    parser.add_argument('-force',
                        help=f'Force experiments to rerun even if they have already finished',
                        action="store_true")
    parser.add_argument('-list',
                        help=f'Force experiments to rerun even if they have already finished',
                        action="store_true")
    parser.add_argument("-venv",
                        help="Path to virtual environment to run experiments on",
                        type=str,
                        default=os.path.expanduser("~/dev/env/vm"),
                        )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not args.experiment is None:
        experiments = [experiment_dict[args.experiment]]

    return experiments, Options(Path(args.venv), args.list, args.force)

def print_table(experiments:[Experiment]):
    table = texttable.Texttable()
    header = ["Experiment", "Finished"]
    table.header(header)
    experiments.sort(key=lambda e: e.__class__.__name__)
    for e in experiments:
        status = "Y" if e.has_finished() else "N"
        name = e.__class__.__name__
        name = name[:40]
        table.add_row((name, status))
        # print(f"{name:40}     {status}")
    print(table.draw())

if __name__ == '__main__':

    todo = [

    ]
    if len(todo)>0:
        print("TODO implement experiments:", ",".join([e.__class__.__name__ for e in todo]))


    all_experiments = [
        TrainModels(),# run this first or you'll need to retrain some models
        DataAugmentationClassical(),
        DataAugmentationHandshape(),
        SimpleConvAccuracies(),
        ModelAccuracies(),

        CompareMeasures(),
        CompareSameEquivariance(),
        CompareSameEquivarianceNormalized(),
        CompareSameEquivarianceSimple(),

        CompareGoodfellowAlpha(),
        CompareGoodfellow(),
        Stratified(),

        DatasetSize(),
        DatasetSubset(),
        DatasetTransfer(),

        AggregationFunctionsVariance(),
        AggregationBeforeAfter(),
        AggregationFunctionsDistance(),
        SameEquivarianceNormalization(),

        TransformationDiversity(),
        TransformationComplexity(),

        BatchNormalization(),
        ActivationFunction(),
        MaxPooling(),
        KernelSize(),

        RandomInitialization(),
        RandomWeights(),
        DuringTraining(),
        #VisualizeInvariantFeatureMaps(),

        CompareModels(),
        TIPooling(),

        ValidateMeasure(),
        ValidateGoodfellow(),
    ]
    experiments, o = parse_args(all_experiments)
    if o.show_list:
        print_table(experiments)
    else:
        for e in experiments:
            e.set_venv(o.venv)
            e(force=o.force)
