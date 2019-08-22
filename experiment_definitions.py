#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


from pytorch import variance
import transformation_measure as tm
from pytorch.experiment import training

import util

from typing import List
import numpy as np
from transformation_measure import visualization
import typing
from pytorch import variance
from pytorch.experiment import training
import os,sys
import argparse,argcomplete
from pytorch.variance import DatasetSubset,DatasetParameters
import runner_utils
from pytorch import models
model_names= models.names
# model_names.sort()

measures = tm.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]

datasets= ["mnist", "cifar10"]
venv_path=runner_utils.get_venv_path()

measure=tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum)

def experiment_training(p: training.Parameters):
    if os.path.exists(training.experiment_model_path(p)):
        return

    python_command = f'experiment_training.py -model "{p.model}" -dataset "{p.dataset}" -transformation "{p.transformations.id()}" -verbose False -train_verbose False -num_workers 4'
    runner_utils.run_python(venv_path, python_command)

def experiment_variance(p: variance.Parameters):
    if os.path.exists(variance.results_path(p)):
        return

    python_command = f'experiment_variance.py -mo "{p.model}.pt" -me "{p.measure.id()}" -d "{p.dataset.id()}" -t "{p.transformations.id()}" -verbose False'
    runner_utils.run_python(venv_path, python_command)

def experiment_plot_layers(variance_parameters:[variance.Parameters],id: str):
    variance_paths= [f'"{variance.results_path(p)}"' for p in variance_parameters]
    variance_paths_str= " ".join(variance_paths)
    python_command = f"experiment_plot_layers.py -id \"{id}\" {variance_paths_str}"
    runner_utils.run_python(venv_path, python_command)

def compare_measures():
    epochs= 0
    for model in model_names:
        for dataset in datasets:
            p_dataset= variance.DatasetParameters(dataset, variance.DatasetSubset.test, 1.0)
            for transformation in tm.common_transformations_without_identity():
                p_training= training.Parameters(model, dataset, transformation, epochs, 0)
                experiment_training(p_training)
                variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
                for p_variance in variance_parameters:
                    experiment_variance(p_variance)
                experiment_plot_layers(variance_parameters, "compare_measures")


if __name__ == '__main__':
    compare_measures()