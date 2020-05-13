import os
import typing

import datasets
import transformation_measure as tm
from enum import Enum

from pathlib import Path
import datasets
import torch,config
from experiment import training

from utils.profiler import Profiler

from .parameters import  Parameters,Options,DatasetParameters,MeasureExperimentResult
from .adapt import adapt_dataset

def experiment(p: Parameters, o: Options):
    assert(len(p.transformations)>0)
    use_cuda = torch.cuda.is_available()

    dataset = datasets.get(p.dataset.name)
    if o.verbose:
        print(dataset.summary())

    model, training_parameters, training_options, scores = training.load_model(p.model_path, use_cuda)


    if training_parameters.dataset != p.dataset.name:
        if o.adapt_dataset:
            if o.verbose:
                print(f"Adapting dataset {p.dataset.name} to model trained on dataset {training_parameters.dataset} (resizing spatial dims and channels)")

            adapt_dataset(dataset,training_parameters.dataset)
            if o.verbose:
                print(dataset.summary())
        else:
            print(f"Error: model trained on dataset {training_parameters.dataset}, but requested to measure on dataset {p.dataset.name}; specify the option '-adapt_dataset True' to adapt the test dataset to the model and test anyway.")

    if o.verbose:
        print("### ", model)
        print("### Scores obtained:")
        training.print_scores(scores)

    import transformation_measure as tm

    from pytorch.numpy_dataset import NumpyDataset
    dataset = dataset.reduce_size_stratified(p.dataset.percentage)
    dataset.normalize_features()

    # TODO move the subset enum to ClassificationDataset
    if p.dataset.subset == datasets.DatasetSubset.test:
        x,y = dataset.x_test,dataset.y_test
    elif p.dataset.subset == datasets.DatasetSubset.train:
        x,y = dataset.x_train,dataset.x_test
    else:
        raise ValueError(p.dataset.subset)

    numpy_dataset = NumpyDataset(x)

    if not p.stratified:
        iterator = tm.NormalPytorchActivationsIterator(model, numpy_dataset, p.transformations, o.batch_size,o.num_workers,use_cuda)
        if o.verbose:
            print(f"Calculating measure {p.measure} dataset size {len(numpy_dataset)}...")

        measure_result = p.measure.eval(iterator)
    else:
        if o.verbose:
            print(f"Calculating stratified version of measure {p.measure}...")
        stratified_numpy_datasets = numpy_dataset.stratify_dataset(y)
        stratified_iterators = [tm.NormalPytorchActivationsIterator(model, numpy_dataset, p.transformations, o.batch_size,o.num_workers,use_cuda) for numpy_dataset in stratified_numpy_datasets]
        measure_result = p.measure.eval_stratified(stratified_iterators,dataset.labels)

    del model
    del dataset
    torch.cuda.empty_cache()

    return MeasureExperimentResult(p, measure_result)


def main(p:Parameters,o:Options):
    profiler= Profiler()
    profiler.event("start")
    if o.verbose:
        print(f"Experimenting with parameters: {p}")
    measures_results=experiment(p,o)
    profiler.event("end")
    print(profiler.summary(human=True))
    config.save_results(measures_results)
