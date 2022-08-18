import os
import typing

import datasets
import tmeasures as tm
from enum import Enum

from pathlib import Path
import datasets
import torch,config

from experiments.tasks import train

from utils.profiler import Profiler

from .parameters import  Parameters,Options,DatasetParameters,MeasureExperimentResult,PyTorchParameters,PyTorchMeasureExperimentResult
from .adapt import adapt_dataset

def experiment(p: Parameters, o: Options,model_path:Path):
    
    assert(len(p.transformations)>0)
    

    dataset = datasets.get_classification(p.dataset.name)
    if o.verbose:
        print(dataset.summary())
    if o.verbose:
        print(f"Loading model {model_path}")
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"

    model, training_parameters,  scores = train.load_model(model_path, device=device)

    print(type(training_parameters),dir(training_parameters))
    if training_parameters.dataset != p.dataset.name:
        if o.adapt_dataset:
            if o.verbose:
                print(f"Adapting dataset {p.dataset.name} to model trained on dataset {training_parameters.dataset_name} (resizing spatial dims and channels)")

            adapt_dataset(dataset, training_parameters.dataset_name)
            if o.verbose:
                print(dataset.summary())
        else:
            print(f"Error: model trained on dataset {training_parameters.dataset_name}, but requested to measure on dataset {p.dataset.name}; specify the option '-adapt_dataset True' to adapt the test dataset to the model and test anyway.")

    if o.verbose:
        print("### ", model)
        print("### Scores obtained:")
        train.print_scores(scores)

    import tmeasures as tm

    from pytorch.numpy_dataset import NumpyDataset
    dataset = dataset.reduce_size_stratified(p.dataset.percentage)
    dataset.normalize_features()

    x,y=dataset.get_subset(p.dataset.subset)
    numpy_dataset = NumpyDataset(x)

    if not p.stratified:
        iterator = tm.pytorch.NormalPytorchActivationsIterator(model, numpy_dataset, p.transformations, o.batch_size,o.num_workers,use_cuda)
        if o.verbose:
            print(f"Calculating measure {p.measure} dataset size {len(numpy_dataset)}...")

        measure_result = p.measure.eval(iterator,verbose=False)
    else:
        if o.verbose:
            print(f"Calculating stratified version of measure {p.measure}...")
        stratified_numpy_datasets = numpy_dataset.stratify_dataset(y)


        stratified_iterators = [tm.pytorch.NormalPytorchActivationsIterator(model, numpy_dataset, p.transformations, o.batch_size,o.num_workers,use_cuda) for numpy_dataset in stratified_numpy_datasets]
        measure_result = p.measure.eval_stratified(stratified_iterators,dataset.labels)

    del model
    del dataset
    torch.cuda.empty_cache()

    return MeasureExperimentResult(p, measure_result)


def main(p:Parameters,o:Options,model_path:Path)->MeasureExperimentResult:
    profiler= Profiler()
    profiler.event("start")
    if o.verbose:
        print(f"Experimenting with parameters: {p}")
    measures_results=experiment(p,o,model_path)
    profiler.event("end")
    print(profiler.summary(human=True))
    config.save_experiment_results(measures_results)
    return measures_results



from tmeasures.pytorch.model import FilteredActivationsModule

def experiment_pytorch(p: PyTorchParameters,model_path:Path,verbose=False):
    assert(len(p.transformations)>0)

    dataset = datasets.get_classification(p.dataset.name)
    if verbose:
        print(dataset.summary())
    if verbose:
        print(f"Loading model {model_path}")

    model, training_parameters, scores = train.load_model(model_path, p.options.model_device)
    
    model = FilteredActivationsModule(model,p.model_filter)

    if training_parameters.dataset_name != p.dataset.name:
        if p.adapt_dataset:
            if verbose:
                print(f"Adapting dataset {p.dataset.name} to model trained on dataset {training_parameters.dataset_name} (resizing spatial dims and channels)")

            adapt_dataset(dataset, training_parameters.dataset_name)
            if verbose:
                print(dataset.summary())
        else:
            print(f"Error: model trained on dataset {training_parameters.dataset_name}, but requested to measure on dataset {p.dataset.name}; specify the option '-adapt_dataset True' to adapt the test dataset to the model and test anyway.")

    if verbose:
        print("### ", model)
        print("### Scores obtained:")
        for k,v in scores.items():
            print(f"{k} --→ {v:.3f}")

    
    from pytorch.numpy_dataset import NumpyDataset
    new_size = p.dataset.size.get_size(dataset.size(p.dataset.subset))
    dataset = dataset.reduce_size_stratified_fixed(new_size,p.dataset.subset)
    dataset.normalize_features()

    x,y=dataset.get_subset(p.dataset.subset)
    numpy_dataset = NumpyDataset(x)

    if not p.stratified:
        if verbose:
            print(f"Calculating measure {p.measure} dataset size {len(numpy_dataset)}...")
        measure_result = p.measure.eval(numpy_dataset, p.transformations, model, p.options)
    else:
        if verbose:
            print(f"Calculating stratified version of measure {p.measure}...")
        stratified_numpy_datasets = numpy_dataset.stratify_dataset(y)
        measure_result = p.measure.eval_stratified(stratified_numpy_datasets,dataset.labels)

    del model
    del dataset
    torch.cuda.empty_cache()

    return PyTorchMeasureExperimentResult(p, measure_result)


def main_pytorch(p:PyTorchParameters,model_path:Path,verbose=False)->MeasureExperimentResult:
    profiler= Profiler()
    profiler.event("start")
    
    if verbose:
        print(f"Experimenting with parameters: {p}")
    measures_results=experiment_pytorch(p,model_path,verbose=verbose)
    profiler.event("end")
    print(profiler.summary(human=True))
    # config.save_experiment_results(measures_results)
    return measures_results