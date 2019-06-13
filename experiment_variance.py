#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

## Calculate the variance of each activation in a model.
## NOTE:
## You should run "experiment_training.py" before this script to generate and train the models for
## a given dataset/model/transformation combination


import datasets
import torch
from pytorch import variance


def experiment(p: variance.Parameters):
    verbose = True

    use_cuda = torch.cuda.is_available()

    dataset = datasets.get(p.dataset.name)
    if verbose:
        print(dataset.summary())

    from pytorch.experiment import training
    model, training_parameters, training_options, scores = training.load_model(p.model, use_cuda)

    if verbose:
        print("### ", model)
        print("### Scores obtained:")
        training.print_scores(scores)

    import transformation_measure as tm

    from pytorch.numpy_dataset import NumpyDataset

    dataset = dataset.reduce_size_stratified(p.dataset.percentage)

    if p.dataset.subset == variance.DatasetSubset.test:
        numpy_dataset = NumpyDataset(dataset.x_test, dataset.y_test)
    elif p.dataset.subset == variance.DatasetSubset.train:
        numpy_dataset = NumpyDataset(dataset.x_train, dataset.y_train)
    else:
        raise ValueError(p.dataset.subset)

    iterator = tm.PytorchActivationsIterator(model, numpy_dataset, p.transformations, batch_size=256)
    print(f"Calculating measure {p.measure}...")

    measure_result = p.measure.eval(iterator,model.activation_names())

    print(f"Calculating stratified version of measure {p.measure}...")
    stratified_numpy_datasets = NumpyDataset.stratify_dataset(dataset.y_test, dataset.x_test)
    stratified_iterators = [tm.PytorchActivationsIterator(model, numpy_dataset, p.transformations, batch_size=256) for numpy_dataset in stratified_numpy_datasets]
    stratified_measure_result = p.measure.eval_stratified(stratified_iterators,model.activation_names(),dataset.labels)

    return variance.VarianceExperimentResult(p, measure_result, stratified_measure_result)

if __name__ == "__main__":
    p, o = variance.parse_parameters()
    print(f"Experimenting with parameters: {p}")
    measures_results=experiment(p)
    variance.save_results(measures_results)


