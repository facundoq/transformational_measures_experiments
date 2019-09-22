#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

## Calculate the variance of each activation in a model.
## NOTE:
## You should run "experiment_training.py" before this script to generate and train the model for
## a given dataset/model/transformation combination


import datasets
import torch,config
from experiment import variance, training
import util

def experiment(p: variance.Parameters, o: variance.Options):
    assert(len(p.transformations)>1)
    use_cuda = torch.cuda.is_available()

    dataset = datasets.get(p.dataset.name)
    if o.verbose:
        print(dataset.summary())

    model, training_parameters, training_options, scores = training.load_model(p.model_path, use_cuda)

    if o.verbose:
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


    if not p.stratified:
        iterator = tm.PytorchActivationsIterator(model, numpy_dataset, p.transformations, batch_size=o.batch_size)
        print(f"Calculating measure {p.measure} dataset size {len(numpy_dataset)}...")

        measure_result = p.measure.eval(iterator)
    else:
        print(f"Calculating stratified version of measure {p.measure}...")
        stratified_numpy_datasets = NumpyDataset.stratify_dataset(dataset.y_test, dataset.x_test)
        stratified_iterators = [tm.PytorchActivationsIterator(model, numpy_dataset, p.transformations, batch_size=o.batch_size,num_workers=o.num_workers) for numpy_dataset in stratified_numpy_datasets]
        measure_result = p.measure.eval_stratified(stratified_iterators,dataset.labels)

    return variance.VarianceExperimentResult(p, measure_result)

if __name__ == "__main__":
    profiler=util.Profiler()
    profiler.event("start")
    p, o = variance.parse_parameters()
    print(f"Experimenting with parameters: {p}")
    measures_results=experiment(p,o)
    profiler.event("end")
    print(profiler.summary(human=True))
    config.save_results(measures_results)



