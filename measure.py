#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

## Calculate the variance of each activation in a model.
## NOTE:
## You should run "train.py" before this script to generate and train the model for
## a given dataset/model/transformation combination
#
# import ray
# ray.init()

import datasets
import torch,config
from experiment import training, variance
from utils import profiler
from transformation_measure.iterators.pytorch_image_dataset import ImageDataset


def experiment(p: variance.Parameters, o: variance.Options):
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

            variance.adapt_dataset(dataset,training_parameters.dataset)
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

    if p.dataset.subset == variance.DatasetSubset.test:
        numpy_dataset = NumpyDataset(dataset.x_test, dataset.y_test)
    elif p.dataset.subset == variance.DatasetSubset.train:
        numpy_dataset = NumpyDataset(dataset.x_train, dataset.y_train)
    else:
        raise ValueError(p.dataset.subset)
    image_dataset = ImageDataset(numpy_dataset)
    p.transformations.set_input_shape(dataset.input_shape)
    p.transformations.set_pytorch(True)
    p.transformations.set_cuda(use_cuda)

    if not p.stratified:
        iterator = tm.NormalStrategy(model, image_dataset, p.transformations, batch_size=o.batch_size)
        if o.verbose:
            print(f"Calculating measure {p.measure} dataset size {len(image_dataset)}...")

        measure_result = p.measure.eval(iterator)
    else:
        if o.verbose:
            print(f"Calculating stratified version of measure {p.measure}...")
        stratified_numpy_datasets = NumpyDataset.stratify_dataset(dataset.y_test, dataset.x_test)
        stratified_iterators = [tm.NormalStrategy(model, numpy_dataset, p.transformations, batch_size=o.batch_size, num_workers=o.num_workers) for numpy_dataset in stratified_numpy_datasets]
        measure_result = p.measure.eval_stratified(stratified_iterators,dataset.labels)

    return variance.VarianceExperimentResult(p, measure_result)


def main(p:variance.Parameters,o:variance.Options):
    profiler= profiler.Profiler()
    profiler.event("start")
    if o.verbose:
        print(f"Experimenting with parameters: {p}")
    measures_results=experiment(p,o)
    profiler.event("end")
    print(profiler.summary(human=True))
    config.save_results(measures_results)
if __name__ == "__main__":

    p, o = variance.parse_parameters()
    main(p,o)



