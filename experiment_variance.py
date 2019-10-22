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


import numpy as np
import cv2

def expand_channels(dataset:datasets.ClassificationDataset,c:int):
    if dataset.dataformat=="NHWC":
        axis=3
    else:
        axis=1

    dataset.x_train = np.repeat(dataset.x_train,c,axis=axis)
    dataset.x_test = np.repeat(dataset.x_test, c, axis=axis)


def collapse_channels(dataset:datasets.ClassificationDataset):
    if dataset.dataformat=="NHWC":
        axis=3
    else:
        axis=1
    dataset.x_train = dataset.x_train.mean(axis=axis,keepdims=True)
    dataset.x_test  = dataset.x_test.mean(axis=axis,keepdims=True)


def resize(dataset:datasets.ClassificationDataset,h:int,w:int,c:int):

    if dataset.dataformat=="NCHW":
        dataset.x_train=np.transpose(dataset.x_train,axes=(0,2,3,1))
        dataset.x_test = np.transpose(dataset.x_test, axes=(0, 2, 3, 1))

    subsets = [dataset.x_train, dataset.x_test]
    new_subsets=[np.zeros((s.shape[0],h,w,c)) for s in subsets]

    for (subset,new_subset) in zip(subsets,new_subsets):
        for i in range(subset.shape[0]):
            img=subset[i, :]
            if c==1:
                #remove channel axis, resize, put again
                img=img[:,:,0]
                img= cv2.resize(img, dsize=(h, w))
                img = img[:, :, np.newaxis]
            else:
                #resize
                img = cv2.resize(img, dsize=(h, w))

            new_subset[i,:]=img

    dataset.x_train = new_subsets[0]
    dataset.x_test = new_subsets[1]

    if dataset.dataformat=="NCHW":
        dataset.x_train = np.transpose(dataset.x_train,axes=(0,3,1,2))
        dataset.x_test = np.transpose(dataset.x_test, axes=(0, 3, 1, 2))

def adapt_dataset(dataset:datasets.ClassificationDataset, dataset_template:str):
    dataset_template = datasets.get(dataset_template)
    h,w,c= dataset_template.input_shape
    del dataset_template
    oh,ow,oc=dataset.input_shape

    # fix channels
    if c !=oc and oc==1:
        expand_channels(dataset,c)

    elif c != oc and c ==1:
        collapse_channels(dataset)
    else:
        raise ValueError(f"Cannot transform image with {oc} channels into image with {c} channels.")

    #fix size
    if h!=oh or w!=ow:
        resize(dataset,h,w,c)

    dataset.input_shape=(h,w,c)




def experiment(p: variance.Parameters, o: variance.Options):
    assert(len(p.transformations)>1)
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



