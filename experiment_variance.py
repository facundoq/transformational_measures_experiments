## Calculate the variance of each activation in a model.
## NOTE:
## You should run "experiment_rotation.py" before this script to generate and train the models for
## a given dataset/model combination

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

from pytorch import models
import datasets
import torch
import pytorch.experiment.utils as utils
from  experiment import variance
if __name__ == "__main__":
    p=variance.parse_parameters()


    print(f"Experimenting with parameters: {p}")

    verbose=True

    use_cuda=torch.cuda.is_available()


    dataset = datasets.get(p.dataset)
    if verbose:
        print(dataset.summary())

    from pytorch.experiment import rotation
    unrotated_model, rotated_model, scores, config=rotation.load_models(dataset, p.model, use_cuda)

    if verbose:
        print("### ", unrotated_model)
        print("### ", rotated_model)
        print("### Scores obtained:")
        rotation.print_scores(scores)

    import transformation_measure as tm


    import numpy as np
    from pytorch.numpy_dataset import NumpyDataset

    sample_skip=2
    if sample_skip>1:
        dataset.x_test= dataset.x_test[::sample_skip, ]
        dataset.y_test= dataset.y_test[::sample_skip]


    def experiment(model,dataset,transformations,base_measure,options):
        numpy_dataset = NumpyDataset(dataset.x_test, dataset.y_test)
        iterator = tm.PytorchActivationsIterator(model,numpy_dataset,transformations,batch_size=256)
        measure_result=base_measure(options).eval(iterator)

        stratified_numpy_datasets = NumpyDataset.stratify_dataset(dataset.y_test,dataset.x_test)
        stratified_iterators = [tm.PytorchActivationsIterator(model,numpy_dataset,transformations,batch_size=256) for numpy_dataset in stratified_numpy_datasets]

        variance_measure = lambda iterator: base_measure(options).eval(iterator)
        stratified_measure = tm.StratifiedMeasure(stratified_iterators, variance_measure)
        stratified_measure_result,measure_per_class = stratified_measure.eval()

        measures=dict([(m.source,m) for m in [measure_result,stratified_measure_result]])
        for i,m in enumerate(measure_per_class):
            measures[f"{m.source}"] = m
        return measures

    from experiment import variance

    unrotated_measures_results=experiment(unrotated_model, dataset, p)
    rotated_measures_results=experiment(rotated_model, dataset, p)
    result=variance.VarianceExperimentResult(unrotated_model.name, dataset.name, unrotated_model.activation_names(), dataset.labels, p.transformations,  rotated_measures_results, unrotated_measures_results)
    variance.save_results(result)


