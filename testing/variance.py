from pytorch import models
import datasets
import torch

from pytorch.numpy_dataset import NumpyDataset

dataset_name="mnist"
model_name=models.SimpleConv.__name__

print(f"### Loading dataset {dataset_name} and model {model_name}....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get(dataset_name)


from pytorch.experiment import training
model,rotated_model,scores,config=training.load_models(dataset, model_name, use_cuda)

print(model.name, dataset.name)

from variance_measure.iterators.pytorch_activations_iterator import PytorchActivationsIterator
import numpy as np
from variance_measure import transformations as tf
import matplotlib
matplotlib.use('Agg')

samples=512
dataset.x_test=dataset.x_test[:samples,]
dataset.y_test=dataset.y_test[:samples]
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)

n_rotations=4
rotations = np.linspace(-np.pi, np.pi, n_rotations, endpoint=False)

transformations_parameters={"rotation":rotations,"scale":[(1, 1)],"translation":[(0,10)]}

transformations_parameters_combinations=tf.generate_transformation_parameter_combinations(transformations_parameters)

transformations=tf.generate_transformations(transformations_parameters_combinations,dataset.input_shape[0:2])

iterator = PytorchActivationsIterator(model,numpy_dataset,transformations,batch_size=256 )

from variance_measure.measures import measure

options={"conv_aggregation_function":"sum","var_or_std":"var"}

measure= measure.NormalizedMeasure(iterator, options)

import time

begin = time.time()
variance_result,v_transformation,v_sample = measure.eval()
print(variance_result)
print(f"Time elapsed(normal): {time.time()-begin}")

print("average_per_layer ", variance_result.per_layer_average())
print("weighted_global_average ", variance_result.weighted_global_average())
print("global_average ",variance_result.global_average())

begin = time.time()
stratified_numpy_datasets = NumpyDataset.stratify_dataset(dataset.y_test,dataset.x_test)
stratified_iterators = [PytorchActivationsIterator(model,numpy_dataset,transformations,batch_size=16) for numpy_dataset in stratified_numpy_datasets]

variance_measure = lambda iterator: measure.NormalizedMeasure(iterator, options).eval()
stratified_measure = measure.StratifiedMeasure(stratified_iterators, variance_measure)
stratified_variance_result,class_variance_result = stratified_measure.eval()
print(f"Time elapsed (stratified): {time.time()-begin}")



