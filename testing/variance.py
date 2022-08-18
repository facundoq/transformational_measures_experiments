import models
import datasets
import torch

from pytorch.numpy_dataset import NumpyDataset

dataset_name="mnist"
model_name= models.SimpleConv.__name__

print(f"### Loading dataset {dataset_name} and model {model_name}....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get_classification(dataset_name)

from experiment import model_loading

model,rotated_model,scores,config= model_loading.get_model(model_name, dataset,use_cuda)

print(model.name, dataset.name)

import tmeasures as tm
import numpy as np
import matplotlib
matplotlib.use('Agg')

samples=512
dataset.x_test=dataset.x_test[:samples,]
dataset.y_test=dataset.y_test[:samples]
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)

n_rotations=4
rotations = np.linspace(-np.pi, np.pi, n_rotations, endpoint=False)

iterator = tm.NormalStrategy(model, numpy_dataset, transformations, batch_size=256)




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



