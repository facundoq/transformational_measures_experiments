from pytorch import models
import datasets
import torch

from pytorch.numpy_dataset import NumpyDataset

dataset_name="mnist"
model_name=models.SimpleConv.__name__

print(f"### Loading dataset {dataset_name} and model {model_name}....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get(dataset_name)


from pytorch.experiment import rotation
model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)


from variance_measure.pytorch_activations_iterator import PytorchActivationsIterator
import numpy as np
from tests.utils import plot_image_grid
from variance_measure import transformations as tf
import matplotlib
matplotlib.use('TkAgg')

numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)

n_rotations=16
rotations = np.linspace(-np.pi, np.pi, n_rotations, endpoint=False)
transformations_parameters={"rotation":rotations,"scale":[(1, 0.5)],"translation":[(0,2)]}
transformations_parameters_combinations=tf.generate_transformation_parameter_combinations(transformations_parameters)
transformations=tf.generate_transformations(transformations_parameters_combinations,dataset.input_shape[0:2])

iterator = PytorchActivationsIterator(model,numpy_dataset,transformations,config)

batch_size=64
i=0

for activations,x_transformed in iterator.samples_first():

    x=x_transformed.transpose( (0,2,3,1))
    plot_image_grid(x, torch.zeros((x.shape[0])),show=False,save=f"t{i}.png")
    i = i + 1
    if i ==10:
        break


