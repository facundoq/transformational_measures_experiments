## Calculate the variance of each activation in a model.
## NOTE:
## You should run "experiment_rotation.py" before this script to generate and train the models for
## a given dataset/model combination

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging

from pytorch import models
from pytorch import classification_dataset as datasets
import torch
import pytorch.experiment.utils as utils

if __name__ == "__main__":
    model_name,dataset_name,transformation_names=utils.parse_model_and_dataset("Experiment: accuracy of model for rotated vs unrotated dataset.")
else:
    dataset_name="cifar10"
    model_name=models.AllConvolutional.__name__


print(f"### Loading dataset {dataset_name} and model {model_name}....")
verbose=False

use_cuda=torch.cuda.is_available()

dataset = datasets.get(dataset_name)
if verbose:
    print(dataset.summary())

from pytorch.experiment import rotation
model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)

if verbose:
    print("### ", model)
    print("### ", rotated_model)
    print("### Scores obtained:")
    rotation.print_scores(scores)

n_rotations=16

from variance_measure.pytorch_activations_iterator import PytorchActivationsIterator
import numpy as np


rotations = np.linspace(-180, 180, n_rotations, endpoint=False)
transformations={"rotation":rotations}
iterator = PytorchActivationsIterator(model,dataset,transformations,config)

batch_size=64
for transformation,samples in iterator.transformations_first(batch_size):
    print(transformation)



