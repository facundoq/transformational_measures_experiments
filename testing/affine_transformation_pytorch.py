from pathlib import Path
import models
import datasets
import torch
import numpy as np
from experiment import util
import os
from pytorch.numpy_dataset import NumpyDataset
from experiment import model_loading
import config
import transformation_measure as tm
import matplotlib
from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
matplotlib.use('Agg')


dataset_name="cifar10"

print(f"### Loading dataset {dataset_name} ....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get(dataset_name)
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageDataset(numpy_dataset)
p= util.Profiler()
p.event("start")

transformations=tm.SimpleAffineTransformationGenerator(r=360, s=2, t=3,n_rotations=8)
# transformations=tm.SimpleAffineTransformationGenerator()
# transformations=tm.SimpleAffineTransformationGenerator(r=360)
# transformations=tm.SimpleAffineTransformationGenerator(t=3)


folderpath = config.testing_path() / f"affine_transformation_pytorch/{dataset_name}"
folderpath.mkdir(exist_ok=True,parents=True)
n_t=len(transformations)
print(n_t)
batch_size=5
i=0
x,y= image_dataset.get_batch(range(batch_size))
from testing import utils
for i in range(batch_size):
    print(f"Generating plots for image {i}")
    original = x[i,:]
    #print(original.dtype,original.shape)
    transformed_images = []
    transformed= np.zeros( (n_t,*original.shape) )
    for j,t in enumerate(transformations):
        transformed[j,:]= t.apply_pytorch(original).numpy()

    filepath = folderpath / f"samples_first_{i}.png"
    transformed = transformed.transpose((0,2,3,1))
    utils.plot_image_grid(transformed,samples = n_t,grid_cols=16, show=False,save=filepath)
    if i ==10:
        break

p.event("end")

