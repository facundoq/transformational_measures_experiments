import models
import datasets
import torch
from experiment import util
import os
from pytorch.numpy_dataset import NumpyDataset
from run import model_loading

import transformation_measure as tm
import matplotlib
from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
matplotlib.use('Agg')


dataset_name="mnist"
model_name= models.ResNet.__name__

print(f"### Loading dataset {dataset_name} and model {model_name}....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get(dataset_name)
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageDataset(numpy_dataset)
model, optimizer = model_loading.get_model(model_name, dataset, use_cuda)
p= util.Profiler()
p.event("start")

transformations=tm.SimpleAffineTransformationGenerator(n_rotations=8,n_scales=2,n_translations=2)

iterator = tm.PytorchActivationsIterator(model,image_dataset,transformations,batch_size=64,num_workers=0)

batch_size=64
i=0
from testing import utils
for activations,x_transformed in iterator.samples_first():

    x=x_transformed.transpose( (0,2,3,1))
    filepath=os.path.expanduser(f"~/variance/test/samples_first_{i}.png")
    utils.plot_image_grid(x, torch.zeros((x.shape[0])),show=False,save=filepath)
    i = i + 1
    if i ==10:
        break

p.event("end")

