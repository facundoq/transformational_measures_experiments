import models
import datasets
import torch
import util
from testing import utils
from pytorch.numpy_dataset import NumpyDataset
from experiment import model_loading
import os
import transformation_measure as tm
import matplotlib
from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
matplotlib.use('Agg')


dataset_name="cifar10"
model_name= models.ResNet.__name__

print(f"### Loading dataset {dataset_name} and model {model_name}....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get(dataset_name)
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageDataset(numpy_dataset)
model, optimizer = model_loading.get_model(model_name, dataset, use_cuda)
p=util.Profiler()
p.event("start")

transformations=tm.SimpleAffineTransformationGenerator(n_rotations=8,n_scales=2,n_translations=2)

iterator = tm.PytorchActivationsIterator(model,image_dataset,transformations,batch_size=64,num_workers=8)

p.event("start")
i=0
for transformation,batch_activations in iterator.transformations_first():
    print(transformation)
    for x,batch_activation in batch_activations:
        x=x.transpose(0,2,3,1)
        filepath = os.path.expanduser(f"~/variance/test/transformation_first{i}.png")
        utils.plot_image_grid(x, torch.zeros((x.shape[0])),show=False,save=filepath)
        i=i+1
        break
p.event("end")
#print(p.summary())
