from pytorch import models
import datasets
import torch

from pytorch.numpy_dataset import NumpyDataset

dataset_name="mnist"
model_name=models.SimpleConv.__name__

print(f"### Loading dataset {dataset_name} and model {model_name}....")

use_cuda=torch.cuda.is_available()
dataset = datasets.get(dataset_name)


from pytorch.experiment import model_loading
model, optimizer = model_loading.get_model(model_name, dataset, use_cuda)
from testing.utils import plot_image_grid
import transformation_measure as tm
import matplotlib
matplotlib.use('Agg')

numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)

transformations=tm.SimpleAffineTransformationGenerator(n_translations=2)

iterator = tm.PytorchActivationsIterator(model,numpy_dataset,transformations)

batch_size=64
i=0
for transformation,batch_activations in iterator.transformations_first():
    print(transformation)
    for x,batch_activation in batch_activations:
        x=x.transpose(0,2,3,1)
        plot_image_grid(x, torch.zeros((x.shape[0])),show=False,save=f"t{i}.png")
        i=i+1
        break



