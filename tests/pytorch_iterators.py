import datasets
import numpy as np

from pytorch.numpy_dataset import NumpyDataset
from variance_measure.pytorch_activations_iterator import ImageDataset

dataformat="NCHW"
dataset = datasets.get("mnist",dataformat=dataformat)
print(dataset.summary())

numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageDataset(numpy_dataset,rotation=90,scale=0.5,translation=(-10,0),dataformat=dataformat)
print(image_dataset)

import matplotlib.pyplot as plt

def plot_image_grid(x,y,samples=64):


    initial_sample=0
    samples=min(samples,len(y))
    skip= y.shape[0] // samples

    grid_cols=8
    grid_rows=samples // grid_cols
    if samples % grid_cols >0:
        grid_rows+=1

    f,axes=plt.subplots(grid_rows,grid_cols)
    for axr in axes:
        for ax in axr:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    for i in range(samples):
        i_sample=i*skip+initial_sample
        klass = y[i_sample]
        row=i // grid_cols
        col=i % grid_cols
        ax=axes[row,col]
        if x.shape[3]==1:
            ax.imshow(x[i_sample,:,:,0], cmap='gray')
        else:
            ax.imshow(x[i_sample, :, :,:])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


for i in range(1):
    x,y= image_dataset.get_batch(list(range(i*32,(i+1)*32)))
    print("pytorch_iterators",x.shape,x.dtype)
    x = x.permute(0, 2, 3, 1)
    x=x.data.numpy()
    y=y.data.numpy()
    #from tests.utils import plot_image_grid

    plot_image_grid(x,y)


