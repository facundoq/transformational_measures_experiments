import datasets
from testing.utils import plot_image_grid

from pytorch.numpy_dataset import NumpyDataset
from variance_measure.pytorch_activations_iterator import ImageDataset

dataformat="NCHW"
dataset = datasets.get("mnist",dataformat=dataformat)
print(dataset.summary())

numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
image_dataset=ImageDataset(numpy_dataset,rotation=90,scale=0.5,translation=(-10,0),dataformat=dataformat)
print(image_dataset)

x,y= image_dataset.get_batch(list(range(32)))
print("pytorch_iterators",x.shape,x.dtype)
x = x.permute(0, 2, 3, 1)
plot_image_grid(x,y)


