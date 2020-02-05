import datasets
from testing.util import plot_image_grid

from pytorch.numpy_dataset import NumpyDataset
from transformation_measure.iterators.pytorch.activations_iterator import ImageDataset
import transformation_measure as tm
dataformat="NCHW"
dataset = datasets.get("cifar10",dataformat=dataformat)
print(dataset.summary())

numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
transformations=tm.SimpleAffineTransformationGenerator(r=360, s=5, t=3)
image_dataset=ImageDataset(numpy_dataset,transformations,dataformat=dataformat)

x,y= image_dataset.get_batch(list(range(128 )))
x = x.permute(0, 2, 3, 1).numpy()
print("pytorch_iterators",x.shape,x.dtype,x.min(axis=(0,1,2)),x.max(axis=(0,1,2)) )

filepath=f"testing/{dataset.name}_samples.png"
print(f"Saving transformed image batch to {filepath}")
plot_image_grid(x,y,show=False,save=filepath)



