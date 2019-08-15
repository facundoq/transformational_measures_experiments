import datasets
from testing.utils import plot_image_grid

from pytorch.numpy_dataset import NumpyDataset
from transformation_measure.iterators.pytorch_activations_iterator import ImageDataset
import transformation_measure as tm
dataformat="NCHW"
dataset = datasets.get("mnist",dataformat=dataformat)
print(dataset.summary())

numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
transformations=tm.SimpleAffineTransformationGenerator(n_rotations=8,n_scales=2,n_translations=2)
image_dataset=ImageDataset(numpy_dataset,transformations,dataformat=dataformat)

x,y= image_dataset.get_batch(list(range(64)))
x = x.permute(0, 2, 3, 1).numpy()
print("pytorch_iterators",x.shape,x.dtype,x.min(axis=(0,1,2)),x.max(axis=(0,1,2)) )

filepath=f"testing/{dataset.name}_samples.png"
print(f"Saving transformed image batch to {filepath}")
plot_image_grid(x,y,show=False,save=filepath)



