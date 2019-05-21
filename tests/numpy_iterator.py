from tests.utils import plot_image_grid
import datasets


dataset = datasets.get("mnist")
print(dataset.summary())


from pytorch.numpy_dataset import NumpyDataset
numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)


x,y= numpy_dataset.get_batch(list(range(32)))
print(x.shape,y.shape)

# permute to NHWC order
x = x.permute(0, 2, 3, 1)
plot_image_grid(x,y)