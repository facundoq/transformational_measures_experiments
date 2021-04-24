from testing.util import plot_image_grid
import datasets
import config
from pytorch.pytorch_image_dataset import ImageClassificationDataset
from pytorch.numpy_dataset import NumpyDataset
import itertools
folderpath=config.testing_path()/ "numpy_iterator"
folderpath.mkdir(parents=True,exist_ok=True)
images= 32
preprocessing=True
print(f"Using datasets: {datasets.names}")
for dataset_name,preprocessing,normalize in itertools.product(datasets.names,[True,False],[True,False]):
    print(dataset_name,preprocessing,normalize)
    dataset = datasets.get(dataset_name)
    # print(dataset.summary())

    pre_str = 'preprocessing_' if preprocessing else ""
    normalize_str = 'normalized_' if preprocessing else ""
    filepath = folderpath / f"{pre_str}{normalize_str}{dataset_name}.png"

    numpy_dataset=NumpyDataset(dataset.x_test,dataset.y_test)
    if preprocessing:
        numpy_dataset= ImageClassificationDataset(numpy_dataset)


    x,y= numpy_dataset.get_batch(list(range(images)))
    if not preprocessing:
        x = x.float()/255
    # print(x.shape,y.shape)

    # permute to NHWC order
    x = x.permute(0, 2, 3, 1)
    x = x.numpy()

    plot_image_grid(x,show=False,save=filepath,normalize=normalize)

