import skimage.io
import transformational_measures as tm
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import config

from experiments.common import common_transformations_combined,common_transformations_da
from testing.util import plot_image_grid

def apply_transformation(x:np.ndarray, t:tm.AffineTransformationNumpy):
    x= t(x)
    # if x.shape[3]==1:
    #     x= x[:, :, :, 0]
    x= x[0, :, :,:]
    return (x * 255).astype("uint8")

def get_image(path):
    image = skimage.io.imread(path)
    original_shape = image.shape
    if len(original_shape) == 2:
        image = image[:, :, np.newaxis]
    image = (image.astype(np.float32)) / 255
    image = image[np.newaxis,]
    return image

results_path= config.testing_path() / "dataset_transformation_plots/"
results_path.mkdir(parents=True,exist_ok=True)


samples = {
    "cifar10":Path("testing/samples/cifar1.png"),
    "mnist":Path("testing/samples/mnist.png"),
    "lsa16":Path("testing/samples/lsa16.png"),
    "rwth":Path("testing/samples/rwth.png"),
    "cifar10_da":Path("testing/samples/cifar1.png"),
    "mnist_da":Path("testing/samples/mnist.png"),
}

transformations={
    "cifar10":common_transformations_combined,
    "mnist":common_transformations_combined,
    "lsa16":common_transformations_da,
    "rwth":common_transformations_da,
    "cifar10_da":common_transformations_da,
    "mnist_da":common_transformations_da,
}



rows,cols=3,3
for sample_id,sample_path in samples.items():
    image = get_image(sample_path)
    transformations_sets=transformations[sample_id]
    print(sample_id)
    for transformation_set in transformations_sets:
        _,h,w,c=image.shape
        transformation_set.set_input_shape((h,w,c))
        transformation_set.set_pytorch(False)
        filepath=results_path / f"{sample_id}_{transformation_set.id()}.jpg"
        n_transformations=rows*cols
        n = len(transformation_set)
        #indices=np.random.permutation(n)[:n_transformations]
        indices = np.linspace(0,n,n_transformations,endpoint=False,dtype=int)
        transformation_set = list(transformation_set)
        transformation_set = [transformation_set[i] for i in indices]
        images=[]
        for t in transformation_set:
            transformed_image = apply_transformation(image,t)
            images.append(transformed_image)
        images = np.stack(images,axis=0)
        # n=images.shape[0]
        # random_indices=np.random.permutation(n)
        # print(n,random_indices)
        # images = images[random_indices,:]
        plot_image_grid(images,samples=rows*cols,grid_cols=cols,show=False,save=filepath)

