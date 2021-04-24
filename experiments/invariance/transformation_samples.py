from .common import *
import experiment.measure as measure_package

import skimage.io
import transformational_measures as tm
import numpy as np
from pathlib import Path
import config

from testing.util import plot_image_grid
from transformational_measures.transformations.pytorch.affine import AffineTransformation
import torch

def apply_transformation_pytorch(x:np.ndarray, t:AffineTransformation):
    x = torch.from_numpy(x)
    # NHWC to NCHW
    x = x.permute(0,3,1,2)
    x=t(x)
    # NCHW to NHWC
    x = x.permute(0,2,3,1)
    x= x[0, :, :,:]
    x = x.numpy()
    return (x * 255).astype("uint8")

def get_image(path):
    image = skimage.io.imread(path)
    original_shape = image.shape
    if len(original_shape) == 2:
        image = image[:, :, np.newaxis]
    image = (image.astype(np.float32)) / 255
    image = image[np.newaxis,]
    return image

class DatasetTransformationPlots(InvarianceExperiment):
    def description(self):
        return '''Generate matrices of images w/ different transformations of a sample for various datasets. '''

    def run(self):
        samples = {
            "cifar10":Path("testing/samples/cifar1.png"),
            "mnist":Path("testing/samples/mnist.png"),
            "lsa16":Path("testing/samples/lsa16.png"),
            "rwth":Path("testing/samples/rwth.png"),
            "cifar10_da":Path("testing/samples/cifar1.png"),
            "mnist_da":Path("testing/samples/mnist.png"),
        }

        transformations={
            "cifar10":common_transformations,
            "mnist":common_transformations,
            "lsa16":common_transformations_da,
            "rwth":common_transformations_da,
            "cifar10_da":common_transformations_da,
            "mnist_da":common_transformations_da,
        }

        rows,cols=3,3
        for sample_id,sample_path in samples.items():
            image = get_image(sample_path)
            transformations_sets=transformations[sample_id]
            for transformation_set in transformations_sets:
                filepath= self.plot_folderpath / f"{sample_id}_{transformation_set.id()}.jpg"
                n_transformations=rows*cols
                n = len(transformation_set)
                #indices=np.random.permutation(n)[:n_transformations]
                indices = np.linspace(0,n,n_transformations,endpoint=False,dtype=int)
                transformation_set = list(transformation_set)
                transformation_set = [transformation_set[i] for i in indices]
                images=[]
                for t in transformation_set:
                    transformed_image = apply_transformation_pytorch(image,t)
                    images.append(transformed_image)
                images = np.stack(images,axis=0)
                # n=images.shape[0]
                # random_indices=np.random.permutation(n)
                # print(n,random_indices)
                # images = images[random_indices,:]
                plot_image_grid(images,samples=rows*cols,grid_cols=cols,show=False,save=filepath)


class STMatrixSamples(InvarianceExperiment):

    def description(self):
        return '''Generate transformed images for the st matrix diagram. '''


    def run(self):

        samples = {
            "cifar1":Path("testing/samples/cifar1.png"),
            "cifar2":Path("testing/samples/cifar2.png"),
            "mnist":Path("testing/samples/mnist.png"),
        }

        for sample_id,sample_path in samples.items():
            image = get_image(sample_path)
            transformations_sets=common_transformations
            for transformation_set in transformations_sets:
                transformation_results_path = self.plot_folderpath / Path(transformation_set.id())
                transformation_results_path.mkdir(parents=True,exist_ok=True)
                for t in transformation_set:
                    filepath = transformation_results_path  / f"{sample_id}_{str(t)}.png"
                    transformed_image = apply_transformation_pytorch(image,t)
                    if transformed_image.shape[2]==1:
                        transformed_image=transformed_image[:,:,0]
                    skimage.io.imsave(filepath, transformed_image)
