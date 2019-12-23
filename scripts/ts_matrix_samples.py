import skimage.io
import transformation_measure as tm
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import config


def apply_transformation(image:np.ndarray,t:tm.Transformation, filepath:Path):
    image= t(image)
    image = image.transpose((1,2,0))
    if image.shape[2]==1:
        image=image[:,:,0]
    skimage.io.imsave(filepath, (image*255).astype("uint8"))

def get_image(path):
    image = skimage.io.imread(path)
    original_shape = image.shape
    if len(original_shape) == 2:
        image = image[:, :, np.newaxis]
    image = (image.astype(float)) / 255
    # image -= image.min(axis=(0,1))
    # image /= image.max(axis=(0,1))
    image = image.transpose(2, 0, 1)
    return image

results_path= config.testing_path() / "affine_generator/"
results_path.mkdir(parents=True,exist_ok=True)

samples = {
    "cifar1":Path("testing/samples/cifar1.png"),
    "cifar2":Path("testing/samples/cifar2.png"),
    "mnist":Path("testing/samples/mnist.png"),
}


for sample_id,sample_path in samples.items():
    image = get_image(sample_path)
    transformations_sets=config.all_transformations()
    for transformation_set in transformations_sets:
        print(transformation_set)
        transformation_results_path = results_path / Path(transformation_set.id())
        transformation_results_path.mkdir(parents=True,exist_ok=True)
        for t in transformation_set:
            filepath = transformation_results_path  / f"{sample_id}_{str(t)}.png"
            apply_transformation(image,t,filepath)


