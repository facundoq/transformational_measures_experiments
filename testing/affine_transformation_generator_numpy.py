import skimage.io
import transformational_measures as tm
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import config


def apply_transformation(x:np.ndarray, t:tm.AffineTransformationNumpy, filepath:Path):
    x= t(x)
    if x.shape[3]==1:
        x= x[:, :, :, 0]
    x= x[0, :, :]
    skimage.io.imsave(str(filepath), (x * 255).astype("uint8"))

def get_image(path):
    image = skimage.io.imread(path)
    original_shape = image.shape
    if len(original_shape) == 2:
        image = image[:, :, np.newaxis]
    image = (image.astype(np.float32)) / 255
    image = image[np.newaxis,]
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
    transformations_sets=config.common_transformations()
    for transformation_set in transformations_sets:
        n,h,w,c=image.shape
        transformation_set.set_input_shape((h,w,c))
        transformation_set.set_pytorch(False)

        transformation_results_path = results_path / Path(transformation_set.id())
        transformation_results_path.mkdir(parents=True,exist_ok=True)
        for t in transformation_set:
            filepath = transformation_results_path  / f"{sample_id}_{str(t)}.png"
            apply_transformation(image,t,filepath)


