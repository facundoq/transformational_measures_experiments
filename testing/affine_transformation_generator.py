import skimage.io
import transformation_measure as tm
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
results_path="testing/affine_generator/"
os.makedirs(results_path,exist_ok=True)
source_path="testing/mnist.png"
source_path="testing/godzilla.png"

def apply_transformation(t:tm.Transformation, image_name):
    image= skimage.io.imread(source_path)
    image = image.transpose(2,0,1)
    if len(image.shape)==2:
        image=image[:,:,np.newaxis]
    image = (image.astype(float)) / 255
    image -= image.min(axis=(0,1))
    image /= image.max(axis=(0,1))

    image= t(image)
    image = image.transpose(1,2,0)
    filepath = os.path.join(results_path,f"{image_name}.png")
    skimage.io.imsave(filepath, (image*255).astype("uint8"))


transformations=tm.SimpleAffineTransformationGenerator(n_scales=2)

for t in transformations:
    apply_transformation(t,str(t))
