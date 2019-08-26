import skimage.io
import transformation_measure as tm
import numpy as np
import matplotlib.pyplot as plt
import os

results_path="testing/affine/"
os.makedirs(results_path,exist_ok=True)
source_path="testing/mnist.png"

def apply_transformation(p, image_name):
    a=tm.AffineTransformationCV(p)

    image= skimage.io.imread(source_path)
    image= image.transpose(2, 0, 1)
    image= a(image)
    image= image.transpose(1 , 2, 0)
    filepath = os.path.join(results_path,f"{image_name}.png")
    skimage.io.imsave( filepath, image)

p=(0, (0, 0) , (1, 1))
apply_transformation(p, "identity")

for r in [0,45,90,135,180,360]:
    p=(r/360*2*3.14, (0, 0), (1, 1))
    apply_transformation(p, f"rotation{r}")

for s in [0.1,0.5,0.8]:
    for s2 in [0.1, 0.5, 0.8]:
        p=(0, (0, 0), (s,s2))
        apply_transformation(p, f"resize={s}-{s2}")

p=(0, (50, 50), (1, 1))
apply_transformation(p, "translation50px")