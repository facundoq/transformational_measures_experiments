import skimage.io
import transformation_measure as tm
import numpy as np
import matplotlib.pyplot as plt
import os
import config


results_path = config.testing_folder()/"affine_inverse"
results_path.mkdir(parents=True,exist_ok=True)

source_path="testing/samples/mnist.png"

def apply_transformation_inverse(p, image_name):
    a=tm.AffineTransformationCV(p)
    a = a.inverse()
    image= skimage.io.imread(source_path)
    image = image[:,:,np.newaxis]
    image= image.transpose(2, 0, 1)
    image= a(image)
    image= image.transpose(1 , 2, 0)
    filepath = results_path / f"{image_name}.png"
    skimage.io.imsave( filepath, image)

p=(0, (0, 0) , (1, 1))
apply_transformation_inverse(p, "identity")

for r in [0,45,90,135,180,360]:
    p=(r/360*2*3.14, (0, 0), (1, 1))
    apply_transformation_inverse(p, f"rotation{r}")

for s in [0.1,0.5,0.8]:
    for s2 in [0.1, 0.5, 0.8]:
        p=(0, (0, 0), (s,s2))
        apply_transformation_inverse(p, f"resize={s}-{s2}")

p=(0, (5, 5), (1, 1))
apply_transformation_inverse(p, "translation5px")
