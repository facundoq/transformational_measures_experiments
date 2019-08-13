import skimage.io
import transformation_measure as tm
import numpy as np
import matplotlib.pyplot as plt




def apply_transformation(p, image_name):
    a=tm.AffineTransformationCV(p)

    image= skimage.io.imread("testing/prueba.png")
    image= image.transpose(2, 0, 1)
    image= a(image)
    image= image.transpose(1 , 2, 0)
    skimage.io.imsave(f"testing/result{image_name}.png", image)

p=(0, (0, 0) , (1, 1))
apply_transformation(p, "identity")

p=(45, (0, 0), (1, 1))
apply_transformation(p, "Rotation45")

p=(0, (0, 0), (0.5, 0.5))
apply_transformation(p, "resize 0.5")

p=(0, (50, 50), (1, 1))
apply_transformation(p, "translation 50px")
