from  skimage import transform as skimage_transform
import numpy as np
import itertools

def generate_transformation_parameter_combinations(transformation_parameters):
    rotation = transformation_parameters["rotation"]
    scale = transformation_parameters["scale"]
    translation = transformation_parameters["translation"]
    return itertools.product(rotation, translation, scale)

class AffineTransformation:
    def __init__(self,parameters,image_size):
        self.parameters=parameters
        self.image_size=image_size
        self.transform=self.generate_transformation(parameters,image_size)

    def generate_transformation(self,transformation_parameters, image_size):
        rotation, translation, scale = transformation_parameters
        transformation = skimage_transform.AffineTransform(scale=scale, rotation=rotation, shear=None,
                                                           translation=translation)
        shift_y, shift_x = (np.array(image_size) - 1) / 2.
        shift = skimage_transform.AffineTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage_transform.AffineTransform(translation=[shift_x, shift_y])

        adjusted_transformation = shift + (transformation + shift_inv)
        return adjusted_transformation

    def __call__(self,image):
        return skimage_transform.warp(image, self.transform.inverse,cval=0.0,preserve_range=True)
    def __str__(self):
        return f"Transformation {self.parameters}"

def generate_transformations(transformation_parameters,image_size):
    return [AffineTransformation(parameter, image_size) for parameter in transformation_parameters]


