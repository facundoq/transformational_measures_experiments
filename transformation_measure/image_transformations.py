from  skimage import transform as skimage_transform
import numpy as np
import itertools

def generate_transformation_parameter_combinations(transformation_parameters):
    rotation = transformation_parameters["rotation"]
    scale = transformation_parameters["scale"]
    translation = transformation_parameters["translation"]
    return itertools.product(rotation, translation, scale)

class AffineTransformation:
    def __init__(self,parameters):
        self.parameters=parameters
        self.transform=self.generate_transformation(parameters)

    def generate_transformation(self,transformation_parameters):
        rotation, translation, scale = transformation_parameters
        transformation = skimage_transform.AffineTransform(scale=scale, rotation=rotation, shear=None,
                                                           translation=translation)

        return transformation

    def center_transformation(self,transformation,image_size):
        shift_y, shift_x = (image_size - 1) / 2.
        shift = skimage_transform.AffineTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage_transform.AffineTransform(translation=[shift_x, shift_y])
        return shift + (transformation+ shift_inv)
    def __call__(self,image):
        h,w,c=image.shape
        image_size=np.array([h, w])
        centered_transformation=self.center_transformation(self.transform,image_size)
        return skimage_transform.warp(image, centered_transformation.inverse,cval=0.0,preserve_range=True)

    def __str__(self):
        return f"Transformation {self.parameters}"

def generate_transformations(transformation_parameters,image_size):
    return [AffineTransformation(parameter, image_size) for parameter in transformation_parameters]


