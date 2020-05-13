import cv2
import numpy as np
from skimage import transform as skimage_transform

from transformations.affine import AffineTransformation


class AffineTransformationNumpy(AffineTransformation):
    def __init__(self,parameters,input_shape):
        super().__init__(parameters,input_shape)
        self.transformation_matrix=self.generate_transformation_matrix(parameters, input_shape)

    def generate_transformation_matrix(self, transformation_parameters, input_shape):
        rotation, translation, scale = transformation_parameters
        transformation = skimage_transform.AffineTransform(scale=scale, rotation=rotation, shear=None,
                                                           translation=translation)
        transformation= self.center_transformation(transformation,input_shape[:2])
        return transformation.params

    def center_transformation(self,transformation:skimage_transform.AffineTransform,image_size):
        h,w=image_size
        shift_y, shift_x = (h- 1) / 2., (w- 1) / 2.
        shift = skimage_transform.AffineTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage_transform.AffineTransform(translation=[shift_x, shift_y])
        return shift + (transformation+ shift_inv)

    def single(self,image:np.ndarray)->np.ndarray:
        image_size=tuple(self.input_shape[:2])
        if self.input_shape[2] == 1:
            image = image[:, :, 0]
        image= cv2.warpPerspective(image, self.transformation_matrix, image_size)
        if self.input_shape[2]==1:
           image= image[:, :, np.newaxis]
        return image

    def __call__(self, batch:np.ndarray)->np.ndarray:
        results=[]
        for i in range(batch.shape[0]):
            x= batch[i, :]
            x=self.single(x)
            results.append(x)
        return np.stack(results, axis=0)

    def inverse(self):
        return AffineTransformationNumpy(self.inverse_parameters(), self.input_shape)


    def numpy(self):
        return self