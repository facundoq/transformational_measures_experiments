from  skimage import transform as skimage_transform
import numpy as np
import itertools
from typing import List,Tuple,Iterator
from .transformation import Transformation,TransformationSet

class AffineTransformation(Transformation):
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

    def __call__(self,image:np.ndarray)->np.ndarray:
        h,w,c=image.shape
        image_size=np.array([h, w])
        centered_transformation=self.center_transformation(self.transform,image_size)
        return skimage_transform.warp(image, centered_transformation.inverse,cval=0.0,preserve_range=True,order=1)

    def __str__(self):
        return f"Transformation {self.parameters}"

TranslationParameter=Tuple[int,int]
ScaleParameter=Tuple[float,float]


class AffineTransformationGenerator(TransformationSet):
    def __init__(self, rotations:List[float]=None, scales:List[ScaleParameter]=None, translations:List[TranslationParameter]=None):
        if rotations is None or not rotations:
            rotations=[0]
        if scales is None or not scales:
            scales = [(1.0, 1.0)]
        if translations is None or not translations:
            translations = [(1, 1)]

        self.rotations:List[float]=rotations
        self.scales:List[ScaleParameter]=scales
        self.translations=translations

    def __repr__(self):
        return f"rot={self.rotations}, scales={self.scales}, translations={self.translations}"
    def id(self):
        return f"r{self.rotations}_s{self.scales}_t{self.translations}"

    def __iter__(self)->Iterator[Transformation]:
        transformation_parameters = itertools.product(self.rotations, self.translations, self.scales)
        return [AffineTransformation(parameter) for parameter in transformation_parameters].__iter__()


class SimpleAffineTransformationGenerator(TransformationSet):

    @classmethod
    def common_transformations(cls):
        transformations = [SimpleAffineTransformationGenerator(),
                           SimpleAffineTransformationGenerator(n_rotations=16)]
        transformations = {t.id(): t for t in transformations}
        return transformations

    def __init__(self,n_rotations:int=None,n_scales:int=None,n_translations:int=None):
        if n_rotations is None:
            n_rotations = 0
        if n_scales is None:
            n_scales = 1
        if n_translations is None:
            n_translations = 0
        self.n_rotations=n_rotations
        self.n_translations=n_translations
        self.n_scales=n_scales
        rotations, translations, scales = self.generate_transformation_values()

        self.affine_transformation_generator=AffineTransformationGenerator(rotations=rotations, scales=scales, translations=translations)

    def __repr__(self):
        return f"rot={self.n_rotations}, scales={self.n_scales}, translations={self.n_translations}"

    def id(self):
        return f"r{self.n_rotations}_s{self.n_scales}_t{self.n_translations}"

    def __iter__(self)->Iterator[Transformation]:
        return self.affine_transformation_generator.__iter__()

    def generate_transformation_values(self):
        rotations = list(np.linspace(-np.pi, np.pi, self.n_rotations, endpoint=False))

        scales=[]
        for s in range(self.n_scales):
            r=float(s+1)/self.n_scales
            scales.append( (r,r) )

        translations=[(0,0)]
        for t in range(self.n_translations):
            d=t+1
            translations.append( (0,d) )
            translations.append((0, -d))
            translations.append((d, 0))
            translations.append((-d, 0))
            translations.append((-d, d))
            translations.append((-d, -d))
            translations.append((d, -d))
            translations.append((d, d))

        return rotations,translations,scales