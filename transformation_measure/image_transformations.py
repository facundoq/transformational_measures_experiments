from  skimage import transform as skimage_transform
import numpy as np
import cv2
import itertools
from typing import List,Tuple,Iterator
from .transformation import Transformation,TransformationSet
import torch.nn.functional as F
import torch

import abc

class AffineTransformation(Transformation):
    def __init__(self,parameters,input_shape):
        self.parameters = parameters
        self.input_shape = input_shape
        self.h, self.w, self.c = input_shape

    def __eq__(self, other):
        return self.parameters == other.parameters and self.input_shape == other.input_shape

    def inverse_parameters(self):
        rotation, translation, scale = self.parameters
        rotation=-rotation
        tx,ty=translation
        translation= (-tx,-ty)
        sx,sy=scale
        scale=(1/sx,1/sy)
        parameters = (rotation,translation,scale)
        return parameters
    def __str__(self):
        r, t, s = self.parameters
        return f"Transformation(r={r},t={t},s={s})"

    @abc.abstractmethod
    def inverse(self):
        pass

    @abc.abstractmethod
    def numpy(self):
        pass

class AffineTransformationPytorch(AffineTransformation):

    def __init__(self,parameters,input_shape,use_cuda=torch.cuda.is_available()):
        super().__init__(parameters,input_shape)
        afcv=AffineTransformationNumpy(self.inverse_parameters(), input_shape)
        self.transformation_matrix=self.generate_pytorch_transformation(afcv.transformation_matrix, input_shape)
        tm=self.transformation_matrix.unsqueeze(0)
        self.grid = F.affine_grid(tm, [1,self.c,self.h,self.w])
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.transformation_matrix=self.transformation_matrix.cuda()
            self.grid=self.grid.cuda()

    def generate_pytorch_transformation(self, transformation_matrix, input_shape):
        h, w, c = input_shape
        transformation_matrix = torch.from_numpy(transformation_matrix)
        transformation_matrix = transformation_matrix[:2, :].float()

        self.normalize_transforms(transformation_matrix, w, h)
        return transformation_matrix

    def normalize_transforms(self, transforms, w, h):
        transforms[0, 0] = transforms[0, 0]
        transforms[0, 1] = transforms[0, 1] * h / w
        transforms[0, 2] = transforms[0, 2] * 2 / w + transforms[0, 0] + transforms[0, 1] - 1

        transforms[1, 0] = transforms[1, 0] * w / h
        transforms[1, 1] = transforms[1, 1]
        transforms[1, 2] = transforms[1, 2] * 2 / h + transforms[1, 0] + transforms[1, 1] - 1

    def __call__(self, x: torch.FloatTensor):
        n, c, h, w = x.shape
        grid = self.grid.expand(n,*self.grid.shape[1:])
        x = F.grid_sample(x, grid)
        return x

    def inverse(self):
        return AffineTransformationPytorch(self.inverse_parameters(), self.input_shape,self.use_cuda)

    def numpy(self):
        return AffineTransformationNumpy(self.parameters, self.input_shape)

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

TranslationParameter=Tuple[int,int]
ScaleParameter=Tuple[float,float]


class AffineTransformationGenerator(TransformationSet):
    def __init__(self,rotations:List[float]=None, scales:List[ScaleParameter]=None, translations:List[TranslationParameter]=None,use_cuda:bool=False,pytorch=False,input_shape=None):
        if rotations is None or not rotations:
            rotations=[0]
        if scales is None or not scales:
            scales = [(1.0, 1.0)]
        if translations is None or not translations:
            translations = [(0, 0)]

        self.rotations:List[float]=rotations
        self.scales:List[ScaleParameter]=scales
        self.translations=translations
        self.input_shape=input_shape
        self.pytorch=pytorch
        self.use_cuda=use_cuda

    def set_input_shape(self,input_shape):
        self.input_shape=input_shape

    def __repr__(self):
        return f"rot={self.rotations}, scales={self.scales}, translations={self.translations}"
    def id(self):
        return f"r{self.rotations}_s{self.scales}_t{self.translations}"
    def all_parameter_combinations(self):
        return itertools.product(self.rotations, self.translations, self.scales)

    def __len__(self):
        return len(self.rotations)*len(self.translations)*len(self.scales)

    def __iter__(self)->Iterator[Transformation]:
        assert not self.input_shape is None
        def generator(parameter):
            if self.pytorch:
                return AffineTransformationPytorch(parameter,self.input_shape,self.use_cuda)
            else:
                return AffineTransformationNumpy(parameter, self.input_shape)

        transformation_parameters=self.all_parameter_combinations()
        return [generator(parameter) for parameter in transformation_parameters].__iter__()

#TODO move this class to Config
class SimpleAffineTransformationGenerator(TransformationSet):

    def __init__(self, r:int=None, s:int=None, t:int=None,n_rotations=16,n_translations=None,n_scales=None):
        if r is None:
            r = 0
        if s is None:
            s = 0
        if t is None:
            t = 0
        if n_translations is None:
            n_translations=0
        if n_scales is None:
            n_scales=0
        self.rotation_intensity=r
        self.translation_intensity=t
        self.scale_intensity=s

        self.n_rotations=n_rotations
        self.n_scales = n_scales
        self.n_translations=n_translations

        rotations, translations, scales = self.generate_transformation_values()
        self.affine_transformation_generator=AffineTransformationGenerator(rotations=rotations, scales=scales, translations=translations)

    def set_input_shape(self,input_shape):
        self.affine_transformation_generator.set_input_shape(input_shape)
    def set_cuda(self,use_cuda):
        self.affine_transformation_generator.use_cuda=use_cuda
    def set_pytorch(self,pytorch):
        self.affine_transformation_generator.pytorch=pytorch

    def copy(self):
        a = SimpleAffineTransformationGenerator(r=self.rotation_intensity,s=self.scale_intensity,t=self.translation_intensity,n_rotations=self.n_rotations,n_translations=self.n_translations)
        a.set_pytorch(self.affine_transformation_generator.pytorch)
        a.set_cuda(self.affine_transformation_generator.use_cuda)
        a.set_input_shape(self.affine_transformation_generator.input_shape)
        return a

    def __repr__(self):
        nr= "" if self.n_rotations==16 else f",nr={self.n_rotations}"
        ns = "" if self.n_scales== self.scale_intensity else f",nt={self.n_scales}"
        nt = "" if self.n_translations == self.translation_intensity else f",nt={self.n_translations}"

        return f"Affine(r={self.rotation_intensity},s={self.scale_intensity},t={self.translation_intensity},nr={self.n_rotations}{nr}{ns}{nt})"
    def __eq__(self, other):
        if isinstance(other,self.__class__):
            return self.rotation_intensity == other.rotation_intensity and self.scale_intensity == other.scale_intensity and self.translation_intensity == other.translation_intensity

    def __len__(self):
        return len(self.affine_transformation_generator)

    def id(self):
        return f"Affine(r={self.rotation_intensity},s={self.scale_intensity},t={self.translation_intensity})"

    def __iter__(self)->Iterator[Transformation]:
        return self.affine_transformation_generator.__iter__()

    def generate_transformation_values(self):

        if self.rotation_intensity>0:
            range_rotations= np.pi*(self.rotation_intensity / (360.0 * 2.0))
            n_rotations= max(self.rotation_intensity*self.n_rotations//360,1)
            rotations = list(np.linspace(-range_rotations, range_rotations, n_rotations, endpoint=False))
        else:
            rotations = [0.0]


        scales = [(1.0,1.0)]
        r=1.0
        for i in range(self.n_scales,self.scale_intensity):
            r_upsample = r + i * 0.05
            r_downsample = r - i * 0.10
            scales.append((r_upsample,r_upsample))
            scales.append((r_downsample,r_downsample))

        translations=[(0,0)]
        for t in range(self.n_translations,self.translation_intensity):
            d= 2**(t)
            translations.append( (0,d) )
            translations.append((0, -d))
            translations.append((d, 0))
            translations.append((-d, 0))
            translations.append((-d, d))
            translations.append((-d, -d))
            translations.append((d, -d))
            translations.append((d, d))

        return rotations,translations,scales