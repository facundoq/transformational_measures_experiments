from  skimage import transform as skimage_transform
import numpy as np
import cv2
import itertools
from typing import List,Tuple,Iterator
from .transformation import Transformation,TransformationSet
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn

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



class AffineTransformationCV(Transformation):
    def __init__(self,parameters):
        self.parameters=parameters
        self.transform=self.generate_transformation(parameters)

    def __eq__(self, other):

        return self.parameters == other.parameters

    def generate_transformation(self,transformation_parameters):
        rotation, translation, scale = transformation_parameters
        transformation = skimage_transform.AffineTransform(scale=scale, rotation=rotation, shear=None,
                                                           translation=translation)
        return transformation

    def center_transformation(self,transformation:skimage_transform.AffineTransform,image_size):
        shift_y, shift_x = (image_size - 1) / 2.
        shift = skimage_transform.AffineTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage_transform.AffineTransform(translation=[shift_x, shift_y])
        return shift + (transformation+ shift_inv)

    def __call__(self,image:np.ndarray)->np.ndarray:
        # print(image.min(), image.max(), image.dtype, image.shape)
        image=image.transpose((1,2,0))
        h, w, c= image.shape
        # print(image.min(),image.max(),image.dtype,image.shape)
        transformation= self.center_transformation(self.transform, np.array((h, w)))
        m= transformation.params

        image= cv2.warpPerspective(image, m, (w, h))
        if c==1:
           image= image[:, :, np.newaxis]
        image = image.transpose(2,0,1)
        return image

    def inverse(self):
        rotation, translation, scale = self.parameters
        rotation=-rotation
        tx,ty=translation
        translation= (-tx,-ty)
        sx,sy=scale
        scale=(1/sx,1/sy)
        parameters = (rotation,translation,scale)
        return AffineTransformationCV(parameters)

    def normalize_transforms(self,transforms, W, H):
        transforms[0, 0] = transforms[0, 0]
        transforms[0, 1] = transforms[0, 1] * H / W
        transforms[0, 2] = transforms[0, 2] * 2 / W + transforms[0, 0] + transforms[0, 1] - 1

        transforms[1, 0] = transforms[1, 0] * W / H
        transforms[1, 1] = transforms[1, 1]
        transforms[1, 2] = transforms[1, 2] * 2 / H + transforms[1, 0] + transforms[1, 1] - 1

    def apply_pytorch(self,x:torch.FloatTensor,use_cuda=torch.cuda.is_available()):
        c,h,w=x.shape
        transformation = self.center_transformation(self.transform, np.array((h, w)))

        transformation_matrix = torch.from_numpy(transformation.params)
        transformation_matrix = transformation_matrix[:2,:].float()
        self.normalize_transforms(transformation_matrix,w,h)
        transformation_matrix=transformation_matrix.unsqueeze(0)
        x= x.unsqueeze(0)
        if use_cuda:
            transformation_matrix.cuda()
        grid = F.affine_grid(transformation_matrix , list(x.shape))
        x = F.grid_sample(x, grid)
        return x

    def apply_pytorch_batch(self,x:torch.FloatTensor,use_cuda=torch.cuda.is_available()):
        n,c,h,w=x.shape
        transformation = self.center_transformation(self.transform, np.array((h, w)))

        transformation_matrix = torch.from_numpy(transformation.params)
        transformation_matrix = transformation_matrix[:2,:].float()
        self.normalize_transforms(transformation_matrix,w,h)
        transformation_matrix=transformation_matrix.expand(n,*transformation_matrix.shape)
        #x= x.unsqueeze(0)
        if use_cuda:
            transformation_matrix=transformation_matrix.cuda()
        grid = F.affine_grid(transformation_matrix , list(x.shape)).cuda()
        x = F.grid_sample(x, grid)
        return x

    def __str__(self):
        r, t, s = self.parameters
        return f"Transformation(r={r},t={t},s={s})"

TranslationParameter=Tuple[int,int]
ScaleParameter=Tuple[float,float]


class AffineTransformationGenerator(TransformationSet):
    def __init__(self,rotations:List[float]=None, scales:List[ScaleParameter]=None, translations:List[TranslationParameter]=None):
        if rotations is None or not rotations:
            rotations=[0]
        if scales is None or not scales:
            scales = [(1.0, 1.0)]
        if translations is None or not translations:
            translations = [(0, 0)]

        self.rotations:List[float]=rotations
        self.scales:List[ScaleParameter]=scales
        self.translations=translations

    def __repr__(self):
        return f"rot={self.rotations}, scales={self.scales}, translations={self.translations}"
    def id(self):
        return f"r{self.rotations}_s{self.scales}_t{self.translations}"

    def __iter__(self)->Iterator[Transformation]:
        transformation_parameters = itertools.product(self.rotations, self.translations, self.scales)
        return [AffineTransformationCV(parameter) for parameter in transformation_parameters].__iter__()


class SimpleAffineTransformationGenerator(TransformationSet):

    def __init__(self, r:int=None, s:int=None, t:int=None,n_rotations=16):
        if r is None:
            r = 0
        if s is None:
            s = 0
        if t is None:
            t = 0
        self.rotation_intensity=r
        self.translation_intensity=t
        self.scale_intensity=s
        self.n_rotations=n_rotations
        rotations, translations, scales = self.generate_transformation_values()
        self.affine_transformation_generator=AffineTransformationGenerator(rotations=rotations, scales=scales, translations=translations)

    def __repr__(self):
        return f"Affine(r={self.rotation_intensity},s={self.scale_intensity},t={self.translation_intensity})"
    def __eq__(self, other):
        if isinstance(other,self.__class__):
            return self.rotation_intensity == other.rotation_intensity and self.scale_intensity == other.scale_intensity and self.translation_intensity == other.translation_intensity

    def id(self):
        return f"Affine(r={self.rotation_intensity},s={self.scale_intensity},t={self.translation_intensity})"

    def __iter__(self)->Iterator[Transformation]:
        return self.affine_transformation_generator.__iter__()


    def infinite_binary_progression(self,low=0,high=1):
        yield high
        values=[ (low,high)]
        while True:
            new_values=[]
            for (l,u) in values:
                mid=(l+u)/2
                yield mid
                new_values.append((l,mid))
                new_values.append((mid, l))
            values=new_values
    def infinite_harmonic_series(self):
        value = 1.0
        n=1.0
        while True:
            yield value/n
            n+=1
    def infinite_geometric_series(self,base):
        n=1
        while True:
            yield pow(base,n)

    def generate_transformation_values(self):

        if self.rotation_intensity>0:
            range_rotations= np.pi*(self.rotation_intensity / (360.0 * 2.0))
            n_rotations= max(self.rotation_intensity*self.n_rotations//360,1)
            rotations = list(np.linspace(-range_rotations, range_rotations, n_rotations, endpoint=False))
        else:
            rotations = [0.0]


        scales = [(1.0,1.0)]
        #scale_series = self.infinite_geometric_series(0.5)

        # for s in itertools.islice(scale_series,self.n_scales):
        #     r=1.0 - s
        #     scales.append( (r,r) )
        r=1.0
        for i in range(self.scale_intensity):
            r_upsample = r + i * 0.05
            r_downsample = r - i * 0.10
            scales.append( (r_upsample,r_upsample) )
            scales.append((r_downsample,r_downsample))



        translations=[(0,0)]
        for t in range(self.translation_intensity):
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