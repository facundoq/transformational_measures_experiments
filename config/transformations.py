from transformation_measure import *
import itertools
from transformations.pytorch.affine import AffineGenerator
from typing import Iterator
from .parameters import UniformRotation,ScaleUniform,TranslationUniform
import torch



n_transformations=24
n_r=n_transformations+1 #24+1=25 transformations
n_s=n_transformations//6 # 24/6=4 -> 4*6+1=25 transformations
n_t=n_transformations//8 # 24/8=3 -> 3*8+1=25 transformations
rotation_max_degrees=360
r=UniformRotation(n_r, rotation_max_degrees)
scale_min_downscale=0.5
scale_max_upscale=1.25
s=ScaleUniform(n_s,scale_min_downscale,scale_max_upscale)
translation_max=0.25
t=TranslationUniform(n_t,translation_max)
common_transformations= [AffineGenerator(r=r,),
                         AffineGenerator(s=s),
                         AffineGenerator(t=t),
                         ]

# 8*7*9=504 transformations
hard = AffineGenerator(r=UniformRotation(8,rotation_max_degrees), #8
                       s=ScaleUniform(1,scale_min_downscale,scale_max_upscale), #7=6+1
                       t=TranslationUniform(1,translation_max)) #9=8+1
common_transformations_combined = common_transformations + [hard]
common_transformations_da = common_transformations_combined
identity_transformation = AffineGenerator()


# def rotation_transformations(n:int):
#     # TODO change to range(0,n), 360 = 0
#     return [AffineGenerator(r=i * 360 // n) for i in range(0, n + 1)]
#
# def scale_transformations(n:int):
#     return [AffineGenerator(s=i) for i in range(n)]
#
# def translation_transformations(n:int):
#     return [AffineGenerator(t=i) for i in range(n)]
#
# def combined_transformations(rs=16,ss=5,ts=5):
#     ''' Includes id transformation, ie Affine with r=0,s=0,t=0
#     '''
#     transformations=[]
#
#     for i in range(rs+1):
#         r = i * 360 //rs
#         for s in range(ss+1):
#             for t in range(ts+1):
#                 numbers = itertools.product(range(1,rs+1),range(s+1),range(t+1))
#                 for nr,ns,nt in numbers:
#                     tg =AffineGenerator(r=r, s=s, t=t, n_rotations=nr,
#                                         n_scales=ns, n_translations=nt)
#                     transformations.append(tg)
#     return transformations
#
# def all_transformations():
#     return combined_transformations(ss=6,ts=6) #common_transformations()+rotation_transformations(16)+scale_transformations(6)+translation_transformations(6) +
#




from transformation_measure.transformation import Transformation,TransformationSet
# import numpy as np
# from typing import List,Tuple,Iterator
#
#
# RangeParameter=Tuple[float,float]
#
# RangeParameter2D=Tuple[RangeParameter,RangeParameter]
#
# default_range_translation = ((0, 0), (0, 0))
# default_range_scale = ((1, 1), (1, 1))
#
# def print_range(r:RangeParameter):
#     x,y=r
#     return f"({x},{y})"
# def print_range2d(r:RangeParameter2D):
#     a,b=r
#     return f"({print_range(a)},{print_range(b)})"
#
#
# def generate_transformation_values(self):
#     def deg2rad(d:float)->float: return np.pi * (d / (360.0 * 2.0))
#     # rotation range in radians
#     start,end=deg2rad(self.r[0]),deg2rad(self.r[1])
#     rotations = list(np.linspace(start, end, self.nr, endpoint=False))
#
#     from_x,from_y=self.s[0]
#     to_x,to_y=self.s[1]
#
#     translations_x= list(np.linspace(from_x, to_x, self.ns, endpoint=True))
#     translations_y = list(np.linspace(from_y, to_y, self.ns, endpoint=True))
#     scales=[ (sx,sy) for sx,sy in zip(translations_x,translations_y)]
#
#     from_x, from_y = self.t[0]
#     to_x, to_y = self.t[1]
#
#     translations_x = list(np.linspace(from_x, to_x, self.nt, endpoint=True))
#     translations_y = list(np.linspace(from_y, to_y, self.nt, endpoint=True))
#     translations = [(tx, ty) for tx, ty in zip(translations_x, translations_y)]
#
#     return rotations, translations, scales