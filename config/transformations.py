from transformation_measure import *
import itertools

def common_transformations() -> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator()]
    return transformations+common_transformations_without_identity()

def common_transformations_without_identity()-> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator(r=360),
                       SimpleAffineTransformationGenerator(s=4),
                       SimpleAffineTransformationGenerator(t=3),
                       SimpleAffineTransformationGenerator(r=360, s=4, t=3),
                       ]
    return transformations

def rotation_transformations(n:int):
    # TODO change to range(0,n), 360 = 0
    return [SimpleAffineTransformationGenerator(r=i * 360 // n) for i in range(0, n+1)]

def scale_transformations(n:int):
    return [SimpleAffineTransformationGenerator(s=i) for i in range(n)]

def translation_transformations(n:int):
    return [SimpleAffineTransformationGenerator(t=i) for i in range(n)]

def combined_transformations(rs=16,ss=5,ts=5):
    ''' Includes id transformation, ie Affine with r=0,s=0,t=0
    '''
    transformations=[]

    for i in range(rs+1):
        r = i * 360 //rs
        for s in range(ss+1):
            for t in range(ts+1):
                numbers = itertools.product(range(1,rs+1),range(s+1),range(t+1))
                for nr,ns,nt in numbers:
                    tg =SimpleAffineTransformationGenerator(r=r, s=s, t=t,n_rotations=nr,
                                                        n_scales=ns,n_translations=nt)
                    transformations.append(tg)
    return transformations

def all_transformations():
    return combined_transformations(ss=6,ts=6) #common_transformations()+rotation_transformations(16)+scale_transformations(6)+translation_transformations(6) +


def parse_transformation(t:str)->SimpleAffineTransformationGenerator:

    pass


from transformation_measure.transformation import Transformation,TransformationSet
import numpy as np
from typing import List,Tuple,Iterator


RangeParameter=Tuple[float,float]

RangeParameter2D=Tuple[RangeParameter,RangeParameter]

default_range_translation = ((0, 0), (0, 0))
default_range_scale = ((1, 1), (1, 1))

class UniformAffineTransformationGenerator(TransformationSet):

    def __init__(self, r: (0,0) = None, s: RangeParameter2D = default_range_translation, t: RangeParameter2D = default_range_translation, n_r:int=1, n_t:int=1,
                 n_s:int=1):

        assert (r[1] - r[0]) > 0
        for i in range(2):
            assert (s[i][1] - s[i][0]) > 0
            assert (t[i][1] - t[i][0]) > 0

        assert n_r >0
        assert n_s > 0
        assert n_t > 0

        self.r = r
        self.t = t
        self.s = s

        self.n_r = n_r
        self.n_s = n_s
        self.n_t = n_t

        rotations, translations, scales = self.generate_transformation_values()
        self.affine_transformation_generator = AffineTransformationGenerator(rotations=rotations, scales=scales,
                                                                             translations=translations)

    def set_input_shape(self, input_shape):
        self.affine_transformation_generator.set_input_shape(input_shape)

    def set_cuda(self, use_cuda):
        self.affine_transformation_generator.use_cuda = use_cuda

    def set_pytorch(self, pytorch):
        self.affine_transformation_generator.pytorch = pytorch

    def valid_input(self, shape: Tuple[int,]) -> bool:
        return self.affine_transformation_generator.valid_input(shape)

    def copy(self):
        a = UniformAffineTransformationGenerator(r=self.r, s=self.s,
                                                 t=self.t, n_r=self.n_r,
                                                 n_t=self.n_t)
        a.set_pytorch(self.affine_transformation_generator.pytorch)
        a.set_cuda(self.affine_transformation_generator.use_cuda)
        a.set_input_shape(self.affine_transformation_generator.input_shape)
        return a

    def __repr__(self):
        nr = f",nr={self.n_r}"
        ns = f",ns={self.n_s}"
        nt = f",nt={self.n_t}"
        return f"FlexibleAffine(r={self.r},s={self.s},t={self.t}{nr}{ns}{nt})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            other: UniformAffineTransformationGenerator = other
            return self.r == other.r \
                   and self.s == other.s \
                   and self.t == other.t \
                   and self.n_r == other.n_r \
                   and self.n_s == other.n_s \
                   and self.n_t == other.n_t

    def __len__(self):
        return len(self.affine_transformation_generator)

    def id(self):
        return str(self)

    def __iter__(self) -> Iterator[Transformation]:
        return self.affine_transformation_generator.__iter__()


    def generate_transformation_values(self):
        def deg2rad(d:float)->float: return np.pi * (d / (360.0 * 2.0))
        # rotation range in radians
        start,end=deg2rad(self.r[0]),deg2rad(self.r[1])
        rotations = list(np.linspace(start, end, self.n_r, endpoint=False))

        from_x,from_y=self.s[0]
        to_x,to_y=self.s[1]
        translations_x= list(np.linspace(from_x,to_x,self.n_s,endpoint=True))
        translations_y = list(np.linspace(from_y, to_y, self.n_s, endpoint=True))
        scales=[ (sx,sy) for sx,sy in zip(translations_x,translations_y)]

        from_x, from_y = self.t[0]
        to_x, to_y = self.t[1]
        translations_x = list(np.linspace(from_x, to_x, self.n_t, endpoint=True))
        translations_y = list(np.linspace(from_y, to_y, self.n_t, endpoint=True))
        translations = [(tx, ty) for tx, ty in zip(translations_x, translations_y)]


        return rotations, translations, scales