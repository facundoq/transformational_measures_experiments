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