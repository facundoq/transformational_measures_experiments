from transformation_measure import *


def common_transformations() -> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator()]
    return transformations+common_transformations_without_identity()

def common_transformations_without_identity()-> [TransformationSet]:
    transformations = [SimpleAffineTransformationGenerator(r=360),
                       SimpleAffineTransformationGenerator(s=4),
                       SimpleAffineTransformationGenerator(t=3),
                       # SimpleAffineTransformationGenerator(r=360, s=4, t=3),
                       ]
    return transformations

def rotation_transformations(n:int):
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
                transformations.append(SimpleAffineTransformationGenerator(r=r, s=s, t=t))
    return transformations

def all_transformations():
    return combined_transformations() #common_transformations()+rotation_transformations(16)+scale_transformations(6)+translation_transformations(6) +

