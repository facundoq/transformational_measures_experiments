from tmeasures.pytorch.transformations.affine import AffineGenerator,RotationGenerator,ScaleGenerator,TranslationGenerator
from tmeasures.transformations.parameters import UniformRotation,ScaleUniform,TranslationUniform

n_transformations=24
n_r=n_transformations+1 #24+1=25 transformations
n_s=n_transformations//6 # 24/6=4 -> 4*6+1=25 transformations
n_t=n_transformations//8 # 24/8=3 -> 3*8+1=25 transformations
rotation_max_degrees=1.0
r=UniformRotation(n_r, rotation_max_degrees)
scale_min_downscale=0.5
scale_max_upscale=1.25
s=ScaleUniform(n_s,scale_min_downscale,scale_max_upscale)
translation_max=0.15
t=TranslationUniform(n_t,translation_max)

common_transformations= [RotationGenerator(r=r,) ,
                         ScaleGenerator(s=s),
                        TranslationGenerator(t=t),
                         ]

# 8*7*9=504 transformations
hard = AffineGenerator(r=UniformRotation(8,rotation_max_degrees), #8
                       s=ScaleUniform(1,scale_min_downscale,scale_max_upscale), #7=6+1
                       t=TranslationUniform(1,translation_max)) #9=8+1

common_transformations_combined = common_transformations + [hard]
common_transformations_da = common_transformations_combined
identity_transformation = AffineGenerator()
