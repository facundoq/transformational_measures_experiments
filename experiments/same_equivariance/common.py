import config
import transformational_measures as tm
from .base import SameEquivarianceExperiment
import itertools
from transformational_measures import visualization as tmv
from .. import visualization as vis
from ..tasks import train, Task
import datasets
import torch
from experiment.measure.parameters import PyTorchParameters, DatasetParameters,DatasetSizeFixed,DatasetSizePercentage

default_dataset_percentage = DatasetSizePercentage(0.1)
default_dataset_size = DatasetSizeFixed(1000)
default_subset = datasets.DatasetSubset.test

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
default_measure_options = tm.pytorch.PyTorchMeasureOptions(model_device=device, measure_device=device,
                                                           data_device="cpu", verbose=True,num_workers=0)
# common_models_generators = [
#     config.SimpleConvConfig,
#     config.AllConvolutionalConfig,
#     config.VGG16DConfig,
#     config.ResNetConfig
# ]
#
# small_models_generators = [config.SimpleConvConfig,
#                            config.AllConvolutionalConfig, ]
from .models import *

simple_models_generators = [SimpleConvConfig]
# common_models_generators  = simple_models_generators

ca_none, ca_mean, = tm.pytorch.NoTransformation(), tm.pytorch.AverageFeatureMaps()
# da = tm.numpy.DistanceAggregation(normalize=False, keep_shape=False)
# da_normalize = tm.numpy.DistanceAggregation(normalize=True, keep_shape=False)
# da_normalize_keep = tm.numpy.DistanceAggregation(normalize=True, keep_shape=True)
# da_keep = tm.numpy.DistanceAggregation(normalize=False, keep_shape=True)
#
# df = tm.numpy.DistanceFunction(normalize=False)
# df_normalize = tm.numpy.DistanceFunction(normalize=True)

measures = config.common_measures()
nvi = tm.pytorch.NormalizedVarianceInvariance(ca_mean)
svi = tm.pytorch.SampleVarianceInvariance()
tvi = tm.pytorch.TransformationVarianceInvariance()
# nd = tm.pytorch.NormalizedDistanceInvariance(da_keep, ca_mean)  # TODO change to ca_none, its the same because of da_keep but still..
# dse = tm.pytorch.NormalizedDistanceSameEquivariance(da_normalize_keep)
nvse = tm.pytorch.NormalizedVarianceSameEquivariance(ca_mean)
tvse = tm.pytorch.TransformationVarianceSameEquivariance()
svse = tm.pytorch.SampleVarianceSameEquivariance()
# gf = tm.pytorch.GoodfellowNormalInvariance(alpha=0.99)
# anova = tm.pytorch.ANOVAInvariance(alpha=0.99,bonferroni=True)

normalized_measures_validation = [nvi]  # nd, vse]
normalized_measures = [nvi]  # , vse]
dataset_names = ["mnist", "cifar10"]
handshape_dataset_names = ["lsa16", "rwth"]


def get_ylim_normalized(measure: tm.numpy.NumpyMeasure):
    # TODO dict
    if measure.__class__ == tm.numpy.NormalizedDistanceSameEquivariance:
        return 8
    elif measure.__class__ == tm.numpy.NormalizedVarianceSameEquivariance:
        return 8
    elif measure.__class__ == tm.numpy.NormalizedVarianceInvariance:
        return 1.4
    elif measure.__class__ == tm.numpy.NormalizedDistanceInvariance:
        return 1.4
    else:
        raise ValueError(measure)


from transformational_measures.pytorch.transformations.affine import AffineGenerator, RotationGenerator, ScaleGenerator, \
    TranslationGenerator
from transformational_measures.transformations.parameters import UniformRotation, ScaleUniform, TranslationUniform

n_transformations = 24
n_rotations = n_transformations + 1  # 24+1=25 transformations
n_scales = n_transformations // 6  # 24/6=4 -> 4*6+1=25 transformations
n_translations = n_transformations // 8  # 24/8=3 -> 3*8+1=25 transformations
rotation_max_degrees = 1
default_uniform_rotation = UniformRotation(n_rotations, rotation_max_degrees)
scale_min_downscale = 0.5
scale_max_upscale = 1.25
default_uniform_scale = ScaleUniform(n_scales, scale_min_downscale, scale_max_upscale)
translation_max = 0.15

default_uniform_translation = TranslationUniform(n_translations, translation_max)

common_transformations = [RotationGenerator(r=default_uniform_rotation, ),
                          ScaleGenerator(s=default_uniform_scale),
                          TranslationGenerator(t=default_uniform_translation),
                          ]

# 8*7*9=504 transformations
hard = AffineGenerator(r=UniformRotation(8, rotation_max_degrees),  # 8
                       s=ScaleUniform(1, scale_min_downscale, scale_max_upscale),  # 7=6+1
                       t=TranslationUniform(1, translation_max))  # 9=8+1

common_transformations_combined = common_transformations + [hard]
common_transformations_da = common_transformations_combined
identity_transformation = AffineGenerator()

def simple_conv_sameequivariance_activation_filter(m:ObservableLayersModule,name:str): return m.activation_names().index(name) < 6