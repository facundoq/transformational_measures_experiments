import config
import transformation_measure as tm
from .base import Experiment
from .language import l
from experiment import variance, training
from transformation_measure import visualization
from experiment.variance import VarianceExperimentResult
import models
import numpy as np
import itertools
from pathlib import Path
import os

default_dataset_percentage = 0.5

common_models_generators = [
    config.SimpleConvConfig,
    config.AllConvolutionalConfig,
    config.VGG16DConfig,
    config.ResNetConfig
]

small_models_generators = [config.SimpleConvConfig,
                           config.AllConvolutionalConfig, ]

simple_models_generators = [config.SimpleConvConfig]
# common_models_generators  = simple_models_generators

ca_none, ca_mean, ca_sum,ca_max = tm.ConvAggregation.none, tm.ConvAggregation.mean, tm.ConvAggregation.sum, tm.ConvAggregation.max
da = tm.DistanceAggregation(normalize=False,keep_feature_maps=False)
da_normalize = tm.DistanceAggregation(normalize=True,keep_feature_maps=False)
da_normalize_keep = tm.DistanceAggregation(normalize=True,keep_feature_maps=True)
da_keep = tm.DistanceAggregation(normalize=False,keep_feature_maps=True)

measures = config.common_measures()
nv = tm.NormalizedVariance(ca_mean)
nd = tm.NormalizedDistance(da_keep,ca_mean) # TODO change to ca_none, its the same because of da_keep but still..
se = tm.DistanceSameEquivarianceMeasure(da_normalize_keep)
gf = tm.GoodfellowNormal()


normalized_measures_validation = [nv,nd,se]
normalized_measures = [nv,se]
dataset_names = ["mnist", "cifar10"]
venv_path = ""

common_transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                          tm.SimpleAffineTransformationGenerator(s=4),
                          tm.SimpleAffineTransformationGenerator(t=3),
                          ]
combined=tm.SimpleAffineTransformationGenerator(r=360, s=4, t=3,n_rotations=6,n_translations=1,n_scales=1)
common_transformations_hard = common_transformations+[combined]


def get_ylim_normalized(measure:tm.Measure):
    if measure.__class__ == tm.DistanceSameEquivarianceMeasure:
        return 8
    elif measure.__class__ == tm.NormalizedVariance:
        return 1.4
    elif measure.__class__ == tm.NormalizedDistance:
        return 1.4
    else:
        raise  ValueError(measure)