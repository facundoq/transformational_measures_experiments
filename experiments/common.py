import config
import torch
import transformation_measure as tm
from .base import Experiment
from .language import l
from transformation_measure.iterators.pytorch_image_dataset import TransformationStrategy
import datasets
from experiment import variance, training,accuracy
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
da = tm.DistanceAggregation(normalize=False, keep_shape=False)
da_normalize = tm.DistanceAggregation(normalize=True, keep_shape=False)
da_normalize_keep = tm.DistanceAggregation(normalize=True, keep_shape=True)
da_keep = tm.DistanceAggregation(normalize=False, keep_shape=True)

df = tm.DistanceFunction(normalize=False)
df_normalize = tm.DistanceFunction(normalize=True)

measures = config.common_measures()
nv = tm.NormalizedVariance(ca_mean)
nd = tm.NormalizedDistance(da_keep,ca_mean) # TODO change to ca_none, its the same because of da_keep but still..
dse = tm.NormalizedDistanceSameEquivariance(da_normalize_keep)
vse = tm.NormalizedVarianceSameEquivariance(ca_mean)
gf = tm.GoodfellowNormal()


normalized_measures_validation = [nv, nd, vse]
normalized_measures = [nv, vse]
dataset_names = ["mnist", "cifar10"]
handshape_dataset_names=["lsa16","rwth"]
venv_path = ""

common_transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                          tm.SimpleAffineTransformationGenerator(s=4),
                          tm.SimpleAffineTransformationGenerator(t=3),
                          ]


combined=tm.SimpleAffineTransformationGenerator(r=360, s=4, t=3,n_rotations=6,n_translations=1,n_scales=1)
common_transformations_combined = common_transformations + [combined]
hard = tm.SimpleAffineTransformationGenerator(r=360,s=4,t=3)
common_transformations_hard = common_transformations + [hard]


common_transformations_da = [tm.SimpleAffineTransformationGenerator(r=360),
                   tm.SimpleAffineTransformationGenerator(s=6),
                   tm.SimpleAffineTransformationGenerator(t=5),
                   tm.SimpleAffineTransformationGenerator(r=360,s=6,t=5),
                   ]

def get_ylim_normalized(measure:tm.Measure):
    # TODO dict
    if measure.__class__ == tm.NormalizedDistanceSameEquivariance:
        return 8
    elif measure.__class__ == tm.NormalizedVarianceSameEquivariance:
        return 8
    elif measure.__class__ == tm.NormalizedVariance:
        return 1.4
    elif measure.__class__ == tm.NormalizedDistance:
        return 1.4
    else:
        raise  ValueError(measure)