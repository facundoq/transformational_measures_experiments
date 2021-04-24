import config
import torch
import transformational_measures as tm
from transformational_measures.visualization import plot_collapsing_layers_same_model,plot_collapsing_layers_different_models
from .base import InvarianceExperiment
from ..language import l
from pytorch.pytorch_image_dataset import TransformationStrategy
import datasets
from experiment import measure, training, accuracy
from transformational_measures import visualization
from experiment.measure import MeasureExperimentResult,DatasetParameters,Parameters
import models
import numpy as np
import itertools
from pathlib import Path
import os
from transformational_measures.transformations.pytorch.affine import AffineGenerator

from transformational_measures.transformations.parameters import UniformRotation,ScaleUniform,TranslationUniform

from config.transformations import common_transformations,common_transformations_da,common_transformations_combined,identity_transformation

# TODO restore to 0.5
default_dataset_percentage = 0.01
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

ca_none, ca_mean, ca_sum, ca_max = tm.IdentityTransformation(), tm.AggregateConvolutions(), tm.AggregateTransformation(tm.AggregateFunction.sum),  tm.AggregateTransformation(tm.AggregateFunction.max)
da = tm.DistanceAggregation(normalize=False, keep_shape=False)
da_normalize = tm.DistanceAggregation(normalize=True, keep_shape=False)
da_normalize_keep = tm.DistanceAggregation(normalize=True, keep_shape=True)
da_keep = tm.DistanceAggregation(normalize=False, keep_shape=True)

df = tm.DistanceFunction(normalize=False)
df_normalize = tm.DistanceFunction(normalize=True)

measures = config.common_measures()
nvi = tm.NormalizedVarianceInvariance(ca_mean)
svi = tm.SampleVarianceInvariance()
tvi = tm.TransformationVarianceInvariance()
nd = tm.NormalizedDistanceInvariance(da_keep, ca_mean)  # TODO change to ca_none, its the same because of da_keep but still..
dse = tm.NormalizedDistanceSameEquivariance(da_normalize_keep)
vse = tm.NormalizedVarianceSameEquivariance(ca_mean)
gf = tm.GoodfellowNormalInvariance(alpha=0.99)
anova = tm.ANOVAInvariance(alpha=0.99,bonferroni=True)

normalized_measures_validation = [nvi]# nd, vse]
normalized_measures = [nvi]#, vse]
dataset_names = ["mnist", "cifar10"]
handshape_dataset_names = ["lsa16", "rwth"]
venv_path = ""



def get_ylim_normalized(measure: tm.NumpyMeasure):
    # TODO dict
    if measure.__class__ == tm.NormalizedDistanceSameEquivariance:
        return 8
    elif measure.__class__ == tm.NormalizedVarianceSameEquivariance:
        return 8
    elif measure.__class__ == tm.NormalizedVarianceInvariance:
        return 1.4
    elif measure.__class__ == tm.NormalizedDistanceInvariance:
        return 1.4
    else:
        raise ValueError(measure)
