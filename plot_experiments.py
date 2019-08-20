#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


from pytorch import variance
import transformation_measure as tm
from pytorch.experiment import training

import util

from typing import List
import numpy as np
from transformation_measure import visualization
import typing
from pytorch import variance
import os,sys
import argparse,argcomplete
from pytorch.variance import DatasetSubset,DatasetParameters

model_names=training.get_models()
model_names.sort()

measures = tm.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]

measure=tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum)
dataset=variance.DatasetParameters(p.dataset,DatasetSubset.test,0.1)