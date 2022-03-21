from transformational_measures import *
import itertools
from datasets import DatasetSubset
from experiment import measure

from transformational_measures.numpy import *

def dataset_percentage_for_measure(measure:Measure, subset=DatasetSubset.test)->float:
    if subset == DatasetSubset.test:
        if measure.__class__.__name__.endswith("Invariance"):
            return 0.5
        else:
            return 0.5
    elif subset == DatasetSubset.train:
        if measure.__class__.__name__.endswith("Invariance"):
            return 0.1
        else:
            return 0.1
    else:
        raise ValueError(f"Invalid subset {subset}")

def default_measure_dataset(dataset:str, m:Measure):
    subset = DatasetSubset.test
    return measure.DatasetParameters(dataset, subset, dataset_percentage_for_measure(m, subset))

da = DistanceAggregation(normalize=False, keep_shape=False)
da_normalize = DistanceAggregation(normalize=True, keep_shape=False)
da_normalize_keep = DistanceAggregation(normalize=True, keep_shape=True)
da_keep = DistanceAggregation(normalize=False, keep_shape=True)

df = DistanceFunction(normalize=False)
df_normalize = DistanceFunction(normalize=True)

measure_transformations=[IdentityTransformation(),
                         AggregateTransformation(AggregateFunction.mean),
                         AggregateTransformation(AggregateFunction.sum),
                         AggregateTransformation(AggregateFunction.max),
                         ]

def common_measures()-> [NumpyMeasure]:

    mt_none, mt_mean, mt_sum, mt_max = measure_transformations

    measures=[
        SampleVarianceInvariance()
        ,TransformationVarianceInvariance()
        ,NormalizedVarianceInvariance(mt_none)
        ,NormalizedVarianceInvariance(mt_mean)
        ,ANOVAFInvariance()
        ,ANOVAInvariance(alpha=0.99,bonferroni=True)
        ,NormalizedDistanceInvariance(da,mt_mean)
        ,NormalizedDistanceInvariance(da_keep, mt_none)
        ,NormalizedDistanceSameEquivariance(da_normalize_keep)
    ]
    return measures
