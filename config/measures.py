from transformational_measures import *
import itertools
from datasets import DatasetSubset
from experiment import measure

def dataset_size_for_measure(measure:NumpyMeasure, subset=DatasetSubset.test)->float:
    if subset == DatasetSubset.test:
        if measure.__class__.__name__.endswith("Variance"):
            return 0.5
        else:
            return 0.5
    elif subset == DatasetSubset.train:
        if measure.__class__.__name__.endswith("Variance"):
            return 0.1
        else:
            return 0.1
    else:
        raise ValueError(f"Invalid subset {subset}")

def default_measure_dataset(dataset:str, m:NumpyMeasure):
    subset = DatasetSubset.test
    return measure.DatasetParameters(dataset, subset, dataset_size_for_measure(m, subset))

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

def all_measures()-> [NumpyMeasure]:



    das = [da,da_normalize_keep,da_normalize,da_keep]
    dfs = [df , df_normalize]
    measure_functions = [MeasureFunction.std]
    measures=[]
    measures.append(SampleVarianceInvariance())
    measures.append(TransformationVarianceInvariance())
    measures.append(SampleVarianceSameEquivariance())
    measures.append(TransformationVarianceSameEquivariance())
    for ca in measure_transformations:
        measures.append(NormalizedVarianceInvariance(ca))
        measures.append(NormalizedVarianceSameEquivariance(ca))


    for (d,mf) in itertools.product(das,measure_functions):
        measures.append(SampleDistanceInvariance(d))
        measures.append(TransformationDistanceInvariance(d))
        for ca in measure_transformations:
            measures.append(NormalizedDistanceInvariance(d,ca))
        measures.append(NormalizedDistanceSameEquivariance(d))
        measures.append(SampleDistanceSameEquivariance(d))
        measures.append(TransformationDistanceSameEquivariance(d))

    for d in dfs:
        measures.append(DistanceSameEquivarianceSimple(d))

    for percentage in [0.01,0.001,0.1,0.5,0.05]:
        measures.append(GoodfellowInvariance(activations_percentage=percentage))
        measures.append(GoodfellowNormalInvariance(alpha=1 - percentage))

    measures.append(ANOVAFInvariance())
    alphas=[0.90, 0.95, 0.99, 0.999]
    for alpha in alphas:
            measures.append(ANOVAInvariance(alpha=alpha, bonferroni=True ))
            measures.append(ANOVAInvariance(alpha=alpha, bonferroni=False))
    return measures


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
