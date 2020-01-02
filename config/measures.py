from transformation_measure import *
import itertools
from experiment.variance import DatasetSubset
from experiment import variance

def dataset_size_for_measure(measure:Measure,subset=DatasetSubset.test)->float:
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

def default_measure_dataset(dataset:str,measure:Measure):
    subset = variance.DatasetSubset.test
    return variance.DatasetParameters(dataset, subset, dataset_size_for_measure(measure,subset))

da = DistanceAggregation(normalize=False, keep_feature_maps=False)
da_normalize = DistanceAggregation(normalize=True, keep_feature_maps=False)
da_normalize_keep = DistanceAggregation(normalize=True, keep_feature_maps=True)
da_keep = DistanceAggregation(normalize=False, keep_feature_maps=True)

def all_measures()-> [Measure]:
    cas=[ConvAggregation.none, ConvAggregation.sum,ConvAggregation.mean,ConvAggregation.max,]
    das = [da,da_normalize_keep,da_normalize,da_keep]
    measure_functions = [MeasureFunction.std]
    measures=[]

    for (ca,mf) in itertools.product(cas,measure_functions):
        measures.append(SampleVariance())
        measures.append(TransformationVariance())
        measures.append(NormalizedVariance(ca))
        # measures.append(SampleVarianceMeasure(mf, ))
        # measures.append(TransformationVarianceMeasure(mf))
        # measures.append(NormalizedVarianceMeasure(mf, conv_aggregation=ca))

    for (d,mf) in itertools.product(das,measure_functions):
        measures.append(SampleDistance(d))
        measures.append(TransformationDistance(d))
        for ca in cas:
            measures.append(NormalizedDistance(d,ca))
        measures.append(DistanceSameEquivarianceMeasure(d))

    for percentage in [0.01,0.001,0.1]:
        measures.append(GoodfellowMeasure(activations_percentage=percentage))
        measures.append(GoodfellowNormalMeasure(alpha=1-percentage))

    measures.append(AnovaFMeasure())
    alphas=[0.90, 0.95, 0.99, 0.999]
    for alpha in alphas:
            measures.append(AnovaMeasure(alpha=alpha, bonferroni=True ))
            measures.append(AnovaMeasure(alpha=alpha, bonferroni=False))
    return measures


def common_measures()-> [Measure]:

    mf, ca_sum, ca_mean = MeasureFunction.std, ConvAggregation.sum, ConvAggregation.mean
    ca_none = ConvAggregation.none
    measures=[
        SampleVariance(mf)
        ,TransformationVariance(mf)
        ,NormalizedVariance(ca_none)
        ,NormalizedVariance(ca_mean)
        ,AnovaFMeasure()
        ,AnovaMeasure(alpha=0.99,bonferroni=True)
        ,NormalizedDistance(da,ca_mean)
        ,NormalizedDistance(da_keep, ca_none)
        ,DistanceSameEquivarianceMeasure(da_normalize_keep)
    ]
    return measures
