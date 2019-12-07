from transformation_measure import *
import itertools

def all_measures()-> [Measure]:
    cas=[ConvAggregation.none, ConvAggregation.sum,ConvAggregation.mean,ConvAggregation.max,]
    das = [DistanceAggregation.mean,DistanceAggregation.max]
    measure_functions = [MeasureFunction.std]
    measures=[]

    for (ca,mf) in itertools.product(cas,measure_functions):
        measures.append(SampleVariance())
        measures.append(TransformationVariance())
        measures.append(NormalizedVariance(ca))
        # measures.append(SampleVarianceMeasure(mf, ))
        # measures.append(TransformationVarianceMeasure(mf))
        # measures.append(NormalizedVarianceMeasure(mf, conv_aggregation=ca))

    for (da,mf) in itertools.product(das,measure_functions):
        measures.append(DistanceSampleMeasure(da))
        measures.append(DistanceTransformationMeasure(da))
        measures.append(DistanceMeasure( da))
        measures.append(DistanceSameEquivarianceMeasure(da))

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
    dmean, dmax, = DistanceAggregation.mean, DistanceAggregation.max
    mf, ca_sum, ca_mean = MeasureFunction.std, ConvAggregation.sum, ConvAggregation.mean
    ca_none = ConvAggregation.none
    measures=[
        SampleVariance(mf)
        ,TransformationVariance(mf)
        ,NormalizedVariance(ca_none)
        ,NormalizedVariance(ca_mean)
        ,AnovaFMeasure()
        ,AnovaMeasure(alpha=0.99,bonferroni=True)
        ,DistanceMeasure(dmean)
        ,DistanceSameEquivarianceMeasure(dmean)

    ]
    return measures
