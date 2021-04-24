import enum
import abc
from datasets import DatasetSubset
import transformational_measures as tm

class Language(abc.ABC):
    goodfellow = "Goodfellow"
    ANOVA = "ANOVA"

    def __init__(self):
        self.name = ""
        self.batch_size = ""
        self.message = ""
        self.accuracy = ""
        self.layer = ""
        self.layers = ""
        self.transformation = ""
        self.transformations = ""
        self.rotation = ""
        self.scale = ""
        self.translation = ""
        self.combined = ""
        self.normalized = ""
        self.unnormalized = ""
        self.meandeviation = ""
        self.epoch = ""
        self.train = ""
        self.test = ""
        self.subset = ""
        self.maxpooling= ""
        self.strided_convolution = ""
        self.measure=""
        self.with_bn = ""
        self.without_bn = ""
        self.aggregation=""
        self.before_normalization = ""
        self.after_normalization = ""
        self.normal = ""
        self.feature_map_aggregation = ""
        self.feature_map_distance = ""
        self.no_data_augmentation = ""
        self.normalized_distance = ""
        self.normalized_variance = ""
        self.sample_distance = ""
        self.transformational_distance = ""
        self.sample_variance = ""
        self.transformational_variance = ""

        self.normalized_distance_sameequivariance= ""

        self.sample_distance_sameequivariance= ""
        self.transformational_distance_sameequivariance= ""

        self.sample_variance_sameequivariance= ""
        self.transformational_variance_sameequivariance= ""
        self.normalized_variance_sameequivariance= ""

        self.simple_sameequivariance= ""
        self.stratified=""
        self.non_stratified=""
        self.to=""
        self.random_models = ""
        self.samples = ""
        self.model=""
        self.mean=""
        self.max=""
        self.min=""
        self.sum=""

    def measure_name(self, m:tm.NumpyMeasure):
        dict = {
            tm.NormalizedDistanceInvariance: self.normalized_distance,
            tm.NormalizedVarianceInvariance: self.normalized_variance,
            tm.SampleDistanceInvariance: self.sample_distance,
            tm.TransformationDistanceInvariance: self.transformational_distance,
            tm.TransformationVarianceInvariance: self.transformational_variance,
            tm.SampleVarianceInvariance: self.sample_variance,
            tm.GoodfellowNormalInvariance: self.goodfellow,
            tm.GoodfellowInvariance: self.goodfellow,
            tm.ANOVAInvariance: self.ANOVA,
            tm.NormalizedDistanceSameEquivariance:self.normalized_distance_sameequivariance,
            tm.TransformationDistanceSameEquivariance:self  .transformational_distance_sameequivariance,
            tm.SampleDistanceSameEquivariance:self.sample_distance_sameequivariance,
            tm.SampleVarianceSameEquivariance:self.sample_variance_sameequivariance,
            tm.TransformationVarianceSameEquivariance:self.transformational_variance_sameequivariance,
            tm.NormalizedVarianceSameEquivariance:self.normalized_variance_sameequivariance,
            tm.DistanceSameEquivarianceSimple:self.simple_sameequivariance,
        }
        return dict[m.__class__]

    def format_subset(self,s:DatasetSubset) -> str:
        if s == DatasetSubset.train:
            return self.train
        elif s == DatasetSubset.test:
            return self.test
        else:
            raise ValueError(s)

    def format_aggregation(self, ca:tm.AggregateFunction):

        if ca == tm.AggregateFunction.mean:
            return self.mean
        elif ca == tm.AggregateFunction.max:
            return self.max
        elif ca == tm.AggregateFunction.sum:
            return self.sum
        # elif ca == tm.AggregateFunction.min:
        #     return self.min
        else:
            raise ValueError(ca)