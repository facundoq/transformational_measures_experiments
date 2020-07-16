import enum
import abc
from datasets import DatasetSubset
import transformational_measures as tm

class Language(abc.ABC):
    def __init__(self):
        self.name = ""
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
        self.goodfellow = "Goodfellow"
        self.ANOVA = "ANOVA"
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
            tm.NormalizedDistance: self.normalized_distance,
            tm.NormalizedVariance: self.normalized_variance,
            tm.SampleDistance: self.sample_distance,
            tm.TransformationDistance: self.transformational_distance,
            tm.TransformationVariance: self.transformational_variance,
            tm.SampleVariance: self.sample_variance,
            tm.GoodfellowNormal: self.goodfellow,
            tm.GoodfellowMeasure: self.goodfellow,
            tm.AnovaMeasure: self.ANOVA,
            tm.NormalizedDistanceSameEquivariance:self.normalized_distance_sameequivariance,
            tm.TransformationDistanceSameEquivariance:self.transformational_distance_sameequivariance,
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

    def format_aggregation(self, ca:tm.ConvAggregation):

        if ca == tm.ConvAggregation.mean:
            return self.mean
        elif ca == tm.ConvAggregation.max:
            return self.max
        elif ca == tm.ConvAggregation.sum:
            return self.sum
        elif ca == tm.ConvAggregation.min:
            return self.min
        else:
            raise ValueError(ca)