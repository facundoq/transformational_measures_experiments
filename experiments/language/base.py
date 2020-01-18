import enum
import abc
from experiment.variance import DatasetSubset
import transformation_measure as tm

class Language(abc.ABC):
    def __init__(self):
        self.name = ""
        self.message = ""
        self.accuracy = ""
        self.layer = ""
        self.layers = ""
        self.transformation = ""
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
        self.measure="Measure"
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
        self.distance_equivariance= ""
        self.stratified=""
        self.non_stratified=""
        self.to=""
        self.random_models = ""
        self.samples = ""

    def measure_name(self,m:tm.Measure):
        dict = {
            tm.NormalizedDistance: self.normalized_distance,
            tm.NormalizedVariance: self.normalized_variance,
            tm.SampleDistance: self.sample_distance,
            tm.TransformationDistance: self.transformational_distance,
            tm.TransformationVariance: self.transformational_variance,
            tm.SampleVariance: self.sample_variance,
            tm.GoodfellowNormalMeasure: self.goodfellow,
            tm.GoodfellowMeasure: self.goodfellow,
            tm.AnovaMeasure: self.ANOVA,
            tm.DistanceSameEquivarianceMeasure:self.distance_equivariance,
        }
        return dict[m.__class__]

    def format_subset(self,s:DatasetSubset) -> str:
        if s == DatasetSubset.train:
            return self.train
        elif s == DatasetSubset.test:
            return self.test
        else:
            raise ValueError(s)

