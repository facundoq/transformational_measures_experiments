import enum
import abc
from datasets import DatasetSubset
import tmeasures as tm

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

    def measure_name_numpy(self, m:tm.np.NumpyMeasure):
        dict = {
            tm.np.NormalizedDistanceInvariance: self.normalized_distance,
            tm.np.NormalizedVarianceInvariance: self.normalized_variance,
            tm.np.SampleDistanceInvariance: self.sample_distance,
            tm.np.TransformationDistanceInvariance: self.transformational_distance,
            tm.np.TransformationVarianceInvariance: self.transformational_variance,
            tm.np.SampleVarianceInvariance: self.sample_variance,
            tm.np.GoodfellowNormalInvariance: self.goodfellow,
            tm.np.GoodfellowInvariance: self.goodfellow,
            tm.np.ANOVAInvariance: self.ANOVA,
            tm.np.NormalizedDistanceSameEquivariance:self.normalized_distance_sameequivariance,
            tm.np.TransformationDistanceSameEquivariance:self  .transformational_distance_sameequivariance,
            tm.np.SampleDistanceSameEquivariance:self.sample_distance_sameequivariance,
            tm.np.SampleVarianceSameEquivariance:self.sample_variance_sameequivariance,
            tm.np.TransformationVarianceSameEquivariance:self.transformational_variance_sameequivariance,
            tm.np.NormalizedVarianceSameEquivariance:self.normalized_variance_sameequivariance,
            tm.np.DistanceSameEquivarianceSimple:self.simple_sameequivariance,
        }
        return dict[m.__class__]

    def format_subset(self,s:DatasetSubset) -> str:
        if s == DatasetSubset.train:
            return self.train
        elif s == DatasetSubset.test:
            return self.test
        else:
            raise ValueError(s)

    def format_aggregation(self, ca:tm.np.AggregateFunction):

        if ca == tm.numpy.AggregateFunction.mean:
            return self.mean
        elif ca == tm.numpy.AggregateFunction.max:
            return self.max
        elif ca == tm.numpy.AggregateFunction.sum:
            return self.sum
        # elif ca == tm.numpy.AggregateFunction.min:
        #     return self.min
        else:
            raise ValueError(ca)