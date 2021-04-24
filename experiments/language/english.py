from . import Language

class English(Language):

    def __init__(self):
        super().__init__()
        self.message="hola"
        self.accuracy="Accuracy"
        self.layer="Layer"
        self.batch_size = "Batch Size"
        self.layers="Layers"
        self.transformation="Transformation"
        self.rotation= "Rotation"
        self.scale = "Scale"
        self.translation = "Translation"
        self.combined = "Combined"
        self.normalized= "Normalized"
        self.unnormalized = "Unnormalized"
        self.meandeviation="μ/σ"
        self.epoch= "Epoch"
        self.train= "Train"
        self.test= "Test"
        self.subset="subset"
        self.maxpooling = "MaxPooling"
        self.strided_convolution = "Strided Convolutions"
        self.measure = "Measure"
        self.with_bn = "With BN"
        self.without_bn = "Without BN"
        self.aggregation = "Aggregation"
        self.before_normalization = "Before normalization"
        self.after_normalization = "After normalization"
        self.normal = "Normal"
        self.feature_map_aggregation = "Sum"
        self.feature_map_distance = "Distance"
        self.no_data_augmentation = "No data augmentation"
        self.normalized_distance = "Normalized Distance Invariance"
        self.normalized_variance = "Normalized Variance Invariance"
        self.sample_distance = "Sample Distance Invariance"
        self.transformational_distance = "Transformational Distance Invariance"
        self.sample_variance = "Sample Variance Invariance"
        self.transformational_variance = "Transformational Variance Invariance"

        self.normalized_distance_sameequivariance= "Normalized Distance Same-Equivariance"

        self.sample_distance_sameequivariance= "Sample Distance Same-Equivariance"
        self.transformational_distance_sameequivariance= "Transformational Distance Same-Equivariancel"

        self.sample_variance_sameequivariance= "Sample Variance Same-Equivariance"
        self.transformational_variance_sameequivariance= "Transformational Variance Same-Equivariance"
        self.normalized_variance_sameequivariance= "Normalized Variance Same-Equivariance"
        self.simple_sameequivariance= "Simple Distance Same-Equivariance"

        self.stratified = "Non-stratified"
        self.non_stratified = "Stratified"
        self.to = "to"
        self.random_models = "Random models"
        self.samples = "samples"
        self.model = "Model"
        self.transformed = "Transformed"
        self.transformational = "Transformation"
        self.sample_based = "Sample"


