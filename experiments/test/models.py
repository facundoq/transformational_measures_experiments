import transformational_measures.measure
from ..invariance.common import *
from .base import TestExperiment
import config

class ModelAccuracy(TestExperiment):

    def description(self):
        return """Test pytorch transformations and generate sample images"""

    def run(self):
        model_config_generators = [config.SimpleConvConfig]
        dataset_names = ["mnist","cifar10"]

        transformations= [identity_transformation]+common_transformations_combined

        combinations = itertools.product(model_config_generators, dataset_names, transformations)
        for model_config_generator, dataset,transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)
