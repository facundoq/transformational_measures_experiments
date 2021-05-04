from .common import *
from ..tasks import train, Task


class TrainModels(SameEquivarianceExperiment):

    def description(self):
        return """Train models and check their performance."""

    def run(self):
        savepoints_percentages = [0,1,2,5,10,25,50,75,100]
        model_generators = simple_models_generators
        transformations = common_transformations_combined+[identity_transformation]
        combinations = itertools.product(
            model_generators, dataset_names, transformations)
        task = Task.Regression

        for model_config_generator, dataset, transformations in combinations:
            # train
            mc:SimpleConvConfig = model_config_generator.for_dataset(dataset,task)

            epochs = mc.epochs(dataset, task, transformations)
            min_accuracy = mc.min_accuracy(dataset, task, transformations)

            savepoints = [sp * epochs // 100 for sp in savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
            # Training
            # p_training = training.Parameters(model_config, dataset, transformation, epochs, savepoints=savepoints)
            cc = train.MaxMSEConvergence(min_accuracy)
            tc = train.TrainConfig(epochs,cc,savepoints=savepoints)
            p = train.TrainParameters(mc,tc,dataset,transformations,task)
            train.train(p,self)
