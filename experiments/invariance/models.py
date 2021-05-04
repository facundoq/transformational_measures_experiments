from .common import *
from .weights import DuringTraining

from ..visualization import accuracies
import experiment.measure as measure_package
import datasets




class TrainModels(InvarianceExperiment):

    def description(self):
        return """Analyze the evolution of invariance in models while they are trained."""

    def run(self):
        model_generators = simple_models_generators
        transformations = common_transformations_combined+[identity_transformation]
        combinations = itertools.product(
            model_generators, dataset_names, transformations)
        for model_config_generator, dataset, transformation in combinations:
            # train
            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            savepoints = [sp * epochs // 100 for sp in DuringTraining.savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
            # Training
            p_training = training.Parameters(model_config, dataset, transformation, epochs, savepoints=savepoints)
            self.experiment_training(p_training)

class SimpleConvAccuracies(InvarianceExperiment):
    def description(self):
        return """Compare the accuracies of the SimpleConv model for each set of transformations"""

    def run(self):
        transformations = common_transformations
        transformation_labels = [l.rotation,l.scale,l.translation]
        for dataset in dataset_names:
            transformation_scores = []
            for transformation in transformations:
                model_config = config.SimpleConvConfig.for_dataset(dataset)
                # train
                epochs = config.get_epochs(model_config, dataset, transformation)
                p_training = training.Parameters(model_config, dataset, transformation, epochs)
                self.experiment_training(p_training)
                model, p, o, scores = training.load_model(self.model_path(p_training), load_state=False,
                                                          use_cuda=False)
                loss, acc = scores["test"]
                transformation_scores.append(acc)

            # plot results
            experiment_name = f"{dataset}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            accuracies.plot_accuracies_single_model(plot_filepath, transformation_scores, transformation_labels)


class ModelAccuracies(InvarianceExperiment):
    def description(self):
        return """Compare the accuracies of the models for each set of transformations"""

    def run(self):
        models = common_models_generators
        transformations = common_transformations_combined
        model_names = [m.for_dataset("mnist").name for m in models]
        transformation_labels = [l.rotation,l.scale,l.translation,l.combined]
        for dataset in dataset_names:
            transformation_scores = []
            for transformation in transformations:
                model_scores = []

                for model_config_generator in models:
                    # train
                    model_config = model_config_generator.for_dataset(dataset)
                    # train
                    epochs = config.get_epochs(model_config, dataset, transformation)
                    p_training = training.Parameters(model_config, dataset, transformation, epochs)
                    self.experiment_training(p_training)
                    model, p, o, scores = training.load_model(self.model_path(p_training), load_state=False,
                                                              use_cuda=False)
                    loss, acc = scores["test"]
                    model_scores.append(acc)
                transformation_scores.append(model_scores)
            # plot results
            experiment_name = f"{dataset}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            transformation_scores = np.array(transformation_scores)
            accuracies.plot_accuracies(plot_filepath, transformation_scores, transformation_labels, model_names)


class CompareModels(InvarianceExperiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""

    def run(self):
        measures = normalized_measures

        models = [
            config.SimpleConvConfig,
            config.AllConvolutionalConfig,
            config.VGG16DConfig,
            config.ResNetConfig,

        ]
        transformations = common_transformations
        model_names = [m.for_dataset("mnist").name for m in models]
        combinations = itertools.product(dataset_names, transformations, measures)
        for (dataset, transformation, measure) in combinations:
            variance_parameters = []
            model_configs = []
            for model_config_generator in models:
                # train
                model_config = model_config_generator.for_dataset(dataset)
                model_configs.append(model_config)

                # train
                epochs = config.get_epochs(model_config, dataset, transformation)
                p_training = training.Parameters(model_config, dataset, transformation, epochs)
                self.experiment_training(p_training)
                # generate variance params
                p = config.dataset_size_for_measure(measure)
                p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
                p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                self.experiment_measure(p_variance)

            # plot results
            experiment_name = f"{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            results = self.load_measure_results(self.results_paths(variance_parameters))
            visualization.plot_collapsing_layers_different_models(results, plot_filepath, labels=model_names,
                                                                  markers=self.fc_layers_indices(results),ylim=get_ylim_normalized(measure))

    def fc_layers_indices(self, results: [tm.MeasureResult]) -> [[int]]:
        indices = []
        for r in results:
            layer_names = r.layer_names
            n = len(layer_names)
            index = n
            for i in range(n):
                if "fc" in layer_names[i] or "PoolOut" in layer_names[i] or "fc" in layer_names[i]:
                    index = n
            indices.append(list(range(index, n)))
        return indices
