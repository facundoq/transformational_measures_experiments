from .common import *


class SameEquivarianceNormalization(Experiment):

    def description(self):
        return """Compare the result of DistanceSameEquivariance normalized or not."""

    def run(self):
        measures = [
            tm.NormalizedDistanceSameEquivariance(da_normalize_keep),
            tm.NormalizedDistanceSameEquivariance(da_keep)
        ]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)
        for model_config_generator, dataset, transformation in combinations:
            # train
            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # eval
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 1.0)
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance, model_path)

            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [l.normalized,l.unnormalized]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=400)


