from .common import *

class Stratified(Experiment):
    def description(self):
        return """Determine the differences between stratified and non-stratified measures."""

    def run(self):
        measures = normalized_measures

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        model_names = simple_models_generators
        transformations = common_transformations_hard

        combinations = itertools.product(model_names, dataset_names, transformations, measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            # train
            model_config = model_config_generator.for_dataset(dataset)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)
            # generate variance params
            p = config.dataset_size_for_measure(measure)
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
            p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
            p_variance_stratified = variance.Parameters(p_training.id(), p_dataset, transformation, measure,stratified=True)
            # evaluate variance
            model_path = config.model_path(p_training)
            self.experiment_variance(p_variance, model_path)
            self.experiment_variance(p_variance_stratified, model_path)
            variance_parameters = [p_variance, p_variance_stratified]
            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = ["Non-stratified", "Stratified"]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
