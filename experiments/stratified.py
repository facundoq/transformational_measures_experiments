from .common import *
import experiment.measure as measure_package
import datasets


class Stratified(Experiment):
    def description(self):
        return """Determine the differences between stratified and non-stratified measures."""

    def run(self):
        measures = normalized_measures

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        model_names = simple_models_generators
        transformations = common_transformations

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
            # make 1/number of classes
            p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.train, 0.1)
            p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
            
            p_dataset_variance = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.train, 1.0)
            p_variance_stratified = measure_package.Parameters(p_training.id(), p_dataset_variance, transformation, measure, stratified=True)
            
            # evaluate variance
            model_path = config.model_path(p_training)
            self.experiment_measure(p_variance)
            self.experiment_measure(p_variance_stratified)
            variance_parameters = [p_variance, p_variance_stratified]
            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_measure_results(config.results_paths(variance_parameters))
            
            labels = [l.non_stratified,l.stratified]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))

