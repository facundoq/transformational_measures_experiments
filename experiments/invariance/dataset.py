from .common import *
import experiment.measure as measure_package
import datasets


class DatasetSize(InvarianceExperiment):

    def description(self):
        return '''Vary the test dataset size and see how it affects the numpy's value. That is, vary the size of the dataset used to compute the invariance (not the training dataset) and see how it affects the calculation of the numpy.'''

    def run(self):
        dataset_sizes = [0.01, 0.05, 0.1, 0.5, 1.0]
        model_names = simple_models_generators
        measures = normalized_measures_validation
        combinations = list(itertools.product(
            model_names, dataset_names, common_transformations_combined, measures))
        for i, (model, dataset, transformation, measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}", end=", ")
            model_config = model.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)
            p_datasets = [measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, p) for p in dataset_sizes]
            experiment_name = f"{model_config}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            variance_parameters = [measure_package.Parameters(p_training.id(), p_dataset, transformation, measure) for p_dataset in p_datasets]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance)
            results = config.load_measure_results(config.results_paths(variance_parameters))
            #p_datasets = [r.parameters.dataset for r in results]

            labels = [f"{d * 100:2}%" for d in dataset_sizes]
            n = len(dataset_sizes)
            values = list(range(n))
            values.reverse()
            colors = visualization.get_sequential_colors(values)

            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, colors=colors,ylim=get_ylim_normalized(measure))



class DatasetSubset(InvarianceExperiment):

    def description(self):
        return '''Vary the test dataset subset (either train o testing) and see how it affects the numpy's value.'''

    def run(self):
        dataset_subsets = [datasets.DatasetSubset.test, datasets.DatasetSubset.train]

        model_names = simple_models_generators
        measures = normalized_measures_validation
        combinations = list(itertools.product(
            model_names, dataset_names, common_transformations_combined, measures))

        for i, (model_config_generator, dataset, transformation, measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}", end=", ")

            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)

            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)

            p_datasets = []
            for subset in dataset_subsets:
                p = config.dataset_size_for_measure(measure, subset)
                p_datasets.append(measure_package.DatasetParameters(dataset, subset, p))
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            variance_parameters = [measure_package.Parameters(p_training.id(), p_dataset, transformation, measure) for
                                   p_dataset in p_datasets]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance)
            results = config.load_measure_results(config.results_paths(variance_parameters))
            labels = [f"{l.format_subset(d.subset)}" for d in p_datasets]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))


class DatasetTransfer(InvarianceExperiment):
    def description(self):
        return """Measure invariance with a different dataset than the one used to train the model."""

    def run(self):
        measures = normalized_measures_validation

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations_combined, measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)

            variance_parameters = []
            for dataset_test in dataset_names:
                p = 0.5 if measure.__class__ == tm.ANOVAInvariance else default_dataset_percentage
                p_dataset = measure_package.DatasetParameters(dataset_test, datasets.DatasetSubset.test, p)
                p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                self.experiment_measure(p_variance, adapt_dataset=True)
                variance_parameters.append(p_variance)

            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_measure_results(config.results_paths(variance_parameters))
            labels = dataset_names
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))
