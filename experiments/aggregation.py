from .common import *

class AggregationBeforeAfter(Experiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate whether to aggregate before or after. Convolutional aggregation collapses all the spatial dimensions of feature maps before normalization so that a single variance value for the feature map can be obtained."""

    def run(self):
        before_functions = [ca_mean, ca_max]
        after_functions = [ca_sum,ca_mean, ca_max]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)

        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)

            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            measures = [tm.NormalizedVariance(ca) for ca in before_functions]
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance, model_path)
            normal_results = config.load_results(config.results_paths(variance_parameters))

            ca_none_variance_parameter = variance.Parameters(p_training.id(), p_dataset, transformation,
                                                             tm.NormalizedVariance(ca_none))
            self.experiment_measure(ca_none_variance_parameter, model_path)
            ca_none_variance_parameters = [ca_none_variance_parameter] * len(after_functions)
            ca_none_results = config.load_results(config.results_paths(ca_none_variance_parameters))
            for (ca, r) in zip(after_functions , ca_none_results):
                r.measure_result = r.measure_result.collapse_convolutions(ca)

            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"

            all_results = normal_results + ca_none_results
            labels = [f"{l.aggregation}: {ca.value}, {l.after_normalization}." for ca in before_functions] + [
                f"{l.aggregation}: {ca.value}, {l.before_normalization}." for ca in after_functions ]
            visualization.plot_collapsing_layers_same_model(all_results, plot_filepath, labels=labels)


class AggregationFunctionsVariance(Experiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate their differences. Convolutional aggregation collapses all the spatial dimensions of feature maps before normalization so that a single variance value for the feature map can be obtained."""

    def run(self):
        before_functions = [ca_mean, ca_max]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)

        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)

            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            measures = [tm.NormalizedVariance(ca) for ca in before_functions]
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance, model_path)
            results = config.load_results(config.results_paths(variance_parameters))

            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            labels = [f"{l.aggregation}: {l.format_aggregation(ca)}, {l.before_normalization}." for ca in before_functions]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)

class AggregationFunctionsDistance(Experiment):
    def description(self):
        return """Test different Convolutional Aggregation strategies for Distance-based invariance measures."""

    def run(self):
        measures = [  tm.NormalizedDistance(da, ca_none)
                    , tm.NormalizedDistance(da, ca_mean)
                    , tm.NormalizedDistance(da_keep, ca_none)
                    ]
        labels = [l.normal,l.feature_map_aggregation,l.feature_map_distance]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)

        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)

            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)

            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance, model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            results[0].measure_result=results[0].measure_result.collapse_convolutions(ca_mean)

            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"

            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
