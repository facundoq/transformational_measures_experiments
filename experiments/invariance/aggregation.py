from .common import *

class AggregationBeforeAfter(InvarianceExperiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate whether to aggregate before or after. Convolutional aggregation collapses all the spatial dimensions of feature maps before normalization so that a single variance value for the feature map can be obtained."""

    def run(self):
        before_functions = [ca_mean, ca_max]
        after_functions = [ca_sum,ca_mean, ca_max]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)

        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = measure.DatasetParameters(dataset, datasets.DatasetSubset.test, default_dataset_percentage)

            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            measures = [tm.NormalizedVarianceInvariance(ca) for ca in before_functions]
            variance_parameters = [measure.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            # model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance)
            normal_results = self.load_measure_results(self.results_paths(variance_parameters))

            ca_none_variance_parameter = measure.Parameters(p_training.id(), p_dataset, transformation,
                                                            tm.NormalizedVarianceInvariance(ca_none))
            self.experiment_measure(ca_none_variance_parameter)
            ca_none_variance_parameters = [ca_none_variance_parameter] * len(after_functions)
            ca_none_results = self.load_measure_results(self.results_paths(ca_none_variance_parameters))
            ca_none_results = [ca.apply(r) for ca,r in zip(after_functions , ca_none_results)]


            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            all_results = normal_results + ca_none_results
            labels = [f"{l.aggregation}: {ca.f}, {l.after_normalization}." for ca in before_functions] + [
                f"{l.aggregation}: {ca.f}, {l.before_normalization}." for ca in after_functions ]
            plot_collapsing_layers_same_model(all_results, plot_filepath, labels=labels)


class AggregationFunctionsVariance(InvarianceExperiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate their differences. Convolutional aggregation collapses all the spatial dimensions of feature maps before normalization so that a single variance value for the feature map can be obtained."""

    def run(self):
        before_functions = [ca_mean, ca_max]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)

        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = measure.DatasetParameters(dataset, datasets.DatasetSubset.test, default_dataset_percentage)

            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            measures = [tm.NormalizedVarianceInvariance(ca) for ca in before_functions]
            variance_parameters = [measure.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            # model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance)
            results = self.load_measure_results(self.results_paths(variance_parameters))

            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            labels = [f"{l.aggregation}: {l.format_aggregation(ca.f)}, {l.before_normalization}." for ca in before_functions]

            plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)

class AggregationFunctionsDistance(InvarianceExperiment):
    def description(self):
        return """Test different Convolutional Aggregation strategies for Distance-based invariance measures."""

    def run(self):
        measures = [  tm.NormalizedDistanceInvariance(da, ca_none)
                    , tm.NormalizedDistanceInvariance(da, ca_mean)
                    , tm.NormalizedDistanceInvariance(da_keep, ca_none)
                    ]
        labels = [l.normal,l.feature_map_aggregation,l.feature_map_distance]

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations)

        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = measure.DatasetParameters(dataset, datasets.DatasetSubset.test, default_dataset_percentage)

            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)

            variance_parameters = [measure.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_measure(p_variance)
            results = self.load_measure_results(self.results_paths(variance_parameters))
            results[0]=ca_mean.apply(results[0])

            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
