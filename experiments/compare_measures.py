from .common import *
import experiment.measure as measure_package

class CompareMeasures(Experiment):
    def description(self):
        return """Test different measures for a given dataset/model/transformation combination to evaluate their differences."""

    def run(self):

        measure_sets = {"Variance": [
            tm.TransformationVariance(),
            tm.SampleVariance(),
        ],
            "Distance": [
               tm.TransformationDistance(da),
               tm.SampleDistance(da),
            ],
            "HighLevel": [
                tm.AnovaMeasure(0.99, bonferroni=True),
                tm.NormalizedVariance(ca_sum),
                tm.NormalizedDistance(da_keep, ca_mean),
                tm.GoodfellowNormal(alpha=0.99),
            ],
            "Normalized": [
                tm.NormalizedVariance(ca_sum),
                tm.NormalizedDistance(da_keep, ca_mean),
            ],
            "NormalizedVariance": [
                tm.NormalizedVariance(ca_sum),
            ],
            "Basic": [
                tm.AnovaMeasure(0.99, bonferroni=True),
                tm.GoodfellowNormal(alpha=0.99),
            ],
            "Equivariance": [
                tm.NormalizedVarianceSameEquivariance(ca_mean),
                tm.NormalizedDistanceSameEquivariance(da_normalize_keep),
                tm.DistanceSameEquivarianceSimple(df_normalize),
            ]
        }

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        # model_generators = common_models_generators
        model_generators = simple_models_generators
        # model_names = ["SimpleConv"]
        transformations = common_transformations_combined

        combinations = itertools.product(model_generators, dataset_names, transformations, measure_sets.items())
        for (model_config_generator, dataset, transformation, measure_set) in combinations:
            # train model with data augmentation and without
            variance_parameters_both = []
            for t in [identity_transformation, transformation]:

                model_config = model_config_generator.for_dataset(dataset)
                epochs = config.get_epochs(model_config, dataset, t)
                p_training = training.Parameters(model_config, dataset, t, epochs, 0)
                self.experiment_training(p_training)

                # generate variance params
                variance_parameters = []
                measure_set_name, measures = measure_set
                for measure in measures:
                    p = config.dataset_size_for_measure(measure)
                    p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
                    p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                    variance_parameters.append(p_variance)
                # evaluate variance
                model_path = config.model_path(p_training)


                for p_variance in variance_parameters:
                    self.experiment_measure(p_variance)
                variance_parameters_both.append(variance_parameters)

            variance_parameters_id = variance_parameters_both[0]
            variance_parameters_data_augmentation = variance_parameters_both[1]
            variance_parameters_all = variance_parameters_id + variance_parameters_data_augmentation
            # plot results
            experiment_name = f"{measure_set_name}_{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_measure_results(config.results_paths(variance_parameters_all))
            labels = [l.measure_name(m) + f" ({l.no_data_augmentation})" for m in measures] + [l.measure_name(m) for m in measures]
            n = len(measures)
            # cmap = visualization.discrete_colormap(n=n)
            cmap = visualization.default_discrete_colormap()
            color = cmap(range(n))
            colors = np.vstack([color, color])
            linestyles = ["--" for i in range(n)] + ["-" for i in range(n)]
            ylim = self.get_ylim(measure_set_name, dataset)
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,linestyles=linestyles,colors=colors, ylim=ylim)

    def get_ylim(self, measure_set_name, dataset):
        if measure_set_name == "Distance":
            return 100 if dataset == "mnist" else 2000
        elif measure_set_name == "Variance":
            return 100 if dataset == "mnist" else 175
        elif measure_set_name == "HighLevel":
            return 1.2 if dataset == "mnist" else 1.2
        elif measure_set_name == "Equivariance":
            return None
            # return 8 if dataset == "mnist" else 8
        else:
            return None


class CompareGoodfellowAlpha(Experiment):

    def description(self):
        return """Compare goodfellow alpha values"""

    def run(self):

        model_names = simple_models_generators
        alphas = [0.5, 0.9, 0.95, 0.99, 0.999]
        measures = [tm.GoodfellowNormal(alpha) for alpha in alphas]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            variance_parameters = []
            for measure in measures:
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_measure_results(config.results_paths(variance_parameters))
            labels = [f"α={alpha}" for alpha in alphas]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class CompareGoodfellow(Experiment):

    def description(self):
        return """Compare goodfellow alpha values"""

    def run(self):
        model_names = simple_models_generators
        measure = tm.GoodfellowNormal(0.999)
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        labels = ["Local", "Global"]
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
            result = config.load_experiment_result(config.results_path(p_variance)).measure_result
            local_result, global_result = result.extra_values[tm.GoodfellowNormal.l_key], result.extra_values[
                tm.GoodfellowNormal.g_key]
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            visualization.plot_collapsing_layers_same_model([local_result, global_result], plot_filepath,labels=labels, ylim=0.1)




class CompareSameEquivarianceNormalized(Experiment):

    def description(self):
        return """Compare same equivariance normalized measures"""

    def run(self):

        model_names = simple_models_generators

        measures = [tm.DistanceSameEquivarianceSimple(df_normalize),
                    tm.NormalizedDistanceSameEquivariance(da_keep),
                    tm.NormalizedVarianceSameEquivariance(ca_mean)]
        #labels = [l.simple,l.normalized_distance,l.normalized_variance]
        labels = ["Auto-Equivarianza Simple","AutoEquivarianza por Distancia", "AutoEquivarianza por Varianza"]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            variance_parameters = []
            for measure in measures:
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_measure_results(config.results_paths(variance_parameters))

            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class CompareSameEquivarianceSimple(Experiment):

    def description(self):
        return """Compare same equivariance normalized measures"""

    def run(self):

        model_names = simple_models_generators

        measures = [tm.DistanceSameEquivarianceSimple(df),
                    tm.DistanceSameEquivarianceSimple(df_normalize),
                    ]
        labels = [l.unnormalized,l.normalized]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            variance_parameters = []
            for measure in measures:
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_measure_results(config.results_paths(variance_parameters))

            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
class CompareSameEquivariance(Experiment):

    def description(self):
        return """Compare transformational and sample same equivariance measures"""

    def run(self):

        model_names = simple_models_generators

        labels = [l.transformational,l.sample_based]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            measure = tm.NormalizedVarianceSameEquivariance(ca_mean)
            p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            normalized_results = config.load_measure_result(config.results_path(p_variance))

            transformation_results=normalized_results.extra_values[tm.NormalizedVarianceSameEquivariance.transformation_key]
            sample_results=normalized_results.extra_values[tm.NormalizedVarianceSameEquivariance.sample_key]
            results=[transformation_results,sample_results]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
