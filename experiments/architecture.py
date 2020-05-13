from .common import *
from models.simple_conv import ActivationFunction as af

class BatchNormalization(Experiment):
    def description(self):
        return """Compare invariance of models trained with/without batchnormalization."""

    def run(self):
        measures = normalized_measures
        models = simple_models_generators
        combinations = itertools.product(
            models, dataset_names, common_transformations, measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            # train

            variance_parameters = []
            for bn in [True, False]:
                model_config = model_config_generator.for_dataset(dataset, bn=bn)
                epochs = config.get_epochs(model_config, dataset, transformation)
                p_training = training.Parameters(model_config, dataset, transformation, epochs)
                self.experiment_training(p_training)

                p = config.dataset_size_for_measure(measure)
                p_dataset = measure.DatasetParameters(dataset, measure.DatasetSubset.test, p)
                p_variance = measure.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                batch_size = 64
                if model_config.name.startswith("ResNet"):
                    batch_size = 32
                self.experiment_measure(p_variance, model_path, batch_size=batch_size)
                variance_parameters.append(p_variance)

            # plot results
            bn_result, result = config.load_measure_results(config.results_paths(variance_parameters))
            layer_names = bn_result.layer_names
            bn_indices = [i for i, n in enumerate(layer_names) if n.endswith("bn")]
            # single
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            visualization.plot_collapsing_layers_same_model([bn_result], plot_filepath, mark_layers=bn_indices)

            # comparison
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}_comparison"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            bn_result = bn_result.remove_layers(bn_indices)
            labels = [l.with_bn,l.without_bn]
            visualization.plot_collapsing_layers_same_model([bn_result, result], plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))


class ActivationFunction(Experiment):
    def description(self):
        return """Determine how the activation function affects invariance"""

    def run(self):
        measures = normalized_measures
        activation_functions = list(af)

        combinations = itertools.product(dataset_names, common_transformations, measures)
        for (dataset, transformation, measure) in combinations:

            variance_parameters = []
            for activation_function in activation_functions:
                model_config = config.SimpleConvConfig.for_dataset(dataset, activation=activation_function)
                p_training,p_variance,p_dataset=self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)

            # plot results
            results= config.load_measure_results(config.results_paths(variance_parameters))
            # single
            experiment_name = f"{models.SimpleConv.__name__}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            labels = [a.value for a in activation_functions]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath,labels=labels,ylim=get_ylim_normalized(measure))


class KernelSize(Experiment):
    def description(self):
        return """Determine how the kernel sizes affect invariance"""

    def run(self):
        measures = normalized_measures
        kernel_sizes = [3,5,7]

        combinations = itertools.product(dataset_names, common_transformations, measures)
        for (dataset, transformation, measure) in combinations:

            variance_parameters = []
            for k in kernel_sizes:
                model_config = config.SimpleConvConfig.for_dataset(dataset, k=k)
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)

            # plot results
            results = config.load_measure_results(config.results_paths(variance_parameters))
            # single
            experiment_name = f"{models.SimpleConv.__name__}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            labels = [f"k={k}" for k in kernel_sizes]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))


class MaxPooling(Experiment):
    def description(self):
        return """Determine wheter MaxPooling affects the invariance structure of the network or it is similar to a network with strided convolutions"""

    def run(self):
        measures = normalized_measures
        max_pooling = [True,False]

        combinations = itertools.product(dataset_names, common_transformations, measures)
        for (dataset, transformation, measure) in combinations:

            variance_parameters = []
            for mp in max_pooling :
                model_config = config.SimpleConvConfig.for_dataset(dataset, max_pooling=mp)
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)

            # plot results
            results = config.load_measure_results(config.results_paths(variance_parameters))
            # single
            experiment_name = f"{models.SimpleConv.__name__}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            labels = [l.maxpooling,l.strided_convolution]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))

