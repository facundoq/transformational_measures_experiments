from .common import *
from models.simple_conv import ActivationFunction as af
import experiment.measure as measure_package
import datasets

class BatchNormalization(InvarianceExperiment):
    def description(self):
        return """Compare invariance of models trained with/without batchnormalization."""

    def run(self):
        measures = normalized_measures
        models = simple_models_generators
        combinations = itertools.product(
            models, dataset_names, common_transformations, measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            # train

            results = []
            for bn in [True, False]:
                model_config = model_config_generator.for_dataset(Task.Classification,dataset,  bn=bn)
                mc,tc,p,model_path = self.train_default(Task.Classification,dataset,transformation,model_config)

                result = self.measure_default(dataset,mc.id(),model_path,transformation,measure,default_measure_options,default_dataset_percentage)
                
                results.append(result)

            # plot results
            bn_result, result = results
            layer_names = bn_result.layer_names
            bn_indices = [i for i, n in enumerate(layer_names) if n.endswith("bn")]
            # single
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            tmv.plot_collapsing_layers_same_model([bn_result], plot_filepath, mark_layers=bn_indices)

            # comparison
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}_comparison"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            bn_result = bn_result.remove_layers(bn_indices)
            labels = [l.with_bn,l.without_bn]
            tmv.plot_collapsing_layers_same_model([bn_result, result], plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))


class ActivationFunctionComparison(InvarianceExperiment):
    def description(self):
        return """Determine how the activation function affects invariance"""

    def run(self):
        measures = normalized_measures
        activation_functions = [af.ELU,af.ReLU,af.PReLU,af.Tanh]

        combinations = itertools.product(dataset_names, common_transformations, measures)
        for (dataset, transformation, measure) in combinations:
            
            results = []
            for activation_function in activation_functions:
                model_config = SimpleConvConfig.for_dataset(Task.Classification,dataset,  activation=activation_function)
                mc,tc,p,model_path = self.train_default(Task.Classification,dataset,transformation,model_config)

                result = self.measure_default(dataset,mc.id(),model_path,transformation,measure,default_measure_options,default_dataset_percentage)
                
                results.append(result)
            # single
            experiment_name = f"{SimpleConvConfig.__name__}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            labels = [a.value for a in activation_functions]
            tmv.plot_collapsing_layers_same_model(results, plot_filepath,labels=labels,ylim=get_ylim_normalized(measure))


class KernelSize(InvarianceExperiment):
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
            results = self.load_measure_results(self.results_paths(variance_parameters))
            # single
            experiment_name = f"{models.SimpleConv.__name__}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            labels = [f"k={k}" for k in kernel_sizes]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))


class MaxPooling(InvarianceExperiment):
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
            results = self.load_measure_results(self.results_paths(variance_parameters))
            # single
            experiment_name = f"{models.SimpleConv.__name__}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            labels = [l.maxpooling,l.strided_convolution]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,ylim=get_ylim_normalized(measure))

