from .common import *
from .weights import DuringTraining

class TrainModels(Experiment):

    def description(self):
        return """Analyze the evolution of invariance in models while they are trained."""

    def run(self):
        model_generators = simple_models_generators
        combinations = itertools.product(
            model_generators, dataset_names, common_transformations)
        for model_config_generator, dataset, transformation in combinations:

            # train
            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            savepoints = [sp * epochs // 100 for sp in DuringTraining.savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))


            # Training
            p_training = training.Parameters(model_config, dataset, transformation, epochs, savepoints=savepoints)
            self.experiment_training(p_training)




class ModelAccuracies(Experiment):
    def description(self):
        return """Compare the accuracies of the models for each set of transformations"""

    def run(self):
        models = common_models_generators
        transformations = common_transformations+[tm.SimpleAffineTransformationGenerator(r=360,s=4,t=3)]
        model_names = [m.for_dataset("mnist").name for m in models]
        transformation_labels = ["Rotation","Scale","Translation","Combined"]
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
                    model, p, o, scores = training.load_model(config.model_path(p_training), load_state=False,
                                                              use_cuda=False)
                    loss, acc = scores["test"]
                    model_scores.append(acc)
                transformation_scores.append(model_scores)
            # plot results
            experiment_name = f"{dataset}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"

            visualization.plot_accuracies(plot_filepath, transformation_scores, transformation_labels , model_names)


class CompareModels(Experiment):
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
        # transformations = [tm.SimpleAffineTransformationGenerator(r=360)]

        transformations = common_transformations_hard

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
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                model_path = config.model_path(p_training)
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [m.name for m in model_configs]
            visualization.plot_collapsing_layers_different_models(results, plot_filepath, labels=labels,markers=self.fc_layers_indices(results))

    def fc_layers_indices(self,results:[VarianceExperimentResult])->[[int]]:
        indices=[]
        for r in results:
            layer_names = r.measure_result.layer_names
            n=len(layer_names)
            index = n
            for i in range(n):
                if "fc" in layer_names[i] or "PoolOut" in layer_names[i] or "fc" in layer_names[i]:
                    index=n
            indices.append(list(range(index,n)))
        return indices
