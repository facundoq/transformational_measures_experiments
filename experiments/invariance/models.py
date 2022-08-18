from .common import *
from .weights import DuringTraining

from ..visualization import accuracies
import experiment.measure as measure_package
import datasets

from ..tasks import train, Task


class TrainModels(InvarianceExperiment):

    def description(self):
        return """Train models and record intermediate instances of them to later
         analize the evolution of invariance while they are training."""

    def run(self):
        model_generators = simple_models_generators
        

        transformation_sets = common_transformations_combined+[identity_transformation]
        transformation_labels = [l.rotation, l.scale, l.translation, l.combined, "id"]

        combinations = itertools.product(model_generators,dataset_names)
        task = Task.Classification
        device = torch.device("cpu")

        for model_config_generator, dataset in combinations:
            transformation_scores = []
            for transformations in transformation_sets:
                # train
                mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
                tc,metric = self.get_train_config(mc,dataset,task,transformations,savepoints=True,verbose=True,batch_size=512)
                
                p = train.TrainParameters(mc, tc, dataset, transformations, task)
                self.train(p)
                p, model, metrics = train.load_model(self.model_path_new(p), device=device, load_state=False)
                transformation_scores.append(metrics[f"test_{metric}"])
                
            experiment_name = f"{dataset}_{model_config_generator}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            accuracies.plot_metrics_single_model( transformation_scores, transformation_labels)
            self.savefig(plot_filepath)

class SimpleConvAccuracies(InvarianceExperiment):
    def description(self):
        return """Compare the accuracies of the SimpleConv model for each set of transformations"""

    def run(self):
        model_config_generator = SimpleConvConfig
        transformation_sets = common_transformations
        task = Task.Classification
        transformation_labels = [l.rotation,l.scale,l.translation]
        for dataset in dataset_names:
            transformation_scores = []
            for transformations in transformation_sets:
                mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
                tc,metric = self.get_train_config(mc,dataset,task,transformations,savepoints=True,verbose=True,batch_size=512)
                
                p = train.TrainParameters(mc, tc, dataset, transformations, task)
                self.train(p)

                p, model, metrics = train.load_model(self.model_path_new(p), device=device, load_state=False)
                transformation_scores.append(metrics[f"test_{metric}"])

            # plot results
            experiment_name = f"{dataset}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            accuracies.plot_metrics_single_model(plot_filepath, transformation_scores, transformation_labels)


class ModelAccuracies(InvarianceExperiment):
    def description(self):
        return """Compare the accuracies of the models for each set of transformations"""

    def run(self):
        models = common_models_generators
        transformations = common_transformations_combined
        task = Task.Classification
        model_names = [m.for_dataset(task,"mnist").name() for m in models]
        transformation_labels = [l.rotation,l.scale,l.translation,l.combined]

        for dataset in dataset_names:
            transformation_scores = []
            for transformation in transformations:
                model_scores = []

                for model_config_generator in models:
                    mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
                    tc,metric = self.get_train_config(mc,dataset,task,transformation,savepoints=True,verbose=True,batch_size=256)
                    
                    p = train.TrainParameters(mc, tc, dataset, transformation, task)
                    self.train(p)

                    p, model, metrics = train.load_model(self.model_path_new(p), device=device, load_state=False)
                    model_scores.append(metrics[f"test_{metric}"])
              
                transformation_scores.append(model_scores)
            # plot results
            experiment_name = f"{dataset}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            transformation_scores = np.array(transformation_scores)
            accuracies.plot_accuracies(plot_filepath, transformation_scores, transformation_labels, model_names)


class CompareModels(InvarianceExperiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""

    def run(self):
        measures = normalized_measures

        models = [
            

        ]
        transformations = common_transformations
        model_names = [m.for_dataset("mnist").name for m in models]
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
                p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
                p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                self.experiment_measure(p_variance)

            # plot results
            experiment_name = f"{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            results = self.load_measure_results(self.results_paths(variance_parameters))
            tmv.plot_average_activations_different_models(results, plot_filepath, labels=model_names,
                                                                  markers=self.fc_layers_indices(results),ylim=get_ylim_normalized(measure))

    def fc_layers_indices(self, results: list[tm.MeasureResult]) -> list[list[int]]:
        indices = []
        for r in results:
            layer_names = r.layer_names
            n = len(layer_names)
            index = n
            for i in range(n):
                if "fc" in layer_names[i] or "PoolOut" in layer_names[i] or "fc" in layer_names[i]:
                    index = n
            indices.append(list(range(index, n)))
        return indices
