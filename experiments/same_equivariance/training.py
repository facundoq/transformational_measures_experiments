from .common import *
from ..tasks import train, Task

from ..visualization.accuracies import plot_accuracies,plot_metrics_single_model
import torch


class TrainModels(SameEquivarianceExperiment):

    def description(self):
        return """Train models and check their performance."""

    def run(self):
        model_generators = simple_models_generators
        transformation_sets = common_transformations_combined
        l=self.l
        transformation_labels = [l.rotation, l.scale, l.translation,l.combined]
        task = Task.TransformationRegression
        device = torch.device("cpu")
        for dataset, model_config_generator in itertools.product(dataset_names,model_generators):
            transformation_scores = []
            for transformations in transformation_sets:
                
                # train
                mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
                tc,metric = self.get_train_config(mc,dataset,task,transformations)
                p = train.TrainParameters(mc, tc, dataset, transformations, task)
                self.train(p)

                p, model, metrics = train.load_model(self.model_path_new(p), device=device, load_state=False)
                transformation_scores.append(metrics[f"test_{metric}"])

            experiment_name = f"{model_config_generator.__name__}_{dataset}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            plot_metrics_single_model(plot_filepath, transformation_scores, transformation_labels,metric=metric)

