from .common import *
from ..tasks import train, Task

from ..visualization.accuracies import plot_accuracies,plot_metrics_single_model

class TrainModels(SameEquivarianceExperiment):

    def description(self):
        return """Train models and check their performance."""

    def run(self):
        savepoints_percentages = [0, 1, 2, 5, 10, 25, 50, 75, 100]
        model_generators = simple_models_generators
        transformation_sets = common_transformations_combined
        l=self.l
        transformation_labels = [l.rotation, l.scale, l.translation,l.combined]
        task = Task.TransformationRegression
        metric = "rae"
        for dataset, model_config_generator in itertools.product(dataset_names,model_generators):
            transformation_scores = []
            for transformations in transformation_sets:
                # train
                mc: ModelConfig = model_config_generator.for_dataset(task,dataset)
                transformations:tm.TransformationSet=transformations
                epochs = mc.epochs(dataset, task, transformations)
                cc = train.MaxMetricConvergence(mc.max_rae(dataset, task, transformations),metric)
                savepoints = [sp * epochs // 100 for sp in savepoints_percentages]
                savepoints = sorted(list(set(savepoints)))
                optimizer = dict(optim="adam", lr=0.0001)
                tc = train.TrainConfig(epochs, cc,optimizer=optimizer, savepoints=savepoints,verbose=False,num_workers=4)

                p = train.TrainParameters(mc, tc, dataset, transformations, task)
                self.train(p)

                p, model, metrics = train.load_model(self.model_path_new(p), device="cpu", load_state=False)
                # print(metrics)
                # v = metrics[f"test_{metric}"]
                # # print(f"{v:.3f}")
                # print(metric)
                transformation_scores.append(metrics[f"test_{metric}"])

            experiment_name = f"{model_config_generator.__name__}_{dataset}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            plot_metrics_single_model(plot_filepath, transformation_scores, transformation_labels,metric=metric)


    # def train(self,p:train.TrainParameters):
    #     if not self.model_trained(p):
    #         print(f"Training model {p.id()} for {p.tc.epochs} epochs ({p.tc.convergence_criteria})...")
    #         train.train(p, self)
    #     else:
    #         print(f"Model {p.id()} already trained.")



