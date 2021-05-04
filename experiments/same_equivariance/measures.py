from .common import *



class CompareSameEquivarianceNormalized(SameEquivarianceExperiment):

    def description(self):
        return """Compare same equivariance normalized measures"""

    def run(self):

        model_names = simple_models_generators

        measures = [tm.DistanceSameEquivarianceSimple(df_normalize),
                    # tm.NormalizedDistanceSameEquivariance(da_keep),
                    tm.NormalizedVarianceSameEquivariance(ca_mean)]

        labels = [self.l.simple_sameequivariance,self.l.normalized_variance_sameequivariance]

        combinations = itertools.product(model_names, dataset_names,common_transformations)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(Task.Regression,dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            variance_parameters = []
            for measure in measures:
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure,Task.Regression)
                variance_parameters.append(p_variance)
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            results = self.load_measure_results_p(variance_parameters)
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
