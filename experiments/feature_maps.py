from .common import *
import torch
import datasets
class VisualizeInvariantFeatureMaps(Experiment):
    def description(self):
        return """Visualize the output of invariant feature maps, to analyze qualitatively if they are indeed invariant."""

    def run(self):
        measures = normalized_measures
        # conv_model_names = [m for m in common_model_names if (not "FFNet" in m)]
        conv_model_names = simple_models_generators  # [models.SimpleConv.__name__]

        transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                           tm.SimpleAffineTransformationGenerator(s=4),
                           tm.SimpleAffineTransformationGenerator(t=3),
                           ]
        combinations = itertools.product(
            conv_model_names, dataset_names, transformations, measures)
        for (model_config_generator, dataset_name, transformation, measure) in combinations:
            model_config = model_config_generator.for_dataset(dataset_name)
            # train
            epochs = config.get_epochs(model_config, dataset_name, transformation)
            p_training = training.Parameters(model_config, dataset_name, transformation, epochs)

            experiment_name = f"{model_config.name}_{dataset_name}_{transformation.id()}_{measure.id()}"
            plot_folderpath = self.plot_folderpath / experiment_name
            finished = Path(plot_folderpath) / "finished"
            if finished.exists():
                continue
            # train
            self.experiment_training(p_training)
            p = config.dataset_size_for_measure(measure)
            p_dataset = variance.DatasetParameters(dataset_name, variance.DatasetSubset.test, p)
            p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
            model_path = config.model_path(p_training)
            self.experiment_measure(p_variance, model_path)

            model_filepath = config.model_path(p_training)
            model, p_model, o, scores = training.load_model(model_filepath, use_cuda=torch.cuda.is_available())
            result_filepath = config.results_path(p_variance)
            result = config.load_result(result_filepath)
            dataset = datasets.get(dataset_name)

            plot_folderpath.mkdir(parents=True, exist_ok=True)

            visualization.plot_invariant_feature_maps_pytorch(plot_folderpath, model, dataset, transformation, result,images=2, most_invariant_k=4, least_invariant_k=4,conv_aggregation=ca_mean)
            finished.touch()