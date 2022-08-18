from .common import *
import torch
import datasets
import experiment.measure as measure_package
from pytorch.numpy_dataset import NumpyDataset
from tmeasures.pytorch import NormalPytorchActivationsIterator, ObservableLayersModule


class VisualizeInvariantFeatureMaps(InvarianceExperiment):
    def description(self):
        return """Visualize the output of invariant feature maps, to analyze qualitatively if they are indeed invariant."""

    def run(self):
        measures = normalized_measures
        # conv_model_names = [m for m in common_model_names if (not "FFNet" in m)]
        conv_model_names = simple_models_generators  # [models.SimpleConv.__name__]

        transformations = common_transformations

        combinations = itertools.product(
            conv_model_names, dataset_names, transformations, measures)
        for (model_config_generator, dataset_name, transformation, measure) in combinations:
            model_config = model_config_generator.for_dataset(dataset_name)
            # train
            epochs = config.get_epochs(model_config, dataset_name, transformation)
            p_training = training.Parameters(model_config, dataset_name, transformation, epochs)

            experiment_name = f"{model_config.name}_{dataset_name}_{transformation.id()}_{measure.id()}"
            plot_folderpath = self.folderpath / experiment_name
            finished = Path(plot_folderpath) / "finished"
            if finished.exists():
                continue
            # train
            self.experiment_training(p_training)
            p = config.dataset_size_for_measure(measure)
            p_dataset = measure_package.DatasetParameters(dataset_name, datasets.DatasetSubset.test, p)
            p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
            model_path = self.model_path(p_training)
            self.experiment_measure(p_variance)

            model_filepath = self.model_path(p_training)
            model, p_model, o, scores = training.load_model(model_filepath, use_cuda=torch.cuda.is_available())
            result_filepath = self.results_path(p_variance)
            result = self.load_experiment_result(result_filepath).measure_result
            dataset = datasets.get_classification(dataset_name)

            plot_folderpath.mkdir(parents=True, exist_ok=True)

            self.plot(plot_folderpath, model, dataset, transformation, result, images=2, most_invariant_k=4,
                      least_invariant_k=4, conv_aggregation=ca_mean)
            finished.touch()

    def plot(self, plot_folderpath: Path, model: ObservableLayersModule, dataset: datasets.ClassificationDataset,
             transformations: tm.TransformationSet, result: tm.MeasureResult, images=8, most_invariant_k: int = 4,
             least_invariant_k: int = 4, conv_aggregation=tm.numpy.AggregateTransformation()):

        numpy_dataset = NumpyDataset(dataset.x_test[:images, :], dataset.y_test[:images])
        iterator = tm.pytorch.NormalPytorchActivationsIterator(model, numpy_dataset, transformations, 32, 0,
                                                               torch.cuda.is_available())
        visualization.plot_invariant_feature_maps(plot_folderpath, iterator, result, most_invariant_k,
                                                  least_invariant_k, conv_aggregation)
