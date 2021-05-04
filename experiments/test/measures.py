import transformational_measures.measure
from ..invariance.common import *

class ValidateMeasure(InvarianceExperiment):

    def description(self):
        return """Validate numpy/transformation. Just for testing purposes."""

    def run(self):
        measures = [
            # tm.TransformationVarianceMeasure(ca_sum),
            # tm.SampleVarianceMeasure( ca_sum),
            # tm.NormalizedVarianceMeasure( ca_sum),

            # tm.TransformationMeasure(ca_sum),
            # tm.SampleMeasure( ca_sum),
            # tm.NormalizedMeasure(ca_sum),
            # tm.NormalizedVariance(ca_sum),
            #tm.NormalizedDistance(da,ca_sum)
            tm.GoodfellowNormalInvariance(),
        ]
        # model_names = ["VGG16D","VGG16DBN"]
        model_names = [config.TIPoolingSimpleConvConfig]
        dataset_names = ["mnist"]
        r,s,t=common_transformations
        transformations = r
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, measures, transformations)
        for model_config_generator, dataset, measure, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False, t=transformation)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, tm.SimpleAffineTransformationGenerator(), epochs, 0,
                                             savepoints=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100])
            experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{transformation.id()}"
            p_dataset = measure.DatasetParameters(dataset, measure.DatasetSubset.test, default_dataset_percentage)
            p_measure = measure.Parameters(p_training.id(), p_dataset, transformation, measure)

            self.experiment_training(p_training)
            model_path = config.model_path(p_training)

            self.experiment_measure(p_measure)
            print(experiment_name)
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n transformation: {transformation.id()} "
            result = config.load_experiment_result(config.results_path(p_measure))
            print(config.results_path(p_measure))
            visualization.plot_heatmap(result.measure_result, plot_filepath)


class ValidateGoodfellow(InvarianceExperiment):

    def description(self):
        return """Validate goodfellow. Just for testing purposes."""

    def run(self):

        model_names = [config.SimpleConvConfig]
        dataset_names = ["mnist"]
        r,s,t=common_transformations
        transformations = [r]

        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names,  transformations)
        for model_config_generator, dataset,  transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            measure = tm.GoodfellowNormalInvariance()
            # train
            experiment_name = f"{model_config.name}_{dataset}__{transformation.id()}"

            p_training,p_variance,p_dataset = self.train_measure(model_config,dataset,transformation,measure,p=0.05)

            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            plot_filepath_global = self.folderpath / f"{experiment_name}_global.jpg"
            plot_filepath_local = self.folderpath / f"{experiment_name}_local.jpg"

            experiment_result = config.load_experiment_result(config.results_path(p_variance))

            result: transformational_measures.measure.MeasureResult =experiment_result.measure_result
            global_result: transformational_measures.measure.MeasureResult = result.extra_values[tm.GoodfellowNormalInvariance.g_key]
            local_result: transformational_measures.measure.MeasureResult =result.extra_values[tm.GoodfellowNormalInvariance.l_key]
            visualization.plot_heatmap(result, plot_filepath)
            visualization.plot_heatmap(global_result, plot_filepath_global)
            visualization.plot_heatmap(local_result, plot_filepath_local)

            plot_filepath_thresholds = self.folderpath / f"{experiment_name}_thresholds.jpg"
            thresholds = global_result.extra_values[tm.GoodfellowNormalGlobalInvariance.thresholds_key]
            thresholds_result= transformational_measures.measure.MeasureResult(thresholds, result.layer_names, global_result.measure)
            visualization.plot_heatmap(thresholds_result, plot_filepath_thresholds)

