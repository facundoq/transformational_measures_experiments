from .common import *

class ValidateMeasure(Experiment):

    def description(self):
        return """Validate measure/transformation. Just for testing purposes."""

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
            #tm.GoodfellowNormalMeasure(),
            tm.GoodfellowGlobalVarianceNormal(),
        ]
        # model_names = ["VGG16D","VGG16DBN"]
        model_names = [config.TIPoolingSimpleConvConfig]
        dataset_names = ["mnist"]
        transformations = [tm.SimpleAffineTransformationGenerator(r=360)]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, measures, transformations)
        for model_config_generator, dataset, measure, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False, t=transformation)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, tm.SimpleAffineTransformationGenerator(), epochs, 0,
                                             savepoints=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100])
            experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{transformation.id()}"
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)
            p_measure = variance.Parameters(p_training.id(), p_dataset, transformation, measure)

            self.experiment_training(p_training)
            model_path = config.model_path(p_training)

            self.experiment_variance(p_measure, model_path)
            print(experiment_name)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"

            title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n transformation: {transformation.id()} "
            result = config.load_result(config.results_path(p_measure))
            print(config.results_path(p_measure))
            visualization.plot_heatmap(result.measure_result, plot_filepath)


class ValidateGoodfellow(Experiment):

    def description(self):
        return """Validate goodfellow. Just for testing purposes."""

    def run(self):

        model_names = [config.SimpleConvConfig]
        dataset_names = ["mnist"]
        transformations = [tm.SimpleAffineTransformationGenerator(r=360)]

        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names,  transformations)
        for model_config_generator, dataset,  transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            measure = tm.GoodfellowNormal()
            # train
            experiment_name = f"{model_config.name}_{dataset}__{transformation.id()}"

            p_training,p_variance,p_dataset = self.train_measure(model_config,dataset,transformation,measure,p=0.05)

            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            plot_filepath_global = self.plot_folderpath / f"{experiment_name}_global.jpg"
            plot_filepath_local = self.plot_folderpath / f"{experiment_name}_local.jpg"

            experiment_result = config.load_result(config.results_path(p_variance))

            result:tm.MeasureResult=experiment_result.measure_result
            global_result:tm.MeasureResult = result.extra_values[tm.GoodfellowNormal.g_key]
            local_result:tm.MeasureResult=result.extra_values[tm.GoodfellowNormal.l_key]
            visualization.plot_heatmap(result, plot_filepath)
            visualization.plot_heatmap(global_result, plot_filepath_global)
            visualization.plot_heatmap(local_result, plot_filepath_local)

            plot_filepath_thresholds = self.plot_folderpath / f"{experiment_name}_thresholds.jpg"
            thresholds = global_result.extra_values[tm.GoodfellowGlobalVarianceNormal.thresholds_key]
            thresholds_result=tm.MeasureResult(thresholds,result.layer_names,global_result.measure)
            visualization.plot_heatmap(thresholds_result, plot_filepath_thresholds)
