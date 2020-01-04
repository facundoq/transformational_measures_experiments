from .common import *
from transformation_measure.measure.stats_running import RunningMean

class TIPooling(Experiment):
    def description(self):
        return """Compare SimpleConv with the TIPooling SimpleConv Network"""

    def run(self):
        measures = normalized_measures



        transformations =common_transformations

        combinations = itertools.product(dataset_names, transformations, measures)
        for (dataset, transformation, measure) in combinations:

            siamese = config.TIPoolingSimpleConvConfig.for_dataset(dataset, bn=False, t=transformation)
            siamese_epochs = config.get_epochs(siamese, dataset, transformation)
            p_training_siamese = training.Parameters(siamese, dataset, tm.SimpleAffineTransformationGenerator(),
                                                     siamese_epochs, 0)

            normal = config.SimpleConvConfig.for_dataset(dataset, bn=False)
            normal_epochs = config.get_epochs(normal, dataset, transformation)
            p_training_normal = training.Parameters(normal, dataset, transformation, normal_epochs, 0)

            p_training_parameters = [p_training_siamese, p_training_normal]
            variance_parameters = []
            for p_training in p_training_parameters:
                # train
                self.experiment_training(p_training)
                # generate variance params
                p = config.dataset_size_for_measure(measure)
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                model_path = config.model_path(p_training)
                self.experiment_variance(p_variance, model_path)

            model, _, _, _ = config.load_model(p_training_siamese,use_cuda=False,load_state=False)
            model: models.TIPoolingSimpleConv = model
            results = config.load_results(config.results_paths(variance_parameters))
            results[0].measure_result = self.average_paths_tipooling(model, results[0].measure_result)
            # plot results
            # print("simpleconv",len(results[1].measure_result.layer_names),results[1].measure_result.layer_names)
            labels = ["TIPooling SimpleConv", "SimpleConv"]
            experiment_name = f"{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            mark_layers = range(model.layer_before_pooling_each_transformation())
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,mark_layers=mark_layers)

    def average_paths_tipooling(self, model: models.TIPoolingSimpleConv, result: tm.MeasureResult) -> tm.MeasureResult:
        k = model.layer_before_pooling_each_transformation()
        # print(len(model.activation_names()),model.activation_names())
        # print(len(model.original_conv_names()),model.original_conv_names())
        # print(len(model.fc_names()), model.fc_names())
        # print(len(result.layers),len(result.layer_names))
        m = len(model.transformations)

        # fix layer names
        layer_names = model.original_conv_names()+model.activation_names()[m*k:]
        # average layer values
        layers = result.layers
        means_per_original_layer = [RunningMean() for i in range(k)]
        for i in range(m):
            start = i*k
            end = (i+1)*k
            block_layers = layers[start:end]
            for j in range(k):
                means_per_original_layer[j].update(block_layers[j])
        conv_layers = [m.mean() for m in means_per_original_layer]
        other_layers = layers[m * k:]
        layers = conv_layers + other_layers
        # print(layer_names)
        # print(len(layers),len(layer_names))
        return tm.MeasureResult(layers, layer_names, result.measure)
