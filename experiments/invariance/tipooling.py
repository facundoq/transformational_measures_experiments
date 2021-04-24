import transformational_measures.measure
from .common import *
from transformational_measures.numpy.stats_running import RunningMeanWelford
import experiment.measure as measure_package

class TIPooling(InvarianceExperiment):
    def description(self):
        return """Compare SimpleConv with the TIPooling SimpleConv Network"""

    def run(self):
        measures = normalized_measures



        transformations =common_transformations

        combinations = itertools.product(dataset_names, transformations, measures)
        for (dataset, transformation, measure) in combinations:

            siamese = config.TIPoolingSimpleConvConfig.for_dataset(dataset, bn=False, t=transformation)
            siamese_epochs = config.get_epochs(siamese, dataset, transformation)
            p_training_siamese = training.Parameters(siamese, dataset,identity_transformation,siamese_epochs, 0)

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
                p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
                p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                model_path = config.model_path(p_training)
                self.experiment_measure(p_variance)

            model, _, _, _ = config.load_model(p_training_siamese,use_cuda=False,load_state=False)
            model: models.TIPoolingSimpleConv = model
            results = config.load_measure_results(config.results_paths(variance_parameters))
            results[0] = self.average_paths_tipooling(model, results[0])
            # plot results
            # print("simpleconv",len(results[1].measure_result.layer_names),results[1].measure_result.layer_names)
            labels = ["TIPooling SimpleConv", "SimpleConv"]
            experiment_name = f"{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            mark_layers = range(model.layer_before_pooling_each_transformation())
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,mark_layers=mark_layers,ylim=get_ylim_normalized(measure))

    def average_paths_tipooling(self, model: models.TIPoolingSimpleConv, result: transformational_measures.measure.MeasureResult) -> transformational_measures.measure.MeasureResult:
        # print(len(model.activation_names()),model.activation_names())
        # print(len(model.original_conv_names()),model.original_conv_names())
        # print(len(model.fc_names()), model.fc_names())
        # print(len(result.layers),len(result.layer_names))
        m = len(model.transformations)

        names=result.layer_names
        index=0
        for i,name in enumerate(names):
            if len(name)>5 and name[0]=="t" and name[4]=="_" :
                index=i+1
            else:
                break
        assert index>0
        assert index % m == 0

        k=index//m
        conv_layer_names = [name[5:] for name in names[:k]]
        other_layer_names = names[index:]
        layer_names=conv_layer_names+other_layer_names

        # average layer values
        layers = result.layers
        means_per_original_layer = [RunningMeanWelford() for i in range(k)]
        for i in range(m):
            start = i*k
            end = (i+1)*k
            block_layers = layers[start:end]
            for j in range(k):
                means_per_original_layer[j].update(block_layers[j])
        conv_layers = [m.mean() for m in means_per_original_layer]
        other_layers = layers[m * k:]
        layers = conv_layers + other_layers

        # print(len(layers),len(layer_names))
        # print(layer_names)
        return transformational_measures.measure.MeasureResult(layers, layer_names, result.measure)
