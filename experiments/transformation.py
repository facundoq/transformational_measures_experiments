from .common import *

class TransformationDiversity(Experiment):

    def description(self):
        return '''Vary the type of transformation both when training and computing the measure, and see how it affects the invariance. For example, train with rotations, then measure with translations. Train with translations. measure with scales, and so on. '''

    def run(self):
        measures = normalized_measures

        combinations = itertools.product(simple_models_generators, dataset_names, measures)
        transformations = [tm.SimpleAffineTransformationGenerator()] + common_transformations

        for model_config_generator, dataset, measure in combinations:
            for i, train_transformation in enumerate(transformations):
                # transformation_plot_folderpath = self.plot_folderpath / name
                # transformation_plot_folderpath.mkdir(exist_ok=True,parents=True)
                model_config = model_config_generator.for_dataset(dataset)

                variance_parameters = []
                print(f"Train transformation {train_transformation}")
                epochs = config.get_epochs(model_config, dataset, train_transformation)
                p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                self.experiment_training(p_training)
                for i, test_transformation in enumerate(transformations):
                    print(f"{i}, ", end="")
                    p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)
                    p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                    model_path = config.model_path(p_training)
                    self.experiment_variance(p_variance, model_path)
                    variance_parameters.append(p_variance)
                results = config.load_results(config.results_paths(variance_parameters))
                labels = [str(t.parameters.transformations) for t in results]
                experiment_name = f"{model_config.name}_{dataset}_{train_transformation}_{measure.id()}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
                title = f"Train transformation: {train_transformation.id()}"
                visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,title=title)


class TransformationComplexity(Experiment):

    def description(self):
        return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 8 rotations, measure variance with 2, 4, 8 and 16. """

    def run(self):
        measures = normalized_measures
        combinations = itertools.product(simple_models_generators, dataset_names, measures)

        test_sets = [config.rotation_transformations(4),
                     [tm.SimpleAffineTransformationGenerator(s=i) for i in [1, 3, 5]],
                     [tm.SimpleAffineTransformationGenerator(t=i) for i in [1, 3, 5]],
                     ]
        train_transformations = common_transformations

        for model_config_generator, dataset, measure in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            for train_transformation, transformation_set in zip(train_transformations, test_sets):
                # TRAIN
                epochs = config.get_epochs(model_config, dataset, train_transformation)
                p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                self.experiment_training(p_training)
                # MEASURE
                variance_parameters = []
                for k, test_transformation in enumerate(transformation_set):
                    p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)
                    p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                    model_path = config.model_path(p_training)
                    self.experiment_variance(p_variance, model_path)
                    variance_parameters.append(p_variance)
                # PLOT
                experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
                title = f"Train transformation: {train_transformation.id()}"
                labels = [f"Test transformation: {t}" for t in transformation_set]

                results = config.load_results(config.results_paths(variance_parameters))
                visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,title=title)


class TransformationComplexityDetailed(Experiment):

    def description(self):
        return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 8 rotations, measure variance with 2, 4, 8 and 16. """

    def run(self):
        measures = normalized_measures
        combinations = itertools.product(simple_models_generators, dataset_names, measures)

        names = ["rotation", "translation", "scale"]
        sets = [config.rotation_transformations(8),
                config.translation_transformations(4),
                config.scale_transformations(4)]

        for model_config_generator, dataset, measure in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            for i, (transformation_set, name) in enumerate(zip(sets, names)):
                n_experiments = (len(transformation_set) + 1) * len(transformation_set)
                print(f"    {name}, #experiments:{n_experiments}")
                # include identity the transformation set
                transformation_set = [tm.SimpleAffineTransformationGenerator()] + transformation_set
                for j, train_transformation in enumerate(transformation_set):
                    transformation_plot_folderpath = self.plot_folderpath / name

                    transformation_plot_folderpath.mkdir(exist_ok=True, parents=True)
                    experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
                    plot_filepath = transformation_plot_folderpath / f"{experiment_name}.jpg"
                    variance_parameters = []
                    print(f"{j}, ", end="")
                    epochs = config.get_epochs(model_config, dataset, train_transformation)
                    p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                    self.experiment_training(p_training)
                    for k, test_transformation in enumerate(transformation_set):
                        p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, default_dataset_percentage)
                        p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                        model_path = config.model_path(p_training)
                        self.experiment_variance(p_variance, model_path)
                        variance_parameters.append(p_variance)

                    title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n Train transformation: {train_transformation.id()}"
                    labels = [f"Test transformation: {t}" for t in transformation_set[1:]]

                    results = config.load_results(config.results_paths(variance_parameters[1:]))
                    visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, title=title)
