from .common import *


class RandomWeights(Experiment):
    def description(self):
        return """Analyze the invariance of untrained networks, ie, with random weights."""

    def run(self):
        random_models_folderpath = config.models_folder() / "random"
        random_models_folderpath.mkdir(exist_ok=True, parents=True)
        o = training.Options(False, False, False, 32, 4, torch.cuda.is_available(), False, 0)
        measures = normalized_measures

        # number of random models to generate
        random_model_n = 10

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations, measures)
        for model_config_generator, dataset_name, transformation, measure in combinations:
            model_config = model_config_generator.for_dataset(dataset_name)
            p = config.dataset_size_for_measure(measure)
            # generate `random_model_n` models and save them without training
            models_paths = []
            p_training = training.Parameters(model_config, dataset_name, transformation, 0)
            dataset = datasets.get(dataset_name)
            for i in range(random_model_n):

                model_path = config.model_path(p_training, model_folderpath=random_models_folderpath)

                # append index to model name
                name, ext = os.path.splitext(str(model_path))
                name += f"_random{i:03}"
                model_path = Path(f"{name}{ext}")
                if not model_path.exists():
                    model, optimizer = model_config.make_model_and_optimizer(dataset.input_shape, dataset.num_classes, o.use_cuda)
                    scores = training.eval_scores(model, dataset, p_training.transformations,  TransformationStrategy.random_sample, o.get_eval_options())
                    training.save_model(p_training, o, model, scores, model_path)
                    del model
                models_paths.append(model_path)

            # generate variance params
            variance_parameters = []
            p_dataset = variance.DatasetParameters(dataset_name, variance.DatasetSubset.test, p)

            for model_path in models_paths:
                model_id, ext = os.path.splitext(os.path.basename(model_path))
                p_variance = variance.Parameters(model_id, p_dataset, transformation, measure)
                self.experiment_variance(p_variance, model_path)
                variance_parameters.append(p_variance)

            # plot results
            experiment_name = f"{model_config.name}_{dataset_name}_{transformation.id()}_{measure}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_results(config.results_paths(variance_parameters))
            n = len(results)
            labels = [f"{l.random_models} ({n} {l.samples})."] + ([None] * (n - 1))
            # get alpha colors
            import matplotlib.pyplot as plt
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))
            color[:, 3] = 0.5

            visualization.plot_collapsing_layers_same_model(results, plot_filepath, plot_mean=True, labels=labels,
                                                            colors=color)


class DuringTraining(Experiment):
    savepoints_percentages = [0, 1, 3, 5, 10, 30, 50, 100]

    def description(self):
        return """Analyze the evolution of invariance in models while they are trained."""

    def run(self):
        measures = normalized_measures_validation

        model_generators = simple_models_generators
        combinations = itertools.product(
            model_generators, dataset_names, common_transformations_combined, measures)

        for model_config_generator, dataset, transformation, measure in combinations:
            # train
            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            savepoints = [sp * epochs // 100 for sp in self.savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))

            # Training
            p_training = training.Parameters(model_config, dataset, transformation, epochs, savepoints=savepoints)
            self.experiment_training(p_training)

            # #Measures
            variance_parameters, model_paths = self.measure(p_training, config, dataset, measure, transformation,
                                                            savepoints)

            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_results(config.results_paths(variance_parameters))
            self.plot(results, plot_filepath, model_paths, savepoints, epochs,measure )

    def measure(self, p_training, config, dataset, measure, transformation, savepoints):
        variance_parameters = []
        model_paths = []
        p = config.dataset_size_for_measure(measure)
        p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
        for sp in savepoints:
            model_path = config.model_path(p_training, savepoint=sp)
            model_id = p_training.id(savepoint=sp)
            p_variance = variance.Parameters(model_id, p_dataset, transformation, measure)
            variance_parameters.append(p_variance)
            model_paths.append(model_path)

        for p_variance, model_path in zip(variance_parameters, model_paths):
            self.experiment_variance(p_variance, model_path)
        return variance_parameters, model_paths

    def plot(self, results, plot_filepath, model_paths, savepoints, epochs,measure:tm.Measure):
        # TODO implement a heatmap where the x axis is the training time/epoch
        # and the y axis indicates the layer, and the color indicates the invariance
        # to see it evolve over time.
        accuracies = []
        for model_path in model_paths:
            _, p, _, score = training.load_model(model_path, False, False)
            loss, accuracy = score["test"]
            accuracies.append(accuracy)
        # ({sp * 100 // epochs}%)
        labels = [f"{sp} ({int(accuracy*100)}%)" for (sp, accuracy) in
                  zip(savepoints, accuracies)]
        n = len(savepoints)
        values = list(range(n))
        values.reverse()
        colors = visualization.get_sequential_colors(values)

        legend_location = ("lower left", (0, 0))
        # legend_location= None
        visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,
                                                        legend_location=legend_location, colors=colors,ylim=get_ylim_normalized(measure))


class RandomInitialization(Experiment):
    def description(self):
        return """Test measures with various instances of the same architecture/transformation/dataset to see if the measure is dependent on the random initialization in the training or simply on the architecture"""

    def run(self):
        measures = normalized_measures_validation
        repetitions = 8

        model_generators = simple_models_generators
        transformations = common_transformations

        combinations = itertools.product(model_generators, dataset_names, transformations, measures)
        for (model_generator, dataset, transformation, measure) in combinations:
            # train
            model_config = model_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            training_parameters = []
            for r in range(repetitions):
                p_training = training.Parameters(model_config, dataset, transformation, epochs, 0, suffix=f"rep{r:02}")
                self.experiment_training(p_training)
                training_parameters.append(p_training)
            # generate variance params
            variance_parameters = []
            for p_training in training_parameters:
                model_path = config.model_path(p_training)
                p = config.dataset_size_for_measure(measure)
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
            results = config.load_results(config.results_paths(variance_parameters))

            visualization.plot_collapsing_layers_same_model(results, plot_filepath, plot_mean=True,ylim=get_ylim_normalized(measure))
