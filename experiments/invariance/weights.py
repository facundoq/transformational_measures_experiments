from torch import save
from .common import *
import experiment.measure as measure_package
import datasets

class RandomWeights(InvarianceExperiment):
    def description(self):
        return """Analyze the invariance of untrained networks, ie, with random weights."""

    def run(self):
        # random_models_folderpath = self.models_folder() / "random"
        # random_models_folderpath.mkdir(exist_ok=True, parents=True)
        # o = training.Options(False, 32,4, torch.cuda.is_available(), False, 0)
        measures = normalized_measures

        # number of random models to generate
        random_model_n = 30
        task = Task.Classification
        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations, measures)
        for model_config_generator, dataset, transformations, measure in combinations:
            mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
            results = []
            for i in range(random_model_n):
                suffix = f"random_weight{i:03}"
                tc,metric = self.get_train_config(mc,dataset,task,transformations,suffix=suffix,savepoints=False,epochs=0)
                p = train.TrainParameters(mc, tc, dataset, transformations, task)
                self.train(p)
                model_path = self.model_path_new(p)
                

                result = self.measure_default(dataset,mc.id()+suffix,model_path,transformations,measure,default_measure_options,default_dataset_percentage)
                results.append(result)

            # plot results
            experiment_name = f"{mc.id()}_{dataset}_{transformations.id()}_{measure}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            # results = self.load_measure_results(self.results_paths(variance_parameters))
            n = len(results)
            labels = [f"{l.random_models} ({n} {l.samples})."] + ([None] * (n - 1))
            # get alpha colors
            import matplotlib.pyplot as plt
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))
            color[:, 3] = 0.5
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, plot_mean=True, labels=labels,colors=color)


class DuringTraining(InvarianceExperiment):
    savepoints_percentages = [0, 1, 3, 5, 10, 30, 50, 100]

    def description(self):
        return """Analyze the evolution of invariance in models while they are trained."""

    def run(self):
        measures = normalized_measures_validation

        model_generators = simple_models_generators
        combinations = itertools.product(
            model_generators, dataset_names, common_transformations_combined, measures)
        task = Task.Classification
        for model_config_generator, dataset, transformations, measure in combinations:
            mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
            tc,metric = self.get_train_config(mc,dataset,task,transformations,savepoints=True)
            p = train.TrainParameters(mc, tc, dataset, transformations, task)
            self.train(p)

            # #Measures
            results, model_paths = self.measure_savepoints(p, dataset, measure, transformations)

            # plot results
            experiment_name = f"{mc.id()}_{dataset}_{transformations.id()}_{measure}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            self.plot(results, plot_filepath, model_paths, tc.savepoints, tc.epochs,measure )

    def measure_savepoints(self, p:train.TrainParameters,  dataset, measure, transformations):
        results = []
        model_paths = []
        for sp in p.tc.savepoints:
            model_path = self.model_path_new(p,savepoint=sp)
            model_id = p.mc.id()+f"_sp{sp}"
            result = self.measure_default(dataset,model_id,model_path,transformations,measure,default_measure_options,default_dataset_percentage)
            results.append(result)
            model_paths.append(model_path)
        return results, model_paths

    def plot(self, results, plot_filepath, model_paths, savepoints, epochs, measure:tm.Measure):
        # TODO implement a heatmap where the x axis is the training time/epoch
        # and the y axis indicates the layer, and the color indicates the invariance
        # to see it evolve over time.
        accuracies = []
        for model_path in model_paths:
            _, p, score = train.load_model(model_path, "cpu")
            accuracy = score["test_acc"]
            accuracies.append(accuracy)
        # ({sp * 100 // epochs}%)
        labels = [f"{sp} ({int(accuracy*100)}%)" for (sp, accuracy) in
                  zip(savepoints, accuracies)]
        n = len(savepoints)
        values = list(range(n))
        values.reverse()
        colors = tmv.get_sequential_colors(values)

        legend_location = ("lower left", (0, 0))
        # legend_location= None
        tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,
                                                        legend_location=legend_location, colors=colors,ylim=get_ylim_normalized(measure))


class RandomInitialization(InvarianceExperiment):
    def description(self):
        return """Test measures with various instances of the same architecture/transformation/dataset to see if the numpy is dependent on the random initialization in the training or simply on the architecture"""

    def run(self):
        measures = normalized_measures

        # number of random models to generate
        random_model_n = 30
        task = Task.Classification
        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations, measures)
        for model_config_generator, dataset, transformations, measure in combinations:
            mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
            results = []
            for i in range(random_model_n):
                suffix = f"random{i:03}"
                tc,metric = self.get_train_config(mc,dataset,task,transformations,suffix=suffix,savepoints=False)
                p = train.TrainParameters(mc, tc, dataset, transformations, task)
                self.train(p)
                model_path = self.model_path_new(p)

                result = self.measure_default(dataset,mc.id()+suffix,model_path,transformations,measure,default_measure_options,default_dataset_percentage)
                results.append(result)

            # plot results
            experiment_name = f"{mc.id()}_{dataset}_{transformations.id()}_{measure}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            # results = self.load_measure_results(self.results_paths(variance_parameters))
            n = len(results)
            labels = [f"{l.random_models} ({n} {l.samples})."] + ([None] * (n - 1))
            # get alpha colors
            import matplotlib.pyplot as plt
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))
            color[:, 3] = 0.5
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, plot_mean=True, labels=labels,colors=color)
