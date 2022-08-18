from .common import *
import experiment.measure as measure_package
import datasets


class DatasetSize(InvarianceExperiment):

    def description(self):
        return '''Vary the test dataset size and see how it affects the numpy's value. That is, vary the size of the dataset used to compute the invariance (not the training dataset) and see how it affects the calculation of the numpy.'''

    def run(self):
        dataset_percentages = [0.01, 0.05] #, 0.1, 0.5, 1.0]
        model_names = simple_models_generators
        measures = normalized_measures_validation
        combinations = list(itertools.product(
            model_names, dataset_names, common_transformations_combined, measures))

        for i, (model, dataset, transformation, measure) in enumerate(combinations):
            
            mc,tc,p,model_path = self.train_default(Task.Classification,dataset,transformation,model)
            results = []
            for p in dataset_percentages:
                result = self.measure_default(dataset,mc.id(),model_path,transformation,measure,default_measure_options,DatasetSizePercentage(p))
                
                results.append(result)

            labels = [f"{d * 100:2}%" for d in dataset_percentages]
            experiment_name = f"{mc.id()}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            
            n = len(dataset_percentages)
            values = list(range(n))
            values.reverse()
            colors = tmv.get_sequential_colors(values)
            
            tmv.plot_average_activations_same_model(results, colors=colors,ylim=get_ylim_normalized(measure))
            self.savefig(plot_filepath)



class DatasetSubset(InvarianceExperiment):

    def description(self):
        return '''Vary the test dataset subset (either train o testing) and see how it affects the numpy's value.'''

    def run(self):
        dataset_subsets = [datasets.DatasetSubset.test, datasets.DatasetSubset.train]

        model_names = simple_models_generators
        measures = normalized_measures_validation
        combinations = list(itertools.product(
            model_names, dataset_names, common_transformations_combined, measures))

        for i, (model_config_generator, dataset, transformation, measure) in enumerate(combinations):
            mc,tc,p,model_path = self.train_default(Task.Classification,dataset,transformation,model_config_generator)
            results = []
            for subset in dataset_subsets:
                dataset_percentage = DatasetSizePercentage(config.measures.dataset_percentage_for_measure(measure, subset))
                result = self.measure_default(dataset,mc.id(),model_path,transformation,measure,default_measure_options,dataset_percentage,subset=subset)
                results.append(result)


            labels = [f"{l.format_subset(subset)}" for subset in dataset_subsets]
            experiment_name = f"{mc.id()}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            tmv.plot_average_activations_same_model(results, labels=labels,ylim=get_ylim_normalized(measure))
            self.savefig(plot_filepath)


class DatasetTransfer(InvarianceExperiment):
    def description(self):
        return """Measure invariance with a different dataset than the one used to train the model."""

    def run(self):
        measures = normalized_measures_validation

        combinations = itertools.product(
            simple_models_generators, dataset_names, common_transformations_combined, measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            mc,tc,p,model_path = self.train_default(Task.Classification,dataset,transformation,model_config_generator)
            results = []

            # # train
            # epochs = config.get_epochs(model_config, dataset, transformation)
            # p_training = training.Parameters(model_config, dataset, transformation, epochs)
            # self.experiment_training(p_training)

            results = []
            for dataset_test in dataset_names:
                dataset_percentage = DatasetSizePercentage(config.measures.dataset_percentage_for_measure(measure))
                result = self.measure_default(dataset_test,mc.id(),model_path,transformation,measure,default_measure_options,dataset_percentage,adapt_dataset=True)
                results.append(result)

                # p = 0.5 if measure.__class__ == tm.ANOVAInvariance else default_dataset_percentage
                # p_dataset = measure_package.DatasetParameters(dataset_test, datasets.DatasetSubset.test, p)
                # p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                # model_path = self.model_path(p_training)
                # self.experiment_measure(p_variance, adapt_dataset=True)
                # variance_parameters.append(p_variance)

            # plot results
            experiment_name = f"{mc.id()}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            labels = dataset_names
            tmv.plot_average_activations_same_model(results, labels=labels,ylim=get_ylim_normalized(measure))
            self.savefig(plot_filepath)
