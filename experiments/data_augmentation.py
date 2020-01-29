from .common import *
import datasets

class DataAugmentation(Experiment):
    def description(self):
        return """Compare the accuracies of the models for each set of transformations"""

    def run(self):
        models = common_models_generators
        transformations = common_transformations_hard
        # model_names = [m.for_dataset("mnist").name for m in models]
        # transformation_labels = [l.rotation,l.scale,l.translation,l.combined]
        identity_transformations=tm.SimpleAffineTransformationGenerator()
        labels = ["Train Normal, Test Normal",
                  "Train Normal, Test Transformed",
                  "Train Transformed, Test Normal",
                  "Train Transformed, Test Transformed",
                  ]

        for dataset in dataset_names:
            p_dataset = accuracy.DatasetParameters(dataset, datasets.DatasetSubset.test, 1.0)

            for transformation in transformations:

                model_labels = []
                accuracies = []
                for model_config_generator in models:
                    # Train
                    model_config = model_config_generator.for_dataset(dataset)
                    model_accuracies = []
                    for t_train in [identity_transformations,transformation]:
                        epochs = config.get_epochs(model_config, dataset, t_train)
                        p_training = training.Parameters(model_config, dataset, t_train, epochs)
                        self.experiment_training(p_training)
                        model_path=config.model_path(p_training)
                        # Test
                        for t_test in [identity_transformations,transformation]:
                            p_accuracy = accuracy.Parameters(model_path,p_dataset,t_test)
                            self.experiment_accuracy(p_accuracy)
                            result= config.load_accuracy(config.accuracy_path(p_accuracy))
                            model_accuracies.append(result.accuracy)
                    accuracies.append(model_accuracies)
                    model_labels.append(model_config.name)

                # plot results
                experiment_name = f"{dataset}_{transformation.id()}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
                visualization.plot_accuracies(plot_filepath, accuracies,labels,model_labels  )


