from .common import *
import datasets
import abc

class DataAugmentation(Experiment):
    def description(self):
        return """Compare the accuracies of the models for each set of transformations"""

    @abc.abstractmethod
    def get_datasets(self)->[str]:
        pass

    def run(self, generators=common_models_generators):
        models = generators
        transformations = common_transformations_da

        # model_names = [m.for_dataset("mnist").name for m in models]
        # transformation_labels = [l.rotation,l.scale,l.translation,l.combined]
        identity_transformations=tm.SimpleAffineTransformationGenerator()
        labels = [f"{l.train} {tr}, {l.test} {te}" for tr in [l.normal,l.transformed] for te in [l.normal,l.transformed]]


        for dataset in self.get_datasets():
            p_dataset = accuracy.DatasetParameters(dataset, datasets.DatasetSubset.test, 1.0)

            for transformation in transformations:

                model_labels = []
                accuracies = []
                for model_config_generator in models:
                    # Train
                    if model_config_generator == config.AllConvolutionalConfig:
                        bn=False
                    else:
                        bn=True

                    model_config = model_config_generator.for_dataset(dataset,bn=bn)

                    model_labels.append(model_config.name)
                    model_accuracies = []
                    for t_train in [identity_transformations,transformation]:
                        epochs = config.get_epochs(model_config, dataset, t_train)
                        p_training = training.Parameters(model_config, dataset, t_train, epochs)
                        if dataset == "lsa16" or dataset =="rwth":
                            batch_size=32
                        else:
                            batch_size=256
                        self.experiment_training(p_training,batch_size=batch_size)
                        model_path=config.model_path(p_training)
                        # Test
                        # print("***")
                        for t_test in [identity_transformations,transformation]:
                            p_accuracy = accuracy.Parameters(model_path,p_dataset,t_test)
                            self.experiment_accuracy(p_accuracy)
                            result= config.load_accuracy(config.accuracy_path(p_accuracy))
                            # print(dataset,t_train,model_config,t_test,result.accuracy)
                            model_accuracies.append(result.accuracy)
                    accuracies.append(model_accuracies)

                # print(accuracies)
                # print(model_labels)
                # plot results
                accuracies=np.array(accuracies)
                experiment_name = f"{dataset}_{transformation.id()}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.jpg"
                visualization.plot_accuracies(plot_filepath, accuracies,labels,model_labels  )


class DataAugmentationClassical(DataAugmentation):
    def get_datasets(self) ->[str]:
        return dataset_names

class DataAugmentationHandshape(DataAugmentation):
    def get_datasets(self) ->[str]:
        return handshape_dataset_names