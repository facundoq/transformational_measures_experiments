from .common import *
import datasets
import abc
from ..visualization.accuracies import plot_accuracies
from typing import List
from experiment import accuracy

class DataAugmentation(InvarianceExperiment):
    def description(self):
        return """Compare the accuracies of the models for each set of transformations"""

    @abc.abstractmethod
    def get_datasets(self)->List[str]:
        pass

    def run(self, generators=common_models_generators):
        models = generators
        transformations = common_transformations_da

        # model_names = [m.for_dataset("mnist").name for m in models]
        # transformation_labels = [l.rotation,l.scale,l.translation,l.combined]
        labels = [f"{l.train} {tr}, {l.test} {te}" for tr in [l.normal,l.transformed] for te in [l.normal,l.transformed]]

        task = Task.Classification
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

                    mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset,bn=bn)

                    model_labels.append(mc.name())
                    model_accuracies = []
                    for t_train in [identity_transformation,transformation]:
                        
                        if dataset == "lsa16":# or dataset =="rwth":
                            batch_size=32
                        else:
                            batch_size=128
                        
                        tc,metric = self.get_train_config(mc,dataset,task,t_train,verbose=True,batch_size=batch_size,savepoints=False)
                        
                        p = train.TrainParameters(mc, tc, dataset, t_train, task)
                        self.train(p)
                        model_path=self.model_path_new(p)
                        # Test
                        # print("***")
                        for t_test in [identity_transformation,transformation]:
                            p_accuracy = accuracy.Parameters(model_path,p_dataset,t_test)
                            result =self.experiment_accuracy(p_accuracy)
                            print(result)
                            self.save_accuracy(result)
                            # result= self.load_accuracy(self.accuracy_path(p_accuracy))
                            # print(dataset,t_train,model_config,t_test,result.accuracy)
                            model_accuracies.append(result.accuracy)
                    accuracies.append(model_accuracies)

                # print(accuracies)
                # print(model_labels)
                # plot results
                accuracies=np.array(accuracies)
                experiment_name = f"{dataset}_{transformation.id()}"
                plot_filepath = self.folderpath / f"{experiment_name}.jpg"
                plot_accuracies(plot_filepath, accuracies,labels,model_labels  )


class DataAugmentationClassical(DataAugmentation):
    def get_datasets(self) ->List[str]:
        return dataset_names

class DataAugmentationHandshape(DataAugmentation):
    def get_datasets(self) ->List[str]:
        return handshape_dataset_names