#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import transformation_measure as tm
from experiment import variance, training

import config
import os
import runner_utils
import models
import itertools
from transformation_measure import visualization
model_names= models.names
model_names.sort()
#model_names=["SimpleConv"]

measures = tm.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]x

datasets= ["mnist", "cifar10"]
venv_path=runner_utils.get_venv_path()

measure=tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum)
import abc

class Experiment():
    def __init__(self):
        self.plot_folderpath = os.path.join(config.plots_base_folder(),self.id())
        os.makedirs(self.plot_folderpath, exist_ok=True)
    def id(self):
        return self.__class__.__name__
    def __call__(self, *args, **kwargs):
        print(f"Running experiment {self.id()}")
        self.run()

    @abc.abstractmethod
    def run(self):
        pass

    def experiment_training(self,p: training.Parameters):
        model_path=config.model_path(p)

        if os.path.exists(model_path):
            return

        python_command = f'experiment_training.py -model "{p.model}" -dataset "{p.dataset}" -transformation "{p.transformations.id()}" -verbose False -train_verbose False -num_workers 4'
        runner_utils.run_python(venv_path, python_command)

    def experiment_variance(self,p: variance.Parameters,model_path:str,batch_size:int=64,num_workers:int=2):

        results_path = config.results_path(p)
        if os.path.exists(results_path):
            return

        python_command = f'experiment_variance.py -mo "{model_path}" -me "{p.measure.id()}" -d "{p.dataset.id()}" -t "{p.transformations.id()}" -verbose False -batchsize {batch_size} -num_workers {num_workers}'
        runner_utils.run_python(venv_path, python_command)

    def experiment_plot_layers(self,variance_parameters:[variance.Parameters], plot_filepath: str, experiment_name:str):
        variance_paths= [f'"{config.results_path(p)}"' for p in variance_parameters]
        variance_paths_str= " ".join(variance_paths)
        python_command = f'experiment_plot_layers.py -name "{experiment_name}" -out "{plot_filepath}" {variance_paths_str}'
        runner_utils.run_python(venv_path, python_command)


class CompareMeasures(Experiment):
    def run(self):
        epochs= 0
        model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]
        for model in model_names:
            for dataset in datasets:
                p_dataset= variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
                for transformation in tm.common_transformations_without_identity():
                    experiment_name=f"{model}_{p_dataset.id()}_{transformation.id()}"
                    plot_filepath=os.path.join(self.plot_folderpath,f"{experiment_name}.png")
                    p_training= training.Parameters(model, dataset, transformation, epochs, 0)
                    self.experiment_training(p_training)
                    variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
                    model_path=config.model_path(p_training)
                    for p_variance in variance_parameters:
                        self.experiment_variance(p_variance,model_path)
                    results = config.load_results(config.results_paths(variance_parameters))
                    labels=[m.id() for m in measures]
                    visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)

class MeasureVsDatasetSize(Experiment):
    '''
    Vary the test dataset size and see how it affects the measure's value
    '''
    def run(self):
        dataset_sizes=[0.1,0.5,1.0]
        epochs= 0
        combinations=itertools.product(*[model_names,datasets,tm.common_transformations_without_identity(),measures])
        for model,dataset,transformation,measure in combinations:
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)
            p_datasets = [variance.DatasetParameters(dataset, variance.DatasetSubset.test, p) for p in dataset_sizes]
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, measure) for p_dataset in p_datasets ]
            model_path=config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance,model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            labels=[f"Dataset percentage: {d.percentage*100:2}%" for d in p_datasets]
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels,title=experiment_name)

class MeasureVsDatasetSubset(Experiment):
    '''
    Vary the test dataset subset and see how it affects the measure's value
    '''
    def run(self):
        dataset_sizes=[ (variance.DatasetSubset.test,0.1), (variance.DatasetSubset.train,0.02)]
        epochs= 0
        combinations=list(itertools.product(*[model_names,datasets,tm.common_transformations_without_identity(),measures]))
        for i,(model,dataset,transformation,measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}")
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)
            p_datasets = [variance.DatasetParameters(dataset, subset, p) for (subset,p) in dataset_sizes]
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, measure) for p_dataset in p_datasets]
            model_path=config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance,model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            labels=[f"Dataset subset {d.subset},  (percentage of data {d.percentage*100:2})%" for d in p_datasets]
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels,title=experiment_name)

class InvarianceVsTransformationDiversity(Experiment):
    '''
    Vary the test transformation diversity and see how it affects the invariance
    '''
    def run(self):
        epochs= 0
        measure_function,conv_agg=tm.MeasureFunction.std,tm.ConvAggregation.sum
        measure=tm.NormalizedMeasure(tm.TransformationMeasure(measure_function,conv_agg),tm.SampleMeasure(measure_function,conv_agg))
        combinations=itertools.product(*[model_names,datasets])
        for model,dataset in combinations:
            print(model,dataset)
            sets=[tm.rotation_transformations(),tm.translation_transformations(),tm.scale_transformations()]
            names=["rotation","translation","scale"]
            for i,(transformation_set,name) in enumerate(zip(sets,names)):
                experiment_name = f"{model}_{dataset}_{measure.id()}"
                plot_filepath = os.path.join(self.plot_folderpath,name, f"{experiment_name}.png")
                variance_parameters=[]
                print(f"    {name}")
                for i,transformation in enumerate(transformation_set):
                    print(f"        {i}/{len(transformation_set)}")
                    p_training = training.Parameters(model, dataset, transformation, epochs, 0)
                    self.experiment_training(p_training)
                    p_dataset =variance.DatasetParameters(dataset, variance.DatasetSubset.test,0.1)
                    p_variance=variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                    model_path = config.model_path(p_training)
                    self.experiment_variance(p_variance, model_path)
                    variance_parameters.append(p_variance)
                results=config.load_results(config.results_paths(variance_parameters))
                labels=[t.id for t in transformation_set]
                visualization.plot_collapsing_layers(results, plot_filepath, labels=labels,title=experiment_name)




if __name__ == '__main__':
    experiments=[CompareMeasures(),MeasureVsDatasetSize(),InvarianceVsTransformationDiversity()]
    for e in experiments:
        e()

