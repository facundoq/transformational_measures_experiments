#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import transformation_measure as tm
from experiment import variance, training

import config
import os
import runner_utils
import models
import itertools

model_names= models.names
model_names.sort()
#model_names=["SimpleConv"]

measures = tm.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]x

datasets= ["mnist", "cifar10"]
venv_path=runner_utils.get_venv_path()

measure=tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum)

def experiment_training(p: training.Parameters):
    model_path=config.model_path(p)

    if os.path.exists(model_path):
        return

    python_command = f'experiment_training.py -model "{p.model}" -dataset "{p.dataset}" -transformation "{p.transformations.id()}" -verbose False -train_verbose False -num_workers 4'
    runner_utils.run_python(venv_path, python_command)

def experiment_variance(p: variance.Parameters,model_path:str):

    results_path = config.results_path(p)
    if os.path.exists(results_path):
        return

    python_command = f'experiment_variance.py -mo "{model_path}" -me "{p.measure.id()}" -d "{p.dataset.id()}" -t "{p.transformations.id()}" -verbose False'
    runner_utils.run_python(venv_path, python_command)

def experiment_plot_layers(variance_parameters:[variance.Parameters], plot_filepath: str, experiment_name:str):
    variance_paths= [f'"{config.results_path(p)}"' for p in variance_parameters]
    variance_paths_str= " ".join(variance_paths)
    python_command = f'experiment_plot_layers.py -name "{experiment_name}" -out "{plot_filepath}" {variance_paths_str}'
    runner_utils.run_python(venv_path, python_command)

def compare_measures():
    plot_folderpath=os.path.join(config.plots_base_folder(),"compare_measures")
    os.makedirs(plot_folderpath,exist_ok=True)
    epochs= 0
    # # model_names=["SimpleConv","VGGLike","AllConvolutional"]
    # model_names=["ResNet"]
    for model in model_names:
        for dataset in datasets:
            p_dataset= variance.DatasetParameters(dataset, variance.DatasetSubset.test, 1.0)
            for transformation in tm.common_transformations_without_identity():
                experiment_name=f"{model}_{p_dataset.id()}_{transformation.id()}"
                plot_filepath=os.path.join(plot_folderpath,f"{experiment_name}.png")
                p_training= training.Parameters(model, dataset, transformation, epochs, 0)
                experiment_training(p_training)
                variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
                model_path=config.model_path(p_training)
                for p_variance in variance_parameters:
                    experiment_variance(p_variance,model_path)
                experiment_plot_layers(variance_parameters,plot_filepath,experiment_name)

def measure_vs_dataset_size():
    '''
    Vary the test dataset size and see how it affects the measure's value
    '''
    dataset_sizes=[0.1,0.5,0.1]
    plot_folderpath=os.path.join(config.plots_base_folder(),"measure_vs_dataset_size")
    os.makedirs(plot_folderpath,exist_ok=True)
    epochs= 0
    combinations=itertools.product(*[model_names,datasets,tm.common_transformations_without_identity()])
    for model,dataset,transformation in combinations:
        p_training = training.Parameters(model, dataset, transformation, epochs, 0)
        experiment_training(p_training)
        p_datasets = [variance.DatasetParameters(dataset, variance.DatasetSubset.test, p) for p in dataset_sizes]
        for m in measures:
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{m.id()}"
            plot_filepath = os.path.join(plot_folderpath, f"{experiment_name}.png")
            variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, m) for p_dataset in p_datasets ]
            model_path=config.model_path(p_training)
            for p_variance in variance_parameters:
                experiment_variance(p_variance,model_path)
            experiment_plot_layers(variance_parameters,plot_filepath,experiment_name)


def invariance_vs_transformation_diversity():
    '''
    Vary the test transformation diversity and see how it affects the invariance
    '''

    plot_folderpath=os.path.join(config.plots_base_folder(),"invariance_vs_transformation_diversity")
    os.makedirs(plot_folderpath,exist_ok=True)
    epochs= 0
    measure_function,conv_agg=tm.MeasureFunction.std,tm.ConvAggregation.sum
    measure=tm.NormalizedMeasure(tm.TransformationMeasure(measure_function,conv_agg),tm.SampleMeasure(measure_function,conv_agg))
    combinations=itertools.product(*[model_names,datasets,tm.common_transformations_without_identity()])
    for model,dataset in combinations:
        sets=[tm.rotation_transformations(),tm.translation_transformations(),tm.scale_transformations()]
        names=["rotation","translation","scale"]
        for i,(set,name) in enumerate(zip(sets,names)):
            experiment_name = f"{model}_{dataset}_{measure.id()}"
            plot_filepath = os.path.join(plot_folderpath,name, f"{experiment_name}.png")
            variance_parameters=[]
            for transformation in set:
                p_training = training.Parameters(model, dataset, transformation, epochs, 0)
                experiment_training(p_training)
                p_dataset =variance.DatasetParameters(dataset, variance.DatasetSubset.test,1.0)
                p_variance=variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                experiment_variance(p_variance, model_path)
                variance_parameters.append(p_variance)

            experiment_plot_layers(variance_parameters,plot_filepath,experiment_name)



if __name__ == '__main__':
    compare_measures()
    measure_vs_dataset_size()
    invariance_vs_transformation_diversity()