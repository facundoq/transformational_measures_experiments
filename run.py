#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import transformation_measure as tm
from experiment import variance, training, model_loading, utils_runner
import numpy as np
import config
import os
import models
import itertools
from transformation_measure import visualization

from pathlib import Path

all_model_names = config.model_names
all_model_names = [name for name in all_model_names if name=="SimpleConv"]

bn_model_names = [name for name in all_model_names if name.endswith("BN")]

model_names = [name for name in all_model_names if not name.endswith("BN")]

model_names.sort()
#model_names= [name for name in model_names if not name == "ResNet"]
#model_names=["SimpleConv"]

measures = config.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]x

dataset_names= ["mnist", "cifar10"]
venv_path= utils_runner.get_venv_path()

measure=tm.TransformationMeasure(tm.MeasureFunction.std,tm.ConvAggregation.sum)
import abc


class Experiment():
    def __init__(self):
        self.plot_folderpath = config.plots_base_folder() / self.id()
        os.makedirs(self.plot_folderpath, exist_ok=True)
        with open(os.path.join(self.plot_folderpath,"description.txt"),"w") as f:
            f.write(self.description())

    def id(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        stars="*"*15
        print(f"{stars} Running experiment {self.id()} {stars}")
        self.run()
        print(f"{stars} Finished experiment {self.id()} {stars}")

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def description(self)->str:
        pass

    def experiment_training(self,p: training.Parameters,min_accuracy=None):
        model_path=config.model_path(p)
        if not min_accuracy:
            min_accuracy = config.min_accuracy(p.model, p.dataset)
        if os.path.exists(model_path):
            if p.savepoints==[]:
                return
            else:
                savepoint_missing=[sp for sp in p.savepoints if not os.path.exists(config.model_path(p,sp))]
                if savepoint_missing:
                    print(f"Savepoints {savepoint_missing} missing; rerunning training...")
                else:
                    return

        savepoints=",".join([str(sp) for sp in p.savepoints])
        python_command = f'train.py -model "{p.model}" -dataset "{p.dataset}" -transformation "{p.transformations.id()}" -epochs {p.epochs} -verbose False -train_verbose False -num_workers 4 -min_accuracy {min_accuracy} -max_restarts 5 -savepoints "{savepoints}" '
        utils_runner.run_python(venv_path, python_command)

    def experiment_variance(self,p: variance.Parameters,model_path:str,batch_size:int=64,num_workers:int=2,adapt_dataset=False):

        results_path = config.results_path(p)
        if os.path.exists(results_path):
            return

        python_command = f'measure.py -mo "{model_path}" -me "{p.measure.id()}" -d "{p.dataset.id()}" -t "{p.transformations.id()}" -verbose False -batchsize {batch_size} -num_workers {num_workers} '
        if adapt_dataset:
            python_command=f"{python_command} -adapt_dataset True"

        utils_runner.run_python(venv_path, python_command)




class CompareMeasures(Experiment):
    def description(self):
        return """Test different measures for a given dataset/model/transformation combination to evaluate their differences."""
    def run(self):
        mf,ca_none=tm.MeasureFunction.std,tm.ConvAggregation.none
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measure_sets={"Variance":[
                                # tm.SampleMeasure(mf,ca_none)
                                # tm.TransformationMeasure(mf,ca_none),
                                tm.TransformationVarianceMeasure(mf, ca_none),
                                tm.SampleVarianceMeasure(mf, ca_none),
                                 ],
                      "Distance":[
                          tm.DistanceTransformationMeasure(dmean),
                          tm.DistanceSampleMeasure(dmean),
                      ],
                      "HighLevel":[
                                  tm.AnovaMeasure(ca_none,0.99,bonferroni=True),
                                  tm.NormalizedMeasure(mf,ca_none),
                                  tm.NormalizedVarianceMeasure(mf, ca_none),
                                  tm.DistanceMeasure(dmean),
                                   ],
                "Equivariance":[
                            tm.DistanceSameEquivarianceMeasure(dmean),
                            tm.DistanceMeasure(dmean),
                            ]
                      }

        #model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]
        transformations = config.common_transformations_without_identity()
        #transformations = [ tm.SimpleAffineTransformationGenerator(n_rotations=16)]
        combinations = itertools.product(*[model_names, dataset_names, transformations, measure_sets.items()])
        for (model,dataset,transformation,measure_set) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            p_training= training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # generate variance params
            variance_parameters=[]
            measure_set_name, measures = measure_set
            for m in measures:
                p= 0.5 if m.__class__==tm.AnovaMeasure else 0.1
                p_dataset= variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance=variance.Parameters(p_training.id(), p_dataset, transformation, m)
                variance_parameters.append(p_variance)
            # evaluate variance
            model_path=config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance,model_path)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure_set_name}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            results = config.load_results(config.results_paths(variance_parameters))
            labels=[m.id() for m in measures]
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)

class MeasureVsDatasetSize(Experiment):

    def description(self):
        return '''Vary the test dataset size and see how it affects the measure's value. That is, vary the size of the dataset used to compute the invariance (not the training dataset) and see how it affects the calculation of the measure.'''

    def run(self):
        dataset_sizes = [0.01,0.05,0.1,0.5,1.0]
        combinations = list(itertools.product(*[model_names, dataset_names, config.common_transformations_without_identity(), measures]))
        for i,(model,dataset,transformation,measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}",end=", ")
            epochs = config.get_epochs(model, dataset, transformation)
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

    def description(self):
        return '''Vary the test dataset subset (either train o testing) and see how it affects the measure's value.'''

    def run(self):
        dataset_sizes=[ (variance.DatasetSubset.test,0.1), (variance.DatasetSubset.train,0.02)]

        combinations=list(itertools.product(*[model_names, dataset_names, config.common_transformations_without_identity(), measures]))
        for i,(model,dataset,transformation,measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}",end=", ")
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)

            p_datasets=[]
            for (subset,p) in dataset_sizes:
                if measure.__class__==tm.AnovaMeasure.__class__:
                    p=p*5
                p_datasets.append(variance.DatasetParameters(dataset, subset, p))
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

    def description(self):
        return '''Vary the scale of transformation both when training a computing the measure, and see how it affects the invariance. For example, train with 2 rotations, then measure the invariance with 2 rotations. Train with 4 rotations, measure with 4 rotations, and so on. '''

    def run(self):
        n_transformations=5
        measure_function,conv_agg=tm.MeasureFunction.std,tm.ConvAggregation.none
        measure=tm.NormalizedMeasure(measure_function,conv_agg)
        distance_measure = tm.DistanceMeasure(tm.DistanceAggregation.mean)
        measures=[measure,distance_measure]
        combinations=itertools.product(*[model_names, dataset_names,measures])
        for model,dataset,measure in combinations:
            print(model,dataset)
            sets=[config.rotation_transformations(n_transformations),config.translation_transformations(n_transformations),config.scale_transformations(n_transformations)]
            names=["rotation","translation","scale"]
            for i,(transformation_set,name) in enumerate(zip(sets,names)):
                transformation_plot_folderpath=os.path.join(self.plot_folderpath,name)
                os.makedirs(transformation_plot_folderpath,exist_ok=True)
                experiment_name = f"{model}_{dataset}_{measure.id()}"
                plot_filepath = os.path.join(transformation_plot_folderpath, f"{experiment_name}.png")
                variance_parameters=[]
                print(f"    {name}, experiment:{len(transformation_set)}")
                for i,transformation in enumerate(transformation_set):
                    print(f"{i}, ",end="")
                    epochs = config.get_epochs(model, dataset, transformation)
                    p_training = training.Parameters(model, dataset, transformation, epochs, 0)
                    self.experiment_training(p_training)
                    p_dataset =variance.DatasetParameters(dataset, variance.DatasetSubset.test,0.1)
                    p_variance=variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                    model_path = config.model_path(p_training)
                    self.experiment_variance(p_variance, model_path)
                    variance_parameters.append(p_variance)
                results=config.load_results(config.results_paths(variance_parameters))
                labels=[str(t) for t in transformation_set]
                visualization.plot_collapsing_layers(results, plot_filepath, labels=labels,title=experiment_name)


class InvarianceVsTransformationDifferentScales(Experiment):

    def description(self):
        return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 8 rotations, measure variance with 2, 4, 8 and 16. """

    def run(self):
        n_transformations=5
        measure_function,conv_agg=tm.MeasureFunction.std,tm.ConvAggregation.none
        measure=tm.NormalizedMeasure(measure_function,conv_agg)
        combinations=itertools.product(*[model_names, dataset_names])
        for model,dataset in combinations:
            print(model,dataset)
            sets=[config.rotation_transformations(n_transformations),config.translation_transformations(n_transformations),config.scale_transformations(n_transformations)]
            names=["rotation","translation","scale"]
            for i,(transformation_set,name) in enumerate(zip(sets,names)):
                n_experiments=(len(transformation_set)+1)*len(transformation_set)
                print(f"    {name}, experiment:{n_experiments}")
                for j,train_transformation in enumerate(transformation_set+[tm.SimpleAffineTransformationGenerator()]):
                    transformation_plot_folderpath = os.path.join(self.plot_folderpath, name)
                    os.makedirs(transformation_plot_folderpath, exist_ok=True)
                    experiment_name = f"{model}_{dataset}_{measure.id()}_{train_transformation.id()}"
                    plot_filepath = os.path.join(transformation_plot_folderpath, f"{experiment_name}.png")
                    variance_parameters = []
                    print(f"{j}, ",end="")
                    epochs = config.get_epochs(model, dataset, train_transformation)
                    p_training = training.Parameters(model, dataset, train_transformation, epochs, 0)
                    self.experiment_training(p_training)
                    for k,test_transformation in enumerate(transformation_set):
                        p_dataset  = variance.DatasetParameters(dataset, variance.DatasetSubset.test,0.1)
                        p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation , measure)
                        model_path = config.model_path(p_training)
                        self.experiment_variance(p_variance, model_path)
                        variance_parameters.append(p_variance)

                    title=f"Invariance to \n. Model: {model}, Dataset: {dataset}, Measure {measure.id()} \n Train transformation: {train_transformation.id()} "
                    labels = [f"Test transformation: {t}" for t in transformation_set]
                    results=config.load_results(config.results_paths(variance_parameters))
                    visualization.plot_collapsing_layers(results, plot_filepath, labels=labels,title=title)




class CollapseConvBeforeOrAfter(Experiment):
    def description(self):
        return """Collapse convolutions spatial dims after/before computing variance."""
    def run(self):
        pre_functions=[tm.ConvAggregation.sum,tm.ConvAggregation.mean,tm.ConvAggregation.max]

        measures=[]
        for f in pre_functions:
            measure=tm.NormalizedMeasure(tm.MeasureFunction.std,f)

            measures.append(measure)
        post_functions=[tm.ConvAggregation.mean]

        combinations = itertools.product(
            *[model_names, dataset_names, config.common_transformations_without_identity()])
        for (model, dataset, transformation) in combinations:
            # train

            epochs = config.get_epochs(model, dataset, transformation)
            p_training= training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # eval variance
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path=config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance,model_path)
            post_measure=tm.NormalizedMeasure(tm.MeasureFunction.std,tm.ConvAggregation.none)
            no_aggregation_parameters=variance.Parameters(p_training.id(), p_dataset, transformation, post_measure)
            self.experiment_variance(no_aggregation_parameters, model_path)

            post_result_sets={"all":pre_functions,"mean":post_functions}
            for set,functions in post_result_sets.items():
                post_results= config.load_results(config.results_paths([no_aggregation_parameters]*len(functions)))
                for f,r in zip(functions,post_results):
                    r.measure_result=r.measure_result.collapse_convolutions(f)

                #plot
                experiment_name = f"{model}_{p_dataset.id()}_{transformation.id()}_{set}"
                plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
                results = config.load_results(config.results_paths(variance_parameters))

                labels=[f"Pre : {m.id()}" for m in measures]
                post_labels=[f"Post: {post_measure.id()} ({f})" for f in functions]
                visualization.plot_collapsing_layers(results+post_results, plot_filepath, labels=labels+post_labels, title=experiment_name)

class ComparePreConvAgg(Experiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate their differences. Convolutional aggregation collapses all the spatial dimensions of feature maps so that a single variance value for the feature map can be obtained."""
    def run(self):
        functions=[tm.ConvAggregation.sum,tm.ConvAggregation.mean,tm.ConvAggregation.max,tm.ConvAggregation.none]
        measure_sets_constructors={"nm":tm.NormalizedVarianceMeasure
                      ,"sm":tm.SampleVarianceMeasure
                      ,"tm":tm.TransformationVarianceMeasure}
        measure_sets = []
        for set_name,measure_constructor in measure_sets_constructors.items():
            measure_objects=[measure_constructor(tm.MeasureFunction.std,f) for f in functions]
            measure_sets.append( (set_name,measure_objects) )

        combinations = itertools.product(
            model_names, dataset_names, config.common_transformations_without_identity(),measure_sets)
        for model, dataset, transformation,(set_name,measures) in combinations:
            p_dataset= variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            experiment_name=f"{model}_{p_dataset.id()}_{transformation.id()}_{set_name}"
            plot_filepath=os.path.join(self.plot_folderpath,f"{experiment_name}.png")
            epochs = config.get_epochs(model, dataset, transformation)
            p_training= training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            variance_parameters= [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path=config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance,model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            labels=[m.id() for m in measures]
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)



import datasets
import torch

class InvarianceForRandomNetworks(Experiment):
    def description(self):
        return """Analyze the invariance of random (untrained) networks."""
    def run(self):
        random_models_folderpath=os.path.join(config.models_folder(),"random")
        os.makedirs(random_models_folderpath,exist_ok=True)
        o=training.Options(False,False,False,32,4,torch.cuda.is_available(),False,0)
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        mf, ca_sum, ca_none= tm.MeasureFunction.std, tm.ConvAggregation.sum, tm.ConvAggregation.none
        measures = [
            tm.NormalizedMeasure(mf,ca_none)
            , tm.AnovaMeasure(ca_none, alpha=0.99,bonferroni=True)
            , tm.DistanceMeasure( dmean)
            , tm.DistanceSameEquivarianceMeasure(dmean)

            ]
        dataset_percentages = [0.1, 0.5]
        # number of random models to generate
        random_model_n=30

        mp = zip(measures, dataset_percentages)
        combinations = itertools.product(
            model_names, dataset_names, config.common_transformations_without_identity(), mp)
        for model_name, dataset_name, transformation, (measure, p) in combinations:
            # generate `random_model_n` models and save them without training
            models_paths=[]
            p_training = training.Parameters(model_name, dataset_name, transformation, 0)
            dataset = datasets.get(dataset_name)
            for i in range(random_model_n):

                model_path=config.model_path(p_training,model_folderpath=random_models_folderpath)

                # append index to model name
                name,ext=os.path.splitext(model_path)
                name+=f"_random{i:03}"
                model_path=f"{name}{ext}"
                if not os.path.exists(model_path):
                    model, optimizer = model_loading.get_model(model_name, dataset, use_cuda=o.use_cuda)
                    scores = training.eval_scores(model, dataset, p_training, o)
                    training.save_model(p_training,o,model,scores,model_path)
                    del model
                models_paths.append(model_path)

            # generate variance params
            variance_parameters = []
            p_dataset = variance.DatasetParameters(dataset_name, variance.DatasetSubset.test, p)

            for model_path in models_paths:
                model_id,ext=os.path.splitext(os.path.basename(model_path))
                p_variance = variance.Parameters(model_id, p_dataset, transformation, measure)
                self.experiment_variance(p_variance, model_path)
                variance_parameters.append(p_variance)


            # plot results
            experiment_name = f"{model_name}_{dataset_name}_{transformation.id()}_{measure}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            results = config.load_results(config.results_paths(variance_parameters))
            n=len(results)
            labels=[f"Random models ({n} samples)."]+ ([None]*(n-1))
            # get alpha colors
            import matplotlib.pyplot as plt
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))
            color[:,3]=0.5
            visualization.plot_collapsing_layers(results, plot_filepath, title=experiment_name,plot_mean=True,labels=labels,color=color)

class InvarianceWhileTraining(Experiment):
    def description(self):
        return """Analyze the evolution of invariance in models while they are trained."""
    def run(self):
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        mf, ca_sum, ca_none = tm.MeasureFunction.std, tm.ConvAggregation.sum, tm.ConvAggregation.none

        measures = [tm.NormalizedMeasure(mf,ca_sum)
                    ,tm.AnovaMeasure(ca_none,alpha=0.99,bonferroni=True)
            , tm.DistanceMeasure(dmean)
            , tm.DistanceSameEquivarianceMeasure( dmean)
                    ]
        dataset_percentages=[0.1,0.5]
        n_intermediate_models=10
        step=100//n_intermediate_models
        savepoints=list(range(0,100,step))+[100]

        mp=zip(measures,dataset_percentages)
        combinations = itertools.product(
            model_names, dataset_names, config.common_transformations_without_identity(),mp)
        for model, dataset, transformation, (measure,p) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs, savepoints=savepoints)
            self.experiment_training(p_training)
            # generate variance params
            variance_parameters = []
            model_paths = []
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
            for sp in savepoints:
                model_path = config.model_path(p_training, savepoint=sp)
                model_id=p_training.id(savepoint=sp)
                p_variance = variance.Parameters(model_id, p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                model_paths.append(model_path)

            for p_variance,model_path in zip(variance_parameters,model_paths):
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            results = config.load_results(config.results_paths(variance_parameters))
            # TODO implement a heatmap where the x axis is the training time/epoch
            # and the y axis indicates the layer, and the color indicates the invariance
            # to see it evolve over time.

            labels = [f"{sp}%" for sp in savepoints]
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)


class CompareBN(Experiment):
    def description(self):
        return """Compare invariance of models trained with/without batchnormalization."""
    def run(self):
        mf, ca = tm.MeasureFunction.std, tm.ConvAggregation.sum
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures= [tm.AnovaMeasure(tm.ConvAggregation.none, 0.99), tm.NormalizedMeasure(mf, ca)
        , tm.DistanceMeasure(dmean)
        , tm.DistanceSameEquivarianceMeasure(dmean)
        ]

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]
        model_pairs=zip(bn_model_names,model_names)
        combinations = itertools.product(
            model_pairs, dataset_names, config.common_transformations_without_identity(),measures)
        for (model_pair, dataset, transformation, measure) in combinations:
            # train

            variance_parameters=[]
            for model in model_pair:
                epochs = config.get_epochs(model, dataset, transformation)
                p_training = training.Parameters(model, dataset, transformation, epochs)
                self.experiment_training(p_training)

                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                batch_size=64
                if model.startswith("ResNet"):
                    batch_size=32
                self.experiment_variance(p_variance, model_path,batch_size=batch_size)
                variance_parameters.append(p_variance)

            # evaluate variance
            model,model_bn=model_pair
            # plot results
            experiment_name = f"{model}_{model_bn}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            results = config.load_results(config.results_paths(variance_parameters))
            labels = model_pair
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)


class InvarianceAcrossDatasets(Experiment):
    def description(self):
        return """Measure invariance with a different dataset than the one used to train the model."""
    def run(self):
        mf, ca = tm.MeasureFunction.std, tm.ConvAggregation.sum
        measures= [tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.none, alpha=0.99), tm.NormalizedMeasure(mf, ca)]

        combinations = itertools.product(
            model_names, dataset_names, config.common_transformations_without_identity(),measures)
        for (model, dataset, transformation, measure) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)

            variance_parameters=[]
            for dataset_test in dataset_names:
                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset_test, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                self.experiment_variance(p_variance, model_path,adapt_dataset=True)
                variance_parameters.append(p_variance)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
            results = config.load_results(config.results_paths(variance_parameters))
            labels = dataset_names
            visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)




class InvarianceVsEpochs(Experiment):
    def description(self):
        return """Analyze the number of epochs needed for a model to converge and gain invariance to different types of transformations."""
    def run(self):
        pass


class CompareModelsForInvariance(Experiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""
    def run(self):
        pass

class CompareAnovaMeasures(Experiment):
    def description(self):
        return """Determine which Anova measure is more appropriate"""
    def run(self):
        pass

class InvarianceVsKernelSize(Experiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""
    def run(self):
        pass

class InvarianceVsMaxPooling(Experiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""
    def run(self):
        pass

class VisualizeInvariantFeatureMaps(Experiment):
    def description(self):
        return """Visualize the output of invariant feature maps, to analyze qualitatively if they are indeed invariant."""
    def run(self):

        dmean,dmax, =  tm.DistanceAggregation.mean,tm.DistanceAggregation.max
        mf, ca_sum,ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.sum,tm.ConvAggregation.mean
        ca_none = tm.ConvAggregation.none
        measures = [
                    # tm.AnovaMeasure(ca_none, alpha=0.99,bonferroni=True),
                    # #tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.mean, alpha=0.99,bonferroni=True),
                    # tm.NormalizedMeasure(mf, ca_sum),
                    # tm.NormalizedMeasure(mf, ca_mean),
                    # tm.TransformationMeasure(mf, tm.ConvAggregation.none),
                    # tm.TransformationMeasure(mf, ca_sum),
                    # tm.DistanceTransformationMeasure(mf, dmean),
                    # tm.DistanceTransformationMeasure(mf, dmax),
                    # tm.DistanceMeasure(mf,dmean),
                    # tm.DistanceMeasure(mf, dmax),
                    # tm.DistanceSameEquivarianceMeasure(mf, dmax),
                    tm.DistanceSameEquivarianceMeasure(dmean),

                    ]
        conv_model_names = [m for m in model_names if (not "FFNet" in m)]
        conv_model_names = [models.SimpleConv.__name__]
        combinations = itertools.product(
            conv_model_names, dataset_names, config.common_transformations_without_identity(), measures)
        for (model_name, dataset_name, transformation_set, measure) in combinations:

            experiment_name = f"{model_name}_{dataset_name}_{transformation_set.id()}_{measure.id()}"
            plot_folderpath = self.plot_folderpath / experiment_name
            finished = Path(plot_folderpath) / "finished"
            if finished.exists():
                continue
            # train
            epochs = config.get_epochs(model_name, dataset_name, transformation_set)
            p_training = training.Parameters(model_name, dataset_name, transformation_set, epochs)
            self.experiment_training(p_training)
            p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
            p_dataset = variance.DatasetParameters(dataset_name, variance.DatasetSubset.test, p)
            p_variance = variance.Parameters(p_training.id(), p_dataset, transformation_set, measure)
            model_path = config.model_path(p_training)
            self.experiment_variance(p_variance, model_path)

            model_filepath=config.model_path(p_training)
            model, p_model, o, scores = training.load_model(model_filepath, use_cuda=torch.cuda.is_available())
            result_filepath= config.results_path(p_variance)
            result = config.load_result(result_filepath)
            dataset = datasets.get(dataset_name)


            plot_folderpath.mkdir(parents=True,exist_ok=True)

            visualization.plot_invariant_feature_maps_pytorch(plot_folderpath,model,dataset,transformation_set,result,images=8,most_invariant_k=4,least_invariant_k=4,conv_aggregation=tm.ConvAggregation.mean)
            (finished).touch()








import argparse


def parse_args(experiments:[Experiment])->[Experiment]:


    parser = argparse.ArgumentParser(description="Run experiments with transformation measures.")

    experiment_names=[e.id() for e in experiments]
    experiment_dict=dict(zip(experiment_names,experiments))

    parser.add_argument('-experiment',
                        help=f'Choose an experiment to run',
                        type=str,
                        default=None,
                        required=False,
                        choices=experiment_names,)
    #argcomplete.autocomplete(parser)

    args = parser.parse_args()
    if args.experiment is None:
        return experiments
    else:
        return [experiment_dict[args.experiment]]


if __name__ == '__main__':
    todo = [InvarianceVsEpochs(),
            InvarianceVsMaxPooling(),
            InvarianceVsKernelSize(),
            CompareModelsForInvariance(),
            CompareModelsForInvariance(),
            ]
    print("TODO implement ",",".join([e.__class__.__name__ for e in todo]))


    all_experiments=[

        InvarianceWhileTraining(), # run this first or you'll need to retrain some models
        CompareMeasures(),
        # ComparePreConvAgg(),
        # CollapseConvBeforeOrAfter(),
        # # #
        MeasureVsDatasetSize(),
        InvarianceVsTransformationDiversity(),
        InvarianceVsTransformationDifferentScales(),
        CompareBN(),
        VisualizeInvariantFeatureMaps(),
        InvarianceAcrossDatasets(),
        # InvarianceForRandomNetworks(),

    ]

    experiments = parse_args(all_experiments)

    for e in experiments:
        e()



