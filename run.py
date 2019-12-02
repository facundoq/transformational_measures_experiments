#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import itertools
import os
from pathlib import Path
import texttable
import argcomplete
import numpy as np
from numpy import short

import config
import models
import transformation_measure as tm
from experiment import variance, training, model_loading, utils_runner
from transformation_measure import visualization

all_model_names = config.model_names

common_model_names = [
                      # models.SimplestConv.__name__,
                      models.SimpleConv.__name__,
                      models.AllConvolutional.__name__,
                      # models.VGGLike.__name__,
                      models.ResNet.__name__]
common_model_names.sort()

common_model_names_bn = [models.SimplestConvBN.__name__,
                         models.SimpleConvBN.__name__,
                         models.AllConvolutionalBN.__name__,
                         # models.VGGLikeBN.__name__,
                         models.ResNetBN.__name__]


common_model_names_except_resnet = [name for name in common_model_names if not name.startswith("ResNet")]

small_model_names = [models.SimpleConv.__name__, models.AllConvolutional.__name__]

simple_model_names = [models.SimpleConv.__name__]


measures = config.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]x

dataset_names = ["mnist", "cifar10"]
venv_path = ""

common_transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                       tm.SimpleAffineTransformationGenerator(t=5),
                       tm.SimpleAffineTransformationGenerator(s=5),
#                       tm.SimpleAffineTransformationGenerator(r=360, s=3, t=3),
                       ]

import abc


class Experiment(abc.ABC):

    def __init__(self):
        self.plot_folderpath = config.plots_base_folder() / self.id()
        self.plot_folderpath.mkdir(exist_ok=True,parents=True)
        with open(self.plot_folderpath / "description.txt", "w") as f:
            f.write(self.description())
        self.venv=Path(".")

    def id(self):
        return self.__class__.__name__
    def set_venv(self,venv:Path):
        self.venv=venv

    def __call__(self, force=False,venv=".",*args, **kwargs):
        stars = "*" * 15
        if not self.has_finished() or force:
            print(f"{stars} Running experiment {self.id()} {stars}")
            self.run()
            print(f"{stars} Finished experiment {self.id()} {stars}")
            self.mark_as_finished()
        else:
            print(f"{stars}Experiment {self.id()} already finished, skipping.{stars}")

    def has_finished(self):
        finished_filepath = self.plot_folderpath / "finished"
        return finished_filepath.exists()
    def mark_as_finished(self):
        finished_filepath = self.plot_folderpath / "finished"
        finished_filepath.touch(exist_ok=True)

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    def experiment_finished(self,p: training.Parameters):
        model_path = config.model_path(p)
        if model_path.exists():
            if p.savepoints == []:
                return True
            else:
                savepoint_missing = [sp for sp in p.savepoints if not config.model_path(p, sp).exists()]
                return savepoint_missing == []
        else:
            return False

    def experiment_training(self, p: training.Parameters, min_accuracy=None):
        if not min_accuracy:
            min_accuracy = config.min_accuracy(p.model, p.dataset)
        if self.experiment_finished(p):
            return
        if len(p.suffix)>0:
            suffix = f'-suffix "{p.suffix}"'
        else:
            suffix=""

        savepoints = ",".join([str(sp) for sp in p.savepoints])
        python_command = f'train.py -model "{p.model}" -dataset "{p.dataset}" -transformation "{p.transformations.id()}" -epochs {p.epochs}  -num_workers 4 -min_accuracy {min_accuracy} -max_restarts 5 -savepoints "{savepoints}" {suffix}'
        utils_runner.run_python(self.venv, python_command)

    def experiment_variance(self, p: variance.Parameters, model_path: Path, batch_size: int = 64, num_workers: int = 2,adapt_dataset=False):

        results_path = config.results_path(p)
        if os.path.exists(results_path):
            return
        if p.stratified:
            stratified = "-stratified"
        else:
            stratified = ""
        python_command = f'measure.py -mo "{model_path}" -me "{p.measure.id()}" -d "{p.dataset.id()}" -t "{p.transformations.id()}" -verbose False -batchsize {batch_size} -num_workers {num_workers} {stratified}'
        if adapt_dataset:
            python_command = f"{python_command} -adapt_dataset True"

        utils_runner.run_python(self.venv, python_command)



class CompareMeasures(Experiment):
    def description(self):
        return """Test different measures for a given dataset/model/transformation combination to evaluate their differences."""

    def run(self):
        mf, ca_none, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.none, tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measure_sets = {"Variance": [
            tm.TransformationMeasure(mf,ca_none),
            tm.SampleMeasure(mf, ca_none),
            # tm.TransformationVarianceMeasure(mf, ca_none),
            # tm.SampleVarianceMeasure(mf, ca_none),
        ],
            "Distance": [
                tm.DistanceTransformationMeasure(dmean),
                tm.DistanceSampleMeasure(dmean),
            ],
            "HighLevel": [
                tm.AnovaMeasure(ca_none, 0.99, bonferroni=True),
                tm.NormalizedMeasure(mf, ca_mean),
                # tm.NormalizedVarianceMeasure(mf, ca_none),
                tm.DistanceMeasure(dmean),
                # tm.GoodfellowMeasure()
                # tm.GoodfellowNormalMeasure(alpha=0.99)
            ],
            "Equivariance": [
                tm.DistanceSameEquivarianceMeasure(dmean),
                tm.DistanceTransformationMeasure(dmean),
            ]
        }

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        model_names = common_model_names
        #model_names = ["SimpleConv"]
        transformations = config.common_transformations_without_identity()

        combinations = itertools.product(model_names, dataset_names, transformations, measure_sets.items())
        for (model, dataset, transformation, measure_set) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # generate variance params
            variance_parameters = []
            measure_set_name, measures = measure_set
            for m in measures:
                p = 0.5 if m.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, m)
                variance_parameters.append(p_variance)
            # evaluate variance
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure_set_name}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [m.id() for m in measures]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)



class RandomInitialization(Experiment):
    def description(self):
        return """Test measures with various instances of the same architecture/transformation/dataset to see if the measure is dependent on the random initialization in the training or simply on the architecture"""

    def run(self):
        mf, ca_none, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.none, tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures=[
                    # tm.AnovaMeasure(ca_none, 0.99, bonferroni=True),
                    tm.NormalizedMeasure(mf, ca_mean),
                    tm.DistanceMeasure(dmean),
                    tm.DistanceSameEquivarianceMeasure(dmean),
        ]
        repetitions = 8
        model_names = [models.SimpleConv.__name__,models.SimpleConvBN.__name__]
        transformations = config.common_transformations_without_identity()

        combinations = itertools.product(model_names, dataset_names, transformations,measures)
        for (model, dataset, transformation,measure) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            training_parameters= []
            for r in range(repetitions):
                p_training = training.Parameters(model, dataset, transformation, epochs, 0,suffix=f"rep{r:02}")
                self.experiment_training(p_training)
                training_parameters.append(p_training)
            # generate variance params
            variance_parameters = []
            for p_training in training_parameters:
                model_path = config.model_path(p_training)
                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))

            visualization.plot_collapsing_layers_same_model(results, plot_filepath,plot_mean=True)


class DatasetSize(Experiment):

    def description(self):
        return '''Vary the test dataset size and see how it affects the measure's value. That is, vary the size of the dataset used to compute the invariance (not the training dataset) and see how it affects the calculation of the measure.'''

    def run(self):
        dataset_sizes = [0.01, 0.05, 0.1, 0.5, 1.0]
        model_names = simple_model_names
        mf, ca_none, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.none, tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures = [
            # tm.AnovaMeasure(ca_none, 0.99, bonferroni=True),
            tm.NormalizedMeasure(mf, ca_mean),
            tm.DistanceMeasure(dmean),
            tm.DistanceSameEquivarianceMeasure(dmean),
        ]
        combinations = list(itertools.product(
            model_names, dataset_names, config.common_transformations_without_identity(), measures))
        for i, (model, dataset, transformation, measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}", end=", ")
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)
            p_datasets = [variance.DatasetParameters(dataset, variance.DatasetSubset.test, p) for p in dataset_sizes]
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, measure) for
                                   p_dataset in p_datasets]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [f"Dataset percentage: {d.percentage * 100:2}%" for d in p_datasets]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class DatasetSubset(Experiment):

    def description(self):
        return '''Vary the test dataset subset (either train o testing) and see how it affects the measure's value.'''

    def run(self):
        dataset_sizes = [(variance.DatasetSubset.test, 0.5), (variance.DatasetSubset.train, 0.1)]

        model_names = simple_model_names
        mf, ca_none, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.none, tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures = [
            # tm.AnovaMeasure(ca_none, 0.99, bonferroni=True),
            tm.NormalizedMeasure(mf, ca_mean),
            tm.DistanceMeasure(dmean),
            tm.DistanceSameEquivarianceMeasure(dmean),
        ]
        combinations = list(itertools.product(
            model_names , dataset_names, config.common_transformations_without_identity(), measures))

        for i, (model, dataset, transformation, measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}", end=", ")
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)

            p_datasets = []
            for (subset, p) in dataset_sizes:
                if measure.__class__ == tm.AnovaMeasure.__class__:
                    p = p * 2
                p_datasets.append(variance.DatasetParameters(dataset, subset, p))
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, measure) for
                                   p_dataset in p_datasets]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [f"Dataset subset {d.subset},  (percentage of data {d.percentage * 100:2})%" for d in p_datasets]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class TransformationDiversity(Experiment):

    def description(self):
        return '''Vary the type of transformation both when training and computing the measure, and see how it affects the invariance. For example, train with rotations, then measure with translations. Train with translations. measure with scales, and so on. '''

    def run(self):
        measure_function, conv_agg = tm.MeasureFunction.std, tm.ConvAggregation.none
        measure = tm.NormalizedMeasure(measure_function, conv_agg)
        distance_measure = tm.DistanceMeasure(tm.DistanceAggregation.mean)
        measures = [measure, distance_measure]

        combinations = itertools.product(simple_model_names, dataset_names, measures)
        transformations = [tm.SimpleAffineTransformationGenerator()]+common_transformations

        for model, dataset, measure in combinations:
            for i, train_transformation in enumerate(transformations):
                # transformation_plot_folderpath = self.plot_folderpath / name
                # transformation_plot_folderpath.mkdir(exist_ok=True,parents=True)
                experiment_name = f"{model}_{dataset}_{measure.id()}_{train_transformation}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
                variance_parameters = []
                print(f"Train transformation {train_transformation}")
                epochs = config.get_epochs(model, dataset, train_transformation)
                p_training = training.Parameters(model, dataset, train_transformation, epochs, 0)
                self.experiment_training(p_training)
                for i, test_transformation in enumerate(transformations):
                    print(f"{i}, ", end="")
                    p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
                    p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                    model_path = config.model_path(p_training)
                    self.experiment_variance(p_variance, model_path)
                    variance_parameters.append(p_variance)
                results = config.load_results(config.results_paths(variance_parameters))
                labels = [str(t.parameters.transformations) for t in results]
                visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class TransformationComplexity(Experiment):

    def description(self):
        return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 8 rotations, measure variance with 2, 4, 8 and 16. """

    def run(self):
        n_transformations = 5
        measure_function, conv_agg = tm.MeasureFunction.std, tm.ConvAggregation.mean
        dmean = tm.DistanceAggregation.mean
        measures = [tm.NormalizedMeasure(measure_function, conv_agg),
                    tm.DistanceMeasure(dmean),
                    tm.DistanceSameEquivarianceMeasure(dmean ),
                    ]
        combinations = itertools.product(simple_model_names, dataset_names,measures)

        names = ["rotation", "translation", "scale"]
        sets = [config.rotation_transformations(8), config.translation_transformations(3),
                config.scale_transformations(3)]

        for model, dataset,measure in combinations:

            for i, (transformation_set, name) in enumerate(zip(sets, names)):
                n_experiments = (len(transformation_set) + 1) * len(transformation_set)
                print(f"    {name}, #experiments:{n_experiments}")
                # include identity the transformation set
                transformation_set = [tm.SimpleAffineTransformationGenerator()] + transformation_set
                for j, train_transformation in enumerate(transformation_set):
                    transformation_plot_folderpath = self.plot_folderpath / name

                    transformation_plot_folderpath.mkdir(exist_ok=True,parents=True)
                    experiment_name = f"{model}_{dataset}_{measure.id()}_{train_transformation.id()}"
                    plot_filepath = transformation_plot_folderpath / f"{experiment_name}.png"
                    variance_parameters = []
                    print(f"{j}, ", end="")
                    epochs = config.get_epochs(model, dataset, train_transformation)
                    p_training = training.Parameters(model, dataset, train_transformation, epochs, 0)
                    self.experiment_training(p_training)
                    for k, test_transformation in enumerate(transformation_set):
                        p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
                        p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                        model_path = config.model_path(p_training)
                        self.experiment_variance(p_variance, model_path)
                        variance_parameters.append(p_variance)

                    title = f"Invariance to \n. Model: {model}, Dataset: {dataset}, Measure {measure.id()} \n Train transformation: {train_transformation.id()} "
                    labels = [f"Test transformation: {t}" for t in transformation_set]
                    results = config.load_results(config.results_paths(variance_parameters))
                    visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, title=title)


class CollapseConvBeforeOrAfter(Experiment):
    def description(self):
        return """Collapse convolutions spatial dims after/before computing variance."""

    def run(self):
        cas = [tm.ConvAggregation.sum, tm.ConvAggregation.mean, tm.ConvAggregation.max]
        model_names = ["SimpleConv"] #simple_model_names#small_model_names

        measure_sets_names=[
                            "NormalizedVariance",
                            # "Anova"
                            ]
        measure_sets={
            "NormalizedVariance" : [tm.NormalizedMeasure(tm.MeasureFunction.std, ca) for ca in cas],
            # "Anova" :[tm.AnovaMeasure(conv_aggregation=ca) for ca in cas],
        }
        measure_sets_no_agg = {
            "NormalizedVariance": tm.NormalizedMeasure(tm.MeasureFunction.std, tm.ConvAggregation.none),
            # "Anova": tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.none) ,
        }
        post_functions = [tm.ConvAggregation.mean]

        combinations = itertools.product(
            model_names , dataset_names, config.common_transformations_without_identity(),measure_sets_names)
        for (model, dataset, transformation,measure_set_name) in combinations:
            # train
            measures= measure_sets[measure_set_name]
            no_agg_measure=measure_sets_no_agg[measure_set_name]
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # eval variance
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)

            no_aggregation_parameters = variance.Parameters(p_training.id(), p_dataset, transformation, no_agg_measure)
            self.experiment_variance(no_aggregation_parameters, model_path)

            post_result_sets = {"all": cas, "mean": post_functions}
            for set, functions in post_result_sets.items():
                post_results = config.load_results(config.results_paths([no_aggregation_parameters]*len(functions)))
                for ca, r in zip(functions, post_results):
                    r.measure_result = r.measure_result.collapse_convolutions(ca)

                # plot
                experiment_name = f"{model}_{p_dataset.id()}_{transformation.id()}_{set}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
                results = config.load_results(config.results_paths(variance_parameters))

                labels = [f"Pre : {m.id()}" for m in measures]
                post_labels = [f"Post: {no_agg_measure.id()} ({f})" for f in functions]
                visualization.plot_collapsing_layers_same_model(results + post_results, plot_filepath, labels=labels + post_labels)


class ComparePreConvAgg(Experiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate their differences. Convolutional aggregation collapses all the spatial dimensions of feature maps so that a single variance value for the feature map can be obtained."""

    def run(self):
        functions = [tm.ConvAggregation.sum, tm.ConvAggregation.mean, tm.ConvAggregation.max, tm.ConvAggregation.none]
        measure_sets_constructors = {
            "nm": tm.NormalizedMeasure
            , "sm": tm.SampleMeasure
            , "tm": tm.TransformationMeasure}
        measure_sets = []
        for set_name, measure_constructor in measure_sets_constructors.items():
            measure_objects = [measure_constructor(tm.MeasureFunction.std, f) for f in functions]
            measure_sets.append((set_name, measure_objects))

        combinations = itertools.product(
            simple_model_names , dataset_names, config.common_transformations_without_identity(), measure_sets)
        for model, dataset, transformation, (set_name, measures) in combinations:
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            experiment_name = f"{model}_{p_dataset.id()}_{transformation.id()}_{set_name}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [m.id() for m in measures]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


import datasets
import torch


class RandomWeights(Experiment):
    def description(self):
        return """Analyze the invariance of untrained networks, ie, with random weights."""

    def run(self):
        random_models_folderpath = config.models_folder() / "random"
        random_models_folderpath.mkdir(exist_ok=True,parents=True)
        o = training.Options(False, False, False, 32, 4, torch.cuda.is_available(), False, 0)
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        mf, ca_mean, ca_none = tm.MeasureFunction.std, tm.ConvAggregation.mean, tm.ConvAggregation.none
        measures = [
            tm.NormalizedMeasure(mf, ca_mean),
            # tm.AnovaMeasure(ca_none, alpha=0.99, bonferroni=True),
            tm.DistanceMeasure(dmean),
            tm.DistanceSameEquivarianceMeasure(dmean),
        ]
        dataset_percentages = [0.1, 0.5,0.5]
        # number of random models to generate
        random_model_n = 30

        mp = zip(measures, dataset_percentages)
        combinations = itertools.product(
            simple_model_names, dataset_names, config.common_transformations_without_identity(), mp)
        for model_name, dataset_name, transformation, (measure, p) in combinations:
            # generate `random_model_n` models and save them without training
            models_paths = []
            p_training = training.Parameters(model_name, dataset_name, transformation, 0)
            dataset = datasets.get(dataset_name)
            for i in range(random_model_n):

                model_path = config.model_path(p_training, model_folderpath=random_models_folderpath)

                # append index to model name
                name, ext = os.path.splitext(str(model_path))
                name += f"_random{i:03}"
                model_path = Path(f"{name}{ext}")
                if not model_path.exists():
                    model, optimizer = model_loading.get_model(model_name, dataset, use_cuda=o.use_cuda)
                    scores = training.eval_scores(model, dataset, p_training, o)
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
            experiment_name = f"{model_name}_{dataset_name}_{transformation.id()}_{measure}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            n = len(results)
            labels = [f"Random models ({n} samples)."] + ([None] * (n - 1))
            # get alpha colors
            import matplotlib.pyplot as plt
            color = plt.cm.hsv(np.linspace(0.1, 0.9, n))
            color[:, 3] = 0.5
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, plot_mean=True,
                                                 labels=labels, color=color)


class DuringTraining(Experiment):
    def description(self):
        return """Analyze the evolution of invariance in models while they are trained."""

    def run(self):
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        mf, ca_sum, ca_none = tm.MeasureFunction.std, tm.ConvAggregation.sum, tm.ConvAggregation.none
        ca_mean = tm.ConvAggregation.mean
        measures = [tm.NormalizedMeasure(mf, ca_mean)
            # , tm.AnovaMeasure(ca_none, alpha=0.99, bonferroni=True)
            , tm.DistanceMeasure(dmean)
            , tm.DistanceSameEquivarianceMeasure(dmean)
                    ]
        dataset_percentages = [0.1, 0.5]

        savepoints_percentages = [0,1,2,3,4,5,10,20,30,40,50,100]
        mp = zip(measures, dataset_percentages)
        combinations = itertools.product(
            simple_model_names, dataset_names, config.common_transformations_without_identity(), mp)
        for model, dataset, transformation, (measure, p) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            savepoints = [ sp * epochs // 100 for sp in savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
            p_training = training.Parameters(model, dataset, transformation, epochs, savepoints=savepoints)
            self.experiment_training(p_training)
            # generate variance params
            variance_parameters = []
            model_paths = []
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
            for sp in savepoints:
                model_path = config.model_path(p_training, savepoint=sp)
                model_id = p_training.id(savepoint=sp)
                p_variance = variance.Parameters(model_id, p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                model_paths.append(model_path)

            for p_variance, model_path in zip(variance_parameters, model_paths):
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            # TODO implement a heatmap where the x axis is the training time/epoch
            # and the y axis indicates the layer, and the color indicates the invariance
            # to see it evolve over time.
            accuracies=[]
            for model_path in model_paths:
                _,_,_,score = training.load_model(model_path,torch.cuda.is_available(),False)
                loss,accuracy=score["test"]
                accuracies.append(accuracy)

            labels = [f"Epoch {sp*epochs//100} ({sp}%), accuracy {accuracy}" for (sp,accuracy) in zip(savepoints,accuracies)]

            legend_location = ("center right", (1.25,0.5))
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,legend_location=legend_location)


class BatchNormalization(Experiment):
    def description(self):
        return """Compare invariance of models trained with/without batchnormalization."""

    def run(self):
        mf, ca = tm.MeasureFunction.std, tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures = [
            # tm.AnovaMeasure(tm.ConvAggregation.none, 0.99),
            tm.NormalizedMeasure(mf, ca),
            tm.DistanceMeasure(dmean),
            tm.DistanceSameEquivarianceMeasure(dmean),
        ]

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]
        model_pairs = [ ("SimpleConv","SimpleConvBN")] #zip(common_model_names_bn, common_model_names)

        combinations = itertools.product(
            model_pairs, dataset_names, config.common_transformations_without_identity(), measures)
        for (model_pair, dataset, transformation, measure) in combinations:
            # train

            variance_parameters = []
            for model in model_pair:
                epochs = config.get_epochs(model, dataset, transformation)
                p_training = training.Parameters(model, dataset, transformation, epochs)
                self.experiment_training(p_training)

                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                batch_size = 64
                if model.startswith("ResNet"):
                    batch_size = 32
                self.experiment_variance(p_variance, model_path, batch_size=batch_size)
                variance_parameters.append(p_variance)

            # evaluate variance
            model, model_bn = model_pair
            # plot results
            experiment_name = f"{model}_{model_bn}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath/ f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = model_pair
            visualization.plot_collapsing_layers_different_models(results, plot_filepath, labels=labels)


class DatasetTransfer(Experiment):
    def description(self):
        return """Measure invariance with a different dataset than the one used to train the model."""

    def run(self):
        mf, ca = tm.MeasureFunction.std, tm.ConvAggregation.mean
        measures = [
                    # tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.none, alpha=0.99),
                    tm.NormalizedMeasure(mf, ca),
                    tm.DistanceMeasure(tm.DistanceAggregation.mean)
        ]

        combinations = itertools.product(
            simple_model_names , dataset_names, config.common_transformations_without_identity(), measures)
        for (model, dataset, transformation, measure) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs)
            self.experiment_training(p_training)

            variance_parameters = []
            for dataset_test in dataset_names:
                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset_test, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                self.experiment_variance(p_variance, model_path, adapt_dataset=True)
                variance_parameters.append(p_variance)

            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = dataset_names
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)





class CompareModels(Experiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""

    def run(self):
        mf, ca_none = tm.MeasureFunction.std, tm.ConvAggregation.none
        ca_mean = tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures  = [tm.NormalizedMeasure(mf, ca_mean),
                     tm.DistanceMeasure(dmean),
                     ]

        model_names = common_model_names
        transformations = [tm.SimpleAffineTransformationGenerator(r=360)]
        # TODO change back when first round finishes
        #transformations =config.common_transformations_without_identity()

        combinations = itertools.product( dataset_names, transformations, measures)
        for (dataset, transformation, measure) in combinations:
            variance_parameters = []
            for model in model_names:
                # train
                epochs = config.get_epochs(model, dataset, transformation)
                p_training = training.Parameters(model, dataset, transformation, epochs, 0)
                self.experiment_training(p_training)
                # generate variance params
                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
                # evaluate variance
                model_path = config.model_path(p_training)
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [m for m in model_names]
            visualization.plot_collapsing_layers_different_models(results, plot_filepath, labels=labels)

class CompareAnovaMeasures(Experiment):
    def description(self):
        return """Determine which Anova measure is more appropriate"""

    def run(self):
        pass


class KernelSize(Experiment):
    def description(self):
        return """Determine how the kernel sizes affect invariance"""

    def run(self):
        pass

class Stratified(Experiment):
    def description(self):
        return """Determine the differences between stratified and non-stratified measures."""

    def run(self):
        mf, ca_none, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.none, tm.ConvAggregation.mean
        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        measures = [
            # tm.TransformationMeasure(mf, ca_none),
            # tm.SampleMeasure(mf, ca_none),
            tm.AnovaMeasure(ca_mean, 0.99, bonferroni=True),
            tm.NormalizedMeasure(mf, ca_mean),
            # tm.DistanceTransformationMeasure(dmean),
            # tm.DistanceSampleMeasure(dmean),
            tm.DistanceMeasure(dmean),
            tm.DistanceSameEquivarianceMeasure(dmean),
        ]

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        model_names = simple_model_names
        transformations = config.common_transformations_without_identity()

        combinations = itertools.product(model_names, dataset_names, transformations, measures)
        for (model, dataset, transformation, measure) in combinations:
            # train
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # generate variance params
            p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
            p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
            p_variance_stratified = variance.Parameters(p_training.id(), p_dataset, transformation, measure,stratified=True)
            # evaluate variance
            model_path = config.model_path(p_training)
            self.experiment_variance(p_variance, model_path)
            self.experiment_variance(p_variance_stratified, model_path)
            variance_parameters=[p_variance,p_variance_stratified]
            # plot results
            experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = ["Non-stratified", "Stratified"]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)



class MaxPooling(Experiment):
    def description(self):
        return """Determine wheter MaxPooling affects the invariance structure of the network or it is similar to a network with strided convolutions"""

    def run(self):
        pass


class VisualizeInvariantFeatureMaps(Experiment):
    def description(self):
        return """Visualize the output of invariant feature maps, to analyze qualitatively if they are indeed invariant."""

    def run(self):

        dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        mf, ca_sum, ca_mean = tm.MeasureFunction.std, tm.ConvAggregation.sum, tm.ConvAggregation.mean
        ca_none = tm.ConvAggregation.none
        measures = [
            # tm.AnovaMeasure(ca_none, alpha=0.99,bonferroni=True),
            # #tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.mean, alpha=0.99,bonferroni=True),
            # tm.NormalizedMeasure(mf, ca_sum),
            tm.NormalizedMeasure(mf, ca_mean),
            # tm.TransformationMeasure(mf, tm.ConvAggregation.none),
            # tm.TransformationMeasure(mf, ca_sum),
            # tm.DistanceTransformationMeasure(mf, dmean),
            # tm.DistanceTransformationMeasure(mf, dmax),
            tm.DistanceMeasure(dmean),
            # tm.DistanceMeasure(dmax),
            # tm.DistanceSameEquivarianceMeasure(dmax),
            tm.DistanceSameEquivarianceMeasure(dmean),

        ]
        #conv_model_names = [m for m in common_model_names if (not "FFNet" in m)]
        conv_model_names = simple_model_names # [models.SimpleConv.__name__]
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

            model_filepath = config.model_path(p_training)
            model, p_model, o, scores = training.load_model(model_filepath, use_cuda=torch.cuda.is_available())
            result_filepath = config.results_path(p_variance)
            result = config.load_result(result_filepath)
            dataset = datasets.get(dataset_name)

            plot_folderpath.mkdir(parents=True, exist_ok=True)

            visualization.plot_invariant_feature_maps_pytorch(plot_folderpath, model, dataset, transformation_set,result, images=2, most_invariant_k=4, least_invariant_k=4,conv_aggregation=tm.ConvAggregation.mean)
            finished.touch()


class ValidateMeasure(Experiment):

    def description(self):
        return """Validate measure/transformation. Just for testing purposes."""

    def run(self):
        measure_function, conv_agg = tm.MeasureFunction.std, tm.ConvAggregation.none
        measures = [
            # tm.TransformationVarianceMeasure(measure_function, conv_agg),
            # tm.SampleVarianceMeasure(measure_function, conv_agg),
            # tm.NormalizedVarianceMeasure(measure_function, conv_agg),

            # tm.TransformationMeasure(measure_function, conv_agg),
            # tm.SampleMeasure(measure_function, conv_agg),
            # tm.NormalizedMeasure(measure_function, conv_agg),
            tm.NormalizedMeasure(measure_function, tm.ConvAggregation.mean),

            # tm.GoodfellowNormalMeasure(),
        ]
        model_names = ["SimplestConv"]
        dataset_names = ["cifar10"]
        transformations = [tm.SimpleAffineTransformationGenerator(r=360)]
        #transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, measures, transformations)
        for model, dataset, measure, transformation in combinations:
            experiment_name = f"{model}_{dataset}_{measure.id()}_{transformation.id()}"
            epochs = config.get_epochs(model, dataset, transformation)
            p_training = training.Parameters(model, dataset, transformation, epochs, 0,savepoints=[0,10,20,30,40,50,60,70,80,100])
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            p_measure = variance.Parameters(p_training.id(), p_dataset, transformation,measure)

            self.experiment_training(p_training)
            model_path = config.model_path(p_training)


            self.experiment_variance(p_measure, model_path)
            print(experiment_name)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"

            title = f"Invariance to \n. Model: {model}, Dataset: {dataset}, Measure {measure.id()} \n transformation: {transformation.id()} "
            result = config.load_result(config.results_path(p_measure))
            print(config.results_path(p_measure))
            visualization.plot_heatmap(result.measure_result,plot_filepath,title=title)

class Options:
    def __init__(self,venv:Path,show_list:bool,force:bool):
        self.venv=venv
        self.show_list=show_list
        self.force = force



def parse_args(experiments: [Experiment]) -> ([Experiment],Options):
    parser = argparse.ArgumentParser(description="Run experiments with transformation measures.")

    experiment_names = [e.id() for e in experiments]
    experiment_dict = dict(zip(experiment_names, experiments))

    parser.add_argument('-experiment',
                        help=f'Choose an experiment to run',
                        type=str,
                        default=None,
                        required=False,
                        choices=experiment_names, )
    parser.add_argument('-force',
                        help=f'Force experiments to rerun even if they have already finished',
                        action="store_true")
    parser.add_argument('-list',
                        help=f'Force experiments to rerun even if they have already finished',
                        action="store_true")
    parser.add_argument("-venv",
                        help="Path to virtual environment to run experiments on",
                        type=str,
                        default=os.path.expanduser("~/dev/env/vm"),
                        )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not args.experiment is None:
        experiments = [experiment_dict[args.experiment]]

    return experiments,Options(Path(args.venv),args.list,args.force)

import warnings

if __name__ == '__main__':
    # warnings.simplefilter('error', UserWarning)

    todo = [
            MaxPooling(),
            KernelSize(),


            ]
    print("TODO implement ", ",".join([e.__class__.__name__ for e in todo]))

    all_experiments = [

        DuringTraining(),  # run this first or you'll need to retrain some models
        CompareMeasures(),
        Stratified(),
        CompareModels(),
        VisualizeInvariantFeatureMaps(),

        RandomInitialization(),
        RandomWeights(),

        DatasetSize(),
        DatasetSubset(),
        DatasetTransfer(),

        ComparePreConvAgg(),
        CollapseConvBeforeOrAfter(),

        TransformationDiversity(),
        TransformationComplexity(),

        BatchNormalization(),


        ValidateMeasure(),


    ]

    experiments, o = parse_args(all_experiments)
    if o.show_list:
        table = texttable.Texttable()
        header = ["Experiment", "Finished"]
        table.header(header)
        experiments.sort(key=lambda e: e.__class__.__name__)
        for e in experiments:
            status = "Y" if e.has_finished() else "N"
            name = e.__class__.__name__
            name = name[:40]
            table.add_row((name,status))
            #print(f"{name:40}     {status}")
        print(table.draw())
    else:
        for e in experiments:
            e.set_venv(o.venv)
            e(force=o.force)
