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
from experiment import variance, training,utils_runner
from transformation_measure import visualization


common_models_generators = [
                  config.SimpleConvConfig,
                  config.AllConvolutionalConfig,
                  config.VGG16DConfig,
                  config.ResNetConfig]


small_models_generators = [config.SimpleConvConfig,
                           config.AllConvolutionalConfig, ]

simple_models_generators = [config.SimpleConvConfig]
common_models_generators  = simple_models_generators

measures = config.common_measures()

dataset_names = ["mnist", "cifar10"]
venv_path = ""

common_transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                       tm.SimpleAffineTransformationGenerator(t=5),
                       tm.SimpleAffineTransformationGenerator(s=5),
                      tm.SimpleAffineTransformationGenerator(r=360, s=3, t=3),
                       ]

ca_none, ca_mean,ca_sum =  tm.ConvAggregation.none, tm.ConvAggregation.mean, tm.ConvAggregation.mean
da_mean, da_max, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max

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


        measure_sets = {"Variance": [
            tm.TransformationVariance(),
            tm.SampleVariance(),
            # tm.TransformationVarianceMeasure(mf, ca_none),
            # tm.SampleVarianceMeasure(mf, ca_none),
        ],
            "Distance": [
                tm.DistanceTransformationMeasure(da_mean),
                tm.DistanceSampleMeasure(da_mean),
            ],
            "HighLevel": [
                tm.AnovaMeasure(0.99, bonferroni=True),
                tm.NormalizedVariance(ca_sum),
                # tm.NormalizedVarianceMeasure(mf, ca_none),
                tm.DistanceMeasure(da_mean),
                # tm.GoodfellowMeasure()
                # tm.GoodfellowNormalMeasure(alpha=0.99)
            ],
            "Equivariance": [
                tm.DistanceSameEquivarianceMeasure(da_mean),
                # tm.DistanceTransformationMeasure(dmean),
            ]
        }

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        #model_generators = common_models_generators
        model_generators = simple_models_generators
        #model_names = ["SimpleConv"]
        transformations = config.common_transformations_without_identity()

        combinations = itertools.product(model_generators, dataset_names, transformations, measure_sets.items())
        for (model_config_generator, dataset, transformation, measure_set) in combinations:
            model_config=model_config_generator.for_dataset(dataset)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            # generate variance params
            variance_parameters = []
            measure_set_name, measures = measure_set
            for measure in measures:
                p = config.dataset_size_for_measure(measure)
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
            # evaluate variance
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)

            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure_set_name}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = [m.id() for m in measures]
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)



class RandomInitialization(Experiment):
    def description(self):
        return """Test measures with various instances of the same architecture/transformation/dataset to see if the measure is dependent on the random initialization in the training or simply on the architecture"""

    def run(self):
        measures=[
                    # tm.AnovaMeasure(ca_none, 0.99, bonferroni=True),
                    tm.NormalizedVariance(ca_mean),
                    tm.DistanceMeasure(da_mean),
                    tm.DistanceSameEquivarianceMeasure(da_mean),
        ]
        repetitions = 8

        model_generators = simple_models_generators
        transformations = config.common_transformations_without_identity()

        combinations = itertools.product(model_generators , dataset_names, transformations,measures)
        for (model_generator, dataset, transformation,measure) in combinations:
            # train
            epochs = config.get_epochs(model_generator, dataset, transformation)
            training_parameters= []
            for r in range(repetitions):
                p_training = training.Parameters(model_generator, dataset, transformation, epochs, 0,suffix=f"rep{r:02}")
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
            experiment_name = f"{model_generator}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))

            visualization.plot_collapsing_layers_same_model(results, plot_filepath,plot_mean=True)


class DatasetSize(Experiment):

    def description(self):
        return '''Vary the test dataset size and see how it affects the measure's value. That is, vary the size of the dataset used to compute the invariance (not the training dataset) and see how it affects the calculation of the measure.'''

    def run(self):
        dataset_sizes = [0.01, 0.05, 0.1, 0.5, 1.0]
        model_names = simple_models_generators
        measures = [
            # tm.AnovaMeasure(ca_none, 0.99, bonferroni=True),
            tm.NormalizedVariance(ca_mean),
            tm.DistanceMeasure(da_mean),
            tm.DistanceSameEquivarianceMeasure(da_mean),
        ]
        combinations = list(itertools.product(
            model_names, dataset_names, config.common_transformations_without_identity(), measures))
        for i, (model, dataset, transformation, measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}", end=", ")
            model_config = model.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)
            p_datasets = [variance.DatasetParameters(dataset, variance.DatasetSubset.test, p) for p in dataset_sizes]
            experiment_name = f"{model_config}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, measure) for p_dataset in p_datasets]
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
        dataset_subsets = [(variance.DatasetSubset.test, 1), (variance.DatasetSubset.train, 0.2)]

        model_names = simple_models_generators
        measures = [
            tm.AnovaMeasure(0.99),
            tm.NormalizedVariance(ca_mean),
            tm.DistanceMeasure(da_mean),
            tm.DistanceSameEquivarianceMeasure(da_mean),
        ]
        combinations = list(itertools.product(
            model_names , dataset_names, config.common_transformations_without_identity(), measures))

        for i, (model_config_generator, dataset, transformation, measure) in enumerate(combinations):
            print(f"{i}/{len(combinations)}", end=", ")

            model_config=model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)

            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)

            p_datasets = []
            for subset, multiplier in dataset_subsets:
                p = config.dataset_size_for_measure(measure) * multiplier
                p_datasets.append(variance.DatasetParameters(dataset, subset, p))
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, measure) for p_dataset in p_datasets]
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
        measure = tm.NormalizedVariance(ca_sum)
        distance_measure = tm.DistanceMeasure(tm.DistanceAggregation.mean)
        measures = [measure, distance_measure]

        combinations = itertools.product(simple_models_generators, dataset_names, measures)
        transformations = [tm.SimpleAffineTransformationGenerator()]+common_transformations

        for model_config_generator, dataset, measure in combinations:
            for i, train_transformation in enumerate(transformations):
                # transformation_plot_folderpath = self.plot_folderpath / name
                # transformation_plot_folderpath.mkdir(exist_ok=True,parents=True)
                model_config = model_config_generator.for_dataset(dataset)
                experiment_name = f"{model_config .name}_{dataset}_{measure.id()}_{train_transformation}"
                plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
                variance_parameters = []
                print(f"Train transformation {train_transformation}")
                epochs = config.get_epochs(model_config, dataset, train_transformation)
                p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
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
        measures = [tm.NormalizedVariance(ca_sum),
                    # tm.DistanceMeasure(da_mean),
                    # tm.DistanceSameEquivarianceMeasure(da_mean),
                    ]
        combinations = itertools.product(simple_models_generators, dataset_names, measures)


        test_sets = [config.rotation_transformations(4),
                [tm.SimpleAffineTransformationGenerator(s=i) for i in [1,3,5]],
                [tm.SimpleAffineTransformationGenerator(t=i) for i in [1,3,5]],
                ]
        train_transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                               tm.SimpleAffineTransformationGenerator(s=5),
                               tm.SimpleAffineTransformationGenerator(t=5),
                            ]
        for model_config_generator, dataset,measure in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            for train_transformation,transformation_set in zip(train_transformations,test_sets):
                    # TRAIN
                    epochs = config.get_epochs(model_config, dataset, train_transformation)
                    p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                    self.experiment_training(p_training)
                    # MEASURE
                    variance_parameters = []
                    for k, test_transformation in enumerate(transformation_set):
                        p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
                        p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                        model_path = config.model_path(p_training)
                        self.experiment_variance(p_variance, model_path)
                        variance_parameters.append(p_variance)
                    # PLOT
                    experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
                    plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
                    title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n Train transformation: {train_transformation.id()}"
                    labels = [f"Test transformation: {t}" for t in transformation_set[1:] ]

                    results = config.load_results(config.results_paths(variance_parameters[1:]))
                    visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, title=title)

class TransformationComplexityDetailed(Experiment):

    def description(self):
        return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 8 rotations, measure variance with 2, 4, 8 and 16. """

    def run(self):
        measures = [tm.NormalizedVariance(ca_sum),
                    tm.DistanceMeasure(da_mean),
                    tm.DistanceSameEquivarianceMeasure(da_mean),
                    ]
        combinations = itertools.product(simple_models_generators, dataset_names, measures)

        names = ["rotation", "translation", "scale"]
        sets = [config.rotation_transformations(8),
                config.translation_transformations(4),
                config.scale_transformations(4)]

        for model_config_generator, dataset,measure in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            for i, (transformation_set, name) in enumerate(zip(sets, names)):
                n_experiments = (len(transformation_set) + 1) * len(transformation_set)
                print(f"    {name}, #experiments:{n_experiments}")
                # include identity the transformation set
                transformation_set = [tm.SimpleAffineTransformationGenerator()] + transformation_set
                for j, train_transformation in enumerate(transformation_set):
                    transformation_plot_folderpath = self.plot_folderpath / name

                    transformation_plot_folderpath.mkdir(exist_ok=True,parents=True)
                    experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
                    plot_filepath = transformation_plot_folderpath / f"{experiment_name}.png"
                    variance_parameters = []
                    print(f"{j}, ", end="")
                    epochs = config.get_epochs(model_config, dataset, train_transformation)
                    p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                    self.experiment_training(p_training)
                    for k, test_transformation in enumerate(transformation_set):
                        p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
                        p_variance = variance.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                        model_path = config.model_path(p_training)
                        self.experiment_variance(p_variance, model_path)
                        variance_parameters.append(p_variance)

                    title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n Train transformation: {train_transformation.id()}"
                    labels = [f"Test transformation: {t}" for t in transformation_set[1:] ]

                    results = config.load_results(config.results_paths(variance_parameters[1:]))
                    visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, title=title)


# class CollapseConvBeforeOrAfter(Experiment):
#     def description(self):
#         return """Collapse convolutions spatial dims after/before computing variance."""
#
#     def run(self):
#         cas = [tm.ConvAggregation.sum, tm.ConvAggregation.mean, tm.ConvAggregation.max]
#         model_names = ["SimpleConv"] #simple_model_names#small_model_names
#
#         measure_sets_names=[
#                             "NormalizedVariance",
#                             # "Anova"
#                             ]
#         measure_sets={
#             "NormalizedVariance" : [tm.NormalizedVariance(ca) for ca in cas],
#             # "Anova" :[tm.AnovaMeasure(conv_aggregation=ca) for ca in cas],
#         }
#         measure_sets_no_agg = {
#             "NormalizedVariance": tm.NormalizedVariance(ca_none),
#             # "Anova": tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.none) ,
#         }
#         post_functions = [tm.ConvAggregation.mean]
#
#         combinations = itertools.product(
#             model_names , dataset_names, config.common_transformations_without_identity(),measure_sets_names)
#         for (model_config_generator, dataset, transformation,measure_set_name) in combinations:
#             model_config = model_config_generator.for_dataset(dataset)
#             # train
#             measures= measure_sets[measure_set_name]
#             no_agg_measure=measure_sets_no_agg[measure_set_name]
#             epochs = config.get_epochs(model_config, dataset, transformation)
#             p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
#             self.experiment_training(p_training)
#             # eval variance
#             p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
#             variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
#             model_path = config.model_path(p_training)
#             for p_variance in variance_parameters:
#                 self.experiment_variance(p_variance, model_path)
#
#             no_aggregation_parameters = variance.Parameters(p_training.id(), p_dataset, transformation, no_agg_measure)
#             self.experiment_variance(no_aggregation_parameters, model_path)
#
#             post_result_sets = {"all": cas, "mean": post_functions}
#             for set, functions in post_result_sets.items():
#                 post_results = config.load_results(config.results_paths([no_aggregation_parameters]*len(functions)))
#                 for ca, r in zip(functions, post_results):
#                     r.measure_result = r.measure_result.collapse_convolutions(ca)
#
#                 # plot
#                 experiment_name = f"{model_config.name}_{p_dataset.id()}_{transformation.id()}_{set}"
#                 plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
#                 results = config.load_results(config.results_paths(variance_parameters))
#
#                 labels = [f"Pre : {m.id()}" for m in measures]
#                 post_labels = [f"Post: {no_agg_measure.id()} ({f})" for f in functions]
#                 visualization.plot_collapsing_layers_same_model(results + post_results, plot_filepath, labels=labels + post_labels)


class AggregationFunctionsVariance(Experiment):
    def description(self):
        return """Test different Convolutional Aggregation (sum,mean,max) functions to evaluate their differences. Convolutional aggregation collapses all the spatial dimensions of feature maps before normalization so that a single variance value for the feature map can be obtained."""

    def run(self):
        functions = [ca_sum,ca_mean,tm.ConvAggregation.max]
        combinations = itertools.product(
            simple_models_generators , dataset_names, config.common_transformations_without_identity())
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            measures = [tm.NormalizedVariance(ca) for ca in functions]
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)
            normal_results = config.load_results(config.results_paths(variance_parameters))

            ca_none_variance_parameter = variance.Parameters(p_training.id(), p_dataset, transformation, tm.NormalizedVariance(ca_none))
            self.experiment_variance(ca_none_variance_parameter , model_path)
            ca_none_variance_parameters = [ca_none_variance_parameter]*3
            ca_none_results = config.load_results(config.results_paths(ca_none_variance_parameters))
            for (ca,r) in zip(functions,ca_none_results):
                r.measure_result = r.measure_result.collapse_convolutions(ca)

            all_results = normal_results + ca_none_results
            labels = [f"Aggregation {ca} before normalization." for ca in functions] + [f"Aggregation {ca} after normalization." for ca in functions]
            visualization.plot_collapsing_layers_same_model(all_results, plot_filepath, labels=labels)


class CompareSameEquivarianceNormalization(Experiment):
    def description(self):
        return """Compare the result of DistanceSameEquivariance normalized or not."""

    def run(self):
        measures = [
                    tm.DistanceSameEquivarianceMeasure(da_mean,normalized=True),
                    tm.DistanceSameEquivarianceMeasure(da_mean, normalized=False)
                    ]

        combinations = itertools.product(
            simple_models_generators , dataset_names, config.common_transformations_without_identity())
        for model_config_generator, dataset, transformation in combinations:
            # train
            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)
            #eval
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            variance_parameters = [variance.Parameters(p_training.id(), p_dataset, transformation, m) for m in measures]
            model_path = config.model_path(p_training)
            for p_variance in variance_parameters:
                self.experiment_variance(p_variance, model_path)

            experiment_name = f"{model_config.name}_{p_dataset.id()}_{transformation.id()}_Distance"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = ["Normalized", "Unnormalized"]
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
        measures = [
            tm.NormalizedVariance(ca_mean),
            # tm.AnovaMeasure(ca_none, alpha=0.99, bonferroni=True),
            tm.DistanceMeasure(da_mean),
            tm.DistanceSameEquivarianceMeasure(da_mean),
        ]

        # number of random models to generate
        random_model_n = 10

        combinations = itertools.product(
            simple_models_generators, dataset_names, config.common_transformations_without_identity(),measures)
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
                    model, optimizer = model_config.make_model(dataset.input_shape, dataset.num_classes, o.use_cuda)
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
            experiment_name = f"{model_config.name}_{dataset_name}_{transformation.id()}_{measure}"
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
        measures = [tm.NormalizedVariance( ca_mean)
            # , tm.AnovaMeasure(ca_none, alpha=0.99, bonferroni=True)
            , tm.DistanceMeasure(da_mean)
            , tm.DistanceSameEquivarianceMeasure(da_mean)
                    ]


        savepoints_percentages = [0,1,2,3,4,5,10,20,30,40,50,100]

        combinations = itertools.product(
            simple_models_generators, dataset_names, config.common_transformations_without_identity(), measures)
        for model_config_generator, dataset, transformation, measure in combinations:
            p = config.dataset_size_for_measure(measure)
            # train
            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            savepoints = [ sp * epochs // 100 for sp in savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
            p_training = training.Parameters(model_config, dataset, transformation, epochs, savepoints=savepoints)
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
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure}"
            print(experiment_name)
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
        measures = [
            # tm.AnovaMeasure(tm.ConvAggregation.none, 0.99),
            tm.NormalizedVariance(ca_sum),
            tm.DistanceMeasure(da_mean),
            tm.DistanceSameEquivarianceMeasure(da_mean),
        ]
        models = [config.SimpleConvConfig]
        combinations = itertools.product(
            models, dataset_names, config.common_transformations_without_identity(), measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            # train

            variance_parameters = []
            for bn in [True,False]:
                model_config=model_config_generator.for_dataset(dataset,bn=bn)
                epochs = config.get_epochs(model_config, dataset, transformation)
                p_training = training.Parameters(model_config, dataset, transformation, epochs)
                self.experiment_training(p_training)

                p = 0.5 if measure.__class__ == tm.AnovaMeasure else 0.1
                p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
                p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
                model_path = config.model_path(p_training)
                batch_size = 64
                if model_config.name.startswith("ResNet"):
                    batch_size = 32
                self.experiment_variance(p_variance, model_path, batch_size=batch_size)
                variance_parameters.append(p_variance)

            # plot results
            experiment_name = f"{model_config.name}_bn_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath/ f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = ["Without BN", "With BN"]
            visualization.plot_collapsing_layers_different_models(results, plot_filepath, labels=labels)


class DatasetTransfer(Experiment):
    def description(self):
        return """Measure invariance with a different dataset than the one used to train the model."""

    def run(self):
        mf, ca = tm.MeasureFunction.std, tm.ConvAggregation.mean
        measures = [
                    # tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.none, alpha=0.99),
                    tm.NormalizedVariance(ca_sum),
                    # tm.DistanceMeasure(tm.DistanceAggregation.mean)
        ]

        combinations = itertools.product(
            simple_models_generators , dataset_names, config.common_transformations_without_identity(), measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
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
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"
            results = config.load_results(config.results_paths(variance_parameters))
            labels = dataset_names
            visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)





class CompareModels(Experiment):
    def description(self):
        return """Determine which model is more invariant. Plots invariance of models as layers progress"""

    def run(self):
        measures  = [tm.NormalizedVariance(ca_mean),
                     tm.DistanceMeasure(da_mean),
                     ]

        models = [
            config.SimpleConvConfig,
            config.AllConvolutionalConfig,
            config.VGG16DConfig,
            config.ResNetConfig]
        #transformations = [tm.SimpleAffineTransformationGenerator(r=360)]

        transformations =config.common_transformations_without_identity()

        combinations = itertools.product( dataset_names, transformations, measures)
        for (dataset, transformation, measure) in combinations:
            variance_parameters = []
            model_configs = []
            for model_config_generator in models:
                # train
                model_config = model_config_generator.for_dataset(dataset)
                model_configs.append(model_config)

                # train
                epochs = config.get_epochs(model_config, dataset, transformation)
                p_training = training.Parameters(model_config, dataset, transformation, epochs)
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
            labels = [m.name for m in model_configs]
            visualization.plot_collapsing_layers_different_models(results, plot_filepath, labels=labels)


class CompareTIPooling(Experiment):
    def description(self):
        return """Compare SimpleConv with the SiameseSimpleConv Network"""

    def run(self):
        measures  = [tm.NormalizedVariance(ca_mean),
                     tm.DistanceMeasure(da_mean),
                     ]

        model_config_generators = [config.SiameseSimpleConvConfig,config.SimpleConvConfig]
        transformations = [tm.SimpleAffineTransformationGenerator(r=360)]

        #transformations =config.common_transformations_without_identity()

        combinations = itertools.product( dataset_names, transformations, measures)
        for (dataset, transformation, measure) in combinations:

            siamese=config.SiameseSimpleConvConfig.for_dataset(dataset,bn=False,t=transformation)
            siamese_epochs = config.get_epochs(siamese,dataset,transformation)
            p_training_siamese = training.Parameters(siamese, dataset, tm.SimpleAffineTransformationGenerator(), siamese_epochs,0)


            normal = config.SimpleConvConfig.for_dataset(dataset,bn=False)
            normal_epochs = config.get_epochs(normal, dataset, transformation)
            p_training_normal = training.Parameters(normal, dataset, transformation, normal_epochs, 0)

            p_training_parameters = [p_training_siamese,p_training_normal]
            variance_parameters = []
            for p_training in p_training_parameters:
                # train
                self.experiment_training(p_training)
                # generate variance params
                p = config.dataset_size_for_measure(measure)
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
            labels = [m.name for m in [siamese,normal]]
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
        measures = [
            # tm.TransformationMeasure(mf, ca_none),
            # tm.SampleMeasure(mf, ca_none),
            tm.AnovaMeasure( 0.99, bonferroni=True),
            tm.NormalizedVariance(ca_sum),
            # tm.DistanceTransformationMeasure(dmean),
            # tm.DistanceSampleMeasure(dmean),
            tm.DistanceMeasure(da_mean),
            tm.DistanceSameEquivarianceMeasure(da_mean),
        ]

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        model_names = simple_models_generators
        transformations = config.common_transformations_without_identity()

        combinations = itertools.product(model_names, dataset_names, transformations, measures)
        for (model_config_generator, dataset, transformation, measure) in combinations:
            # train
            model_config = model_config_generator.for_dataset(dataset)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs)
            self.experiment_training(p_training)
            # generate variance params
            p = config.dataset_size_for_measure(measure)
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
            p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
            p_variance_stratified = variance.Parameters(p_training.id(), p_dataset, transformation, measure,stratified=True)
            # evaluate variance
            model_path = config.model_path(p_training)
            self.experiment_variance(p_variance, model_path)
            self.experiment_variance(p_variance_stratified, model_path)
            variance_parameters=[p_variance,p_variance_stratified]
            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}_{measure.id()}"
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
            # tm.AnovaMeasure( alpha=0.99,bonferroni=True),
            # #tm.AnovaMeasure(conv_aggregation=tm.ConvAggregation.mean, alpha=0.99,bonferroni=True),
            # tm.NormalizedMeasure(mf, ca_sum),
            tm.NormalizedVariance(ca_sum),
            # tm.TransformationMeasure(mf),
            # tm.TransformationMeasure(mf),
            # tm.DistanceTransformationMeasure(mf, dmean),
            # tm.DistanceTransformationMeasure(mf, dmax),
            tm.DistanceMeasure(dmean),
            # tm.DistanceMeasure(dmax),
            # tm.DistanceSameEquivarianceMeasure(dmax),
            tm.DistanceSameEquivarianceMeasure(dmean),

        ]
        #conv_model_names = [m for m in common_model_names if (not "FFNet" in m)]
        conv_model_names = simple_models_generators # [models.SimpleConv.__name__]

        transformations = [tm.SimpleAffineTransformationGenerator(r=360),
                       tm.SimpleAffineTransformationGenerator(t=5),
                       tm.SimpleAffineTransformationGenerator(s=5)]
        combinations = itertools.product(
            conv_model_names, dataset_names, transformations, measures)
        for (model_config_generator, dataset_name, transformation, measure) in combinations:
            model_config = model_config_generator.for_dataset(dataset_name)
            # train
            epochs = config.get_epochs(model_config, dataset_name, transformation)
            p_training = training.Parameters(model_config, dataset_name, transformation, epochs)

            experiment_name = f"{model_config.name}_{dataset_name}_{transformation.id()}_{measure.id()}"
            plot_folderpath = self.plot_folderpath / experiment_name
            finished = Path(plot_folderpath) / "finished"
            if finished.exists():
                continue
            # train
            self.experiment_training(p_training)
            p = config.dataset_size_for_measure(measure)
            p_dataset = variance.DatasetParameters(dataset_name, variance.DatasetSubset.test, p)
            p_variance = variance.Parameters(p_training.id(), p_dataset, transformation, measure)
            model_path = config.model_path(p_training)
            self.experiment_variance(p_variance, model_path)

            model_filepath = config.model_path(p_training)
            model, p_model, o, scores = training.load_model(model_filepath, use_cuda=torch.cuda.is_available())
            result_filepath = config.results_path(p_variance)
            result = config.load_result(result_filepath)
            dataset = datasets.get(dataset_name)

            plot_folderpath.mkdir(parents=True, exist_ok=True)

            visualization.plot_invariant_feature_maps_pytorch(plot_folderpath, model, dataset, transformation,result, images=2, most_invariant_k=4, least_invariant_k=4,conv_aggregation=tm.ConvAggregation.mean)
            finished.touch()


class ValidateMeasure(Experiment):

    def description(self):
        return """Validate measure/transformation. Just for testing purposes."""

    def run(self):
        measures = [
            # tm.TransformationVarianceMeasure(ca_sum),
            # tm.SampleVarianceMeasure( ca_sum),
            # tm.NormalizedVarianceMeasure( ca_sum),

            # tm.TransformationMeasure(ca_sum),
            # tm.SampleMeasure( ca_sum),
            # tm.NormalizedMeasure(ca_sum),
            tm.NormalizedVariance(ca_sum),

            # tm.GoodfellowNormalMeasure(),
        ]
        #model_names = ["VGG16D","VGG16DBN"]
        model_names=[config.SiameseSimpleConvConfig]
        dataset_names = ["mnist"]
        transformations = [tm.SimpleAffineTransformationGenerator(r=360)]
        #transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, measures, transformations)
        for model_config_generator, dataset, measure, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset,bn=False,t=transformation)
            # train
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, tm.SimpleAffineTransformationGenerator(), epochs, 0,savepoints=[0,10,20,30,40,50,60,70,80,100])
            experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{transformation.id()}"
            p_dataset = variance.DatasetParameters(dataset, variance.DatasetSubset.test, 0.1)
            p_measure = variance.Parameters(p_training.id(), p_dataset, transformation,measure)

            self.experiment_training(p_training)
            model_path = config.model_path(p_training)


            self.experiment_variance(p_measure, model_path)
            print(experiment_name)
            plot_filepath = self.plot_folderpath / f"{experiment_name}.png"

            title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n transformation: {transformation.id()} "
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

        AggregationFunctionsVariance(),
        # CollapseConvBeforeOrAfter(),

        TransformationDiversity(),
        TransformationComplexity(),

        BatchNormalization(),
        CompareSameEquivarianceNormalization(),

        CompareTIPooling(),

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
