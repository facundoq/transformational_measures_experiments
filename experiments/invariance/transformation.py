from .common import *
import experiment.measure as measure_package


import config.transformations
class TransformationSetSize(InvarianceExperiment):

    def description(self):
        return """Train with transformations of the same complexity but more densely sampled"""

    def run(self):
        measures = normalized_measures
        combinations = itertools.product(simple_models_generators, dataset_names, measures)
        nr,ns,nt=config.transformations.n_r,config.transformations.n_s,config.transformations.n_t
        n_rotations= [nr-20,nr-10,nr,nr+10,nr+20,nr+30]
        n_scales=  [ns-2,ns-1,ns,ns+2,ns+4,nt+6]
        n_translations =  [nt-2,nt-1,nt,nt+1,nt+2, nt+3]
        r = config.transformations.rotation_max_degrees
        down,up=config.transformations.scale_min_downscale,config.transformations.scale_max_upscale
        t = config.transformations.translation_max
        test_sets = [[AffineGenerator(r=UniformRotation(i,r)) for i in n_rotations],
                     [AffineGenerator(s=ScaleUniform(i,down,up)) for i in n_scales],
                     [AffineGenerator(t=TranslationUniform(i, t)) for i in n_translations],
                     ]

        labels = [ [f"{len(s)} {l.transformations}" for s in set] for set in test_sets ]


        train_transformations = common_transformations

        for model_config_generator, dataset, measure in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            for train_transformation, transformation_set,set_labels in zip(train_transformations, test_sets,labels):
                # TRAIN
                epochs = config.get_epochs(model_config, dataset, train_transformation)
                p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                model_path = self.model_path(p_training)
                self.experiment_training(p_training)
                # MEASURE
                variance_parameters = []
                for k, test_transformation in enumerate(transformation_set):
                    p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, default_dataset_percentage)
                    p_variance = measure_package.Parameters(model_config.id(), p_dataset, test_transformation, measure)
                    #model_path
                    self.experiment_measure(p_variance,model_path=model_path)
                    variance_parameters.append(p_variance)

                # PLOT
                experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
                plot_filepath = self.folderpath / f"{experiment_name}.jpg"
                results = self.load_measure_results(self.results_paths(variance_parameters))
                visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=set_labels,ylim=1.4)



class TransformationComplexity(InvarianceExperiment):

    def description(self):
        return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 16 rotations, measure variance with 2, 4, 8 and 16. """

    def run(self):
        measures = normalized_measures
        combinations = itertools.product(simple_models_generators, dataset_names, measures)

        n_complexities=4
        rotations=np.linspace(90,360,n_complexities)
        #[(0.875,1.0625),(0.75,1.125),(0.75,1.125),(0.5,1.25)
        upscale=np.linspace(1,1.25,n_complexities+1)[1:]
        downscale=np.flip(np.linspace(0.5,1,n_complexities,endpoint=False))
        scaling= list(zip(downscale,upscale))
        translation = np.linspace(0.05,0.2,n_complexities)

        test_sets = [[AffineGenerator(r=UniformRotation(25,r)) for r in rotations],
                     [AffineGenerator(s=ScaleUniform(4,*s)) for s in scaling],
                     [AffineGenerator(t=TranslationUniform(3,t)) for t in translation],
                     ]
        labels = [
                 [f"0° {l.to} {d}°" for d in rotations],
                 [f"{d} {l.to} {u}" for (d,u) in scaling],
                 [f"0 {l.to} {i}" for i in translation],
        ]

        train_transformations = common_transformations

        for model_config_generator, dataset, measure in combinations:
            model_config = model_config_generator.for_dataset(dataset)

            for train_transformation, transformation_set,set_labels in zip(train_transformations, test_sets,labels):
                # TRAIN
                epochs = config.get_epochs(model_config, dataset, train_transformation)
                p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                self.experiment_training(p_training)
                # MEASURE
                variance_parameters = []
                for k, test_transformation in enumerate(transformation_set):
                    p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, default_dataset_percentage)
                    p_variance = measure_package.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                    model_path = self.model_path(p_training)
                    self.experiment_measure(p_variance)
                    variance_parameters.append(p_variance)
                # PLOT
                experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
                plot_filepath = self.folderpath / f"{experiment_name}.jpg"
                #title = f" transformation: {train_transformation.id()}"

                results = self.load_measure_results(self.results_paths(variance_parameters))

                visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=set_labels,ylim=1.4)


# class TransformationComplexityDetailed(Experiment):
#
#     def description(self):
#         return """Train a model/dataset with a transformation of scale X and then test with scales Y and Z of the same type, where Y<X and Z>X. Ie, train with 8 rotations, test variance with 2, 4, 8 and 16. """
#
#     def run(self):
#         measures = normalized_measures
#         combinations = itertools.product(simple_models_generators, dataset_names, measures)
#
#         names = [l.rotation,l.scale,l.translation]
#         sets = [config.rotation_transformations(8),
#                 config.translation_transformations(4),
#                 config.scale_transformations(4)]
#
#         for model_config_generator, dataset, measure in combinations:
#             model_config = model_config_generator.for_dataset(dataset)
#
#             for i, (transformation_set, name) in enumerate(zip(sets, names)):
#                 n_experiments = (len(transformation_set) + 1) * len(transformation_set)
#                 print(f"    {name}, #invariance:{n_experiments}")
#                 # include identity the transformation set
#                 transformation_set = [tm.SimpleAffineTransformationGenerator()] + transformation_set
#                 for j, train_transformation in enumerate(transformation_set):
#                     transformation_plot_folderpath = self.plot_folderpath / name
#
#                     transformation_plot_folderpath.mkdir(exist_ok=True, parents=True)
#                     experiment_name = f"{model_config.name}_{dataset}_{measure.id()}_{train_transformation.id()}"
#                     plot_filepath = transformation_plot_folderpath / f"{experiment_name}.jpg"
#                     variance_parameters = []
#                     print(f"{j}, ", end="")
#                     epochs = config.get_epochs(model_config, dataset, train_transformation)
#                     p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
#                     self.experiment_training(p_training)
#                     for k, test_transformation in enumerate(transformation_set):
#                         p_dataset = measure.DatasetParameters(dataset, measure.DatasetSubset.test, default_dataset_percentage)
#                         p_variance = measure.Parameters(p_training.id(), p_dataset, test_transformation, measure)
#                         model_path = config.model_path(p_training)
#                         self.experiment_measure(p_variance)
#                         variance_parameters.append(p_variance)
#
#                     title = f"Invariance to \n. Model: {model_config.name}, Dataset: {dataset}, Measure {measure.id()} \n Train transformation: {train_transformation.id()}"
#                     labels = [f"Test transformation: {t}" for t in transformation_set[1:]]
#
#                     results = config.load_measure_results(config.results_paths(variance_parameters[1:]))
#                     visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, title=title)


class TransformationDiversity(InvarianceExperiment):

    def description(self):
        return '''Vary the type of transformation both when training and computing the numpy, and see how it affects the invariance. For example, train with rotations, then test with translations. Train with translations. numpy with scales, and so on. '''

    def run(self):
        measures = normalized_measures

        combinations = itertools.product(simple_models_generators, dataset_names, measures)
        transformations = common_transformations

        labels = [l.rotation, l.scale, l.translation]
        for model_config_generator, dataset, measure in combinations:
            for i, train_transformation in enumerate(transformations):
                # transformation_plot_folderpath = self.plot_folderpath / name
                # transformation_plot_folderpath.mkdir(exist_ok=True,parents=True)
                model_config = model_config_generator.for_dataset(dataset)

                variance_parameters = []
                print(f"{l.train}: {train_transformation}")
                epochs = config.get_epochs(model_config, dataset, train_transformation)

                p_training = training.Parameters(model_config, dataset, train_transformation, epochs, 0)
                self.experiment_training(p_training)
                for i, test_transformation in enumerate(transformations):
                    print(f"{i}, ", end="")
                    p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, default_dataset_percentage)
                    p_variance = measure_package.Parameters(p_training.id(), p_dataset, test_transformation, measure)
                    model_path = self.model_path(p_training)
                    self.experiment_measure(p_variance)
                    variance_parameters.append(p_variance)
                results = self.load_measure_results(self.results_paths(variance_parameters))
                experiment_name = f"{model_config.name}_{dataset}_{train_transformation}_{measure.id()}"
                plot_filepath = self.folderpath / f"{experiment_name}.jpg"
                # title = f"Train transformation: {train_transformation.id()}"
                visualization.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)

