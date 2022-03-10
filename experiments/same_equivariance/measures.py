from .common import *


class CompareSameEquivarianceNormalized(SameEquivarianceExperiment):

    def description(self):
        return """Compare same equivariance normalized measures"""

    def run(self):

        model_names = simple_models_generators

        measures = [
            tvse,
            svse,
            nvse,
            tm.pytorch.NormalizedVarianceSameEquivariance()
        ]

        labels = [m.abbreviation() for m in measures]
        task = Task.TransformationRegression
        combinations = itertools.product(model_names, dataset_names, common_transformations)
        for model_config_generator, dataset, transformations in combinations:
            mc: ModelConfig = model_config_generator.for_dataset(task, dataset)
            tc, metric = self.get_train_config(mc, dataset, task, transformations)
            p = train.TrainParameters(mc, tc, dataset, transformations, task)
            self.train(p)
            # train
            experiment_name = f"{mc.id()}_{dataset}_{transformations.id()}"
            p_dataset = DatasetParameters(dataset, default_subset, default_dataset_percentage)
            results = []
            model_path = self.model_path_new(p)

            for measure in measures:
                mp = PyTorchParameters(mc.id(), p_dataset, transformations, measure, default_measure_options,
                                       model_filter=simple_conv_sameequivariance_activation_filter)
                result = self.measure(model_path, mp, verbose=False).numpy()
                results.append(result)

            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class TransformationSampleSizes(SameEquivarianceExperiment):

    def description(self):
        return """Compare error of the measure with different sizes of the sample and transformation sets"""

    def run(self):

        model_names = simple_models_generators

        measures = [
            tvse,
            svse,
            tm.pytorch.NormalizedVarianceSameEquivariance(),
            nvse,
        ]
        
        sample_sizes = [24, 96, 384, 1536] 
        rotations = [RotationGenerator(UniformRotation(n, rotation_max_degrees)) for n in sample_sizes]
        scales = [ScaleGenerator(ScaleUniform(n // 6, scale_min_downscale, scale_max_upscale)) for n in sample_sizes]
        translations = [TranslationGenerator(TranslationUniform(n // 8, translation_max)) for n in sample_sizes]
        transformations_sets = zip(common_transformations,[rotations,scales,translations])
        # [
        #     (default_uniform_scale, rotations),
        #     (scales[0], scales),
        #     (translations[0], translations)
        # ]
        task = Task.TransformationRegression
        combinations = itertools.product(model_names, dataset_names, measures, transformations_sets)
        labels_samples = [f"{i}" for i in sample_sizes]
        for model_config_generator, dataset, measure, transformation_set in combinations:
            train_transformation, test_transformations = transformation_set
            mc: ModelConfig = model_config_generator.for_dataset(task, dataset)
            tc, metric = self.get_train_config(mc, dataset, task, train_transformation, savepoints=False)
            p = train.TrainParameters(mc, tc, dataset, train_transformation, task)
            self.train(p)
            model_path = self.model_path_new(p)
            # train
            experiment_name = f"{mc.id()}_{dataset}_{train_transformation.id()}_{measure}"
            s_n = len(sample_sizes)
            t_n = len(test_transformations)
            results = np.empty((s_n, t_n), dtype=tm.pytorch.PyTorchMeasureResult)
            
            for i, sample_size in enumerate(sample_sizes):
                p_dataset = DatasetParameters(dataset, default_subset, DatasetSizeFixed(sample_size))
                for j, transformation in enumerate(test_transformations):
                    mp = PyTorchParameters(mc.id(), p_dataset, transformation, measure, default_measure_options,model_filter=simple_conv_sameequivariance_activation_filter)
                    results[i, j] = self.measure(model_path, mp, verbose=False).numpy()


            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            labels_transformations = [f"{len(ts)}" for ts in test_transformations]
            tmv.plot_relative_error_heatmap(results, results[-1, -1], plot_filepath, labels_samples,
                                            labels_transformations)
