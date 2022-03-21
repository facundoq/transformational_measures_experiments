from .common import *
import experiment.measure as measure_package
import transformational_measures.visualization as tmv

import config.transformations as ct


class TransformationSampleSizes(InvarianceExperiment):

    def description(self):
        return """Compare error of the measure with different sizes of the sample and transformation sets"""

    def run(self):
        
        # measures = [nvi, gf_percent, svi,tvi] # anova,
        # measures = [ gf_percent ]
        measures = [nvi] 

        sample_sizes = [24, 96, 384, 1536,2304] 
        rotations = [RotationGenerator(UniformRotation(n, rotation_max_degrees)) for n in sample_sizes]
        scales = [ScaleGenerator(ScaleUniform(n // 6, scale_min_downscale, scale_max_upscale)) for n in sample_sizes]
        translations = [TranslationGenerator(TranslationUniform(n // 8, translation_max)) for n in sample_sizes]
        transformations_sets = zip(common_transformations,[rotations,scales,translations])
        
        combinations = itertools.product(simple_models_generators,  measures, dataset_names,transformations_sets)
        labels_samples = [f"{i}" for i in sample_sizes]
        task = Task.Classification
        
        for model_config_generator, measure,dataset, transformation_set in combinations:
            train_transformation, test_transformations = transformation_set
            # model_config = model_config_generator.for_dataset(task,dataset)
            mc: train.ModelConfig = model_config_generator.for_dataset(task,dataset)
            tc,metric = self.get_train_config(mc,dataset,task,train_transformation)
            p = train.TrainParameters(mc, tc, dataset, train_transformation, task)
            self.train(p)
            model_path = self.model_path_new(p)

            experiment_name = f"{mc.id()}_{dataset}_{train_transformation.id()}_{measure}"
            s_n = len(sample_sizes)
            t_n = len(test_transformations)
            results = np.empty((s_n, t_n), dtype=tm.pytorch.PyTorchMeasureResult)
            experiment_name = f"{mc.id()}_{dataset}_{train_transformation.id()}_{measure}"

            for i, sample_size in enumerate(sample_sizes):
                p_dataset = DatasetParameters(dataset, default_subset, DatasetSizeFixed(sample_size))
                for j, transformation in enumerate(test_transformations):
                    mp = PyTorchParameters(mc.id(), p_dataset, transformation, measure, default_measure_options)
                    results[i, j] = self.measure(model_path, mp, verbose=False).numpy()

                # PLOT
                
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            labels_transformations = [f"{len(ts)}" for ts in test_transformations]
            heatmap = tmv.get_relative_errors(results, results[-1, -1])
            tmv.plot_relative_error_heatmap(heatmap, plot_filepath, labels_samples,labels_transformations)
                