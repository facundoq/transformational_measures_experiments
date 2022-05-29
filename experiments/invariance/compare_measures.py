from .common import *
import experiment.measure as measure_package
import transformational_measures.visualization as tmv



class MeasureCorrelationWithTransformation(InvarianceExperiment):

    def description(self):
        return """Train models M1, M2, Mn with transformation of scales X1,X2,..Xn respectively and then test all models with scale Xn, where Xi<Xi+1. Ie, train with rotations of 0, 30, 60, 90,.. 360 degrees, and then test with rotations of 360 degrees. """

    def run(self):
        measures = normalized_measures
        combinations = itertools.product(simple_models_generators, dataset_names, measures)

        n=4
        rotations=np.linspace(0,rotation_max_degrees,n)
        # [(0.875,1.0625),(0.75,1.125),(0.75,1.125),(0.5,1.25)
        upscale=np.linspace(1,scale_max_upscale,n)
        downscale=np.flip(np.linspace(scale_min_downscale,1,n))
        scaling= list(zip(downscale,upscale))
        translation = np.linspace(0,translation_max,n)
        n_r,n_s,n_t = n_rotations, n_scales,n_translations
        train_sets = [[AffineGenerator(r=UniformRotation(n_r,r)) for r in rotations],
                     [AffineGenerator(s=ScaleUniform(n_s,*s)) for s in scaling],
                     [AffineGenerator(t=TranslationUniform(n_t,t)) for t in translation],
                     ]
        labels = [
            [f"0° {l.to} {int(d*360)}°" for d in rotations],
            [f"{int(d*100)}% {l.to} {int(u*100)}%" for (d,u) in scaling],
            [f"0% {l.to} {int(i*100)}%" for i in translation],
        ]
        labels[0][0]="No rotation"
        labels[1][0]="No scaling"
        labels[2][0]="No translation"
        test_transformations = common_transformations

        for model_config_generator, dataset, measure in combinations:
            
            for train_set, test_transformation,set_labels in zip(train_sets, test_transformations,labels):
                # TRAIN
                results = []
                for k, train_transformation in enumerate(train_set):
                    print(train_transformation.id(),test_transformation.id())
                    mc,tc,p,model_path = self.train_default(Task.Classification,dataset,train_transformation,model_config_generator)
                # MEASURE
                    result = self.measure_default(dataset,mc.id(),model_path,test_transformation,
                    measure,default_measure_options,default_dataset_percentage)
                    results.append(result)
                # PLOT
                experiment_name = f"{mc.id()}_{dataset}_{measure.id()}_{test_transformation.id()}"
                plot_filepath = self.folderpath / f"{experiment_name}.jpg"
                #title = f" transformation: {train_transformation.id()}"
                tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=set_labels,ylim=1.4)


class InvarianceMeasureCorrelation(InvarianceExperiment):
    def description(self):
        return """Measure the correlation between different the invariance measure and the invariance of the model via training with different DataAugmentation intensities."""

    def run(self):

        measure_sets = {"Variance": [
            tvi,
            svi
        ],
        "ANOVA": [

        ],
        "NormalizedVariance": [
            nvi
        ],
        "Goodfellow": [
            gf_normal,
            gf_percent
        ],
        }

        model_generators = simple_models_generators
        
        transformations = common_transformations_combined

        combinations = itertools.product(model_generators, dataset_names, transformations, measure_sets.items())
        for (model_config_generator, dataset, transformations, measure_set) in combinations:
            # train model with data augmentation and without
            results_no_augmentation = []
            results_augmentation = []
            for t in [identity_transformation, transformations]:
                mc,tc,p,model_path = self.train_default(Task.Classification,dataset,t,model_config_generator)
                # MEASURE
                   

                # generate variance params
                variance_parameters = []
                measure_set_name, measures = measure_set
                for measure in measures:
                    # measure without augmentation
                    result = self.measure_default(dataset,mc.id(),model_path,identity_transformation, measure,default_measure_options,default_dataset_percentage)
                    # measure with augmentation
                    results_no_augmentation.append(result)
                    result = self.measure_default(dataset,mc.id(),model_path,transformations, measure,default_measure_options,default_dataset_percentage)
                    results_augmentation.append(result)
                # evaluate variance


            # plot results
            experiment_name = f"{measure_set_name}_{mc.id()}_{dataset}_{transformations.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            results = results_no_augmentation+results_augmentation
            labels = [l.measure_name(m) + f" ({l.no_data_augmentation})" for m in measures] + [l.measure_name(m) for m in measures]
            n = len(measures)

            cmap = tmv.default_discrete_colormap()
            color = cmap(range(n))
            colors = np.vstack([color, color])
            linestyles = ["--" for i in range(n)] + ["-" for i in range(n)]
            # ylim = self.get_ylim(measure_set_name, dataset)
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,linestyles=linestyles,colors=colors)
            #TODO also plot conv only

    # def get_ylim(self, measure_set_name, dataset):
    #     if measure_set_name == "Goo":
    #         return 100 if dataset == "mnist" else 2000
    #     elif measure_set_name == "Variance":
    #         return 100 if dataset == "mnist" else 175
    #     elif measure_set_name == "HighLevel":
    #         return 1.2 if dataset == "mnist" else 1.2
    #     elif measure_set_name == "Equivariance":
    #         return None
    #         # return 8 if dataset == "mnist" else 8
    #     else:
    #         return None


class CompareMeasures(InvarianceExperiment):
    def description(self):
        return """Test different measures for a given dataset/model/transformation combination to evaluate their differences."""

    def run(self):

        measure_sets = {
            "Variance": [
                tm.TransformationVarianceInvariance(),
                tm.SampleVarianceInvariance(),
            ],
            # "Distance": [
            #    tm.TransformationDistanceInvariance(da),
            #    tm.SampleDistanceInvariance(da),
            # ],
            "Normalized": [
                    tm.ANOVAInvariance(),
                    tm.NormalizedVarianceInvariance(ca_sum),
                    tm.GoodfellowNormalInvariance(alpha=0.99),
            ],
            # "Equivariance": [
            #     tm.NormalizedVarianceSameEquivariance(ca_mean),
            #     tm.NormalizedDistanceSameEquivariance(da_normalize_keep),
            #     tm.DistanceSameEquivarianceSimple(df_normalize),
            # ]
        }

        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]

        # model_generators = common_models_generators
        model_generators = simple_models_generators
        # model_names = ["SimpleConv"]
        transformations = common_transformations

        combinations = itertools.product(model_generators, dataset_names, transformations, measure_sets.items())
        for (model_config_generator, dataset, transformation, measure_set) in combinations:
            # train model with data augmentation and without

            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)

            # generate variance params
            variance_parameters = []
            measure_set_name, measures = measure_set
            for measure in measures:
                p = config.dataset_size_for_measure(measure)
                p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, p)
                p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, measure)
                variance_parameters.append(p_variance)
            # evaluate variance
            model_path = self.model_path(p_training)

            for p_variance in variance_parameters:
                self.experiment_measure(p_variance,model_path=model_path)

            # plot results
            experiment_name = f"{measure_set_name}_{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            results = self.load_measure_results(self.results_paths(variance_parameters))
            labels =[l.measure_name(m) for m in measures]
            n = len(measures)

            cmap = tmv.default_discrete_colormap()
            color = cmap(range(n))
            colors = np.vstack([color, color])
      #      linestyles = ["--" for i in range(n)] + ["-" for i in range(n)]
            ylim = self.get_ylim(measure_set_name, dataset)
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels, ylim=ylim)

    def get_ylim(self, measure_set_name, dataset):
        if measure_set_name == "Distance":
            return 100 if dataset == "mnist" else 2000
        elif measure_set_name == "Variance":
            return 100 if dataset == "mnist" else 175
        elif measure_set_name == "HighLevel":
            return 1.2 if dataset == "mnist" else 1.2
        elif measure_set_name == "Equivariance":
            return None
            # return 8 if dataset == "mnist" else 8
        else:
            return None


class CompareGoodfellowAlpha(InvarianceExperiment):

    def description(self):
        return """Compare goodfellow alpha values"""

    def run(self):

        model_names = simple_models_generators
        alphas = [0.5, 0.9, 0.95, 0.99, 0.999]
        measures = [tm.GoodfellowNormalInvariance(alpha) for alpha in alphas]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            variance_parameters = []
            for measure in measures:
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            results = self.load_measure_results(self.results_paths(variance_parameters))
            labels = [f"α={alpha}" for alpha in alphas]
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)


class CompareGoodfellow(InvarianceExperiment):

    def description(self):
        return """Compare goodfellow global vs local"""

    def run(self):
        model_names = simple_models_generators
        measure = tm.GoodfellowNormalInvariance(0.999)
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        labels = ["Local", "Global"]
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
            result = self.load_experiment_result(self.results_path(p_variance)).measure_result
            local_result, global_result = result.extra_values[tm.GoodfellowNormalInvariance.l_key], result.extra_values[
                tm.GoodfellowNormalInvariance.g_key]
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            tmv.plot_collapsing_layers_same_model([local_result, global_result], plot_filepath,labels=labels, ylim=0.1)





class CompareSameEquivarianceSimple(InvarianceExperiment):

    def description(self):
        return """Compare same equivariance normalized measures"""

    def run(self):

        model_names = simple_models_generators

        measures = [tm.DistanceSameEquivarianceSimple(df),
                    tm.DistanceSameEquivarianceSimple(df_normalize),
                    ]
        labels = [l.unnormalized,l.normalized]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            variance_parameters = []
            for measure in measures:
                p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
                variance_parameters.append(p_variance)
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            results = self.load_measure_results(self.results_paths(variance_parameters))

            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)
class CompareSameEquivariance(InvarianceExperiment):

    def description(self):
        return """Compare transformational and sample same equivariance measures"""

    def run(self):

        model_names = simple_models_generators

        labels = [l.transformational,l.sample_based]
        # transformations=config.common_transformations()
        combinations = itertools.product(model_names, dataset_names, common_transformations_combined)
        for model_config_generator, dataset, transformation in combinations:
            model_config = model_config_generator.for_dataset(dataset, bn=False)
            # train
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            measure = tm.NormalizedVarianceSameEquivariance(ca_mean)
            p_training, p_variance, p_dataset = self.train_measure(model_config, dataset, transformation, measure)
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"
            normalized_results = self.load_measure_result(self.results_path(p_variance))

            transformation_results=normalized_results.extra_values[tm.NormalizedVarianceSameEquivariance.transformation_key]
            sample_results=normalized_results.extra_values[tm.NormalizedVarianceSameEquivariance.sample_key]
            results=[transformation_results,sample_results]
            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels)



class DistanceApproximation(InvarianceExperiment):
    def description(self):
        return """Test different batch sizes to compute the DistanceInvariance measures to evaluate how they approximate the VarianceInvariance measure."""

    def run(self):


        # model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # model_names=["ResNet"]
        batch_sizes = [32, 128, 512, 1024]
        # model_generators = common_models_generators
        model_generators = simple_models_generators
        # model_names = ["SimpleConv"]
        transformations = [common_transformations[0]]

        dataset_names = ["mnist","cifar10"]
        combinations = itertools.product(model_generators, dataset_names,transformations)
        for (model_config_generator, dataset, transformation) in combinations:
            # train model with data augmentation and without

            model_config = model_config_generator.for_dataset(dataset)
            epochs = config.get_epochs(model_config, dataset, transformation)
            p_training = training.Parameters(model_config, dataset, transformation, epochs, 0)
            self.experiment_training(p_training)


            variance_measure = tm.NormalizedVarianceInvariance()
            distance_measure = tm.NormalizedDistanceInvariance(tm.DistanceAggregation(True,False))
            percentage_dataset = config.dataset_size_for_measure(variance_measure)
            p_dataset = measure_package.DatasetParameters(dataset, datasets.DatasetSubset.test, percentage_dataset)
            p_variance = measure_package.Parameters(p_training.id(), p_dataset, transformation, variance_measure)

            # evaluate variance
            results=[self.experiment_measure(p_variance)]
            for b in batch_sizes:
                p = measure_package.Parameters(p_training.id(), p_dataset, transformation, distance_measure,suffix=f"batch_size={b}")
                variance_result=self.experiment_measure(p,batch_size=b)
                results.append(variance_result)
            # plot results
            experiment_name = f"{model_config.name}_{dataset}_{transformation.id()}"
            plot_filepath = self.folderpath / f"{experiment_name}.jpg"

            labels = [l.measure_name(variance_measure)]+[l.measure_name(distance_measure)+f"(b={b})" for b in batch_sizes]
            n = len(results)

            cmap = tmv.default_discrete_colormap()
            color = cmap(range(n))
            colors = np.vstack([color])
            linestyles = ["--"] + ["-" for b in batch_sizes]

            tmv.plot_collapsing_layers_same_model(results, plot_filepath, labels=labels,linestyles=linestyles,colors=colors)
