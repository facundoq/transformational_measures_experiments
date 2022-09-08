from .common import *
import experiment.measure as measure_package
import tmeasures.visualization as tmv

class VisualizeMeasures(InvarianceExperiment):
    def description(self):
        return """Visualize measures by layer and with heatmap."""

    def run(self):

        measures = all_invariance_measures

        model_generators = simple_models_generators
        
        transformations = common_transformations_combined

        combinations = itertools.product(model_generators, dataset_names, transformations, measures)
        for (model_config_generator, dataset, transformations, measure) in combinations:
            
            mc,tc,p,model_path = self.train_default(Task.Classification,dataset,transformations,model_config_generator)
            result = self.measure_default(dataset,mc.id(),model_path,transformations, measure,default_measure_options,default_dataset_percentage)

            # plot results
            experiment_name = f"{measure.id()}_{mc.id()}_{dataset}_{transformations.id()}"
            bylayer_filepath = self.folderpath / f"{experiment_name}_bylayer.jpg"
            heatmap_filepath = self.folderpath / f"{experiment_name}_heatmap.jpg"

            tmv.plot_average_activations_same_model([result], )
            self.savefig(bylayer_filepath)
            tmv.plot_heatmap(result)
            self.savefig(heatmap_filepath)