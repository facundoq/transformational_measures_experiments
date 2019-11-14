#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import transformation_measure as tm
from experiment import variance, training, utils_runner

import config
import os


all_model_names = config.model_names
bn_model_names = [name for name in all_model_names if name.endswith("BN")]
model_names = [name for name in all_model_names if not name.endswith("BN")]

model_names.sort()
model_names= [name for name in model_names if not name.startswith("ResNet")]
model_names=["SimpleConv"]

measures = config.common_measures()

# dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
# dataset_percentages= [0.1, 0.5, 1.0]x

dataset_names= ["mnist", "cifar10"]

venv_path = utils_runner.get_venv_path()

import abc

class Experiment(abc.ABC):
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
        pass
        # mf,ca_none=tm.MeasureFunction.std,tm.ConvAggregation.none
        # dmean, dmax, = tm.DistanceAggregation.mean, tm.DistanceAggregation.max
        # measure_sets={"Variance":[
        #                         # tm.SampleMeasure(mf,ca_none)
        #                         # tm.TransformationMeasure(mf,ca_none),
        #                         tm.TransformationVarianceMeasure(mf, ca_none),
        #                         tm.SampleVarianceMeasure(mf, ca_none),
        #                          ],
        #               "Distance":[
        #                   tm.DistanceTransformationMeasure(dmean),
        #                   tm.DistanceSampleMeasure(dmean),
        #               ],
        #               "HighLevel":[
        #                           tm.AnovaMeasure(ca_none,0.99,bonferroni=True),
        #                           tm.NormalizedMeasure(mf,ca_none),
        #                           tm.NormalizedVarianceMeasure(mf, ca_none),
        #                           tm.DistanceMeasure(dmean),
        #                            ],
        #         "Equivariance":[
        #                     tm.DistanceSameEquivarianceMeasure(dmean),
        #                     tm.DistanceMeasure(dmean),
        #                     ]
        #               }
        #
        # #model_names=["SimpleConv","VGGLike","AllConvolutional"]
        # # model_names=["ResNet"]
        # transformations = config.common_transformations_without_identity()
        #
        # combinations = itertools.product(*[model_names, dataset_names, transformations, measure_sets.items()])
        # for (model,dataset,transformation,measure_set) in combinations:
        #     # train
        #     epochs = config.get_epochs(model, dataset, transformation)
        #     p_training= training.Parameters(model, dataset, transformation, epochs, 0)
        #     self.experiment_training(p_training)
        #     # generate variance params
        #     variance_parameters=[]
        #     measure_set_name, measures = measure_set
        #     for m in measures:
        #         p= 0.5 if m.__class__==tm.AnovaMeasure else 0.1
        #         p_dataset= variance.DatasetParameters(dataset, variance.DatasetSubset.test, p)
        #         p_variance=variance.Parameters(p_training.id(), p_dataset, transformation, m)
        #         variance_parameters.append(p_variance)
        #     # evaluate variance
        #     model_path=config.model_path(p_training)
        #     for p_variance in variance_parameters:
        #         self.experiment_variance(p_variance,model_path)
        #
        #     # plot results
        #     experiment_name = f"{model}_{dataset}_{transformation.id()}_{measure_set_name}"
        #     plot_filepath = os.path.join(self.plot_folderpath, f"{experiment_name}.png")
        #     results = config.load_results(config.results_paths(variance_parameters))
        #     labels=[m.id() for m in measures]
        #     visualization.plot_collapsing_layers(results, plot_filepath, labels=labels, title=experiment_name)

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

import sys
if __name__ == '__main__':
    sys.exit()

    all_experiments=[
        CompareMeasures(),
        # ComparePreConvAgg(),
        # CollapseConvBeforeOrAfter(),
        # # #

        # InvarianceForRandomNetworks(),

    ]

    experiments = parse_args(all_experiments)

    for e in experiments:
        e()



