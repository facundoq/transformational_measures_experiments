#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import sys
import argcomplete

from experiments.same_equivariance import *
from experiments import language
from typing import Dict,List

class Options:
    def __init__(self, show_list: bool, force: bool):
        self.show_list = show_list
        self.force = force




def parse_args(experiments: Dict[str,List[SameEquivarianceExperiment]]) -> ([SameEquivarianceExperiment], Options):
    parser = argparse.ArgumentParser(description="Run invariance with transformation measures.")
    group_names = list(experiments.keys())

    experiments_plain = [e for g in experiments.values() for e in g]
    experiment_names = [e.id() for e in experiments_plain]
    experiment_dict = dict(zip(experiment_names, experiments_plain))

    parser.add_argument('-experiment',
                        help=f'Choose an experiment to run',
                        type=str,
                        default=None,
                        required=False,choices=experiment_names, )
    parser.add_argument('-group',
                        help=f'Choose an experiment group to run',
                        type=str,
                        default=None,
                        required=False,choices=group_names, )
    parser.add_argument('-force',
                        help=f'Force invariance to rerun even if they have already finished',
                        action="store_true")
    parser.add_argument('-list',
                        help=f'List invariance and status',
                        action="store_true")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if not args.experiment is None and not args.group is None:
        sys.exit("Cant specify both experiment and experiment group")
    selected_experiments = experiments_plain
    if not args.experiment is None:
        selected_experiments = [experiment_dict[args.experiment]]
    if not args.group is None:
        selected_experiments = experiments[args.group]

    return selected_experiments, Options(args.list, args.force)



if __name__ == '__main__':
    language.set_language(language.English())
    todo = [

    ]
    if len(todo)>0:
        print("TODO implement invariance:", ",".join([e.__class__.__name__ for e in todo]))


    all_experiments = {
        "Initial":[
        TrainModels(),
        ],
        "SamplePlots":[
        DatasetTransformationPlots(),
        STMatrixSamples(),

        # DataAugmentationClassical(),
        # DataAugmentationHandshape(),
        ],
        "Accuracies":[
        SimpleConvAccuracies(),
        ModelAccuracies(),
        ],
        "Measures":[
        # InvarianceMeasureCorrelation(),
        MeasureCorrelationWithTransformation(),
        CompareMeasures(),
        DistanceApproximation(),
        # CompareSameEquivariance(),
        #   CompareSameEquivarianceNormalized(),
        # CompareSameEquivarianceSimple(),
        ],
        "Weights":[
        RandomInitialization(),
        RandomWeights(),
        DuringTraining(),

        ],
        "Dataset":[
        DatasetSize(),
        DatasetSubset(),
        DatasetTransfer(),
        # #
        ]
        ,"Variants":[
        AggregationFunctionsVariance(),
        AggregationBeforeAfter(),
        # AggregationFunctionsDistance(),
        Stratified(),
        # SameEquivarianceNormalization(),
        #
        ]
        ,"Transformations":[
        TransformationDiversity(),
        TransformationComplexity(),
        TransformationSetSize(),
        ]
        ,"Hiperparameters":[
        BatchNormalization(),
        ActivationFunction(),
        MaxPooling(),
        KernelSize(),
        ],
        "Goodfellow":[
            CompareGoodfellowAlpha(),
            CompareGoodfellow(),
        ]
        ,"Models":[
        CompareModels(),
        TIPooling(),
        ]
        ,"Validate":[
        # VisualizeInvariantFeatureMaps(),
        # ValidateMeasure(),
        # ValidateGoodfellow(),
        ]
    }
    experiments, o = parse_args(all_experiments)
    if o.show_list:
        print_table(experiments)
    else:
        for e in experiments:
            e(force=o.force)
