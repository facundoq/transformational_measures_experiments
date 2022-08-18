#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from experiments.invariance import *
from experiments import language
from experiments.base import Experiment


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
        # SimpleConvAccuracies(),
        # ModelAccuracies(),
        ],
        "Measures":[
        VisualizeMeasures(),
        TransformationSampleSizes(),            
        # # # InvarianceMeasureCorrelation(),
        
        MeasureCorrelationWithTransformation(),
        # # CompareMeasures(),
        # # DistanceApproximation(),
        # # # CompareSameEquivariance(),
        # # #   CompareSameEquivarianceNormalized(),
        # # # CompareSameEquivarianceSimple(),
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
        # # # #
        ],
        # # ,"Variants":[
        # # AggregationFunctionsVariance(),
        # # AggregationBeforeAfter(),
        # # # AggregationFunctionsDistance(),
        # # Stratified(),
        # # # SameEquivarianceNormalization(),
        # # #
        
        "Transformations":[
        TransformationDiversity(),
        TransformationComplexity(),
        TransformationSetSize(),
        ]
        ,"Hiperparameters":[
        BatchNormalization(),
        ActivationFunctionComparison(),
        MaxPooling(),
        KernelSize(),
        ],
        # # "Goodfellow":[
        # #     CompareGoodfellowAlpha(),
        # #     CompareGoodfellow(),
        # # ]
        # # ,"Models":[
        # # CompareModels(),
        # # TIPooling(),
        # ],
        "Validate":[
        # VisualizeInvariantFeatureMaps(),
        ]
    }

    experiments, o = Experiment.parse_args(all_experiments)
    if o.show_list:
        Experiment.print_table(experiments)
    else:
        for e in experiments:
            e(force=o.force)