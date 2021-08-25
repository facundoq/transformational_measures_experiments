#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from experiments.same_equivariance import *
from experiments import language,Experiment

if __name__ == '__main__':
    language.set_language(language.English())


    all_experiments = {
        "Initial":[
        TrainModels(),
        # CompareSameEquivarianceNormalized(),
        TransformationSampleSizes(),

        ],
    }
    experiments, o = Experiment.parse_args(all_experiments)
    if o.show_list:
        Experiment.print_table(experiments)
    else:
        for e in experiments:
            e(force=o.force)
