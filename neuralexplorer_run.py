#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import experiments.neuralexplorer as ne
from experiments.base import Experiment
from experiments.neuralexplorer.invariance import Invariance


if __name__ == '__main__':

    all_experiments = {
        "Initial":[
        Invariance(),
        ],
    }
    Experiment.run(all_experiments)
    