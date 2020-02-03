#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

## Calculate the variance of each activation in a model.
## NOTE:
## You should run "train.py" before this script to generate and train the model for
## a given dataset/model/transformation combination
#
# import ray
# ray.init()


import config
from experiment import  accuracy
from utils import profiler



if __name__ == "__main__":
    profiler= profiler.Profiler()
    p, o = accuracy.parse_parameters()
    profiler.event("start")
    if o.verbose:
        print(f"Experimenting with parameters: {p}")
    accuracy_results=accuracy.experiment(p,o)
    profiler.event("end")
    print(profiler.summary(human=True))
    config.save_accuracy(accuracy_results)



