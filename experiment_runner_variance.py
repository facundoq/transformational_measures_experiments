#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import os
import utils
def run_experiment(model_path, dataset_name, transformation_name, measure_name, dataset_subset, dataset_percentage, venv_path):
    python_command=f"experiment_variance.py -mo \"{model_path}\"" \
        f" -me \"{measure_name}\" -d \"{dataset_name}({dataset_subset},p={dataset_percentage})\" -t \"{transformation_name}\" -verbose False"
    utils.run_python(venv_path,python_command)


# DATASET
import datasets
from pytorch.experiment import training
import transformation_measure as tm
if __name__ == '__main__':
    venv_path=utils.get_venv_path()
    model_names=training.get_models()
    measures = [tm.TransformationMeasure(tm.MeasureFunction.std, tm.ConvAggregation.sum)
        , tm.NormalizedMeasure(tm.TransformationMeasure(tm.MeasureFunction.std, tm.ConvAggregation.sum)
                               , tm.SampleMeasure(tm.MeasureFunction.std, tm.ConvAggregation.sum))
        , tm.SampleMeasure(tm.MeasureFunction.std, tm.ConvAggregation.sum)
                ]
    measure_names=[m.id() for m in measures]
    dataset_subsets= ["train", "test"]
    dataset_percentages= ["0.1", "0.5", "1.0"]
    message=f"""Running experiments
    Models: {", ".join(model_names)}
"""
    for model_path in model_names:
        for measure_name in measure_names:
            for dataset_subset in dataset_subsets:
                for dataset_percentage in dataset_percentages:
                    model,p,o,scores=training.load_model(model_path, True)
                    run_experiment(model_path, p.dataset, p.transformations, measure_name, dataset_subset, dataset_percentage, venv_path)
