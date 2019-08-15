#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import os
import runner_utils
from pytorch import variance
import transformation_measure as tm

def run_experiment(model_path:str, measure:tm.Measure,d:variance.DatasetParameters, transformation:tm.TransformationSet, venv_path:str):
    python_command=f"experiment_variance.py -mo \"{model_path}\"" \
        f" -me \"{measure.id()}\" -d \"{d.id()}\" -t \"{transformation.id()}\" -verbose False"
    runner_utils.run_python(venv_path, python_command)


# DATASET
import datasets
from pytorch.experiment import training
if __name__ == '__main__':
    venv_path=runner_utils.get_venv_path()
    model_names=training.get_models()
    measures = tm.common_measures()
    dataset_subsets=  [variance.DatasetSubset.train,variance.DatasetSubset.test]
    dataset_percentages= [0.1, 0.5, 1.0]
    message=f"""Running experiments
    Models: {", ".join(model_names)}
"""
    for model_path in model_names:
        for measure in measures:
            for dataset_subset in dataset_subsets:
                for dataset_percentage in dataset_percentages:
                    model,p,o,scores=training.load_model(model_path, True)
                    dataset=variance.DatasetParameters(p.dataset,dataset_subset,dataset_percentage)
                    if len(p.transformations)>1:
                        run_experiment(model_path, measure, dataset, p.transformations, venv_path)
                    else:
                        #Model trained without transforms. Test with all transforms
                        for t in tm.common_transformations_without_identity():
                            run_experiment(model_path,measure,dataset,t,venv_path)
