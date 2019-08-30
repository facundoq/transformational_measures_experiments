#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import runner_utils
import transformation_measure as tm
from experiment import variance, training,model_loading
import config
import util

def run_experiment(model_path:str, measure:tm.Measure, d: variance.DatasetParameters, transformation:tm.TransformationSet, venv_path:str):
    python_command=f"experiment_variance.py -mo \"{model_path}\"" \
        f" -me \"{measure.id()}\" -d \"{d.id()}\" -t \"{transformation.id()}\" -verbose False"
    runner_utils.run_python(venv_path, python_command)


# DATASET

if __name__ == '__main__':
    venv_path=runner_utils.get_venv_path()
    model_names=config.get_models_filepaths()
    model_names.sort()
    measures = config.common_measures()
    dataset_subsets=  [variance.DatasetSubset.train, variance.DatasetSubset.test]
    dataset_percentages= [0.1, 0.5, 1.0]
    message=f"""Running variance measure experiments.
    Models: {", ".join(model_names)}
"""
    configurations=[]
    for model_path in model_names:
        for measure in measures:
            for dataset_subset in dataset_subsets:
                for dataset_percentage in dataset_percentages:
                    model,p,o,scores= training.load_model(model_path, False, load_state=False)
                    dataset= variance.DatasetParameters(p.dataset, dataset_subset, dataset_percentage)
                    if len(p.transformations)>1:
                        configurations.append((model_path,measure,p.transformations,dataset))
                    else:
                        #Model trained without transforms. Test with all transforms
                        for t in config.common_transformations_without_identity():
                            configurations.append( (model_path, measure, t, dataset) )

    n=len(configurations)
    p_all = util.Profiler()
    p_all.event("start")
    for i,(model_path,measure,transformations,dataset) in enumerate(configurations):
        p = util.Profiler()
        p.event(f"{i}/{n} start")
        model, _, _, _ = training.load_model(model_path, True)
        print(f"{i}/{n}")
        run_experiment(model_path, measure, dataset, transformations, venv_path)
        p.event(f"end")
        print(p.summary(human=True))
    p_all.event("end")
    print(p_all.summary(human=True))