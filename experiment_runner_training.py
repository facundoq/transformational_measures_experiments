#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import os
import utils
def run_experiment(experiment, model_name, dataset_name,transformation_name, venv_path):
    python_command=f"{experiment}.py -m {model_name} -d {dataset_name} -t {transformation_name} -verbose False"
    utils.run_python(venv_path,python_command)


# DATASET
import datasets

from pytorch.experiment import model_loading
if __name__ == '__main__':
    venv_path=utils.get_venv_path()
    model_names=model_loading.get_model_names()
    model_names=["AllConvolutional","SimpleConv","ResNet","VGGLike"]
    dataset_names=datasets.names
    dataset_names=["cifar10"]
    transformation_names=["r16_s1_t0","r0_s1_t0"]
    train=True
    experiments=["experiment_variance"]

    message=f"""Running experiments, train={train}
    Experiments: {", ".join(experiments)}
    Models: {", ".join(model_names)}
    Datasets: {", ".join(dataset_names)}
    transformations: {", ".join(transformation_names)}
    """

    for model_name in model_names:
        for dataset_name in dataset_names:
            for transformation_name in transformation_names:
                if train:
                    run_experiment("experiment_training",model_name,dataset_name, transformation_name, venv_path)
                # for experiment in experiments:
                        # run_experiment(experiment,model_name,dataset_name,venv_path)
