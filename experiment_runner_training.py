#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import os
import utils
def run_experiment(experiment:str, model_name:str, dataset_name:str,transformation_name:str, venv_path:str):
    python_command=f"{experiment}.py -m {model_name} -d {dataset_name} -t {transformation_name} -verbose False"
    utils.run_python(venv_path,python_command)


# DATASET
import datasets
import transformation_measure as tm
from pytorch.experiment import model_loading
if __name__ == '__main__':
    venv_path=utils.get_venv_path()
    model_names=model_loading.get_model_names()
    #model_names=["AllConvolutional","SimpleConv","ResNet","VGGLike"]
    #dataset_names=datasets.names
    dataset_names=["mnist","cifar10"]

    transformations=tm.common_transformations()
    transformation_names=[t.id() for t in transformations]
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
