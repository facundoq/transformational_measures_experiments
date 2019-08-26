#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import runner_utils
def run_experiment(experiment:str, model_name:str, dataset_name:str,transformation_name:str, venv_path:str):
    python_command=f"{experiment}.py -model {model_name} -dataset {dataset_name} -transformation \"{transformation_name}\" -verbose False -train_verbose False -num_workers 4"
    runner_utils.run_python(venv_path, python_command)


# DATASET
import transformation_measure as tm
from experiment import model_loading

if __name__ == '__main__':
    venv_path=runner_utils.get_venv_path()
    model_names= model_loading.get_model_names()
    #model_names=["AllConvolutional","SimpleConv","ResNet","VGGLike"]
    #dataset_names=datasets.names
    dataset_names=["mnist","cifar10"]

    transformations=tm.common_transformations()
    transformation_names=[t.id() for t in transformations]
    experiments=["experiment_variance"]

    message=f"""Training all combinations of:
    Models: {", ".join(model_names)}
    Datasets: {", ".join(dataset_names)}
    transformations: {", ".join(transformation_names)}
    """
    print(message)

    for model_name in model_names:
        for dataset_name in dataset_names:
            for transformation_name in transformation_names:
                    run_experiment("experiment_training",model_name,dataset_name, transformation_name, venv_path)
                # for experiment in experiments:
                        # run_experiment(experiment,model_name,dataset_name,venv_path)
