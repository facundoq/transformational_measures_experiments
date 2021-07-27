#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import pathlib
import os
from experiment import measure, training
from experiments.invariance.base import InvarianceExperiment
from experiments.same_equivariance.base import SameEquivarianceExperiment
import texttable
import config
import argparse
from experiments.tasks import train
class MockInvarianceExperiment(InvarianceExperiment):
    def run(self):
        pass
    def description(self):
        return ""

    def get_row(self, model_path):
        model, p, o, scores = training.load_model(model_path, False, load_state=False)
        train_accuracy = scores["train"][1]
        test_accuracy = scores["test"][1]

        row = (model.name, p.dataset, p.transformations.id(), p.epochs, train_accuracy, test_accuracy)

        del model
        del scores
        del p
        del o

        return row

class MockSameEquivarianceExperiment(SameEquivarianceExperiment):
    def run(self):
        pass
    def description(self):
        return ""

    def get_row(self,model_path):
        p, model, scores = train.load_model(model_path, "cpu", load_state=False)
        header = [k for k in scores.keys() if k.startswith("test_") or k.startswith("train_")]
        # print(header)
        values = [scores[k] for k in header]
        # print(values)

        # train_accuracy = scores["train"][1]
        # test_accuracy = scores["test"][1]
        p: train.TrainParameters = p

        row = (model.name, p.dataset_name, p.transformations.id(), p.tc.epochs, *values)
        del model
        del scores
        del p

        return row,header




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Show metrics of models")
    experiments = {'Invariance':MockInvarianceExperiment(),
                    'SameEquivariance':MockSameEquivarianceExperiment()}

    parser.add_argument('experiment',
                        choices=list(experiments.keys()),
                        type=str,
                        help="Experiment's models")
    args = parser.parse_args()
    experiment = experiments[args.experiment]
    models_folderpath = experiment.models_folder()
    models_filepaths = list(pathlib.Path(models_folderpath).glob('**/*.pt'))
    models_filepaths.sort()
    message=f"""Training results for experiment {args.experiment} from {models_folderpath}:"""
    print(message)
    table=texttable.Texttable(max_width=120)
    header=["models","dataset","transform","epochs"]


    data=[]
    for model_path in models_filepaths:
        # Avoid intermediate savepoint models.
        if "savepoint" in model_path.name:
            continue
        row,row_header = experiment.get_row(model_path)
        data.append(row)
    if len(data)>0:
        header = header + row_header
        table.header(header)
        table.add_rows(data,header=False)
        table_str=table.draw()
        print(table_str)

        with open(os.path.join(config.base_path(), f"{args.experiment}_latest_training_results.txt"), "w") as f:
            f.write(table_str)
    else:
        print("No models found")
