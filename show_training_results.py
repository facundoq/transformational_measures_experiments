#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import os
from experiment import variance, training
import texttable
import config

if __name__ == '__main__':
    models_filepaths= config.get_models_filepaths()
    models_filepaths.sort()
    message=f"""Training results"""
    print(message)
    table=texttable.Texttable()
    header=["models","dataset","transform","epochs","train","test"]
    table.header(header)

    data=[]
    for model_path in models_filepaths:
        model, p, o, scores = training.load_model(model_path, False, load_state=False)
        train_accuracy=scores["train"][1]
        test_accuracy = scores["test"][1]
        row=(model.name,p.dataset,p.transformations.id(),p.epochs,train_accuracy,test_accuracy)
        data.append(row)

    table.add_rows(data,header=False)
    table_str=table.draw()
    print(table_str)

    with open(os.path.join(config.base_path(), "latest_training_results.txt"), "w") as f:
        f.write(table_str)
