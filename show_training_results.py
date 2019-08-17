#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from pytorch import variance
import os
from pytorch.experiment import training
import texttable

if __name__ == '__main__':
    model_names=training.get_models()
    model_names.sort()
    message=f"""Training results"""
    print(message)
    table=texttable.Texttable()
    header=["model","dataset","transform","epochs","train","test"]
    table.header(header)

    data=[]
    for model_path in model_names:
        model, p, o, scores = training.load_model(model_path, False,load_state=False)
        train_accuracy=scores["train"][1]
        test_accuracy = scores["test"][1]
        row=(model.name,p.dataset,p.transformations.id(),p.epochs,train_accuracy,test_accuracy)
        data.append(row)

    table.add_rows(data,header=False)
    table_str=table.draw()
    print(table_str)
with open(os.path.join(variance.base_folder(),"latest_training_results.txt"),"w") as f:
    f.write(table_str)
