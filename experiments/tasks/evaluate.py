
import tmeasures as tm
from pathlib import Path
import torch
from poutyne import Model

from . import Task
from .train import prepare_dataset
import datasets
class EvalConfig:
    def __init__(self,device:torch.device,batch_size:int):
        self.device=device
        self.batch_size=batch_size
class EvalParameters:
    def __init__(self,transformations:tm.TransformationSet,metrics:list[str],dataset:str,task:Task,subset: datasets.DatasetSubset,device:torch.device,eval_config:EvalConfig):
        
        self.metrics = metrics
        self.transformations = transformations
        self.dataset = dataset
        self.task = task
        self.subset = subset
        self.eval_config=eval_config
     

from .train import load_model
def evaluate(model_filepath:Path,c:EvalParameters):
    
    model,p,original_scores = load_model((model_filepath,c.eval_config.device))

    poutyne_model = Model(model,epoch_metrics=c.metrics)
    
    train_dataset, test_dataset, input_shape, dim_output = prepare_dataset(c.transformations,c.dataset,c.task)
    if c.subset == datasets.DatasetSubset.test:
        scores = poutyne_model.evaluate_dataset(test_dataset,batch_size=c.eval_config.batch_size)
    elif c.subset == datasets.DatasetSubset.train:
        scores = poutyne_model.evaluate_dataset(train_dataset,batch_size=c.eval_config.batch_size)
    else:
        raise ValueError(c.subset)
    return scores







