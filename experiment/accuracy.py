import typing
from experiment import training
import datasets, torch
from torch import nn
import transformational_measures as tm
from pathlib import Path
from pytorch.pytorch_image_dataset import TransformationStrategy
from utils.profiler import Profiler

class DatasetParameters:
    def __init__(self,name:str,subset: datasets.DatasetSubset,percentage:float):
        assert(percentage>0)
        assert(percentage<=1)
        self.name=name
        self.subset=subset
        self.percentage=percentage

    def __repr__(self):
        return f"{self.name}({self.subset.value},p={self.percentage:.2})"
    def id(self):
        return str(self)

class Parameters:
    def __init__(self, model_path:Path, dataset:DatasetParameters, transformations:tm.TransformationSet,transformation_strategy:TransformationStrategy=TransformationStrategy.random_sample):
        self.model_path=model_path
        self.dataset=dataset
        self.transformations=transformations
        self.transformation_strategy=transformation_strategy

    def model_name(self):
        return self.model_path.stem

    def id(self):
        return f"{self.model_name()}_{self.dataset}_{self.transformations.id()}_{self.transformation_strategy.value}"

    def __repr__(self):

        return f"AccuracyExperiment parameters: models={self.model_name()}, dataset={self.dataset} transformations={self.transformations}, transformation strategy={self.transformation_strategy.value}"

class Options:
    def __init__(self,verbose:bool,batch_size:int,num_workers:int,device:str):
        self.verbose=verbose
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.device=device

    def get_eval_options(self):
        return training.EvalOptions(self.device,self.batch_size,self.num_workers)

class AccuracyExperimentResult:
    def __init__(self, parameters:Parameters,accuracy:float):
        self.parameters=parameters
        self.accuracy=accuracy

    def __repr__(self):
        description = f"{AccuracyExperimentResult.__name__}, params: {self.parameters}, accuracy:{self.accuracy}"
        return description

    def id(self):
        return f"{self.parameters.id()}"


import config


def experiment(p: Parameters, o: Options):
    assert(len(p.transformations)>0)
    

    model, training_parameters, scores = training.load_model(p.model_path, device=o.device)

    if o.verbose:
        print("### ", model)
        print("### Scores obtained:")
        training.print_scores(scores)

    dataset = datasets.get_classification(p.dataset.name)
    dataset.normalize_features()
    dataset = dataset.reduce_size_stratified(p.dataset.percentage)
    if o.verbose:
        print(dataset.summary())

    if o.verbose:
        print(f"Measuring accuracy with transformations {p.transformations} on dataset {p.dataset} of size {dataset.size(p.dataset.subset)}...")

    result:float=measure(model,dataset,p.transformations,o,p.dataset.subset)

    del model
    del dataset
    torch.cuda.empty_cache()

    return AccuracyExperimentResult(p, result)

def measure(model:nn.Module,dataset:datasets.ClassificationDataset,transformations:tm.TransformationSet,o:Options,subset:datasets.DatasetSubset)-> float:

    scores = training.eval_scores(model,dataset,transformations,TransformationStrategy.random_sample,o.get_eval_options(),subsets=subset.value)
    loss, accuracy = scores[subset.value]
    return accuracy

def main(p:Parameters,o:Options):
    profiler=Profiler()
    profiler.event("start")
    if o.verbose:
        print(f"Experimenting with parameters: {p}")
    accuracy_results=experiment(p,o)
    profiler.event("end")
    print(profiler.summary(human=True))
    
    return accuracy_results
