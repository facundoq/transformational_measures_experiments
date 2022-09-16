from .. import TMExperiment

import config

from ..language import English

from ..tasks.train import TrainParameters,Task
from ..tasks import train
import tmeasures as tm
from experiment import measure

from ..models import ModelConfig
from pathlib import Path

default_base_folderpath = Path("~/same_equivariance").expanduser()
default_base_folderpath.mkdir(parents=True,exist_ok=True)

class SameEquivarianceExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        super().__init__(default_base_folderpath)



    def get_train_config(self,mc: ModelConfig, dataset: str, task: Task,
                                   transformations: tm.pytorch.PyTorchTransformationSet,savepoints=True,verbose=False):
        epochs = mc.epochs(dataset, task, transformations)
        if savepoints:
            savepoints_percentages = [0, 1, 2, 5, 10, 25, 50, 75, 100]
            savepoints_epochs = []#[0,1,2,3]
            savepoints = savepoints_epochs + [sp * epochs // 100 for sp in savepoints_percentages]
            savepoints = sorted(list(set(savepoints)))
        else:
            savepoints=[]

        metric = "rae"
        cc = train.MaxMetricConvergence(mc.max_rae(dataset, task, transformations), metric)
        optimizer = dict(optim="adam", lr=0.0001)
        tc = train.TrainConfig(epochs, cc, optimizer=optimizer, savepoints=savepoints, verbose=verbose, num_workers=4)
        return tc, metric

    


