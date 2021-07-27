from .. import TMExperiment

import config

from ..language import English

from ..tasks.train import TrainParameters,Task
from ..tasks import train


class SameEquivarianceExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        base_folderpath = config.base_path() / "same_equivariance"
        super().__init__(base_folderpath)


    def model_trained(self, p: TrainParameters):
        # custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filepaths = [self.model_path_new(p)] + [self.model_path_new(p, s) for s in p.tc.savepoints]
        exist = [p.exists() for p in filepaths]
        return all(exist)

    def model_path_new(self, p: TrainParameters, savepoint=None, custom_models_folderpath=None):
        custom_models_folderpath = self.models_folder() if custom_models_folderpath is None else custom_models_folderpath
        filename = f"{p.id(savepoint)}.pt"
        folder = p.mc.__class__.__name__
        filepath = custom_models_folderpath / folder / filename
        return filepath


    def train(self,p:train.TrainParameters):
        if not self.model_trained(p):
            print(f"Training model {p.id()} for {p.tc.epochs} epochs ({p.tc.convergence_criteria})...")
            train.train(p, self)
        else:
            print(f"Model {p.id()} already trained.")

    def measure(self,):
        pass