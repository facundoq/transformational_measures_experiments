from .. import TMExperiment, train
from pathlib import Path
from experiment import measure, training, accuracy
import transformational_measures as tm
import datasets
import config
import torch
from ..language import English

class InvarianceExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        base_folderpath = config.base_path() / "invariance"
        super().__init__(base_folderpath)

