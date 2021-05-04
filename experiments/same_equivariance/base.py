from .. import TMExperiment

import config

from ..language import English

class SameEquivarianceExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        base_folderpath = config.base_path() / "same_equivariance"
        super().__init__(base_folderpath)

