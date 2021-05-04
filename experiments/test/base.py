from .. import TMExperiment, train
import config
from ..language import English

class TestExperiment(TMExperiment):

    def __init__(self,l=English()):
        self.l=l
        base_folderpath = config.base_path() / "test"
        super().__init__(base_folderpath)



