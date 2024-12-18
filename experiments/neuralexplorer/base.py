
from pathlib import Path
from experiment import measure
from ..base import Experiment


default_base_folderpath = Path("~/neuralexplorer").expanduser()
default_base_folderpath.mkdir(parents=True,exist_ok=True)

class NeuralExplorerExperiment(Experiment):

    def __init__(self):
        super().__init__(default_base_folderpath)
    
    def description(self) -> str:
        return "NeuralExplorer Experiment"


