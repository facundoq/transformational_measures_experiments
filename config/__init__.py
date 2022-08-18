import os

from pathlib import Path

import tmeasures.measure
from .datasets import *
from .transformations import *
from tmeasures.transformations import *

def base_path():
    return Path(os.path.expanduser("~/"))

def testing_path():
    return base_path() / "testing"

