import os

from pathlib import Path

import transformational_measures.measure
from .models import *
from .datasets import *
from .measures import *
from .transformations import *
from transformational_measures.transformations import *

def base_path():
    return Path(os.path.expanduser("~/"))

def testing_path():
    return base_path() / "testing"

