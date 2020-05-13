from .base import Language
from .english import English
from .spanish import Spanish

l=English()

def set_language(lang:Language):
    l=lang