from .general import *
from .nn import *
from .diagnostics import *

del nn # we don't want torch's nn here
from . import data, nn, competition