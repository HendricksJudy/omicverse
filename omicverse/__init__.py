r"""
OmicVerse package initialization.
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:  # pragma: no cover - fallback for old Python
    from pkg_resources import get_distribution

    version = lambda name: get_distribution(name).version  # type: ignore

from . import bulk, single, utils, bulk2single, pp, space, pl, externel
from .utils._data import read
from .utils._plot import palette, ov_plot_set, plot_set
from .utils._env_checker import check_environment, environment_wizard

name = "omicverse"
__version__ = version(name)

from ._settings import settings

import matplotlib.pyplot as plt  # re-export
plt = plt

import numpy as np
np = np

import pandas as pd
pd = pd
