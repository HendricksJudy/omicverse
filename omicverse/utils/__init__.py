r"""Utility submodule for OmicVerse."""

from ._data import *
from ._plot import *
# from ._genomics import *  # optional
from ._mde import *
from ._syn import *
from ._scatterplot import *
from ._knn import *
from ._heatmap import *
from ._roe import roe
from ._paga import cal_paga, plot_paga
from ._cluster import cluster, LDA_topic, filtered, refine_label
from ._venn import venny4py
from ._lsi import *
from ._neighboors import neighbors
from ._env_checker import check_environment, environment_wizard
