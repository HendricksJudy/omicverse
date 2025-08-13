import json
import os
from typing import Iterable, Tuple
import pandas as pd


def load_cl_popv_mapping(cl_obo_file: str) -> Tuple[dict, dict]:
    """Load Cell Ontology mapping dictionaries from ``cl_popv.json``.

    Parameters
    ----------
    cl_obo_file
        Path to ``cl_popv.json`` or a directory containing it.

    Returns
    -------
    tuple
        Two dictionaries ``(name2id, id2name)`` mapping lowercase cell type
        names to Cell Ontology IDs and vice versa.
    """
    if os.path.isdir(cl_obo_file):
        cl_obo_file = os.path.join(cl_obo_file, "cl_popv.json")

    with open(cl_obo_file) as f:
        cl_dict = json.load(f)

    name2id = {k.lower(): v for k, v in cl_dict.get("lbl_2_id", {}).items()}
    id2name = cl_dict.get("id_2_lbl", {})
    return name2id, id2name


def map_celltypes_to_ontology(celltypes: Iterable[str], cl_obo_file: str,
                              return_name: bool = True) -> pd.Series | pd.DataFrame:
    """Map cell type names to Cell Ontology identifiers.

    Parameters
    ----------
    celltypes
        Iterable of cell type strings.
    cl_obo_file
        Path to ``cl_popv.json`` or directory containing it.
    return_name
        When ``True``, return a :class:`~pandas.DataFrame` with ``cl_id`` and
        ``cl_name``. Otherwise only ``cl_id`` as a Series is returned.
    """
    name2id, id2name = load_cl_popv_mapping(cl_obo_file)
    series = pd.Series(list(celltypes))
    cl_id = series.astype(str).str.lower().map(name2id)
    if not return_name:
        cl_id.index = range(len(series))
        return cl_id
    cl_name = cl_id.map(id2name)
    df = pd.DataFrame({"cl_id": cl_id, "cl_name": cl_name}, index=series.index)
    return df
