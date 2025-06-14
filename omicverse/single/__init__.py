"""Single-cell omics analysis utilities."""

# Heavy functionality lives in submodules and is imported lazily.

from ._cosg import cosg
from ._anno import pySCSA,MetaTiME,scanpy_lazy,scanpy_cellanno_from_dict,get_celltype_marker
from ._nocd import scnocd
from ._mofa import pyMOFAART,pyMOFA,GLUE_pair,factor_exact,factor_correlation,get_weights,glue_pair
from ._scdrug import autoResolution,writeGEP,Drug_Response
from ._cpdb import (cpdb_network_cal,cpdb_plot_network,
                    cpdb_plot_interaction,
                    cpdb_interaction_filtered,
                    cpdb_submeans_exacted,cpdb_exact_target,cpdb_exact_source)
from ._scgsea import (geneset_aucell,pathway_aucell,pathway_aucell_enrichment,
                      geneset_aucell_tmp,pathway_aucell_tmp,pathway_aucell_enrichment_tmp,
                      pathway_enrichment,pathway_enrichment_plot,)
from ._via import pyVIA,scRNA_hematopoiesis
from ._simba import pySIMBA
from ._tosica import pyTOSICA
from ._atac import atac_concat_get_index,atac_concat_inner,atac_concat_outer
from ._batch import batch_correction
from ._cellfategenie import Fate,gene_trends,mellon_density
from ._ltnn import scLTNN,plot_origin_tesmination,find_related_gene
from ._traj import TrajInfer,fle
from ._diffusionmap import diffmap
from ._cefcon import pyCEFCON,convert_human_to_mouse_network,load_human_prior_interaction_network,mouse_hsc_nestorowa16
from ._aucell import aucell
from ._metacell import MetaCell,plot_metacells,get_obs_value
from ._mdic3 import pyMDIC3
from ._cnmf import *
from ._gptcelltype import gptcelltype,gpt4celltype,get_cluster_celltype
from ._cytotrace2 import cytotrace2
from ._gptcelltype_local import gptcelltype_local
from ._sccaf import SCCAF_assessment,plot_roc,SCCAF_optimize_all,color_long
from ._multimap import TFIDF_LSI,Wrapper,Integration,Batch
from ._scdiffusion import scDiffusion
from ._cellvote import get_cluster_celltype,CellVote
from ._deg_ct import DCT,DEG
from ._lazy_function import lazy
from ._lazy_report import generate_scRNA_report

