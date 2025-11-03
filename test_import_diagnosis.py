#!/usr/bin/env python
"""
Diagnostic script to identify which import in omicverse.utils.__init__ is causing issues.
"""

import sys
print(f"Python version: {sys.version}")

# Test each import individually
imports_to_test = [
    ("_data", "from omicverse.utils._data import *"),
    ("_plot", "from omicverse.utils._plot import *"),
    ("_mde", "from omicverse.utils._mde import *"),
    ("_syn", "from omicverse.utils._syn import *"),
    ("_scatterplot", "from omicverse.utils._scatterplot import *"),
    ("_knn", "from omicverse.utils._knn import *"),
    ("_heatmap", "from omicverse.utils._heatmap import *"),
    ("_roe", "from omicverse.utils._roe import roe, roe_plot_heatmap"),
    ("_odds_ratio", "from omicverse.utils._odds_ratio import odds_ratio, plot_odds_ratio_heatmap"),
    ("_shannon_diversity", "from omicverse.utils._shannon_diversity import shannon_diversity, compare_shannon_diversity, plot_shannon_diversity"),
    ("_resolution", "from omicverse.utils._resolution import optimal_resolution, plot_resolution_optimization, resolution_stability_analysis"),
    ("_paga", "from omicverse.utils._paga import cal_paga, plot_paga"),
    ("_cluster", "from omicverse.utils._cluster import cluster, LDA_topic, filtered, refine_label"),
    ("_venn", "from omicverse.utils._venn import venny4py"),
    ("_lsi", "from omicverse.utils._lsi import *"),
    ("_neighboors", "from omicverse.utils._neighboors import neighbors"),
    ("smart_agent", "from omicverse.utils import smart_agent"),
]

print("\n" + "="*70)
print("Testing individual imports from omicverse.utils")
print("="*70)

failed_imports = []
for module_name, import_statement in imports_to_test:
    try:
        exec(import_statement)
        print(f"✅ {module_name:20s} - OK")
    except Exception as e:
        print(f"❌ {module_name:20s} - FAILED: {type(e).__name__}: {str(e)[:60]}")
        failed_imports.append((module_name, e))

print("\n" + "="*70)
print("Testing full omicverse.utils import")
print("="*70)

try:
    import omicverse.utils
    print("✅ omicverse.utils imported successfully")

    # Check if smart_agent is accessible
    if hasattr(omicverse.utils, 'smart_agent'):
        print("✅ smart_agent is accessible via omicverse.utils.smart_agent")
    else:
        print("❌ smart_agent NOT accessible via omicverse.utils.smart_agent")
        print(f"   Available attributes: {[x for x in dir(omicverse.utils) if not x.startswith('_')][:10]}...")

except Exception as e:
    print(f"❌ Failed to import omicverse.utils: {e}")

print("\n" + "="*70)
print("Summary")
print("="*70)
if failed_imports:
    print(f"\n{len(failed_imports)} import(s) failed:")
    for module_name, error in failed_imports:
        print(f"  - {module_name}: {type(error).__name__}")
else:
    print("\n✅ All imports successful!")

sys.exit(len(failed_imports))
