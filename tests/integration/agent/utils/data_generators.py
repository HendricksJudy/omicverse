"""
Generate reference data from tutorial notebooks.

This module provides utilities to:
- Execute tutorial notebooks
- Extract intermediate outputs
- Save as reference data for integration tests
"""

import scanpy as sc
import omicverse as ov
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List


def generate_pbmc3k_references(output_dir: Path, force: bool = False):
    """
    Generate reference data for PBMC3k workflows.

    Executes standard PBMC3k preprocessing pipeline and saves
    intermediate outputs for test validation.

    Args:
        output_dir: Directory to save reference files
        force: Overwrite existing references

    Returns:
        dict: Metadata about generated references
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    metrics_file = output_dir / 'reference_metrics.json'
    if metrics_file.exists() and not force:
        print(f"References already exist at {output_dir}. Use force=True to regenerate.")
        with open(metrics_file) as f:
            return json.load(f)

    print("Generating PBMC3k reference data...")

    # Load raw data
    print("Loading raw PBMC3k...")
    adata = sc.datasets.pbmc3k()

    metadata = {
        'raw': {
            'n_obs': int(adata.n_obs),
            'n_vars': int(adata.n_vars)
        }
    }

    # ===================
    # QC step
    # ===================
    print("Performing QC filtering...")
    adata_qc = adata.copy()

    # Calculate QC metrics
    adata_qc.var['mt'] = adata_qc.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(
        adata_qc,
        qc_vars=['mt'],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    # Filter cells
    adata_qc = adata_qc[adata_qc.obs.n_genes_by_counts > 500, :]
    adata_qc = adata_qc[adata_qc.obs.pct_counts_mt < 20, :]

    # Filter genes
    sc.pp.filter_genes(adata_qc, min_cells=3)

    # Save QC reference
    adata_qc.write_h5ad(output_dir / 'qc.h5ad')
    metadata['qc'] = {
        'n_obs': int(adata_qc.n_obs),
        'n_vars': int(adata_qc.n_vars),
        'thresholds': {
            'min_genes': 500,
            'max_mt_pct': 20,
            'min_cells_per_gene': 3
        }
    }
    print(f"  QC: {adata_qc.n_obs} cells × {adata_qc.n_vars} genes")

    # ===================
    # Preprocessing
    # ===================
    print("Preprocessing and HVG selection...")
    adata_pp = adata_qc.copy()

    # Normalize
    sc.pp.normalize_total(adata_pp, target_sum=1e4)
    sc.pp.log1p(adata_pp)

    # Store raw counts in layer
    adata_pp.raw = adata_pp.copy()

    # HVG selection
    sc.pp.highly_variable_genes(
        adata_pp,
        n_top_genes=2000,
        flavor='seurat'
    )

    # Filter to HVGs
    adata_pp = adata_pp[:, adata_pp.var.highly_variable]

    # Save preprocessing reference
    adata_pp.write_h5ad(output_dir / 'preprocessed.h5ad')
    metadata['preprocessed'] = {
        'n_obs': int(adata_pp.n_obs),
        'n_vars': int(adata_pp.n_vars),
        'n_hvg': 2000
    }
    print(f"  Preprocessed: {adata_pp.n_obs} cells × {adata_pp.n_vars} HVGs")

    # ===================
    # Clustering
    # ===================
    print("Clustering and dimensionality reduction...")
    adata_clust = adata_pp.copy()

    # PCA
    sc.pp.scale(adata_clust, max_value=10)
    sc.tl.pca(adata_clust, n_comps=50, svd_solver='arpack')

    # Neighbors and clustering
    sc.pp.neighbors(adata_clust, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(adata_clust, resolution=1.0, random_state=0)

    # UMAP
    sc.tl.umap(adata_clust, random_state=0)

    # Save clustering reference
    adata_clust.write_h5ad(output_dir / 'clustered.h5ad')
    metadata['clustered'] = {
        'n_obs': int(adata_clust.n_obs),
        'n_vars': int(adata_clust.n_vars),
        'n_clusters': int(adata_clust.obs['leiden'].nunique()),
        'pca_components': 50,
        'leiden_resolution': 1.0
    }
    print(f"  Clustered: {adata_clust.obs['leiden'].nunique()} clusters")

    # ===================
    # Save metadata
    # ===================
    with open(output_dir / 'reference_metrics.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Reference data saved to {output_dir}")
    print(f"   - qc.h5ad")
    print(f"   - preprocessed.h5ad")
    print(f"   - clustered.h5ad")
    print(f"   - reference_metrics.json")

    return metadata


def generate_bulk_deg_reference(
    output_dir: Path,
    counts_file: Optional[Path] = None,
    force: bool = False
):
    """
    Generate reference DEG data for bulk RNA-seq tests.

    Args:
        output_dir: Directory to save reference files
        counts_file: Path to count matrix (if None, uses mock data)
        force: Overwrite existing references

    Returns:
        dict: Metadata about generated references
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    deg_file = output_dir / 'reference_deg.csv'
    if deg_file.exists() and not force:
        print(f"Bulk DEG references already exist. Use force=True to regenerate.")
        return {'status': 'exists'}

    print("Generating bulk DEG reference data...")

    if counts_file and counts_file.exists():
        # Load user-provided data
        print(f"Loading counts from {counts_file}")
        # Implementation depends on file format
        pass
    else:
        # Generate mock DEG data for testing
        print("Generating mock DEG data...")
        n_genes = 1000
        genes = [f"GENE_{i}" for i in range(n_genes)]

        # Simulate DEG results
        np.random.seed(42)
        deg_data = {
            'gene': genes,
            'log2FC': np.random.randn(n_genes) * 2,
            'pvalue': np.random.random(n_genes),
            'qvalue': np.random.random(n_genes),
            'BaseMean': np.random.exponential(100, n_genes)
        }

        import pandas as pd
        deg_df = pd.DataFrame(deg_data)

        # Add significance flag
        deg_df['significant'] = deg_df['qvalue'] < 0.05

        # Save
        deg_df.to_csv(output_dir / 'reference_deg.csv', index=False)

        metadata = {
            'n_genes': n_genes,
            'n_significant': int(deg_df['significant'].sum()),
            'p_threshold': 0.05
        }

        with open(output_dir / 'reference_metrics.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"✅ Bulk DEG references saved to {output_dir}")
    return metadata


def generate_all_references(base_dir: Path = None, force: bool = False):
    """
    Generate all reference data for integration tests.

    Args:
        base_dir: Base directory (defaults to tests/integration/agent/data/)
        force: Overwrite existing references

    Returns:
        dict: Summary of generated references
    """
    if base_dir is None:
        # Assume running from project root
        base_dir = Path(__file__).parents[1] / 'data'

    print("="*70)
    print("Generating Reference Data for Agent Integration Tests")
    print("="*70)

    results = {}

    # PBMC3k
    print("\n[1/2] PBMC3k single-cell workflow...")
    pbmc_dir = base_dir / 'pbmc3k'
    try:
        results['pbmc3k'] = generate_pbmc3k_references(pbmc_dir, force=force)
        results['pbmc3k']['status'] = 'success'
    except Exception as e:
        print(f"❌ Failed to generate PBMC3k references: {e}")
        results['pbmc3k'] = {'status': 'failed', 'error': str(e)}

    # Bulk DEG
    print("\n[2/2] Bulk RNA-seq DEG workflow...")
    bulk_dir = base_dir / 'bulk_deg'
    try:
        results['bulk_deg'] = generate_bulk_deg_reference(bulk_dir, force=force)
        results['bulk_deg']['status'] = 'success'
    except Exception as e:
        print(f"❌ Failed to generate bulk DEG references: {e}")
        results['bulk_deg'] = {'status': 'failed', 'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("Reference Data Generation Complete")
    print("="*70)

    success_count = sum(
        1 for r in results.values()
        if r.get('status') in ['success', 'exists']
    )
    print(f"✅ {success_count}/{len(results)} datasets generated successfully")

    return results


if __name__ == '__main__':
    """
    Run as script to generate reference data:

    python -m tests.integration.agent.utils.data_generators
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate reference data for agent integration tests"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory (default: tests/integration/agent/data/)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing references'
    )

    args = parser.parse_args()

    results = generate_all_references(
        base_dir=args.output,
        force=args.force
    )

    # Exit with error if any failed
    if any(r.get('status') == 'failed' for r in results.values()):
        exit(1)
