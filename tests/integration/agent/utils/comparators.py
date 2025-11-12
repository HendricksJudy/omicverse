"""
Comparison utilities for agent outputs vs references.

Provides functions to compare complex outputs like:
- AnnData objects
- Clustering results
- DEG tables
- Gene sets
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


def compare_adata_structures(
    output,
    reference,
    check_X: bool = False,
    check_obs: List[str] = None,
    check_var: List[str] = None,
    check_obsm: List[str] = None,
    check_varm: List[str] = None,
    check_uns: List[str] = None
) -> Dict[str, bool]:
    """
    Compare structural elements of two AnnData objects.

    Args:
        output: Output AnnData
        reference: Reference AnnData
        check_X: Compare expression matrices (expensive)
        check_obs: List of .obs columns to compare
        check_var: List of .var columns to compare
        check_obsm: List of .obsm keys to compare
        check_varm: List of .varm keys to compare
        check_uns: List of .uns keys to compare

    Returns:
        dict: Comparison results (True = match)
    """
    results = {}

    # Shape comparison
    results['n_obs_match'] = output.n_obs == reference.n_obs
    results['n_vars_match'] = output.n_vars == reference.n_vars

    # Expression matrix
    if check_X:
        X_out = output.X.toarray() if hasattr(output.X, 'toarray') else output.X
        X_ref = reference.X.toarray() if hasattr(reference.X, 'toarray') else reference.X
        results['X_match'] = np.allclose(X_out, X_ref, rtol=1e-5)

    # .obs columns
    if check_obs:
        for col in check_obs:
            key = f'obs_{col}'
            if col in output.obs and col in reference.obs:
                results[key] = output.obs[col].equals(reference.obs[col])
            else:
                results[key] = False

    # .var columns
    if check_var:
        for col in check_var:
            key = f'var_{col}'
            if col in output.var and col in reference.var:
                results[key] = output.var[col].equals(reference.var[col])
            else:
                results[key] = False

    # .obsm keys
    if check_obsm:
        for key_name in check_obsm:
            key = f'obsm_{key_name}'
            if key_name in output.obsm and key_name in reference.obsm:
                results[key] = np.allclose(
                    output.obsm[key_name],
                    reference.obsm[key_name],
                    rtol=1e-3,
                    equal_nan=True
                )
            else:
                results[key] = False

    # .varm keys
    if check_varm:
        for key_name in check_varm:
            key = f'varm_{key_name}'
            if key_name in output.varm and key_name in reference.varm:
                results[key] = np.allclose(
                    output.varm[key_name],
                    reference.varm[key_name],
                    rtol=1e-3
                )
            else:
                results[key] = False

    # .uns keys
    if check_uns:
        for key_name in check_uns:
            key = f'uns_{key_name}'
            results[key] = (
                key_name in output.uns and
                key_name in reference.uns
            )

    return results


def compute_clustering_similarity(
    labels1: np.ndarray,
    labels2: np.ndarray,
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute clustering similarity metrics.

    Args:
        labels1: First clustering labels
        labels2: Second clustering labels
        metrics: Which metrics to compute
                 Options: ['ari', 'ami', 'nmi', 'fowlkes_mallows', 'silhouette']

    Returns:
        dict: Computed metrics
    """
    if metrics is None:
        metrics = ['ari']

    results = {}

    try:
        from sklearn import metrics as sk_metrics

        if 'ari' in metrics:
            results['ari'] = sk_metrics.adjusted_rand_score(labels1, labels2)

        if 'ami' in metrics:
            results['ami'] = sk_metrics.adjusted_mutual_info_score(
                labels1, labels2
            )

        if 'nmi' in metrics:
            results['nmi'] = sk_metrics.normalized_mutual_info_score(
                labels1, labels2
            )

        if 'fowlkes_mallows' in metrics:
            results['fowlkes_mallows'] = sk_metrics.fowlkes_mallows_score(
                labels1, labels2
            )

    except ImportError:
        results['error'] = 'sklearn not available'

    return results


def compare_gene_sets(
    genes1: List[str],
    genes2: List[str],
    top_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare two gene sets/lists.

    Args:
        genes1: First gene list
        genes2: Second gene list
        top_n: Only compare top N genes from each list

    Returns:
        dict: Comparison metrics
    """
    if top_n:
        genes1 = genes1[:top_n]
        genes2 = genes2[:top_n]

    set1 = set(genes1)
    set2 = set(genes2)

    intersection = set1 & set2
    union = set1 | set2

    results = {
        'n_genes1': len(set1),
        'n_genes2': len(set2),
        'n_common': len(intersection),
        'n_unique1': len(set1 - set2),
        'n_unique2': len(set2 - set1),
        'jaccard': len(intersection) / len(union) if union else 0,
        'overlap_pct': len(intersection) / min(len(set1), len(set2))
                       if set1 and set2 else 0
    }

    # Order-aware comparison (for ranked lists)
    if top_n:
        # Spearman correlation of ranks
        try:
            from scipy.stats import spearmanr

            # Create rank dictionaries
            rank1 = {g: i for i, g in enumerate(genes1)}
            rank2 = {g: i for i, g in enumerate(genes2)}

            # Get common genes' ranks
            common = list(intersection)
            if len(common) > 1:
                ranks1 = [rank1[g] for g in common]
                ranks2 = [rank2[g] for g in common]
                rho, pval = spearmanr(ranks1, ranks2)
                results['spearman_rho'] = rho
                results['spearman_pval'] = pval

        except ImportError:
            pass

    return results


def compare_deg_results(
    deg1: pd.DataFrame,
    deg2: pd.DataFrame,
    gene_col: str = 'gene',
    fc_col: str = 'log2FC',
    pval_col: str = 'qvalue',
    sig_threshold: float = 0.05,
    top_n: int = 100
) -> Dict[str, Any]:
    """
    Compare two DEG result tables.

    Args:
        deg1: First DEG table
        deg2: Second DEG table (reference)
        gene_col: Gene identifier column
        fc_col: Fold change column
        pval_col: Adjusted p-value column
        sig_threshold: Significance threshold
        top_n: Number of top genes to compare

    Returns:
        dict: Comparison metrics
    """
    results = {}

    # Significant genes
    sig1 = set(deg1[deg1[pval_col] < sig_threshold][gene_col])
    sig2 = set(deg2[deg2[pval_col] < sig_threshold][gene_col])

    results['n_significant1'] = len(sig1)
    results['n_significant2'] = len(sig2)
    results['n_common_sig'] = len(sig1 & sig2)
    results['sig_jaccard'] = (
        len(sig1 & sig2) / len(sig1 | sig2) if (sig1 | sig2) else 0
    )

    # Top N by absolute fold change
    top_genes1 = set(
        deg1.nlargest(top_n, fc_col, keep='all')[gene_col]
    )
    top_genes2 = set(
        deg2.nlargest(top_n, fc_col, keep='all')[gene_col]
    )

    results['top_n'] = top_n
    results['n_common_top'] = len(top_genes1 & top_genes2)
    results['top_jaccard'] = (
        len(top_genes1 & top_genes2) / len(top_genes1 | top_genes2)
    )
    results['top_overlap_pct'] = len(top_genes1 & top_genes2) / top_n

    # Fold change correlation for common genes
    common_genes = set(deg1[gene_col]) & set(deg2[gene_col])
    if len(common_genes) > 10:
        try:
            from scipy.stats import pearsonr, spearmanr

            deg1_sub = deg1[deg1[gene_col].isin(common_genes)].set_index(gene_col)
            deg2_sub = deg2[deg2[gene_col].isin(common_genes)].set_index(gene_col)

            # Align by gene
            common_sorted = sorted(common_genes)
            fc1 = deg1_sub.loc[common_sorted, fc_col].values
            fc2 = deg2_sub.loc[common_sorted, fc_col].values

            pearson_r, pearson_p = pearsonr(fc1, fc2)
            spearman_r, spearman_p = spearmanr(fc1, fc2)

            results['fc_pearson_r'] = pearson_r
            results['fc_pearson_p'] = pearson_p
            results['fc_spearman_r'] = spearman_r
            results['fc_spearman_p'] = spearman_p

        except ImportError:
            pass

    return results


def compare_embeddings(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    labels1: Optional[np.ndarray] = None,
    labels2: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compare two low-dimensional embeddings (e.g., UMAP, t-SNE).

    Args:
        embedding1: First embedding (n_cells, n_dims)
        embedding2: Second embedding (n_cells, n_dims)
        labels1: Optional cluster labels for first embedding
        labels2: Optional cluster labels for second embedding

    Returns:
        dict: Comparison metrics
    """
    results = {}

    # Shape check
    assert embedding1.shape == embedding2.shape, \
        f"Shape mismatch: {embedding1.shape} vs {embedding2.shape}"

    # Procrustes analysis (align embeddings, then measure distance)
    try:
        from scipy.spatial import procrustes

        mtx1, mtx2, disparity = procrustes(embedding1, embedding2)
        results['procrustes_disparity'] = disparity

    except ImportError:
        pass

    # If labels provided, compare neighborhood preservation
    if labels1 is not None and labels2 is not None:
        # Compute silhouette scores for both
        try:
            from sklearn.metrics import silhouette_score

            sil1 = silhouette_score(embedding1, labels1, metric='euclidean')
            sil2 = silhouette_score(embedding2, labels2, metric='euclidean')

            results['silhouette1'] = sil1
            results['silhouette2'] = sil2
            results['silhouette_diff'] = abs(sil1 - sil2)

        except ImportError:
            pass

    # Distance correlation
    try:
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import pearsonr

        # Compute pairwise distances
        dist1 = pdist(embedding1)
        dist2 = pdist(embedding2)

        # Correlate distance matrices
        r, p = pearsonr(dist1, dist2)
        results['distance_correlation'] = r
        results['distance_correlation_p'] = p

    except ImportError:
        pass

    return results


def summarize_comparison(
    comparison_results: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[bool, str]:
    """
    Summarize comparison results and determine pass/fail.

    Args:
        comparison_results: Dict of metric name -> value
        thresholds: Dict of metric name -> minimum acceptable value

    Returns:
        tuple: (passed: bool, summary: str)
    """
    if thresholds is None:
        thresholds = {
            'ari': 0.85,
            'jaccard': 0.75,
            'overlap_pct': 0.80,
            'procrustes_disparity': 0.15,  # lower is better
        }

    passed = True
    issues = []

    for metric, threshold in thresholds.items():
        if metric in comparison_results:
            value = comparison_results[metric]

            # Lower is better for disparity metrics
            if 'disparity' in metric or 'diff' in metric:
                if value > threshold:
                    passed = False
                    issues.append(
                        f"{metric}={value:.3f} > {threshold} (threshold)"
                    )
            else:
                # Higher is better for similarity metrics
                if value < threshold:
                    passed = False
                    issues.append(
                        f"{metric}={value:.3f} < {threshold} (threshold)"
                    )

    if passed:
        summary = "✅ All metrics passed thresholds"
    else:
        summary = "❌ Failed metrics:\n  " + "\n  ".join(issues)

    return passed, summary


if __name__ == '__main__':
    """
    Test comparator functions with synthetic data.
    """
    import numpy as np

    print("Testing comparator functions...")

    # Test clustering similarity
    labels1 = np.array([0, 0, 1, 1, 2, 2])
    labels2 = np.array([0, 0, 1, 1, 2, 2])  # Perfect match
    labels3 = np.array([1, 1, 0, 0, 2, 2])  # Relabeled but same structure

    sim_perfect = compute_clustering_similarity(labels1, labels2)
    sim_relabeled = compute_clustering_similarity(labels1, labels3)

    print(f"Perfect match ARI: {sim_perfect.get('ari', 'N/A')}")
    print(f"Relabeled ARI: {sim_relabeled.get('ari', 'N/A')}")

    # Test gene set comparison
    genes1 = ['GeneA', 'GeneB', 'GeneC', 'GeneD']
    genes2 = ['GeneB', 'GeneC', 'GeneD', 'GeneE']

    gene_comp = compare_gene_sets(genes1, genes2)
    print(f"\nGene set Jaccard: {gene_comp['jaccard']:.3f}")
    print(f"Common genes: {gene_comp['n_common']}")

    print("\n✅ Comparator tests complete")
