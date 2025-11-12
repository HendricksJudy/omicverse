"""
Output validation utilities for agent integration tests.

Provides classes and functions to validate agent outputs against:
- Expected data shapes and structures
- Reference datasets
- Statistical thresholds
- File outputs
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from pathlib import Path


class OutputValidator:
    """Validate agent outputs match reference expectations."""

    def __init__(self, verbose: bool = False):
        """
        Initialize validator.

        Args:
            verbose: If True, print detailed validation messages
        """
        self.verbose = verbose

    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[Validator] {message}")

    def validate_adata_shape(
        self,
        adata,
        expected_cells: int,
        expected_genes: int,
        tolerance: float = 0.05
    ):
        """
        Check AnnData dimensions within tolerance.

        Args:
            adata: AnnData object to validate
            expected_cells: Expected number of cells
            expected_genes: Expected number of genes
            tolerance: Allowed relative error (default 5%)

        Raises:
            AssertionError: If dimensions outside tolerance
        """
        cell_error = abs(adata.n_obs - expected_cells) / expected_cells
        gene_error = abs(adata.n_vars - expected_genes) / expected_genes

        self._log(f"Cell count: {adata.n_obs} vs {expected_cells} "
                 f"(error: {cell_error*100:.2f}%)")
        self._log(f"Gene count: {adata.n_vars} vs {expected_genes} "
                 f"(error: {gene_error*100:.2f}%)")

        assert cell_error <= tolerance, (
            f"Cell count off by {cell_error*100:.1f}%: "
            f"{adata.n_obs} vs {expected_cells} expected"
        )
        assert gene_error <= tolerance, (
            f"Gene count off by {gene_error*100:.1f}%: "
            f"{adata.n_vars} vs {expected_genes} expected"
        )

    def validate_columns_exist(
        self,
        adata,
        required_columns: List[str],
        location: str = 'obs'
    ):
        """
        Check required columns present in .obs or .var.

        Args:
            adata: AnnData object
            required_columns: List of column names to check
            location: 'obs' or 'var'

        Raises:
            AssertionError: If any required columns missing
        """
        df = getattr(adata, location)
        missing = set(required_columns) - set(df.columns)

        if missing:
            available = list(df.columns)
            raise AssertionError(
                f"Missing columns in .{location}: {missing}\n"
                f"Available columns: {available}"
            )

        self._log(f"All required columns present in .{location}: "
                 f"{required_columns}")

    def validate_obsm_keys(
        self,
        adata,
        required_keys: List[str]
    ):
        """
        Check required embedding keys present in .obsm.

        Args:
            adata: AnnData object
            required_keys: List of .obsm keys to check

        Raises:
            AssertionError: If any required keys missing
        """
        missing = set(required_keys) - set(adata.obsm.keys())

        if missing:
            available = list(adata.obsm.keys())
            raise AssertionError(
                f"Missing .obsm keys: {missing}\n"
                f"Available keys: {available}"
            )

        # Check shapes
        for key in required_keys:
            arr = adata.obsm[key]
            assert arr.shape[0] == adata.n_obs, (
                f".obsm['{key}'] has wrong number of rows: "
                f"{arr.shape[0]} vs {adata.n_obs} cells"
            )

        self._log(f"All required .obsm keys present: {required_keys}")

    def validate_clustering(
        self,
        adata,
        cluster_column: str = 'leiden',
        min_clusters: int = 1,
        max_clusters: int = 100
    ):
        """
        Validate clustering results.

        Args:
            adata: AnnData object with clustering
            cluster_column: Name of cluster column
            min_clusters: Minimum expected clusters
            max_clusters: Maximum expected clusters

        Raises:
            AssertionError: If clustering invalid
        """
        assert cluster_column in adata.obs.columns, (
            f"Cluster column '{cluster_column}' not found in .obs\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

        n_clusters = adata.obs[cluster_column].nunique()
        self._log(f"Found {n_clusters} clusters in '{cluster_column}'")

        assert min_clusters <= n_clusters <= max_clusters, (
            f"Expected {min_clusters}-{max_clusters} clusters, "
            f"got {n_clusters}"
        )

        # Check no NaN assignments
        n_nan = adata.obs[cluster_column].isna().sum()
        assert n_nan == 0, (
            f"{n_nan} cells not assigned to clusters (NaN values)"
        )

    def validate_deg_table(
        self,
        deg_df: pd.DataFrame,
        required_columns: List[str],
        min_significant: Optional[int] = None,
        p_column: str = 'qvalue',
        p_threshold: float = 0.05
    ):
        """
        Validate DEG results table.

        Args:
            deg_df: DEG results DataFrame
            required_columns: Expected columns
            min_significant: Minimum number of significant genes
            p_column: Column name for p-values
            p_threshold: Significance threshold

        Raises:
            AssertionError: If DEG table invalid
        """
        # Check columns
        missing = set(required_columns) - set(deg_df.columns)
        if missing:
            raise AssertionError(
                f"Missing DEG columns: {missing}\n"
                f"Available: {list(deg_df.columns)}"
            )

        self._log(f"DEG table has {len(deg_df)} genes")

        # Check significant genes
        if min_significant is not None and p_column in deg_df.columns:
            n_sig = (deg_df[p_column] < p_threshold).sum()
            self._log(f"Found {n_sig} significant genes "
                     f"({p_column} < {p_threshold})")

            assert n_sig >= min_significant, (
                f"Expected ≥{min_significant} significant genes, "
                f"got {n_sig}"
            )

    def validate_hvg_selection(
        self,
        adata,
        expected_n_hvg: int,
        hvg_column: str = 'highly_variable'
    ):
        """
        Validate highly variable gene selection.

        Args:
            adata: AnnData object
            expected_n_hvg: Expected number of HVGs
            hvg_column: Column name in .var

        Raises:
            AssertionError: If HVG selection invalid
        """
        assert hvg_column in adata.var.columns, (
            f"HVG column '{hvg_column}' not found in .var"
        )

        n_hvg = adata.var[hvg_column].sum()
        self._log(f"Found {n_hvg} highly variable genes")

        assert n_hvg == expected_n_hvg, (
            f"Expected {expected_n_hvg} HVGs, got {n_hvg}"
        )

    def validate_no_nan(
        self,
        adata,
        check_X: bool = True,
        check_obs: List[str] = None,
        check_obsm: List[str] = None
    ):
        """
        Validate no NaN values in critical locations.

        Args:
            adata: AnnData object
            check_X: Check main expression matrix
            check_obs: List of .obs columns to check
            check_obsm: List of .obsm keys to check

        Raises:
            AssertionError: If NaN values found
        """
        if check_X:
            # Handle sparse matrices
            if hasattr(adata.X, 'toarray'):
                X = adata.X.toarray()
            else:
                X = adata.X

            n_nan = np.isnan(X).sum()
            assert n_nan == 0, f"Found {n_nan} NaN values in .X"
            self._log(".X has no NaN values")

        if check_obs:
            for col in check_obs:
                n_nan = adata.obs[col].isna().sum()
                assert n_nan == 0, (
                    f"Found {n_nan} NaN values in .obs['{col}']"
                )
            self._log(f"No NaN in .obs columns: {check_obs}")

        if check_obsm:
            for key in check_obsm:
                n_nan = np.isnan(adata.obsm[key]).sum()
                assert n_nan == 0, (
                    f"Found {n_nan} NaN values in .obsm['{key}']"
                )
            self._log(f"No NaN in .obsm keys: {check_obsm}")

    def compare_to_reference(
        self,
        output_adata,
        reference_adata,
        compare_clustering: bool = True,
        cluster_column: str = 'leiden',
        min_ari: float = 0.85
    ) -> Dict[str, Any]:
        """
        Compare output to reference, compute similarity metrics.

        Args:
            output_adata: Agent output
            reference_adata: Reference data
            compare_clustering: Compute clustering similarity (ARI)
            cluster_column: Cluster column name
            min_ari: Minimum acceptable ARI score

        Returns:
            dict: Similarity metrics

        Raises:
            AssertionError: If similarity below thresholds
        """
        results = {}

        # Shape comparison
        results['cell_count_match'] = (
            output_adata.n_obs == reference_adata.n_obs
        )
        results['gene_count_match'] = (
            output_adata.n_vars == reference_adata.n_vars
        )

        self._log(f"Shape match - Cells: {results['cell_count_match']}, "
                 f"Genes: {results['gene_count_match']}")

        # Clustering similarity
        if compare_clustering:
            if (cluster_column in output_adata.obs and
                cluster_column in reference_adata.obs):

                try:
                    from sklearn.metrics import adjusted_rand_score
                    ari = adjusted_rand_score(
                        output_adata.obs[cluster_column],
                        reference_adata.obs[cluster_column]
                    )
                    results['clustering_ari'] = ari
                    self._log(f"Clustering ARI: {ari:.3f}")

                    assert ari >= min_ari, (
                        f"Clustering similarity too low: "
                        f"ARI={ari:.3f} < {min_ari}"
                    )
                except ImportError:
                    self._log("sklearn not available, skipping ARI")

        return results

    def validate_complete_workflow(
        self,
        adata,
        check_preprocessing: bool = True,
        check_clustering: bool = True,
        check_embeddings: bool = True
    ):
        """
        Validate a complete workflow produced expected outputs.

        Args:
            adata: Final AnnData from workflow
            check_preprocessing: Validate preprocessing outputs
            check_clustering: Validate clustering
            check_embeddings: Validate dimensionality reduction

        Raises:
            AssertionError: If workflow incomplete
        """
        # Check basic structure
        assert adata.n_obs > 0, "Empty AnnData (0 cells)"
        assert adata.n_vars > 0, "No genes in AnnData"

        self._log(f"Complete workflow validation - "
                 f"{adata.n_obs} cells × {adata.n_vars} genes")

        # Check preprocessing
        if check_preprocessing:
            expected_obs = ['n_counts']
            expected_var = ['highly_variable']

            for col in expected_obs:
                if col in adata.obs:
                    assert not adata.obs[col].isna().all(), (
                        f"Column '{col}' is all NaN"
                    )

            for col in expected_var:
                if col in adata.var:
                    assert not adata.var[col].isna().all(), (
                        f"Column '{col}' is all NaN"
                    )

            self._log("Preprocessing outputs validated")

        # Check clustering
        if check_clustering:
            cluster_cols = ['leiden', 'louvain', 'clusters']
            found = [c for c in cluster_cols if c in adata.obs]

            assert found, (
                f"No clustering found. Expected one of: {cluster_cols}"
            )

            for col in found:
                self.validate_clustering(adata, col)

        # Check embeddings
        if check_embeddings:
            expected_obsm = ['X_pca', 'X_umap']

            for key in expected_obsm:
                if key in adata.obsm:
                    arr = adata.obsm[key]
                    assert arr.shape[0] == adata.n_obs, (
                        f"Shape mismatch in .obsm['{key}']"
                    )
                    # Check finite values
                    assert np.isfinite(arr).all(), (
                        f".obsm['{key}'] contains NaN or Inf"
                    )

            self._log("Embeddings validated")

    def validate_file_output(
        self,
        file_path: Union[str, Path],
        expected_size_mb: Optional[float] = None,
        check_readable: bool = True
    ):
        """
        Validate file output exists and is valid.

        Args:
            file_path: Path to output file
            expected_size_mb: Expected file size in MB (with tolerance)
            check_readable: Try to read file to validate format

        Raises:
            AssertionError: If file invalid
        """
        file_path = Path(file_path)

        assert file_path.exists(), f"Output file not found: {file_path}"

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        self._log(f"File '{file_path.name}' size: {file_size_mb:.2f} MB")

        if expected_size_mb is not None:
            tolerance = 0.5  # ±50%
            lower = expected_size_mb * (1 - tolerance)
            upper = expected_size_mb * (1 + tolerance)

            assert lower <= file_size_mb <= upper, (
                f"File size outside expected range: "
                f"{file_size_mb:.2f} MB not in "
                f"[{lower:.2f}, {upper:.2f}] MB"
            )

        if check_readable:
            suffix = file_path.suffix.lower()

            try:
                if suffix == '.csv':
                    pd.read_csv(file_path, nrows=5)
                elif suffix in ['.xlsx', '.xls']:
                    pd.read_excel(file_path, nrows=5)
                elif suffix == '.h5ad':
                    import scanpy as sc
                    sc.read_h5ad(file_path)
                elif suffix == '.pdf':
                    # Just check it's not empty
                    assert file_size_mb > 0.001

                self._log(f"File '{file_path.name}' is readable")

            except Exception as e:
                raise AssertionError(
                    f"File '{file_path}' not readable: {e}"
                )


def compare_deg_tables(
    output_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    gene_column: str = 'gene',
    fc_column: str = 'log2FC',
    p_column: str = 'qvalue',
    top_n: int = 100,
    min_overlap: float = 0.80
) -> Dict[str, Any]:
    """
    Compare two DEG tables for consistency.

    Args:
        output_df: Agent-generated DEG table
        reference_df: Reference DEG table
        gene_column: Column with gene names
        fc_column: Fold change column
        p_column: P-value column
        top_n: Number of top genes to compare
        min_overlap: Minimum required overlap (Jaccard)

    Returns:
        dict: Comparison metrics

    Raises:
        AssertionError: If overlap too low
    """
    # Get top N genes by absolute fold change
    output_top = set(
        output_df.nlargest(top_n, fc_column, keep='all')[gene_column]
    )
    ref_top = set(
        reference_df.nlargest(top_n, fc_column, keep='all')[gene_column]
    )

    # Jaccard similarity
    intersection = output_top & ref_top
    union = output_top | ref_top
    jaccard = len(intersection) / len(union) if union else 0

    # Overlap percentage
    overlap_pct = len(intersection) / top_n

    metrics = {
        'jaccard': jaccard,
        'overlap_pct': overlap_pct,
        'n_common': len(intersection),
        'n_output_unique': len(output_top - ref_top),
        'n_ref_unique': len(ref_top - output_top),
    }

    assert overlap_pct >= min_overlap, (
        f"Top {top_n} gene overlap too low: "
        f"{overlap_pct*100:.1f}% < {min_overlap*100:.1f}%\n"
        f"Common genes: {len(intersection)}/{top_n}"
    )

    return metrics
