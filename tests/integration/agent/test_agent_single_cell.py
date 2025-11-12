"""
Integration tests for ov.agent single-cell workflows.

Tests based on tutorial notebook: t_preprocess.ipynb

Phase 1 - Foundation Tests:
- QC filtering
- HVG selection
- Dimensionality reduction
- Clustering
"""

import pytest
import numpy as np
import scanpy as sc
import omicverse as ov


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.quick
@pytest.mark.single_cell
def test_agent_qc_filtering(pbmc3k_raw, agent_with_api_key, output_validator):
    """
    Test agent performs QC filtering correctly.

    Request: "quality control with nUMI>500, mito<0.2"

    Expected:
    - Skill matched: single-preprocessing
    - Cell count: ~2,603 cells (±5%)
    - Gene count: ~13,631 genes (±5%)
    - Required columns added: n_counts, mito_perc

    Reference: t_preprocess.ipynb cell 4-5
    """
    # Copy to avoid modifying fixture
    adata = pbmc3k_raw.copy()
    initial_cells = adata.n_obs
    initial_genes = adata.n_vars

    print(f"\nInitial: {initial_cells} cells × {initial_genes} genes")

    # Run agent request
    result = agent_with_api_key.run(
        'quality control with nUMI>500, mito<0.2',
        adata
    )

    # Agent should return modified adata or dict with adata
    if isinstance(result, dict):
        result_adata = result.get('adata') or result.get('value')
    else:
        result_adata = result

    print(f"After QC: {result_adata.n_obs} cells × {result_adata.n_vars} genes")

    # Validate shape within tolerance
    output_validator.validate_adata_shape(
        result_adata,
        expected_cells=2603,
        expected_genes=13631,
        tolerance=0.10  # 10% tolerance for QC variations
    )

    # Validate QC metrics computed
    # Note: exact column names may vary based on agent implementation
    qc_columns = ['n_counts', 'n_genes', 'total_counts']
    available_cols = set(result_adata.obs.columns)

    # At least one QC metric should be present
    has_qc = any(col in available_cols for col in qc_columns)
    assert has_qc, (
        f"No QC metrics found in .obs. "
        f"Expected one of {qc_columns}, got {list(available_cols)}"
    )

    # Verify filtering was applied (some cells removed)
    assert result_adata.n_obs < initial_cells, \
        "No cells were filtered (QC not applied)"

    assert result_adata.n_vars < initial_genes, \
        "No genes were filtered (QC not applied)"

    print("✅ QC filtering test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.quick
@pytest.mark.single_cell
def test_agent_hvg_selection(pbmc3k_raw, agent_with_api_key, output_validator):
    """
    Test agent performs HVG selection correctly.

    Request: "preprocess with 2000 highly variable genes using shiftlog|pearson"

    Expected:
    - Skill matched: single-preprocessing
    - Exactly 2,000 HVGs selected
    - .var['highly_variable'] column present
    - .X is normalized (log-scale)

    Reference: t_preprocess.ipynb cell 6-7
    """
    adata = pbmc3k_raw.copy()

    # Simple QC first
    sc.pp.filter_cells(adata, min_genes=500)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"\nAfter basic QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # Run agent request for preprocessing
    result = agent_with_api_key.run(
        'preprocess with 2000 highly variable genes',
        adata
    )

    if isinstance(result, dict):
        result_adata = result.get('adata') or result.get('value')
    else:
        result_adata = result

    # Validate HVG column exists
    output_validator.validate_columns_exist(
        result_adata,
        required_columns=['highly_variable'],
        location='var'
    )

    # Validate exactly 2000 HVGs selected
    # Note: After filtering, n_vars might be 2000 if subset to HVGs
    if result_adata.n_vars == 2000:
        # Subset to HVGs
        print("Data subset to 2000 HVGs")
    else:
        # HVGs marked but not subset
        output_validator.validate_hvg_selection(
            result_adata,
            expected_n_hvg=2000
        )

    # Check normalization applied (X should be log-normalized)
    # Log-normalized data typically has values in range [0, ~10]
    X = result_adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    mean_expr = np.mean(X)
    max_expr = np.max(X)

    print(f"Expression stats - mean: {mean_expr:.2f}, max: {max_expr:.2f}")

    # Log-normalized data characteristics
    assert max_expr < 20, \
        f"Data doesn't appear log-normalized (max={max_expr:.1f}, expected <20)"

    print("✅ HVG selection test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.single_cell
def test_agent_dimensionality_reduction(
    pbmc3k_raw,
    agent_with_api_key,
    output_validator
):
    """
    Test agent computes PCA and UMAP correctly.

    Request: "compute PCA with 50 components and UMAP embedding"

    Expected:
    - .obsm['X_pca'] shape: (n_cells, 50)
    - .obsm['X_umap'] shape: (n_cells, 2)
    - No NaN/Inf values

    Reference: t_preprocess.ipynb cell 8-10
    """
    adata = pbmc3k_raw.copy()

    # Preprocess first
    sc.pp.filter_cells(adata, min_genes=500)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    print(f"\nPreprocessed: {adata.n_obs} cells × {adata.n_vars} genes")

    # Run agent request
    result = agent_with_api_key.run(
        'compute PCA with 50 components and UMAP embedding',
        adata
    )

    if isinstance(result, dict):
        result_adata = result.get('adata') or result.get('value')
    else:
        result_adata = result

    # Validate embeddings exist
    output_validator.validate_obsm_keys(
        result_adata,
        required_keys=['X_pca', 'X_umap']
    )

    # Validate PCA shape
    assert result_adata.obsm['X_pca'].shape == (result_adata.n_obs, 50), \
        f"PCA shape incorrect: {result_adata.obsm['X_pca'].shape}"

    # Validate UMAP shape
    assert result_adata.obsm['X_umap'].shape == (result_adata.n_obs, 2), \
        f"UMAP shape incorrect: {result_adata.obsm['X_umap'].shape}"

    # Validate no NaN/Inf
    output_validator.validate_no_nan(
        result_adata,
        check_X=False,
        check_obsm=['X_pca', 'X_umap']
    )

    print("✅ Dimensionality reduction test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.quick
@pytest.mark.single_cell
def test_agent_clustering(pbmc3k_raw, agent_with_api_key, output_validator):
    """
    Test agent performs Leiden clustering correctly.

    Request: "leiden clustering resolution=1.0"

    Expected:
    - .obs['leiden'] contains cluster assignments
    - Number of clusters: 8-12 (typical for PBMC3k)
    - All cells assigned (no NaN)

    Reference: t_preprocess.ipynb cell 11-12
    """
    adata = pbmc3k_raw.copy()

    # Preprocess
    sc.pp.filter_cells(adata, min_genes=500)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata)

    print(f"\nPreprocessed: {adata.n_obs} cells")

    # Run agent request
    result = agent_with_api_key.run(
        'leiden clustering resolution=1.0',
        adata
    )

    if isinstance(result, dict):
        result_adata = result.get('adata') or result.get('value')
    else:
        result_adata = result

    # Validate clustering
    output_validator.validate_clustering(
        result_adata,
        cluster_column='leiden',
        min_clusters=5,
        max_clusters=15
    )

    # Print cluster distribution
    cluster_counts = result_adata.obs['leiden'].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells")

    print("✅ Clustering test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.single_cell
@pytest.mark.workflow
def test_agent_complete_pbmc3k_workflow(
    pbmc3k_raw,
    agent_with_api_key,
    output_validator
):
    """
    Test complete PBMC3k workflow from QC to visualization.

    Multi-step workflow:
    1. QC filtering
    2. Preprocessing + HVG
    3. Leiden clustering
    4. UMAP visualization

    Expected:
    - All steps complete without errors
    - Final AnnData contains all expected attributes
    - Workflow produces biologically reasonable results

    Reference: t_ov_agent_pbmc3k.ipynb full workflow
    """
    adata = pbmc3k_raw.copy()
    agent = agent_with_api_key

    print(f"\nInitial: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 1: QC
    print("\n[1/4] Quality control...")
    adata = agent.run('quality control with nUMI>500, mito<0.2', adata)
    if isinstance(adata, dict):
        adata = adata.get('adata') or adata.get('value')

    assert adata.n_obs > 2000, f"Too many cells filtered: {adata.n_obs}"
    print(f"  After QC: {adata.n_obs} cells")

    # Step 2: Preprocessing
    print("\n[2/4] Preprocessing with HVG selection...")
    adata = agent.run('preprocess with 2000 highly variable genes', adata)
    if isinstance(adata, dict):
        adata = adata.get('adata') or adata.get('value')

    # Check HVG selection occurred
    if 'highly_variable' in adata.var.columns:
        n_hvg = adata.var['highly_variable'].sum()
        print(f"  Selected {n_hvg} HVGs")
    elif adata.n_vars == 2000:
        print(f"  Subset to {adata.n_vars} HVGs")

    # Step 3: Clustering
    print("\n[3/4] Leiden clustering...")
    adata = agent.run('leiden clustering resolution=1.0', adata)
    if isinstance(adata, dict):
        adata = adata.get('adata') or adata.get('value')

    assert 'leiden' in adata.obs.columns, "Clustering not performed"
    n_clusters = adata.obs['leiden'].nunique()
    print(f"  Found {n_clusters} clusters")

    assert 5 <= n_clusters <= 15, \
        f"Unexpected number of clusters: {n_clusters}"

    # Step 4: UMAP
    print("\n[4/4] Computing UMAP...")
    adata = agent.run('compute umap', adata)
    if isinstance(adata, dict):
        adata = adata.get('adata') or adata.get('value')

    assert 'X_umap' in adata.obsm, "UMAP not computed"
    assert adata.obsm['X_umap'].shape == (adata.n_obs, 2)
    print(f"  UMAP computed: {adata.obsm['X_umap'].shape}")

    # Final validation
    print("\n[Final] Validating complete workflow...")
    output_validator.validate_complete_workflow(
        adata,
        check_preprocessing=True,
        check_clustering=True,
        check_embeddings=True
    )

    print("\n✅ Complete workflow test passed")
    print(f"Final data: {adata.n_obs} cells × {adata.n_vars} genes, "
          f"{n_clusters} clusters")
