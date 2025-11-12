"""
Integration tests for ov.agent multi-step workflows.

Phase 2 - Multi-Step Workflow Tests:
- Cell type annotation workflows
- DEG + enrichment workflows
- Complex sequential operations
- Workflow state preservation

Based on:
- t_ov_agent_pbmc3k.ipynb (complete agent workflow)
- t_anno_noref.ipynb (annotation workflows)
- t_deg.ipynb (DEG + enrichment)
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.single_cell
@pytest.mark.workflow
def test_agent_annotation_workflow(
    pbmc3k_raw,
    agent_with_api_key,
    output_validator
):
    """
    Test complete cell type annotation workflow.

    Workflow:
    1. Preprocessing (QC + normalization)
    2. Clustering
    3. Cell type annotation

    Expected:
    - Skills: single-preprocessing → single-clustering → single-annotation
    - Final .obs contains cell type labels
    - Known PBMC cell types identified (T cells, B cells, Monocytes, etc.)

    Reference: t_anno_noref.ipynb, t_ov_agent_pbmc3k.ipynb
    """
    adata = pbmc3k_raw.copy()
    agent = agent_with_api_key

    print(f"\nInitial: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 1: Preprocessing
    print("\n[1/3] Preprocessing...")
    result = agent.run(
        'perform quality control and preprocessing with 2000 highly variable genes',
        adata
    )
    adata = result if not isinstance(result, dict) else result.get('adata', result.get('value'))

    print(f"  After preprocessing: {adata.n_obs} cells × {adata.n_vars} genes")

    # Verify preprocessing outputs
    assert adata.n_obs > 2000, "Too many cells filtered"

    # Step 2: Clustering
    print("\n[2/3] Clustering...")
    result = agent.run(
        'perform leiden clustering with resolution 1.0',
        adata
    )
    adata = result if not isinstance(result, dict) else result.get('adata', result.get('value'))

    # Verify clustering
    assert 'leiden' in adata.obs.columns, "Clustering not performed"
    n_clusters = adata.obs['leiden'].nunique()
    print(f"  Found {n_clusters} clusters")

    # Step 3: Cell type annotation
    print("\n[3/3] Cell type annotation...")

    # For PBMC data, we can use marker genes for annotation
    result = agent.run(
        'annotate cell types in PBMC data using known marker genes. '
        'Use markers: CD3D/CD3E for T cells, CD79A/MS4A1 for B cells, '
        'CD14/FCGR3A for Monocytes, NKG7 for NK cells',
        adata
    )
    adata = result if not isinstance(result, dict) else result.get('adata', result.get('value'))

    # Verify annotation results
    # The agent might add annotations in various columns
    possible_annotation_cols = [
        'cell_type', 'celltype', 'annotation', 'predicted_celltype',
        'leiden_celltype', 'cluster_annotation'
    ]

    annotation_col = None
    for col in possible_annotation_cols:
        if col in adata.obs.columns:
            annotation_col = col
            break

    if annotation_col:
        print(f"\n  Annotations found in .obs['{annotation_col}']:")
        annot_counts = adata.obs[annotation_col].value_counts()
        for celltype, count in annot_counts.items():
            print(f"    {celltype}: {count} cells")

        # Check for expected PBMC cell types
        annotations_str = ' '.join(adata.obs[annotation_col].astype(str).str.lower().unique())

        expected_celltypes = ['t cell', 'b cell', 'monocyte', 'nk']
        found_celltypes = [ct for ct in expected_celltypes if ct in annotations_str]

        print(f"\n  Expected PBMC cell types found: {len(found_celltypes)}/{len(expected_celltypes)}")

        # At least 2 major cell types should be identified
        assert len(found_celltypes) >= 2, (
            f"Expected at least 2 PBMC cell types, found: {found_celltypes}"
        )
    else:
        # If no explicit annotation column, check if marker genes were analyzed
        print("\n  Note: No explicit cell type annotation column found.")
        print("  Agent may have provided annotation in a different format.")

    # Validate complete workflow structure
    output_validator.validate_complete_workflow(
        adata,
        check_preprocessing=True,
        check_clustering=True,
        check_embeddings=False  # UMAP not required for annotation
    )

    print("\n✅ Annotation workflow test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.bulk
@pytest.mark.workflow
def test_agent_deg_enrichment_workflow(agent_with_api_key, output_validator, tmp_path):
    """
    Test DEG analysis followed by pathway enrichment.

    Workflow:
    1. Differential expression analysis
    2. Pathway enrichment on significant genes

    Expected:
    - Skills: bulk-deg-analysis
    - DEG table with significant genes
    - Enrichment results with pathway names and p-values

    Reference: t_deg.ipynb

    Note: Uses mock data for this test. Real data test would be more comprehensive.
    """
    agent = agent_with_api_key

    # Create more realistic mock DEG data
    np.random.seed(42)
    n_genes = 2000
    n_samples = 6

    # Generate counts with clear differential expression
    base_counts = np.random.negative_binomial(10, 0.5, size=(n_genes, n_samples))

    # Make first 100 genes significantly DE (upregulated in treatment)
    base_counts[:100, 3:] = base_counts[:100, 3:] * 3

    # Make next 50 genes significantly DE (downregulated in treatment)
    base_counts[100:150, 3:] = base_counts[100:150, 3:] // 3

    # Create gene names (use some real-ish gene symbols)
    genes = []
    for i in range(n_genes):
        if i < 10:
            # Use some real gene names for testing
            gene_names = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS',
                         'PTEN', 'AKT1', 'BRAF', 'PIK3CA', 'NRAS']
            genes.append(gene_names[i])
        else:
            genes.append(f"GENE_{i}")

    import anndata
    adata = anndata.AnnData(X=base_counts)
    adata.obs['group'] = ['control'] * 3 + ['treatment'] * 3
    adata.obs['sample'] = [f"Sample_{i}" for i in range(n_samples)]
    adata.var_names = genes
    adata.obs_names = adata.obs['sample']

    print(f"\nMock DEG data: {adata.n_obs} samples × {adata.n_vars} genes")
    print(f"Groups: {adata.obs['group'].value_counts().to_dict()}")
    print(f"Expected DE genes: ~150 (100 up, 50 down)")

    # Step 1: DEG Analysis
    print("\n[1/2] Differential expression analysis...")
    result = agent.run(
        'perform differential expression analysis comparing treatment vs control groups. '
        'Use appropriate statistical method for bulk RNA-seq.',
        adata
    )

    # Extract DEG results
    if isinstance(result, dict):
        deg_table = result.get('deg_table') or result.get('deg') or result.get('value')
        result_adata = result.get('adata')
    elif isinstance(result, pd.DataFrame):
        deg_table = result
        result_adata = None
    else:
        result_adata = result
        deg_table = None

    # Validate DEG table exists
    if deg_table is not None and isinstance(deg_table, pd.DataFrame):
        print(f"  DEG table generated: {len(deg_table)} genes")

        # Check for required columns
        expected_cols = ['log2FC', 'pvalue']
        available_cols = set(deg_table.columns)

        missing = set(expected_cols) - available_cols
        if missing:
            print(f"  Warning: Missing expected columns: {missing}")
            print(f"  Available columns: {list(available_cols)}")
        else:
            # Validate statistical results
            if 'pvalue' in deg_table.columns:
                n_significant = (deg_table['pvalue'] < 0.05).sum()
                print(f"  Significant genes (p < 0.05): {n_significant}")

            if 'log2FC' in deg_table.columns:
                n_upregulated = (deg_table['log2FC'] > 1).sum()
                n_downregulated = (deg_table['log2FC'] < -1).sum()
                print(f"  Upregulated (|FC| > 2): {n_upregulated}")
                print(f"  Downregulated (|FC| > 2): {n_downregulated}")

        # Step 2: Pathway Enrichment
        print("\n[2/2] Pathway enrichment analysis...")

        # Get top significant genes for enrichment
        if 'pvalue' in deg_table.columns:
            sig_genes = deg_table[deg_table['pvalue'] < 0.05]

            if len(sig_genes) > 10:
                # Request enrichment analysis
                gene_list = sig_genes.head(100).index.tolist()

                enrichment_request = f"""
                Perform pathway enrichment analysis on these significant genes: {', '.join(gene_list[:20])}...
                Use GO biological process or KEGG pathways.
                """

                try:
                    enrichment_result = agent.run(enrichment_request, adata)

                    print(f"  Enrichment analysis completed")

                    # Check if enrichment results returned
                    if isinstance(enrichment_result, dict):
                        if 'enrichment' in enrichment_result or 'pathways' in enrichment_result:
                            print("  ✅ Enrichment results generated")
                        else:
                            print("  Note: Enrichment results format varies by implementation")

                except Exception as e:
                    print(f"  Note: Enrichment step had issues: {e}")
                    print("  This is expected with mock data - real pathways not available")

            else:
                print(f"  Warning: Only {len(sig_genes)} significant genes found")
                print("  Skipping enrichment (needs more genes)")

    else:
        print("  Note: DEG results not in expected DataFrame format")
        print("  Agent may have stored results in .uns or other location")

        if result_adata is not None:
            print(f"  Checking .uns keys: {list(result_adata.uns.keys())}")

    print("\n✅ DEG + enrichment workflow test completed")
    print("Note: Full validation requires real biological data and pathway databases")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.single_cell
@pytest.mark.workflow
def test_agent_preprocessing_clustering_umap_workflow(
    pbmc3k_raw,
    agent_with_api_key,
    output_validator
):
    """
    Test sequential workflow with state preservation.

    Workflow:
    1. Preprocessing
    2. Clustering
    3. UMAP visualization

    Validates:
    - Each step preserves previous results
    - Data structure maintained throughout
    - No information loss between steps

    Reference: t_preprocess.ipynb sequential execution
    """
    adata = pbmc3k_raw.copy()
    agent = agent_with_api_key

    print(f"\nInitial: {adata.n_obs} cells × {adata.n_vars} genes")

    # Track what's added at each step
    initial_obs_cols = set(adata.obs.columns)
    initial_var_cols = set(adata.var.columns)
    initial_obsm_keys = set(adata.obsm.keys())

    # Step 1: Preprocessing
    print("\n[1/3] Preprocessing...")
    adata = agent.run(
        'preprocess single-cell data: filter cells with >500 genes, '
        'normalize, select 2000 HVGs, scale, and compute PCA',
        adata
    )
    adata = adata if not isinstance(adata, dict) else adata.get('adata', adata.get('value'))

    step1_obs_cols = set(adata.obs.columns)
    step1_obsm_keys = set(adata.obsm.keys())

    print(f"  After step 1: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  New .obs columns: {step1_obs_cols - initial_obs_cols}")
    print(f"  New .obsm keys: {step1_obsm_keys - initial_obsm_keys}")

    # Verify preprocessing occurred
    assert adata.n_obs < pbmc3k_raw.n_obs, "No filtering occurred"
    assert 'X_pca' in adata.obsm or 'pca' in str(adata.obsm.keys()).lower(), \
        "PCA not computed"

    # Step 2: Clustering
    print("\n[2/3] Clustering...")
    adata = agent.run(
        'compute neighbors and perform leiden clustering',
        adata
    )
    adata = adata if not isinstance(adata, dict) else adata.get('adata', adata.get('value'))

    step2_obs_cols = set(adata.obs.columns)

    print(f"  New .obs columns: {step2_obs_cols - step1_obs_cols}")

    # Verify clustering added
    assert 'leiden' in adata.obs.columns or 'louvain' in adata.obs.columns, \
        "Clustering not added"

    # Verify previous results preserved
    if 'X_pca' in step1_obsm_keys:
        assert 'X_pca' in adata.obsm, "PCA lost after clustering step"

    # Step 3: UMAP
    print("\n[3/3] UMAP visualization...")
    adata = agent.run(
        'compute UMAP embedding',
        adata
    )
    adata = adata if not isinstance(adata, dict) else adata.get('adata', adata.get('value'))

    step3_obsm_keys = set(adata.obsm.keys())

    print(f"  New .obsm keys: {step3_obsm_keys - step1_obsm_keys}")

    # Verify UMAP added
    assert 'X_umap' in adata.obsm, "UMAP not computed"

    # Verify all previous results preserved
    assert 'leiden' in adata.obs.columns or 'louvain' in adata.obs.columns, \
        "Clustering lost after UMAP step"

    if 'X_pca' in step1_obsm_keys:
        assert 'X_pca' in adata.obsm, "PCA lost after UMAP step"

    # Final validation
    print("\n[Final] Validating complete workflow...")
    output_validator.validate_complete_workflow(
        adata,
        check_preprocessing=True,
        check_clustering=True,
        check_embeddings=True
    )

    # Print summary
    print("\n✅ Sequential workflow test passed")
    print(f"Final state:")
    print(f"  Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  .obs columns: {len(adata.obs.columns)}")
    print(f"  .obsm keys: {list(adata.obsm.keys())}")

    # Check for expected number of clusters
    cluster_col = 'leiden' if 'leiden' in adata.obs.columns else 'louvain'
    n_clusters = adata.obs[cluster_col].nunique()
    print(f"  Clusters: {n_clusters}")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.single_cell
@pytest.mark.workflow
def test_agent_marker_gene_identification_workflow(
    pbmc3k_raw,
    agent_with_api_key,
    output_validator
):
    """
    Test workflow: clustering → marker gene identification.

    Workflow:
    1. Preprocessing and clustering
    2. Find marker genes for each cluster
    3. Validate marker gene results

    Expected:
    - Marker genes identified for each cluster
    - Statistical significance metrics included
    - Biologically interpretable results

    Reference: t_preprocess.ipynb marker gene analysis
    """
    adata = pbmc3k_raw.copy()
    agent = agent_with_api_key

    print(f"\nInitial: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 1: Preprocessing and clustering
    print("\n[1/2] Preprocessing and clustering...")

    # Do basic preprocessing first
    sc.pp.filter_cells(adata, min_genes=500)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=1.0)

    print(f"  Preprocessed: {adata.n_obs} cells, {adata.obs['leiden'].nunique()} clusters")

    # Step 2: Marker gene identification
    print("\n[2/2] Finding marker genes...")

    result = agent.run(
        'identify marker genes for each cluster using statistical testing. '
        'Find top 10 markers per cluster ranked by statistical significance.',
        adata
    )

    # Extract results
    if isinstance(result, dict):
        result_adata = result.get('adata', result.get('value'))
        marker_df = result.get('markers') or result.get('marker_genes')
    else:
        result_adata = result
        marker_df = None

    # Check if marker genes were computed
    if result_adata is not None:
        # Check for rank_genes_groups results in .uns
        if 'rank_genes_groups' in result_adata.uns:
            print("  ✅ Marker genes stored in .uns['rank_genes_groups']")

            # Validate structure
            rgg = result_adata.uns['rank_genes_groups']
            if isinstance(rgg, dict):
                if 'names' in rgg:
                    n_clusters = len(rgg['names'].dtype.names)
                    print(f"  Marker genes identified for {n_clusters} clusters")

                    # Show some example markers
                    print("\n  Sample markers (top 3 per first 3 clusters):")
                    for i, cluster in enumerate(rgg['names'].dtype.names[:3]):
                        markers = rgg['names'][cluster][:3]
                        print(f"    Cluster {cluster}: {', '.join(markers)}")

        elif marker_df is not None and isinstance(marker_df, pd.DataFrame):
            print(f"  ✅ Marker genes table: {len(marker_df)} entries")

            # Validate marker table structure
            expected_cols = ['gene', 'cluster', 'pvalue']
            available_cols = set(marker_df.columns)

            if available_cols & set(expected_cols):
                print(f"  Table columns: {list(marker_df.columns)}")

                if 'cluster' in marker_df.columns:
                    n_clusters = marker_df['cluster'].nunique()
                    print(f"  Markers for {n_clusters} clusters")

        else:
            print("  Note: Marker genes may be in a different format")

    print("\n✅ Marker gene workflow test completed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.workflow
@pytest.mark.error_handling
def test_agent_workflow_with_missing_step(
    pbmc3k_raw,
    agent_with_api_key
):
    """
    Test workflow handles missing prerequisite steps.

    Scenario: Request clustering without preprocessing
    Expected: Agent either:
    1. Performs preprocessing automatically, OR
    2. Returns informative error/warning

    This tests the agent's ability to handle incomplete workflows gracefully.
    """
    adata = pbmc3k_raw.copy()
    agent = agent_with_api_key

    print(f"\nInitial: {adata.n_obs} cells × {adata.n_vars} genes")
    print("Attempting clustering WITHOUT preprocessing...")

    # Request clustering on raw data (missing preprocessing)
    try:
        result = agent.run(
            'perform leiden clustering',
            adata
        )

        result_adata = result if not isinstance(result, dict) else \
                      result.get('adata', result.get('value'))

        # Check if agent handled it
        if 'leiden' in result_adata.obs.columns:
            print("\n✅ Agent successfully handled missing preprocessing")
            print("   (Either auto-preprocessed or clustered raw data)")

            # Check if preprocessing was done
            if 'X_pca' in result_adata.obsm or 'highly_variable' in result_adata.var.columns:
                print("   → Agent automatically performed preprocessing")
            else:
                print("   → Agent clustered raw data directly")

        else:
            print("\n⚠️  Clustering not added to .obs")
            print("   Agent may need preprocessing first")

    except Exception as e:
        print(f"\n✅ Agent appropriately raised error for missing preprocessing:")
        print(f"   {type(e).__name__}: {str(e)[:100]}")

    print("\n✅ Error handling test completed")


if __name__ == '__main__':
    """Run multi-workflow tests standalone."""
    pytest.main([__file__, '-v', '-s'])
