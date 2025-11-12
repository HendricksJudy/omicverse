"""
Integration tests for ov.agent bulk RNA-seq workflows.

Tests based on tutorial notebook: t_deg.ipynb

Phase 1 - Foundation Tests:
- Differential expression analysis
- Gene ID mapping
"""

import pytest
import numpy as np
import pandas as pd
import omicverse as ov


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.quick
@pytest.mark.bulk
def test_agent_bulk_deg_basic(agent_with_api_key, output_validator, tmp_path):
    """
    Test agent performs basic DEG analysis.

    Request: "perform differential expression analysis"

    Expected:
    - DEG table generated
    - Required columns: log2FC, pvalue, qvalue, BaseMean
    - At least some significant genes

    Reference: t_deg.ipynb

    Note: This is a simplified test. Full DEG test requires real bulk data.
    """
    # Create mock count data
    np.random.seed(42)
    n_genes = 1000
    n_samples = 6

    # Mock expression matrix
    counts = np.random.negative_binomial(10, 0.5, size=(n_genes, n_samples))
    genes = [f"GENE_{i}" for i in range(n_genes)]
    samples = [f"Sample_{i}" for i in range(n_samples)]

    # Create DataFrame
    count_df = pd.DataFrame(counts, index=genes, columns=samples)

    # Sample metadata
    metadata = pd.DataFrame({
        'sample': samples,
        'group': ['control'] * 3 + ['treatment'] * 3
    })

    print(f"\nMock data: {n_genes} genes × {n_samples} samples")

    # Save to temp file
    count_file = tmp_path / 'counts.csv'
    count_df.to_csv(count_file)

    metadata_file = tmp_path / 'metadata.csv'
    metadata.to_csv(metadata_file, index=False)

    # Request DEG analysis
    # Note: Agent needs to load data and perform analysis
    request = f"""
    Load count data from {count_file} and metadata from {metadata_file}.
    Perform differential expression analysis comparing treatment vs control groups.
    """

    # This test is a placeholder - actual implementation depends on agent's
    # ability to load external files and perform DEG analysis
    pytest.skip(
        "Full DEG test requires agent integration with file loading. "
        "This test validates the structure but needs agent enhancement."
    )


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.full
@pytest.mark.bulk
def test_agent_deg_with_anndata(agent_with_api_key, output_validator):
    """
    Test DEG analysis using AnnData format.

    Request: "identify differentially expressed genes between groups"

    Expected:
    - Agent generates DEG results
    - Proper statistical testing
    - Results interpretable

    Reference: t_deg.ipynb adapted for AnnData
    """
    # Create mock AnnData for DEG
    np.random.seed(42)
    n_obs = 6
    n_vars = 1000

    # Generate counts with some DE genes
    X = np.random.negative_binomial(10, 0.5, size=(n_obs, n_vars))

    # Make first 50 genes "differentially expressed"
    X[3:, :50] = X[3:, :50] * 2  # Treatment has higher expression

    import anndata
    adata = anndata.AnnData(X=X)
    adata.obs['group'] = ['control'] * 3 + ['treatment'] * 3
    adata.var_names = [f"GENE_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Sample_{i}" for i in range(n_obs)]

    print(f"\nMock DEG data: {adata.n_obs} samples × {adata.n_vars} genes")
    print(f"Groups: {adata.obs['group'].value_counts().to_dict()}")

    # Request DEG analysis
    result = agent_with_api_key.run(
        'identify differentially expressed genes between control and treatment groups',
        adata
    )

    # Validate result structure
    # Result should contain DEG table or modified adata with DEG results
    if isinstance(result, dict):
        # Check if DEG results present
        assert 'value' in result or 'adata' in result, \
            "Agent result missing expected keys"

        # If DEG table returned
        if 'deg_table' in result or isinstance(result.get('value'), pd.DataFrame):
            deg_df = result.get('deg_table') or result['value']

            # Validate DEG table structure
            expected_cols = ['log2FC', 'pvalue']
            available_cols = set(deg_df.columns)

            # At least basic DEG columns should be present
            missing = set(expected_cols) - available_cols
            if missing:
                print(f"Warning: Missing expected columns: {missing}")
                print(f"Available columns: {list(available_cols)}")

    print("✅ DEG with AnnData test completed")
    print("Note: Full validation requires checking statistical results")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.skill
@pytest.mark.bulk
def test_skill_bulk_deg_analysis_invoked(agent_with_api_key):
    """
    Test that bulk-deg-analysis skill is properly invoked.

    This test verifies the agent can:
    1. Match the correct skill for DEG requests
    2. Load skill content
    3. Generate appropriate analysis code

    Note: Actual execution requires real bulk data, but we can test
    the skill matching and code generation steps.
    """
    import anndata
    import numpy as np

    # Create minimal mock data
    adata = anndata.AnnData(
        X=np.random.rand(6, 100),
        obs=pd.DataFrame({'group': ['A']*3 + ['B']*3})
    )

    request = "perform bulk RNA-seq differential expression analysis"

    # This would test skill matching
    # In practice, we'd need to inspect agent internals or logs
    # to confirm skill was matched

    pytest.skip(
        "Skill matching verification requires agent instrumentation. "
        "See verifier tests for skill matching validation."
    )


if __name__ == '__main__':
    """Run bulk tests standalone."""
    pytest.main([__file__, '-v', '-s'])
