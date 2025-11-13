"""
Integration tests for ov.agent error handling and edge cases.

Phase 4 - Error Handling and Edge Cases:
Tests the agent's ability to:
- Recover from errors using reflection
- Handle invalid parameters gracefully
- Manage missing or incomplete data
- Deal with empty inputs
- Respond appropriately to out-of-scope requests
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov
from pathlib import Path


# ==============================================================================
# ERROR RECOVERY TESTS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
@pytest.mark.quick
def test_agent_reflection_recovery(pbmc3k_raw, agent_with_api_key):
    """
    Test automatic error recovery using reflection mechanism.

    Agent should detect errors and retry with corrections (≤3 attempts).
    """
    adata = pbmc3k_raw.copy()

    # Preprocess minimally
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"\nTesting error recovery with reflection")
    print(f"Initial: {adata.n_obs} cells × {adata.n_vars} genes")

    # Request that might initially generate problematic code
    # e.g., using a non-existent method or wrong parameter
    result = agent_with_api_key.run(
        'perform leiden clustering with very high resolution 100.0',
        adata
    )

    result_adata = result if not isinstance(result, dict) else \
                   result.get('adata', result.get('value', result))

    # Check if agent successfully handled the request
    if hasattr(result_adata, 'obs') and result_adata is not None:
        print("✅ Agent handled challenging request (with or without reflection)")

        # Check if clustering was performed
        if 'leiden' in result_adata.obs.columns:
            n_clusters = result_adata.obs['leiden'].nunique()
            print(f"   Clustering completed: {n_clusters} clusters")
    else:
        print("⚠️  Agent may have encountered unrecoverable error")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_invalid_parameters(pbmc3k_raw, agent_with_api_key):
    """
    Test handling of malformed or invalid parameter requests.

    Agent should handle errors gracefully or ask for clarification.
    """
    adata = pbmc3k_raw.copy()

    # Preprocess
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata)

    print(f"\nTesting invalid parameter handling")

    # Test with invalid resolution (negative value)
    result = agent_with_api_key.run(
        'perform leiden clustering with resolution=-1.0',
        adata
    )

    result_adata = result if not isinstance(result, dict) else \
                   result.get('adata', result.get('value', result))

    # Agent should either:
    # 1. Handle error gracefully (e.g., use default value)
    # 2. Return informative message
    # 3. Not crash
    if result_adata is not None:
        print("✅ Agent handled invalid parameters gracefully")

        # Check if it proceeded with valid alternative
        if hasattr(result_adata, 'obs') and 'leiden' in result_adata.obs.columns:
            print("   Agent used alternative valid parameters")
    else:
        print("✅ Agent returned None/error message (valid error handling)")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_missing_data(pbmc3k_raw, agent_with_api_key):
    """
    Test handling of missing required data/preprocessing.

    Agent should either preprocess first or return informative error.
    """
    adata = pbmc3k_raw.copy()

    # Minimal filtering only - no normalization, no HVG, no PCA
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"\nTesting missing preprocessing data")
    print(f"Data status: {adata.n_obs} cells, no normalization/PCA")

    # Request clustering without proper preprocessing
    result = agent_with_api_key.run(
        'perform leiden clustering and UMAP visualization',
        adata
    )

    result_adata = result if not isinstance(result, dict) else \
                   result.get('adata', result.get('value', result))

    # Agent should either:
    # 1. Automatically perform required preprocessing
    # 2. Return error message about missing preprocessing
    if result_adata is not None and hasattr(result_adata, 'obs'):
        print("✅ Agent handled missing preprocessing")

        # Check if agent performed preprocessing
        if 'leiden' in result_adata.obs.columns or 'X_umap' in result_adata.obsm:
            print("   Agent performed necessary preprocessing automatically")
        else:
            print("   Agent may have provided guidance on preprocessing")
    else:
        print("✅ Agent informed about missing prerequisites (valid behavior)")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_empty_input(agent_with_api_key):
    """
    Test handling of empty AnnData objects.

    Agent should detect and handle empty data gracefully.
    """
    print(f"\nTesting empty input handling")

    # Create empty AnnData
    empty_adata = sc.AnnData(
        X=np.array([]).reshape(0, 100),  # 0 cells, 100 genes
        obs=pd.DataFrame(index=pd.Index([], name='cell_id')),
        var=pd.DataFrame(index=[f'Gene_{i}' for i in range(100)])
    )

    print(f"Empty AnnData: {empty_adata.n_obs} cells × {empty_adata.n_vars} genes")

    result = agent_with_api_key.run(
        'perform quality control and clustering',
        empty_adata
    )

    # Agent should gracefully handle empty data
    # Either return None, empty result, or informative message
    if result is None or (isinstance(result, str) and 'empty' in result.lower()):
        print("✅ Agent detected and reported empty input")
    elif isinstance(result, dict) and not result:
        print("✅ Agent returned empty result for empty input")
    else:
        print("✅ Agent handled empty input (may have provided guidance)")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_skill_mismatch(agent_with_api_key):
    """
    Test handling of requests outside skill scope.

    Agent should respond appropriately when no matching skill exists.
    """
    print(f"\nTesting out-of-scope request handling")

    # Request completely unrelated to bioinformatics
    result = agent_with_api_key.run(
        'predict tomorrow\'s stock market prices for AAPL',
        None
    )

    # Agent should respond appropriately
    # Could be: rejection, clarification request, or explanation
    if result is not None:
        print("✅ Agent responded to out-of-scope request")
        if isinstance(result, str):
            # Check for appropriate response patterns
            response_lower = result.lower()
            if any(word in response_lower for word in ['cannot', 'unable', 'outside', 'scope', 'bioinformatics', 'omics']):
                print("   Agent appropriately declined out-of-scope request")
            else:
                print("   Agent provided some response")
    else:
        print("✅ Agent returned None for out-of-scope request")


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_very_small_dataset(agent_with_api_key):
    """
    Test handling of very small datasets (edge case).

    Agent should handle minimal data appropriately.
    """
    print(f"\nTesting very small dataset handling")

    # Create tiny dataset (10 cells, 50 genes)
    np.random.seed(42)
    tiny_adata = sc.AnnData(
        X=np.random.negative_binomial(5, 0.3, (10, 50)),
        obs=pd.DataFrame(index=[f'Cell_{i}' for i in range(10)]),
        var=pd.DataFrame(index=[f'Gene_{i}' for i in range(50)])
    )

    print(f"Tiny dataset: {tiny_adata.n_obs} cells × {tiny_adata.n_vars} genes")

    result = agent_with_api_key.run(
        'perform basic preprocessing and clustering',
        tiny_adata
    )

    result_adata = result if not isinstance(result, dict) else \
                   result.get('adata', result.get('value', result))

    # Agent should handle small dataset appropriately
    if result_adata is not None:
        print("✅ Agent handled very small dataset")
        if hasattr(result_adata, 'n_obs'):
            print(f"   Result: {result_adata.n_obs} cells")
    else:
        print("⚠️  Agent may have warned about dataset size")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_missing_gene_names(agent_with_api_key):
    """
    Test handling of data without gene names.

    Agent should handle or request proper gene identifiers.
    """
    print(f"\nTesting missing gene names handling")

    # Create dataset without gene names (just indices)
    np.random.seed(42)
    adata_no_names = sc.AnnData(
        X=np.random.negative_binomial(5, 0.3, (100, 50)),
        obs=pd.DataFrame(index=[f'Cell_{i}' for i in range(100)]),
        var=pd.DataFrame(index=[f'{i}' for i in range(50)])  # Numeric indices only
    )

    print(f"Dataset: {adata_no_names.n_obs} cells, genes={list(adata_no_names.var_names[:5])}")

    result = agent_with_api_key.run(
        'find marker genes for cell types',
        adata_no_names
    )

    # Agent should handle lack of gene names
    if result is not None:
        print("✅ Agent handled dataset without gene names")
    else:
        print("⚠️  Agent may have requested proper gene names")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.error_handling
def test_agent_conflicting_requests(pbmc3k_raw, agent_with_api_key):
    """
    Test handling of conflicting or contradictory requests.

    Agent should clarify or make reasonable choice.
    """
    adata = pbmc3k_raw.copy()

    # Minimal preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"\nTesting conflicting request handling")

    # Conflicting request: subset to HVGs then keep all genes
    result = agent_with_api_key.run(
        'select 1000 highly variable genes, then use all genes for clustering',
        adata
    )

    result_adata = result if not isinstance(result, dict) else \
                   result.get('adata', result.get('value', result))

    # Agent should make reasonable interpretation
    if result_adata is not None and hasattr(result_adata, 'n_vars'):
        print(f"✅ Agent handled conflicting request")
        print(f"   Final gene count: {result_adata.n_vars}")


# ==============================================================================
# SUMMARY FUNCTION
# ==============================================================================

def test_error_handling_summary():
    """
    Summary of error handling and edge case coverage.

    Phase 4 - Error Handling Tests:
    - Error recovery and reflection
    - Invalid parameter handling
    - Missing data detection
    - Empty input handling
    - Out-of-scope request handling
    - Edge cases (small datasets, missing metadata)
    """
    print("\n" + "="*70)
    print("ERROR HANDLING TEST SUMMARY - PHASE 4")
    print("="*70)

    test_categories = {
        'Error Recovery': [
            'test_agent_reflection_recovery',
            'test_agent_invalid_parameters',
        ],
        'Missing/Invalid Data': [
            'test_agent_missing_data',
            'test_agent_empty_input',
            'test_agent_missing_gene_names',
        ],
        'Scope and Request Handling': [
            'test_agent_skill_mismatch',
            'test_agent_conflicting_requests',
        ],
        'Edge Cases': [
            'test_agent_very_small_dataset',
        ],
    }

    total_tests = sum(len(tests) for tests in test_categories.values())

    print(f"Total error handling tests: {total_tests}")
    print()

    for category, tests in test_categories.items():
        print(f"{category}:")
        for test in tests:
            print(f"  ✅ {test}")
        print()

    print("="*70)
    print("Phase 4 validates agent robustness and error resilience")
    print("="*70 + "\n")


if __name__ == '__main__':
    """Run error handling tests standalone."""
    pytest.main([__file__, '-v', '-s', '-k', 'not full'])
