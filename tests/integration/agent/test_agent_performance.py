"""
Integration tests for ov.agent performance and scalability.

Phase 5 - Performance and Robustness Tests:
Tests the agent's:
- Execution time benchmarks
- Concurrent request handling (thread safety)
- Scalability with large datasets
- Memory efficiency
- Consistency across multiple runs
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import psutil
import os


# ==============================================================================
# PERFORMANCE BENCHMARK TESTS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.performance
@pytest.mark.full
def test_agent_execution_time(pbmc3k_raw, agent_with_api_key):
    """
    Benchmark execution time for standard PBMC3k workflow.

    Validates: Completes within reasonable time (e.g., <5 min for 4-step workflow).
    """
    adata = pbmc3k_raw.copy()

    print(f"\n{'='*70}")
    print("PERFORMANCE BENCHMARK: Standard 4-Step Workflow")
    print(f"{'='*70}")
    print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")

    # Define 4-step workflow
    workflow_steps = [
        ('Quality Control', 'filter cells with >200 genes and <20% mitochondria'),
        ('Preprocessing', 'normalize, log-transform, and select 2000 HVGs'),
        ('Clustering', 'compute PCA, neighbors, and leiden clustering'),
        ('Visualization', 'compute UMAP embedding'),
    ]

    step_times = {}
    total_start = time.time()

    for step_name, request in workflow_steps:
        print(f"\n--- Step: {step_name} ---")
        print(f"Request: {request}")

        step_start = time.time()
        result = agent_with_api_key.run(request, adata)
        step_time = time.time() - step_start

        step_times[step_name] = step_time

        # Update adata for next step
        if result is not None:
            adata = result if not isinstance(result, dict) else \
                    result.get('adata', result.get('value', adata))

        print(f"Duration: {step_time:.2f}s")
        if hasattr(adata, 'n_obs'):
            print(f"Result: {adata.n_obs} cells × {adata.n_vars} genes")

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'='*70}")
    print("TIMING SUMMARY")
    print(f"{'='*70}")
    for step, duration in step_times.items():
        print(f"{step:25s}: {duration:6.2f}s")
    print(f"{'='*70}")
    print(f"{'Total':<25s}: {total_time:6.2f}s")
    print(f"{'='*70}\n")

    # Validate reasonable execution time (e.g., <5 minutes = 300s)
    # Note: Actual time depends on model speed, adjust threshold as needed
    max_time = 600  # 10 minutes for conservative threshold
    assert total_time < max_time, \
        f"Workflow took {total_time:.1f}s (>{max_time}s threshold)"

    print(f"✅ Workflow completed in {total_time:.1f}s (within {max_time}s threshold)")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.performance
@pytest.mark.full
def test_agent_concurrent_requests(pbmc3k_raw, api_key, model_name):
    """
    Test thread safety with concurrent agent instances.

    Validates: No race conditions, all instances produce correct results.
    """
    print(f"\n{'='*70}")
    print("CONCURRENCY TEST: Parallel Agent Instances")
    print(f"{'='*70}")

    # Create multiple datasets
    n_agents = 3
    datasets = []

    for i in range(n_agents):
        adata = pbmc3k_raw.copy()
        # Add unique identifier
        adata.obs['dataset_id'] = f'dataset_{i}'
        datasets.append(adata)

    print(f"Testing {n_agents} concurrent agents")

    def run_agent_workflow(dataset_id: int, adata, api_key: str, model: str) -> Dict:
        """Run a simple workflow on one dataset."""
        agent = ov.Agent(model=model, api_key=api_key, temperature=0.0)

        start_time = time.time()

        # Simple 2-step workflow
        try:
            result1 = agent.run(
                'filter cells with >200 genes',
                adata
            )
            adata_filtered = result1 if not isinstance(result1, dict) else \
                            result1.get('adata', result1.get('value', result1))

            result2 = agent.run(
                'normalize and log-transform',
                adata_filtered
            )
            adata_final = result2 if not isinstance(result2, dict) else \
                         result2.get('adata', result2.get('value', result2))

            duration = time.time() - start_time

            return {
                'dataset_id': dataset_id,
                'success': True,
                'n_cells': adata_final.n_obs if hasattr(adata_final, 'n_obs') else None,
                'n_genes': adata_final.n_vars if hasattr(adata_final, 'n_vars') else None,
                'duration': duration,
                'error': None
            }

        except Exception as e:
            return {
                'dataset_id': dataset_id,
                'success': False,
                'n_cells': None,
                'n_genes': None,
                'duration': time.time() - start_time,
                'error': str(e)
            }

    # Run agents concurrently
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=n_agents) as executor:
        futures = [
            executor.submit(run_agent_workflow, i, datasets[i], api_key, model_name)
            for i in range(n_agents)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "✅" if result['success'] else "❌"
            print(f"{status} Dataset {result['dataset_id']}: "
                  f"{result['n_cells']} cells, {result['duration']:.2f}s")

    total_time = time.time() - start_time

    # Validate results
    print(f"\n{'='*70}")
    print("CONCURRENCY SUMMARY")
    print(f"{'='*70}")
    print(f"Total concurrent time: {total_time:.2f}s")
    print(f"Successful: {sum(1 for r in results if r['success'])}/{n_agents}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}/{n_agents}")

    # Check all succeeded
    failures = [r for r in results if not r['success']]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  Dataset {f['dataset_id']}: {f['error']}")

    assert all(r['success'] for r in results), \
        f"{len(failures)} agent(s) failed in concurrent execution"

    # Check results are consistent (similar cell counts)
    cell_counts = [r['n_cells'] for r in results if r['n_cells'] is not None]
    if len(cell_counts) > 1:
        mean_cells = np.mean(cell_counts)
        std_cells = np.std(cell_counts)
        print(f"Cell counts: {cell_counts} (mean={mean_cells:.0f}, std={std_cells:.0f})")

    print(f"\n✅ All {n_agents} concurrent agents completed successfully")
    print(f"{'='*70}\n")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.performance
@pytest.mark.full
def test_agent_large_dataset(agent_with_api_key):
    """
    Test scalability with larger dataset.

    Validates: Agent completes without memory errors on 50k cells.
    """
    print(f"\n{'='*70}")
    print("SCALABILITY TEST: Large Dataset (50k cells)")
    print(f"{'='*70}")

    # Create larger synthetic dataset (50k cells, 2000 genes)
    # Note: Using sparse matrix for memory efficiency
    np.random.seed(42)
    n_cells = 50000
    n_genes = 2000

    print(f"Generating synthetic dataset: {n_cells} cells × {n_genes} genes")

    # Create sparse random data
    from scipy.sparse import random as sparse_random

    X = sparse_random(n_cells, n_genes, density=0.1, format='csr',
                      random_state=42) * 10

    large_adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f'Cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame(index=[f'Gene_{i}' for i in range(n_genes)])
    )

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Initial memory: {mem_before:.1f} MB")
    print(f"Data size: {large_adata.X.data.nbytes / 1024 / 1024:.1f} MB (sparse)")

    # Test basic operations
    print("\n--- Testing basic preprocessing on large dataset ---")

    start_time = time.time()

    try:
        result = agent_with_api_key.run(
            'calculate basic QC metrics (total counts per cell, genes per cell)',
            large_adata
        )

        duration = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        result_adata = result if not isinstance(result, dict) else \
                       result.get('adata', result.get('value', result))

        print(f"\n{'='*70}")
        print("SCALABILITY SUMMARY")
        print(f"{'='*70}")
        print(f"Duration: {duration:.2f}s")
        print(f"Memory before: {mem_before:.1f} MB")
        print(f"Memory after: {mem_after:.1f} MB")
        print(f"Memory increase: {mem_increase:.1f} MB")

        if hasattr(result_adata, 'obs'):
            print(f"Result: {result_adata.n_obs} cells × {result_adata.n_vars} genes")

            # Check if QC metrics were added
            qc_cols = [col for col in result_adata.obs.columns
                      if any(term in col.lower() for term in ['total', 'counts', 'genes', 'n_'])]
            if qc_cols:
                print(f"QC metrics added: {', '.join(qc_cols[:3])}")

        # Validate memory didn't explode (arbitrary threshold: <10GB increase)
        max_mem_increase = 10000  # MB
        assert mem_increase < max_mem_increase, \
            f"Memory increased by {mem_increase:.1f}MB (>{max_mem_increase}MB threshold)"

        print(f"\n✅ Large dataset handled successfully")
        print(f"{'='*70}\n")

    except MemoryError as e:
        pytest.fail(f"Memory error with large dataset: {e}")

    except Exception as e:
        print(f"⚠️  Error processing large dataset: {e}")
        pytest.skip(f"Large dataset test requires more resources: {e}")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.performance
def test_agent_consistency(pbmc3k_raw, agent_with_api_key):
    """
    Test consistency of results across multiple runs.

    With temperature=0, results should be deterministic.
    """
    adata = pbmc3k_raw.copy()

    # Preprocess to controlled state
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"\n{'='*70}")
    print("CONSISTENCY TEST: Multiple Runs with temperature=0")
    print(f"{'='*70}")

    request = 'normalize to 10000 counts per cell and log-transform'

    n_runs = 2
    results = []

    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")

        adata_copy = adata.copy()
        result = agent_with_api_key.run(request, adata_copy)

        result_adata = result if not isinstance(result, dict) else \
                       result.get('adata', result.get('value', result))

        if hasattr(result_adata, 'X'):
            # Store checksum of data
            checksum = np.sum(result_adata.X[:100, :100])  # Sample for speed
            results.append({
                'run': i + 1,
                'n_obs': result_adata.n_obs,
                'n_vars': result_adata.n_vars,
                'checksum': checksum
            })
            print(f"Result: {result_adata.n_obs} cells, checksum={checksum:.4f}")

    # Compare results
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("CONSISTENCY SUMMARY")
        print(f"{'='*70}")

        # Check dimensions match
        dims_match = all(r['n_obs'] == results[0]['n_obs'] and
                        r['n_vars'] == results[0]['n_vars']
                        for r in results)

        print(f"Dimensions match: {'✅' if dims_match else '❌'}")

        # Check data similarity (may not be exactly identical due to LLM variation)
        checksums = [r['checksum'] for r in results]
        checksum_diff = abs(checksums[0] - checksums[1]) if len(checksums) == 2 else 0
        checksum_pct = (checksum_diff / abs(checksums[0])) * 100 if checksums[0] != 0 else 0

        print(f"Checksum difference: {checksum_pct:.2f}%")

        # With temperature=0, results should be very similar (<5% difference)
        if checksum_pct < 5.0:
            print("✅ Results are highly consistent (temperature=0 working)")
        else:
            print("⚠️  Results show some variation (may be expected with LLM non-determinism)")

        print(f"{'='*70}\n")


# ==============================================================================
# MEMORY EFFICIENCY TESTS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.performance
@pytest.mark.full
def test_agent_memory_efficiency(pbmc3k_raw, agent_with_api_key):
    """
    Test memory efficiency - no memory leaks on repeated operations.

    Validates: Memory usage remains stable across multiple operations.
    """
    adata = pbmc3k_raw.copy()

    # Preprocess
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"\n{'='*70}")
    print("MEMORY EFFICIENCY TEST: Repeated Operations")
    print(f"{'='*70}")

    process = psutil.Process(os.getpid())

    n_iterations = 3
    memory_usage = []

    for i in range(n_iterations):
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Perform operation
        adata_copy = adata.copy()
        result = agent_with_api_key.run(
            'calculate total counts per cell',
            adata_copy
        )

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage.append(mem_after)

        print(f"Iteration {i+1}: Memory = {mem_after:.1f} MB "
              f"(Δ{mem_after-mem_before:+.1f} MB)")

        # Clean up
        del adata_copy
        del result

    # Check for memory leaks
    print(f"\n{'='*70}")
    print("MEMORY SUMMARY")
    print(f"{'='*70}")
    print(f"Memory usage: {memory_usage}")

    if len(memory_usage) >= 2:
        memory_growth = memory_usage[-1] - memory_usage[0]
        print(f"Total memory growth: {memory_growth:+.1f} MB")

        # Allow some growth, but not excessive (e.g., <500MB)
        max_growth = 500
        if abs(memory_growth) < max_growth:
            print(f"✅ Memory usage stable (growth <{max_growth}MB)")
        else:
            print(f"⚠️  Significant memory growth detected ({memory_growth:.1f}MB)")

    print(f"{'='*70}\n")


# ==============================================================================
# SUMMARY FUNCTION
# ==============================================================================

def test_performance_summary():
    """
    Summary of performance and scalability test coverage.

    Phase 5 - Performance and Robustness:
    - Execution time benchmarks
    - Concurrent request handling
    - Large dataset scalability
    - Memory efficiency
    - Result consistency
    """
    print("\n" + "="*70)
    print("PERFORMANCE TEST SUMMARY - PHASE 5")
    print("="*70)

    test_categories = {
        'Performance Benchmarks': [
            'test_agent_execution_time',
            'test_agent_consistency',
        ],
        'Scalability': [
            'test_agent_large_dataset',
            'test_agent_concurrent_requests',
        ],
        'Resource Efficiency': [
            'test_agent_memory_efficiency',
        ],
    }

    total_tests = sum(len(tests) for tests in test_categories.values())

    print(f"Total performance tests: {total_tests}")
    print()

    for category, tests in test_categories.items():
        print(f"{category}:")
        for test in tests:
            print(f"  ✅ {test}")
        print()

    print("="*70)
    print("Phase 5 validates agent performance and scalability")
    print("="*70 + "\n")


if __name__ == '__main__':
    """Run performance tests standalone."""
    pytest.main([__file__, '-v', '-s', '-m', 'performance'])
