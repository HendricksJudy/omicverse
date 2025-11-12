"""
Pytest fixtures for agent integration tests.

Provides shared fixtures for:
- Test data (PBMC3k, bulk datasets)
- Agent initialization with API keys
- Output validators
- Reference data loaders
"""

import pytest
import scanpy as sc
import omicverse as ov
import os
import json
from pathlib import Path
from typing import Optional


# Test data directory
DATA_DIR = Path(__file__).parent / 'data'


@pytest.fixture(scope='session')
def pbmc3k_raw():
    """
    Load raw PBMC3k dataset.

    Returns:
        AnnData: Raw PBMC3k with 2,700 cells × 32,738 genes
    """
    try:
        adata = sc.datasets.pbmc3k()
        return adata
    except Exception as e:
        pytest.skip(f"Could not load PBMC3k dataset: {e}")


@pytest.fixture(scope='session')
def pbmc3k_reference_metrics():
    """
    Load reference metrics for PBMC3k workflows.

    Returns:
        dict: Expected metrics at different workflow stages
    """
    metrics_path = DATA_DIR / 'pbmc3k' / 'reference_metrics.json'
    if not metrics_path.exists():
        pytest.skip(f"Reference metrics not found: {metrics_path}")

    with open(metrics_path) as f:
        return json.load(f)


@pytest.fixture
def pbmc3k_qc_reference():
    """Load QC-filtered PBMC3k reference."""
    ref_path = DATA_DIR / 'pbmc3k' / 'qc.h5ad'
    if not ref_path.exists():
        pytest.skip(f"QC reference not found: {ref_path}")
    return sc.read_h5ad(ref_path)


@pytest.fixture
def pbmc3k_preprocessed_reference():
    """Load preprocessed PBMC3k reference (with HVG selection)."""
    ref_path = DATA_DIR / 'pbmc3k' / 'preprocessed.h5ad'
    if not ref_path.exists():
        pytest.skip(f"Preprocessed reference not found: {ref_path}")
    return sc.read_h5ad(ref_path)


@pytest.fixture
def pbmc3k_clustered_reference():
    """Load clustered PBMC3k reference (with UMAP and Leiden)."""
    ref_path = DATA_DIR / 'pbmc3k' / 'clustered.h5ad'
    if not ref_path.exists():
        pytest.skip(f"Clustered reference not found: {ref_path}")
    return sc.read_h5ad(ref_path)


@pytest.fixture
def api_key():
    """
    Get API key from environment.

    Tries multiple providers in order:
    1. OPENAI_API_KEY
    2. ANTHROPIC_API_KEY
    3. GEMINI_API_KEY

    Skips test if no key found.
    """
    key = (os.getenv('OPENAI_API_KEY') or
           os.getenv('ANTHROPIC_API_KEY') or
           os.getenv('GEMINI_API_KEY'))

    if not key:
        pytest.skip("No API key available for integration tests. "
                   "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")

    return key


@pytest.fixture
def model_name():
    """
    Get model name based on available API key.

    Returns cheaper models for faster tests:
    - OpenAI: gpt-4o-mini
    - Anthropic: claude-haiku-3.5
    - Google: gemini-2.5-flash
    """
    if os.getenv('OPENAI_API_KEY'):
        return 'gpt-4o-mini'
    elif os.getenv('ANTHROPIC_API_KEY'):
        return 'anthropic/claude-haiku-3.5'
    elif os.getenv('GEMINI_API_KEY'):
        return 'gemini/gemini-2.5-flash'
    else:
        pytest.skip("No API key available")


@pytest.fixture
def agent_with_api_key(api_key, model_name):
    """
    Initialize OmicVerse agent with API key.

    Uses cheap models for cost-effective testing.
    Temperature set to 0 for deterministic outputs.

    Returns:
        ov.Agent: Initialized agent
    """
    try:
        agent = ov.Agent(
            model=model_name,
            api_key=api_key,
            temperature=0.0  # Deterministic
        )
        return agent
    except Exception as e:
        pytest.skip(f"Could not initialize agent: {e}")


@pytest.fixture
def output_validator():
    """
    Utility for validating agent outputs against references.

    Returns:
        OutputValidator: Validator instance
    """
    from .utils.validators import OutputValidator
    return OutputValidator()


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Temporary directory for test outputs (plots, files, etc.).

    Args:
        tmp_path: pytest built-in fixture

    Returns:
        Path: Temporary directory
    """
    output_dir = tmp_path / "agent_outputs"
    output_dir.mkdir()
    return output_dir


# Markers for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring API keys"
    )
    config.addinivalue_line(
        "markers", "agent: Tests for ov.agent functionality"
    )
    config.addinivalue_line(
        "markers", "quick: Quick integration tests (~5-10 min)"
    )
    config.addinivalue_line(
        "markers", "full: Full integration test suite (~30-60 min)"
    )
    config.addinivalue_line(
        "markers", "single_cell: Single-cell workflow tests"
    )
    config.addinivalue_line(
        "markers", "bulk: Bulk RNA-seq workflow tests"
    )
    config.addinivalue_line(
        "markers", "spatial: Spatial transcriptomics tests"
    )
    config.addinivalue_line(
        "markers", "skill: Individual skill validation tests"
    )
    config.addinivalue_line(
        "markers", "workflow: Multi-step workflow tests"
    )
    config.addinivalue_line(
        "markers", "error_handling: Error recovery tests"
    )


@pytest.fixture
def skip_if_no_reference_data():
    """
    Skip test if reference data not generated yet.

    Use as:
        def test_something(skip_if_no_reference_data):
            skip_if_no_reference_data('pbmc3k')
            # test code...
    """
    def _skip(dataset_name: str):
        ref_dir = DATA_DIR / dataset_name
        if not ref_dir.exists() or not any(ref_dir.iterdir()):
            pytest.skip(
                f"Reference data for '{dataset_name}' not found. "
                f"Run: python scripts/generate_reference_data.py"
            )
    return _skip


# Session-level cleanup
@pytest.fixture(scope='session', autouse=True)
def cleanup_on_exit():
    """Cleanup any temporary files after test session."""
    yield
    # Cleanup code here if needed
    pass
