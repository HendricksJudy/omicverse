# Quick Start Guide: Agent Integration Tests

Get started with OmicVerse agent integration tests in 5 minutes.

## Prerequisites

- Python 3.8+
- OmicVerse installed
- API key from OpenAI, Anthropic, or Google

## Step 1: Set Up API Key

```bash
# Choose one provider:
export OPENAI_API_KEY="sk-..."           # Recommended for tests
# OR
export ANTHROPIC_API_KEY="sk-ant-..."
# OR
export GEMINI_API_KEY="..."
```

## Step 2: Generate Reference Data

```bash
# From project root
python scripts/generate_reference_data.py

# This creates reference outputs in tests/integration/agent/data/
# Takes ~5 minutes for PBMC3k dataset
```

Expected output:
```
Generating PBMC3k reference data...
  QC: 2603 cells × 13631 genes
  Preprocessed: 2603 cells × 2000 HVGs
  Clustered: 11 clusters
✅ Reference data saved to tests/integration/agent/data/pbmc3k
```

## Step 3: Run Quick Tests

```bash
# Run fast tests (~5-10 minutes)
pytest tests/integration/agent/ -m quick -v

# Expected: 5 tests pass
```

## Step 4: Run All Tests

```bash
# Run complete Phase 1-3 tests (~30-60 minutes)
pytest tests/integration/agent/ -v

# Expected: 19+ tests pass
```

## Common Commands

### Test by Phase

```bash
# Phase 1: Foundation tests
pytest tests/integration/agent/test_agent_single_cell.py -v

# Phase 2: Multi-step workflows
pytest tests/integration/agent/test_agent_multiworkflow.py -v

# Phase 3: Individual skills
pytest tests/integration/agent/test_agent_skills.py -v -k "not full"
```

### Test by Category

```bash
# Single-cell tests only
pytest tests/integration/agent/ -m single_cell -v

# Workflow tests only
pytest tests/integration/agent/ -m workflow -v

# Skill tests only
pytest tests/integration/agent/ -m skill -v
```

### Run Specific Test

```bash
# Example: Test QC filtering
pytest tests/integration/agent/test_agent_single_cell.py::test_agent_qc_filtering -v -s
```

## Understanding Test Output

### Successful Test
```
test_agent_qc_filtering PASSED
Initial: 2700 cells × 32738 genes
After QC: 2603 cells × 13631 genes
✅ QC filtering test passed
```

### Skipped Test
```
test_agent_bulk_deg SKIPPED (No API key available)
```
**Solution**: Set API key environment variable

### Failed Test
```
test_agent_clustering FAILED
AssertionError: Expected 8-12 clusters, got 15
```
**Note**: Some variation is expected due to non-deterministic algorithms

## Validation Tolerances

Tests use tolerances for non-deterministic outputs:

| Check | Tolerance | Reason |
|-------|-----------|--------|
| Cell/gene counts | ±5-10% | Algorithm variation |
| Clustering | ARI ≥ 0.85 | Different random seeds |
| DEG overlap | ≥80% | Statistical methods differ |

## Cost Estimation

Using `gpt-4o-mini` (~$0.15 per 1M tokens):

- Quick tests: ~$0.10-0.50
- Full Phase 1-3: ~$1-5
- Single test: ~$0.05-0.10

## Troubleshooting

### "No API key available"
```bash
# Check env variable is set
echo $OPENAI_API_KEY

# If empty, export it
export OPENAI_API_KEY="your-key"
```

### "Reference data not found"
```bash
# Generate reference data
python scripts/generate_reference_data.py

# Verify it was created
ls tests/integration/agent/data/pbmc3k/
# Should show: qc.h5ad, preprocessed.h5ad, clustered.h5ad, reference_metrics.json
```

### "Agent result missing expected keys"
This is usually fine - agent may return results in different formats. Tests handle multiple formats.

### Tests taking too long
```bash
# Run only quick tests
pytest tests/integration/agent/ -m quick

# Or skip computationally intensive tests
pytest tests/integration/agent/ -k "not full"
```

### ImportError or ModuleNotFoundError
```bash
# Install OmicVerse in development mode
pip install -e .

# Or install test dependencies
pip install pytest pytest-asyncio scanpy
```

## Next Steps

### View Coverage Report
```bash
python tests/integration/agent/utils/skill_coverage.py
```

### Generate Custom Reports
```python
from tests.integration.agent.utils.workflow_tracker import create_pbmc_preprocessing_workflow

workflow = create_pbmc_preprocessing_workflow()
# ... execute workflow ...
workflow.print_summary()
```

### Add Custom Tests
See `docs/testing/agent_integration_tests_plan.md` for guidelines on adding new tests.

## Getting Help

- **Documentation**: `tests/integration/agent/README.md`
- **Full Plan**: `docs/testing/agent_integration_tests_plan.md`
- **Issues**: Report at https://github.com/HendricksJudy/omicverse/issues

## Quick Reference

```bash
# Setup
export OPENAI_API_KEY="sk-..."
python scripts/generate_reference_data.py

# Run tests
pytest tests/integration/agent/ -m quick -v          # Fast (~10 min)
pytest tests/integration/agent/ -m "quick or workflow" -v  # Quick + workflows (~20 min)
pytest tests/integration/agent/ -v                   # All tests (~60 min)

# By phase
pytest tests/integration/agent/test_agent_single_cell.py -v     # Phase 1
pytest tests/integration/agent/test_agent_multiworkflow.py -v   # Phase 2
pytest tests/integration/agent/test_agent_skills.py -v          # Phase 3

# Coverage
python tests/integration/agent/utils/skill_coverage.py
```

## Success Criteria

✅ All quick tests pass (5 tests)
✅ API key configured
✅ Reference data generated
✅ No import errors

You're ready to use the agent integration test suite! 🎉
