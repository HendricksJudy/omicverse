# Agent Integration Tests

Real-world integration tests for `ov.agent` based on tutorial notebooks.

## Overview

These tests verify that `ov.agent` can successfully replicate analyses from OmicVerse tutorial notebooks and produce equivalent, validated results.

## Test Structure

```
tests/integration/agent/
├── conftest.py                      # Shared fixtures
├── data/                            # Reference data
│   ├── pbmc3k/                     # PBMC3k references
│   └── bulk_deg/                   # Bulk DEG references
├── utils/                          # Validation utilities
│   ├── validators.py               # Output validators
│   ├── data_generators.py          # Reference data generators
│   └── comparators.py              # Comparison functions
├── test_agent_single_cell.py       # Single-cell tests
├── test_agent_bulk.py              # Bulk RNA-seq tests
└── pytest.ini                      # Pytest configuration
```

## Requirements

### API Keys

Tests require an API key for one of the supported LLM providers:

```bash
# OpenAI (recommended for tests)
export OPENAI_API_KEY="your-key-here"

# Or Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# Or Google
export GEMINI_API_KEY="your-key-here"
```

Tests use cheaper models by default:
- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-haiku-3.5`
- Google: `gemini-2.5-flash`

### Reference Data

Before running tests, generate reference data:

```bash
python scripts/generate_reference_data.py
```

This creates expected outputs from tutorial notebooks for validation.

## Running Tests

### All integration tests

```bash
pytest tests/integration/agent/
```

### Quick tests only (~5-10 min)

```bash
pytest tests/integration/agent/ -m quick
```

### Full test suite (~30-60 min)

```bash
pytest tests/integration/agent/ -m full
```

### Single-cell tests only

```bash
pytest tests/integration/agent/ -m single_cell
```

### Bulk RNA-seq tests only

```bash
pytest tests/integration/agent/ -m bulk
```

### Specific test

```bash
pytest tests/integration/agent/test_agent_single_cell.py::test_agent_qc_filtering -v
```

## Test Markers

- `integration` - All integration tests (requires API keys)
- `agent` - Tests for ov.agent functionality
- `quick` - Fast subset for CI/PR checks
- `full` - Comprehensive tests for nightly runs
- `single_cell` - Single-cell workflow tests
- `bulk` - Bulk RNA-seq tests
- `spatial` - Spatial transcriptomics tests
- `skill` - Individual skill validation
- `workflow` - Multi-step workflows
- `error_handling` - Error recovery tests

## Test Coverage

### Phase 1: Foundation Tests (Implemented)

Single-cell:
- ✅ QC filtering (`test_agent_qc_filtering`)
- ✅ HVG selection (`test_agent_hvg_selection`)
- ✅ Dimensionality reduction (`test_agent_dimensionality_reduction`)
- ✅ Clustering (`test_agent_clustering`)
- ✅ Complete workflow (`test_agent_complete_pbmc3k_workflow`)

Bulk RNA-seq:
- ⚠️  DEG analysis (basic structure, needs real data)
- ⚠️  Gene ID mapping (placeholder)

### Phase 2-5: To Be Implemented

See `docs/testing/agent_integration_tests_plan.md` for full roadmap.

## Validation Strategy

Tests validate outputs using:

1. **Shape checks**: Cell/gene counts within tolerance (±5-10%)
2. **Structure checks**: Required columns/keys present
3. **Range checks**: Values within expected ranges
4. **Similarity metrics**: Compare to references (ARI, Jaccard)
5. **Statistical checks**: Proper distributions, no NaN/Inf

Tolerances account for:
- Non-deterministic algorithms (UMAP, leiden)
- LLM output variation
- Different random seeds

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on:
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # Nightly

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Generate references
        run: python scripts/generate_reference_data.py
      - name: Run quick tests
        run: pytest tests/integration/agent/ -m quick
```

## Cost Optimization

- Use `gpt-4o-mini` (~$0.15/1M tokens) for tests
- Cache reference data (no re-execution)
- Estimated cost per full run: $1-5

## Troubleshooting

### Tests Skipped

**Issue**: Tests skipped with "No API key available"

**Solution**: Set API key environment variable:
```bash
export OPENAI_API_KEY="your-key"
```

### Reference Data Not Found

**Issue**: Tests skipped with "Reference data not found"

**Solution**: Generate reference data:
```bash
python scripts/generate_reference_data.py
```

### Agent Returns Unexpected Format

**Issue**: Tests fail with "Agent result missing expected keys"

**Solution**: Agent return format may vary. Tests handle both:
- Direct AnnData: `adata = agent.run(...)`
- Dict with adata: `result['adata']` or `result['value']`

### Validation Tolerance Too Strict

**Issue**: Test fails due to minor differences

**Solution**: Adjust tolerance in validator:
```python
output_validator.validate_adata_shape(
    adata,
    expected_cells=2603,
    tolerance=0.10  # Increase from 0.05 to 0.10
)
```

## Contributing

To add new tests:

1. Identify tutorial notebook to test
2. Extract key operations and expected outputs
3. Write test in appropriate file
4. Add reference data if needed
5. Update this README with test coverage

See `docs/testing/agent_integration_tests_plan.md` for detailed guidelines.

## References

- Plan: `docs/testing/agent_integration_tests_plan.md`
- Tutorial notebooks: `omicverse_guide/docs/Tutorials-*/`
- Agent implementation: `omicverse/utils/smart_agent.py`
- Skills: `omicverse/.claude/skills/`
