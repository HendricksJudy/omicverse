# Production Deployment Checklist

Use this checklist to deploy agent integration tests to production CI/CD.

## Pre-Deployment

### ✅ 1. Environment Setup

- [ ] Python 3.8+ installed
- [ ] OmicVerse installed in development mode (`pip install -e .`)
- [ ] pytest and pytest-asyncio installed
- [ ] scanpy, numpy, pandas available
- [ ] API key obtained (OpenAI, Anthropic, or Google)

### ✅ 2. Reference Data

- [ ] Run `python scripts/generate_reference_data.py`
- [ ] Verify data exists: `ls tests/integration/agent/data/pbmc3k/`
- [ ] Check data files:
  - [ ] `qc.h5ad` (QC-filtered PBMC3k)
  - [ ] `preprocessed.h5ad` (HVG-selected)
  - [ ] `clustered.h5ad` (with Leiden + UMAP)
  - [ ] `reference_metrics.json` (expected metrics)

### ✅ 3. Local Test Run

- [ ] Set API key: `export OPENAI_API_KEY="..."`
- [ ] Run quick tests: `pytest tests/integration/agent/ -m quick -v`
- [ ] Verify 5+ tests pass
- [ ] Check execution time (<15 minutes)
- [ ] Review cost (should be <$1)

## CI/CD Setup

### ✅ 4. GitHub Secrets

- [ ] Navigate to: `Settings → Secrets and variables → Actions`
- [ ] Add secret: `OPENAI_API_KEY` (or alternative)
- [ ] Verify secret is available in organization/repository

### ✅ 5. Workflow Configuration

- [ ] Copy template: `cp .github/workflows/agent-integration-tests.yml.template .github/workflows/agent-integration-tests.yml`
- [ ] Review workflow triggers:
  - [ ] Pull request trigger configured
  - [ ] Nightly schedule set (or disabled if not needed)
  - [ ] Manual dispatch enabled
- [ ] Adjust timeouts if needed
- [ ] Configure notification preferences

### ✅ 6. Caching Strategy

- [ ] pip cache enabled (reduces install time)
- [ ] Reference data cache enabled (prevents regeneration)
- [ ] Cache key configured correctly

## Initial Deployment

### ✅ 7. First Commit

```bash
# Add workflow file
git add .github/workflows/agent-integration-tests.yml
git commit -m "Add agent integration tests CI/CD"
git push
```

### ✅ 8. Verify Workflow

- [ ] Go to: `Actions` tab in GitHub
- [ ] Verify workflow appears
- [ ] Manually trigger a test run
- [ ] Monitor execution
- [ ] Check logs for errors
- [ ] Verify tests complete successfully

### ✅ 9. Test on Pull Request

- [ ] Create a test PR
- [ ] Verify workflow runs automatically
- [ ] Check quick tests pass
- [ ] Review execution time and cost
- [ ] Confirm PR comment added (if configured)

## Monitoring & Maintenance

### ✅ 10. Cost Monitoring

- [ ] Set up billing alerts in API provider dashboard
- [ ] Monitor monthly costs
- [ ] Adjust test frequency if needed
- [ ] Consider model downgrade if costs high

**Cost targets** (with gpt-4o-mini):
- Quick tests per PR: $0.10-0.50
- Nightly full tests: $1-5
- Monthly total: <$150

### ✅ 11. Test Maintenance

- [ ] Review test failures weekly
- [ ] Update reference data quarterly
- [ ] Add new tests for new features
- [ ] Archive obsolete tests

### ✅ 12. Documentation

- [ ] Share QUICKSTART.md with team
- [ ] Document any custom configurations
- [ ] Update troubleshooting guide
- [ ] Create runbook for common issues

## Optimization

### ✅ 13. Performance

- [ ] Enable parallel test execution if possible
- [ ] Use faster models for simple tests
- [ ] Cache more aggressively
- [ ] Skip tests on non-agent file changes

### ✅ 14. Coverage

- [ ] Review skill coverage report quarterly
- [ ] Add tests for untested skills
- [ ] Prioritize high-value skills
- [ ] Target 60%+ skill coverage

### ✅ 15. Integration

- [ ] Integrate with existing CI/CD
- [ ] Add to project dashboard
- [ ] Include in release process
- [ ] Document in contribution guidelines

## Advanced Configuration

### ✅ 16. Multi-Provider Testing

- [ ] Test with multiple LLM providers
- [ ] Compare results across providers
- [ ] Document provider-specific issues
- [ ] Set up fallback providers

### ✅ 17. Custom Notifications

```yaml
# Example: Slack notification
- name: Notify on Slack
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "Agent tests ${{ job.status }}"
      }
```

### ✅ 18. Test Matrix

```yaml
# Example: Test multiple Python versions
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
```

## Troubleshooting

### Common Issues

**Workflow not triggering**:
- Check file paths in workflow `on.paths`
- Verify branch names match
- Ensure secrets are set

**Tests failing in CI but passing locally**:
- Check Python version matches
- Verify all dependencies installed
- Review environment differences
- Check reference data generation

**API rate limits**:
- Reduce test frequency
- Add retry logic with backoff
- Use multiple API keys
- Cache results more aggressively

**High costs**:
- Switch to cheaper model (gpt-4o-mini)
- Reduce nightly test frequency
- Skip full tests on small PRs
- Implement smarter test selection

## Sign-Off

### Final Checks Before Production

- [ ] All quick tests pass
- [ ] All full tests pass
- [ ] Reference data generated
- [ ] CI/CD workflow configured
- [ ] API keys secured
- [ ] Team trained on troubleshooting
- [ ] Documentation complete
- [ ] Cost monitoring in place

### Approval

**Deployed by**: _________________

**Date**: _________________

**Approved by**: _________________

**Notes**: ___________________________________________________________

---

## Quick Reference Commands

```bash
# Local testing
export OPENAI_API_KEY="sk-..."
python scripts/generate_reference_data.py
pytest tests/integration/agent/ -m quick -v

# CI/CD setup
cp .github/workflows/agent-integration-tests.yml.template \
   .github/workflows/agent-integration-tests.yml
git add .github/workflows/agent-integration-tests.yml
git commit -m "Add agent integration tests CI/CD"
git push

# Monitoring
python tests/integration/agent/utils/skill_coverage.py
pytest tests/integration/agent/ -v --tb=short

# Maintenance
python scripts/generate_reference_data.py --force
pytest tests/integration/agent/ -v --lf  # Re-run last failures
```

## Support

- **Documentation**: `tests/integration/agent/README.md`
- **Quick Start**: `tests/integration/agent/QUICKSTART.md`
- **Full Plan**: `docs/testing/agent_integration_tests_plan.md`
- **Issues**: https://github.com/HendricksJudy/omicverse/issues
