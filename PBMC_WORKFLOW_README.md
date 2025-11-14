# PBMC Multi-head Clustering Workflow with Stability Analysis

## Overview

This workflow implements a comprehensive end-to-end analysis for PBMC single-cell RNA-seq data, featuring:

1. **KB-python alignment guidance** - CLI commands for processing raw 10x FASTQs
2. **PBMC-grade QC pipeline** - Rigorous quality control with documented thresholds
3. **Multi-method clustering** - Leiden, Louvain, GMM, and LDA across multiple resolutions
4. **UMAP drift metrics** - Novel stability quantification approach
5. **Comprehensive reporting** - Automated tables, plots, and markdown reports

## Skills Utilized

This workflow demonstrates the integration of multiple OmicVerse agent skills:

- `single-preprocessing`: QC/HVG/scaling guidance
- `single-clustering`: Multi-head resolution sweeps
- `data-viz-plots`: Stability diagnostics visualization
- `single-downstream-analysis`: Analysis support

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n pbmc_workflow python=3.9 -y
conda activate pbmc_workflow

# Install requirements
pip install -r pbmc_workflow_requirements.txt
```

### 2. Run the Workflow

```bash
# Execute the main workflow script
python pbmc_multihead_clustering_workflow.py
```

### 3. Expected Outputs

The workflow generates:
- `pbmc_clustering_summary.csv` - Complete results table
- `pbmc_multihead_clustering_results.h5ad` - Annotated AnnData object
- `pbmc_clustering_stability_diagnostics.png` - Visualization dashboard
- `pbmc_workflow_report.md` - Detailed markdown report

## Workflow Details

### Step 1: KB-python Alignment (for raw FASTQs)

The workflow provides CLI commands for processing raw 10x Chromium data:

```bash
# Install kb-python
pip install kb-python

# Download human reference
kb ref -d human -i index.idx -g t2g.txt

# Run alignment
kb count \\
    -i index.idx \\
    -g t2g.txt \\
    -x 10xv3 \\
    -o output_dir/ \\
    --h5ad \\
    -t 8 \\
    pbmc_R1.fastq.gz \\
    pbmc_R2.fastq.gz
```

### Step 2: QC Parameters (PBMC-optimized)

Based on `single-preprocessing` skill recommendations:

| Parameter | Threshold | Rationale |
|-----------|-----------|-----------|
| Min UMIs | 500 | Remove low-quality cells |
| Min genes | 250 | Remove empty droplets |
| Max mito % | 20% | Remove dying cells |
| HVGs | 2000 | Balance signal vs noise |

### Step 3: Multi-method Clustering

#### Leiden & Louvain
- **Resolutions tested**: 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0
- **Basis**: Graph-based community detection
- **Best for**: Well-separated cell types

#### Gaussian Mixture Models (GMM)
- **Components tested**: 5, 8, 10, 12, 15, 18, 20
- **Basis**: Probabilistic clustering on PCA space
- **Best for**: Continuous/gradient populations

#### Latent Dirichlet Allocation (LDA)
- **Topics tested**: 5, 8, 10, 12, 15, 18, 20
- **Basis**: Topic modeling on count matrix
- **Best for**: Overlapping transcriptional programs

### Step 4: UMAP Drift Stability Metric

**Novel approach to quantify clustering stability:**

```python
def compute_umap_drift(adata, cluster_key, n_bootstrap=10, subsample_frac=0.8):
    """
    UMAP Drift Metric:
    1. Subsample cells (80% of data)
    2. Recompute UMAP on subsample
    3. Measure average Euclidean distance between original and recomputed coordinates
    4. Normalize to [0, 1] range

    Lower drift → more stable/reproducible clustering
    """
```

**Interpretation:**
- **< 0.2**: Excellent stability
- **0.2 - 0.4**: Good stability
- **> 0.4**: Unstable clustering (likely overfitting)

### Step 5: Automated Recommendations

The workflow automatically identifies optimal resolutions by:
1. Computing drift for all method×resolution combinations
2. Selecting minimum drift configuration per method
3. Generating ranked recommendations table

## Example Output

### Summary Table Format

| Method | Resolution | Clusters | Drift | Recommended |
|--------|------------|----------|-------|-------------|
| Leiden | 0.8 | 9 | 0.142 | ✓ |
| Leiden | 1.0 | 11 | 0.156 | |
| Louvain | 0.5 | 8 | 0.138 | ✓ |
| GMM | 10 | 10 | 0.167 | ✓ |
| LDA | 12 | 12 | 0.183 | ✓ |

### Visualization Dashboard

The workflow generates a multi-panel figure showing:

**Panel A**: Drift across resolutions (line plot per method)
**Panel B**: Number of clusters vs drift (scatter plot)
**Panel C**: Recommended configurations (bar plot)
**Panels D-G**: UMAP embeddings for each recommended clustering

## Customization Options

### Modify Resolution Ranges

```python
# In the script, edit these lines:
resolutions = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]  # Leiden/Louvain
gmm_components = [5, 8, 10, 12, 15, 18, 20]  # GMM
lda_topics = [5, 8, 10, 12, 15, 18, 20]  # LDA
```

### Adjust QC Thresholds

```python
# Modify these filters:
adata = adata[adata.obs['total_counts'] > 500, :]  # Min UMIs
adata = adata[adata.obs['n_genes_by_counts'] > 250, :]  # Min genes
adata = adata[adata.obs['pct_counts_mt'] < 20, :]  # Max mito %
```

### Change Stability Parameters

```python
# Adjust drift computation:
drift, drift_std = compute_umap_drift(
    adata,
    key,
    n_bootstrap=10,  # Number of bootstrap iterations
    subsample_frac=0.8  # Subsample fraction (0-1)
)
```

## Performance Notes

### Computational Requirements

| Step | Time (3k cells) | Memory | Notes |
|------|----------------|--------|-------|
| QC & preprocessing | ~2 min | ~2 GB | Fast |
| Leiden/Louvain | ~1 min | ~1 GB | Per resolution |
| GMM | ~5 min | ~2 GB | Per component |
| LDA | ~10 min | ~4 GB | Per topic |
| UMAP drift | ~15 min | ~3 GB | Parallelizable |

### Optimization Tips

1. **Reduce bootstrap iterations** for faster drift computation:
   ```python
   n_bootstrap=5  # Instead of 10
   ```

2. **Use fewer resolutions** for initial exploration:
   ```python
   resolutions = [0.5, 1.0, 1.5]  # Instead of 7 values
   ```

3. **Parallelize LDA** with more workers:
   ```python
   lda = LatentDirichletAllocation(..., n_jobs=-1)  # Use all cores
   ```

## Interpretation Guidelines

### 1. Stability vs Granularity Trade-off

- **Low resolution + low drift**: Robust major cell types
- **High resolution + low drift**: Fine-grained subpopulations
- **High resolution + high drift**: Likely overfitting

### 2. Cross-method Validation

If multiple methods agree on cluster count at similar drift levels, this suggests:
- Strong biological signal
- Well-separated populations
- Robust clustering structure

### 3. Method-specific Considerations

**Leiden/Louvain:**
- Similar results expected (Leiden is improved Louvain)
- Differences indicate graph structure sensitivity

**GMM vs Graph-based:**
- GMM better for continuous gradients
- Graph methods better for discrete types

**LDA:**
- Higher drift typical (topics are overlapping)
- Good for identifying transcriptional programs

## Troubleshooting

### Issue: High drift across all methods

**Possible causes:**
- Insufficient preprocessing (batch effects)
- Low-quality data (doublets, ambient RNA)
- Heterogeneous population (no clear structure)

**Solutions:**
1. Apply batch correction
2. Stricter QC filtering
3. Consider continuous representations (e.g., diffusion maps)

### Issue: Inconsistent cluster counts across methods

**Expected behavior** - different methods capture different aspects:
- Leiden/Louvain: Graph connectivity
- GMM: Gaussian components
- LDA: Transcriptional programs

**Action:** Cross-validate with marker genes and known biology

### Issue: Memory errors during LDA

**Solution:** Reduce dataset size or use streaming LDA:
```python
lda = LatentDirichletAllocation(
    ...,
    learning_method='online',  # Streaming mode
    batch_size=128  # Process in batches
)
```

## References

### Skills Documentation
- [single-preprocessing](../../.claude/skills/single-preprocessing/)
- [single-clustering](../../.claude/skills/single-clustering/)
- [data-viz-plots](../../.claude/skills/data-viz-plots/)
- [single-downstream-analysis](../../.claude/skills/single-downstream-analysis/)

### Key Papers
- **Leiden algorithm**: Traag et al., Scientific Reports, 2019
- **LDA for scRNA-seq**: Stein-O'Brien et al., Cell Systems, 2019
- **kb-python**: Melsted et al., Nature Biotechnology, 2021

### OmicVerse Documentation
- https://omicverse.readthedocs.io/

## Citation

If you use this workflow, please cite:

```bibtex
@software{pbmc_multihead_workflow,
  title={PBMC Multi-head Clustering Workflow with Stability Analysis},
  author={OmicVerse Agent},
  year={2025},
  note={Skills: single-preprocessing, single-clustering, data-viz-plots, single-downstream-analysis}
}
```

## License

MIT License - Feel free to adapt for your research!

## Support

For issues or questions:
- Check OmicVerse documentation
- Review skill prompts in `.claude/skills/`
- Open issue on GitHub

---

**Workflow Version**: 1.0
**Last Updated**: 2025-11-14
**OmicVerse Agent**: Priority 2 (comprehensive skills-guided workflow)
