# PBMC Multi-head Clustering - Quick Reference

## 🚀 One-Command Setup

```bash
# Install and run
conda create -n pbmc python=3.9 -y
conda activate pbmc
pip install numpy pandas scanpy omicverse matplotlib seaborn scikit-learn
python pbmc_multihead_clustering_workflow.py
```

## 📋 Code Blocks

### 1. KB-python Alignment (Raw FASTQs → AnnData)

```bash
pip install kb-python
kb ref -d human -i index.idx -g t2g.txt

kb count \
    -i index.idx \
    -g t2g.txt \
    -x 10xv3 \
    -o output/ \
    --h5ad \
    -t 8 \
    pbmc_R1.fastq.gz pbmc_R2.fastq.gz
```

### 2. QC & Preprocessing

```python
import scanpy as sc
import omicverse as ov

# Load data
adata = sc.read_h5ad('output/counts_unfiltered/adata.h5ad')

# QC metrics
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

# Filter (PBMC thresholds)
adata = adata[adata.obs['total_counts'] > 500, :]
adata = adata[adata.obs['n_genes_by_counts'] > 250, :]
adata = adata[adata.obs['pct_counts_mt'] < 20, :]

# Store raw
adata.layers['counts'] = adata.X.copy()

# Normalize & HVG
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer='counts')
adata = adata[:, adata.var['highly_variable']]

# Scale & PCA
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50)

# Neighbors & UMAP
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata)
```

### 3. Multi-method Clustering

```python
# Leiden
resolutions = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')

# Louvain
for res in resolutions:
    sc.tl.louvain(adata, resolution=res, key_added=f'louvain_r{res}')

# GMM
from sklearn.mixture import GaussianMixture
components = [5, 8, 10, 12, 15, 18, 20]
for n_comp in components:
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(adata.obsm['X_pca'])
    adata.obs[f'gmm_k{n_comp}'] = pd.Categorical(labels.astype(str))

# LDA
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

count_matrix = adata.layers['counts'].toarray() if hasattr(adata.layers['counts'], 'toarray') else adata.layers['counts']
count_matrix = np.maximum(count_matrix, 0)

topics = [5, 8, 10, 12, 15, 18, 20]
for n_topics in topics:
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, random_state=42, n_jobs=-1)
    topic_dist = lda.fit_transform(count_matrix)
    labels = np.argmax(topic_dist, axis=1)
    adata.obs[f'lda_k{n_topics}'] = pd.Categorical(labels.astype(str))
```

### 4. UMAP Drift Stability Metric

```python
def compute_umap_drift(adata, cluster_key, n_bootstrap=10, subsample_frac=0.8):
    """
    Quantify clustering stability via UMAP coordinate drift.

    Returns:
        drift (float): Normalized drift score [0-1], lower is better
        drift_std (float): Standard deviation across bootstraps
    """
    original_umap = adata.obsm['X_umap'].copy()
    n_cells = adata.n_obs
    n_subsample = int(n_cells * subsample_frac)

    drifts = []
    for i in range(n_bootstrap):
        # Subsample
        np.random.seed(42 + i)
        subsample_idx = np.random.choice(n_cells, size=n_subsample, replace=False)
        adata_sub = adata[subsample_idx, :].copy()

        # Recompute UMAP
        sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata_sub)

        # Measure drift
        original_coords = original_umap[subsample_idx]
        new_coords = adata_sub.obsm['X_umap']
        distances = np.sqrt(np.sum((original_coords - new_coords)**2, axis=1))
        drifts.append(np.mean(distances))

    avg_drift = np.mean(drifts)
    std_drift = np.std(drifts)
    normalized_drift = min(avg_drift / 20.0, 1.0)  # Normalize to [0, 1]

    return normalized_drift, std_drift

# Compute for all clustering results
clustering_keys = [f'leiden_r{r}' for r in resolutions] + \
                  [f'louvain_r{r}' for r in resolutions] + \
                  [f'gmm_k{k}' for k in components] + \
                  [f'lda_k{k}' for k in topics]

results = []
for key in clustering_keys:
    drift, drift_std = compute_umap_drift(adata, key, n_bootstrap=10)
    results.append({'key': key, 'drift': drift, 'drift_std': drift_std})
```

### 5. Summary Table

```python
import pandas as pd

# Create summary DataFrame
summary_data = []
for key in clustering_keys:
    method = key.split('_')[0]
    resolution = float(key.split('r')[-1]) if 'r' in key else int(key.split('k')[-1])
    n_clusters = adata.obs[key].nunique()

    # Get drift from results
    drift_info = next((r for r in results if r['key'] == key), None)
    drift = drift_info['drift'] if drift_info else None

    summary_data.append({
        'Method': method.capitalize(),
        'Resolution': resolution,
        'N_Clusters': n_clusters,
        'Drift': drift
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(['Method', 'Drift'])

# Find recommended (lowest drift per method)
recommendations = {}
for method in summary_df['Method'].unique():
    method_data = summary_df[summary_df['Method'] == method]
    best_idx = method_data['Drift'].idxmin()
    recommendations[method] = method_data.loc[best_idx]

# Display
print("\n" + "="*80)
print("CLUSTERING STABILITY SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))
print("\nRECOMMENDED CONFIGURATIONS:")
for method, rec in recommendations.items():
    print(f"  {method}: res={rec['Resolution']}, clusters={rec['N_Clusters']}, drift={rec['Drift']:.3f}")
```

### 6. Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Drift by resolution
ax = axes[0, 0]
for method in summary_df['Method'].unique():
    method_data = summary_df[summary_df['Method'] == method]
    ax.plot(method_data['Resolution'], method_data['Drift'],
            marker='o', label=method, linewidth=2)
ax.set_xlabel('Resolution / Components')
ax.set_ylabel('UMAP Drift')
ax.set_title('Clustering Stability Across Resolutions')
ax.legend()
ax.grid(alpha=0.3)

# Panel 2: Clusters vs drift
ax = axes[0, 1]
colors = {'Leiden': '#E74C3C', 'Louvain': '#3498DB', 'Gmm': '#2ECC71', 'Lda': '#F39C12'}
for method in summary_df['Method'].unique():
    method_data = summary_df[summary_df['Method'] == method]
    ax.scatter(method_data['N_Clusters'], method_data['Drift'],
               c=colors.get(method, 'gray'), label=method, s=80, alpha=0.7)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('UMAP Drift')
ax.set_title('Clusters vs Stability')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3 & 4: UMAP with best Leiden and Louvain
for idx, method in enumerate(['Leiden', 'Louvain']):
    ax = axes[1, idx]
    rec = recommendations[method]
    key = f"{method.lower()}_r{rec['Resolution']}"

    sc.pl.umap(adata, color=key, ax=ax, show=False)
    ax.set_title(f'{method} (res={rec["Resolution"]}, drift={rec["Drift"]:.3f})')

plt.tight_layout()
plt.savefig('clustering_stability.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 📊 Output Files

```
pbmc_clustering_summary.csv              # Complete results table
pbmc_multihead_clustering_results.h5ad   # Annotated AnnData
pbmc_clustering_stability_diagnostics.png # Visualization dashboard
pbmc_workflow_report.md                   # Detailed report
```

## 🔑 Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Min UMIs | 500 | Filter low-quality cells |
| Min genes | 250 | Remove empty droplets |
| Max mito % | 20% | Remove dying cells |
| HVGs | 2000 | Balance signal vs noise |
| PCA comps | 50 | Capture variance |
| Neighbors | 15 | Graph construction |
| Bootstrap | 10 | Drift estimation |

## 📈 Drift Interpretation

| Drift Score | Stability | Action |
|-------------|-----------|--------|
| < 0.2 | Excellent | Use with confidence |
| 0.2 - 0.4 | Good | Validate with markers |
| > 0.4 | Unstable | Lower resolution or improve QC |

## 🎯 Method Selection Guide

| Method | Best For | Advantages | Limitations |
|--------|----------|------------|-------------|
| **Leiden** | Discrete cell types | Fast, interpretable | Requires clear separation |
| **Louvain** | General clustering | Widely used | Less optimal than Leiden |
| **GMM** | Continuous gradients | Probabilistic | Assumes Gaussian distributions |
| **LDA** | Transcriptional programs | Captures overlaps | Higher computational cost |

## ⚡ Performance Tips

**Speed up drift computation:**
```python
# Reduce bootstraps
drift, _ = compute_umap_drift(adata, key, n_bootstrap=5)

# Or test fewer resolutions
resolutions = [0.5, 1.0, 1.5]  # Instead of 7 values
```

**Reduce memory for LDA:**
```python
lda = LatentDirichletAllocation(
    ...,
    learning_method='online',
    batch_size=128
)
```

## 🔧 Common Fixes

**Error: "No module named 'leidenalg'"**
```bash
pip install leidenalg python-igraph
```

**Warning: "Convergence warning in GMM"**
```python
# Increase iterations
gmm = GaussianMixture(..., max_iter=1000)
```

**High memory usage:**
```python
# Work with subset
adata_sub = adata[np.random.choice(adata.n_obs, 2000, replace=False), :]
```

---

**Full documentation**: See `PBMC_WORKFLOW_README.md`
**Complete script**: `pbmc_multihead_clustering_workflow.py`
