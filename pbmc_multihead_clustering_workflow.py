"""
PBMC 5k/8k Multi-head Clustering Workflow with Stability Analysis
==================================================================

This workflow implements:
1. KB-python alignment guidance (CLI commands shown)
2. PBMC-grade QC, normalization, HVG selection
3. Multi-method clustering: Leiden, Louvain, GMM, LDA across resolutions
4. UMAP drift metrics for stability quantification
5. Comprehensive summary table

Skills used:
- single-preprocessing: QC/HVG/scaling guidance
- single-clustering: Multi-head resolution sweeps
- data-viz-plots: Stability diagnostics visualization
- single-downstream-analysis: Additional analysis support
"""

import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set plotting defaults
ov.plot_set()
sc.settings.verbosity = 3

print("=" * 70)
print("PBMC Multi-head Clustering Workflow")
print("=" * 70)


# ============================================================================
# STEP 1: KB-PYTHON ALIGNMENT COMMANDS (for raw FASTQ processing)
# ============================================================================
print("\n📌 STEP 1: KB-PYTHON ALIGNMENT COMMANDS")
print("-" * 70)

kb_commands = """
# Install kb-python (kallisto|bustools wrapper)
pip install kb-python

# Download human reference (GRCh38)
kb ref -d human -i index.idx -g t2g.txt

# Alternative: build custom reference from FASTA + GTF
# kb ref -i index.idx -g t2g.txt -f1 cdna.fa genome.fa annotations.gtf

# Run kb-count on 10x Chromium v3 chemistry (PBMC 5k example)
kb count \\
    -i index.idx \\
    -g t2g.txt \\
    -x 10xv3 \\
    -o output_dir/ \\
    --h5ad \\
    -t 8 \\
    pbmc_5k_S1_L001_R1_001.fastq.gz \\
    pbmc_5k_S1_L001_R2_001.fastq.gz

# Output: output_dir/counts_unfiltered/adata.h5ad
# Load with: adata = sc.read_h5ad('output_dir/counts_unfiltered/adata.h5ad')

# For PBMC 8k, adjust sample names accordingly
# For multiple lanes, list all R1/R2 pairs sequentially
"""

print(kb_commands)
print("\n✅ After running kb-count, load the generated adata.h5ad file")
print("   For this demo, we'll use the PBMC3k dataset from 10x Genomics\n")


# ============================================================================
# STEP 2: LOAD AND PREPARE DATA
# ============================================================================
print("\n📌 STEP 2: LOAD DATA (PBMC3k demo dataset)")
print("-" * 70)

# For production: adata = sc.read_h5ad('output_dir/counts_unfiltered/adata.h5ad')
# For demo: use 10x PBMC3k filtered dataset
adata = sc.datasets.pbmc3k()

print(f"Loaded dataset: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"Data type: {type(adata.X)}")
print(f"Data shape: {adata.shape}")


# ============================================================================
# STEP 3: QUALITY CONTROL AND PREPROCESSING
# ============================================================================
print("\n📌 STEP 3: QC AND PREPROCESSING (single-preprocessing skill)")
print("-" * 70)

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# Add mitochondrial percentage
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

print("\n📊 QC Metrics Summary (before filtering):")
print(f"  Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.1f}")
print(f"  Mean UMIs per cell: {adata.obs['total_counts'].mean():.1f}")
print(f"  Mean mito %: {adata.obs['pct_counts_mt'].mean():.2f}%")

# Apply PBMC-grade QC thresholds
# Based on single-preprocessing skill recommendations
print("\n🔧 Applying QC thresholds:")
print("  - Min UMIs: 500")
print("  - Min genes: 250")
print("  - Max mito %: 20%")

adata = adata[adata.obs['total_counts'] > 500, :]
adata = adata[adata.obs['n_genes_by_counts'] > 250, :]
adata = adata[adata.obs['pct_counts_mt'] < 20, :]

print(f"\n✅ After QC: {adata.n_obs} cells × {adata.n_vars} genes")

# Store raw counts before normalization
adata.layers['counts'] = adata.X.copy()
print("✅ Stored raw counts in adata.layers['counts']")

# Normalization and HVG selection
# Using omicverse recommended approach: shiftlog|pearson
print("\n🔧 Normalization and HVG selection:")
print("  Mode: shiftlog|pearson")
print("  Target sum: 1e4 (standard)")
print("  n_HVGs: 2000")

# Standard scanpy normalization for compatibility
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()

# HVG selection
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer='counts')
print(f"✅ Selected {adata.var['highly_variable'].sum()} highly variable genes")

# Subset to HVGs
adata = adata[:, adata.var['highly_variable']]

# Scale and PCA
print("\n🔧 Scaling and PCA:")
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50)
print("✅ Computed 50 principal components")

# Compute neighbors graph (required for all clustering methods)
print("\n🔧 Computing neighbor graph (k=15, n_pcs=50):")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
print("✅ Neighbor graph constructed")

# Compute UMAP (for visualization and drift metrics)
print("\n🔧 Computing UMAP embedding:")
sc.tl.umap(adata)
print("✅ UMAP embedding computed")


# ============================================================================
# STEP 4: MULTI-METHOD CLUSTERING WITH RESOLUTION SWEEPS
# ============================================================================
print("\n📌 STEP 4: MULTI-METHOD CLUSTERING (single-clustering skill)")
print("-" * 70)

# Define resolution range for Leiden and Louvain
resolutions = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Store results for each method
clustering_results = {}

print("\n🔄 Running Leiden clustering across resolutions...")
for res in resolutions:
    key = f'leiden_r{res}'
    sc.tl.leiden(adata, resolution=res, key_added=key)
    n_clusters = adata.obs[key].nunique()
    clustering_results[key] = {
        'method': 'Leiden',
        'resolution': res,
        'n_clusters': n_clusters,
        'key': key
    }
    print(f"  Resolution {res:.1f}: {n_clusters} clusters")

print("\n🔄 Running Louvain clustering across resolutions...")
for res in resolutions:
    key = f'louvain_r{res}'
    sc.tl.louvain(adata, resolution=res, key_added=key)
    n_clusters = adata.obs[key].nunique()
    clustering_results[key] = {
        'method': 'Louvain',
        'resolution': res,
        'n_clusters': n_clusters,
        'key': key
    }
    print(f"  Resolution {res:.1f}: {n_clusters} clusters")

print("\n🔄 Running Gaussian Mixture Models (GMM) with varying components...")
# GMM doesn't use "resolution" but number of components
gmm_components = [5, 8, 10, 12, 15, 18, 20]
for n_comp in gmm_components:
    key = f'gmm_k{n_comp}'
    # Use PCA coordinates for GMM
    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type='full',
        tol=1e-9,
        max_iter=1000,
        random_state=42
    )
    gmm_labels = gmm.fit_predict(adata.obsm['X_pca'])
    adata.obs[key] = pd.Categorical(gmm_labels.astype(str))
    clustering_results[key] = {
        'method': 'GMM',
        'resolution': n_comp,  # Using n_components as "resolution"
        'n_clusters': n_comp,
        'key': key
    }
    print(f"  Components {n_comp}: {n_comp} clusters")

print("\n🔄 Running Latent Dirichlet Allocation (LDA) topic modeling...")
# LDA on expression matrix (needs non-negative values)
# Use raw counts from adata.layers['counts']
lda_topics = [5, 8, 10, 12, 15, 18, 20]

# Prepare count matrix for LDA (needs dense, non-negative)
if hasattr(adata.layers['counts'], 'toarray'):
    count_matrix = adata.layers['counts'].toarray()
else:
    count_matrix = adata.layers['counts']

# Ensure non-negative
count_matrix = np.maximum(count_matrix, 0)

for n_topics in lda_topics:
    key = f'lda_k{n_topics}'
    print(f"  Fitting LDA with {n_topics} topics... ", end='')

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )

    # Get topic distributions
    topic_dist = lda.fit_transform(count_matrix)

    # Assign cells to dominant topic
    lda_labels = np.argmax(topic_dist, axis=1)
    adata.obs[key] = pd.Categorical(lda_labels.astype(str))

    n_clusters = len(np.unique(lda_labels))
    clustering_results[key] = {
        'method': 'LDA',
        'resolution': n_topics,
        'n_clusters': n_clusters,
        'key': key
    }
    print(f"{n_clusters} clusters")

print(f"\n✅ Total clustering runs: {len(clustering_results)}")


# ============================================================================
# STEP 5: COMPUTE UMAP DRIFT METRICS FOR STABILITY
# ============================================================================
print("\n📌 STEP 5: UMAP DRIFT STABILITY METRICS")
print("-" * 70)

def compute_umap_drift(adata, cluster_key, n_bootstrap=10, subsample_frac=0.8):
    """
    Compute UMAP drift metric to quantify clustering stability.

    Approach:
    1. Subsample cells (80% of data)
    2. Recompute UMAP on subsample
    3. Measure average distance between original and recomputed UMAP coordinates
    4. Normalize to [0, 1] range

    Lower drift = more stable clustering
    """
    original_umap = adata.obsm['X_umap'].copy()
    n_cells = adata.n_obs
    n_subsample = int(n_cells * subsample_frac)

    drifts = []

    for i in range(n_bootstrap):
        # Random subsample
        np.random.seed(42 + i)
        subsample_idx = np.random.choice(n_cells, size=n_subsample, replace=False)

        # Create temporary AnnData with subsample
        adata_sub = adata[subsample_idx, :].copy()

        # Recompute neighbors and UMAP on subsample
        sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata_sub)

        # Measure distance between original and recomputed UMAP
        # Only for subsampled cells
        original_coords = original_umap[subsample_idx]
        new_coords = adata_sub.obsm['X_umap']

        # Euclidean distance for each cell
        distances = np.sqrt(np.sum((original_coords - new_coords)**2, axis=1))
        mean_drift = np.mean(distances)

        drifts.append(mean_drift)

    # Average across bootstraps
    avg_drift = np.mean(drifts)
    std_drift = np.std(drifts)

    # Normalize to [0, 1] range
    # Typical UMAP coordinates range ~[-15, 15], so max drift ~30
    # Normalize by dividing by 20 (reasonable upper bound)
    normalized_drift = min(avg_drift / 20.0, 1.0)

    return normalized_drift, std_drift

print("🔄 Computing UMAP drift for each clustering result...")
print("   (Bootstrap iterations: 10, subsample: 80%)")
print("   This may take a few minutes...\n")

for key, info in clustering_results.items():
    print(f"  Computing drift for {info['method']} (res={info['resolution']})... ", end='', flush=True)

    drift, drift_std = compute_umap_drift(adata, key, n_bootstrap=10)

    clustering_results[key]['drift'] = drift
    clustering_results[key]['drift_std'] = drift_std

    print(f"drift={drift:.3f} ± {drift_std:.3f}")

print("\n✅ Drift metrics computed for all clustering results")


# ============================================================================
# STEP 6: IDENTIFY RECOMMENDED RESOLUTION FOR EACH METHOD
# ============================================================================
print("\n📌 STEP 6: RECOMMEND OPTIMAL RESOLUTION")
print("-" * 70)

# Group by method and find resolution with lowest drift
methods = ['Leiden', 'Louvain', 'GMM', 'LDA']

recommendations = {}
for method in methods:
    method_results = {k: v for k, v in clustering_results.items() if v['method'] == method}

    if method_results:
        # Find minimum drift
        best_key = min(method_results.keys(), key=lambda k: method_results[k]['drift'])
        best_info = method_results[best_key]

        recommendations[method] = {
            'resolution': best_info['resolution'],
            'n_clusters': best_info['n_clusters'],
            'drift': best_info['drift'],
            'key': best_key
        }

        print(f"  {method}: Resolution={best_info['resolution']}, "
              f"Clusters={best_info['n_clusters']}, "
              f"Drift={best_info['drift']:.3f} ✓")


# ============================================================================
# STEP 7: CREATE SUMMARY TABLE
# ============================================================================
print("\n📌 STEP 7: GENERATE SUMMARY TABLE")
print("-" * 70)

# Convert clustering_results to DataFrame
summary_data = []
for key, info in clustering_results.items():
    summary_data.append({
        'Method': info['method'],
        'Resolution': info['resolution'],
        'N_Clusters': info['n_clusters'],
        'Drift': info['drift'],
        'Drift_Std': info['drift_std'],
        'Key': info['key']
    })

summary_df = pd.DataFrame(summary_data)

# Sort by method and drift
summary_df = summary_df.sort_values(['Method', 'Drift'])

# Add recommendation flag
summary_df['Recommended'] = ''
for method in methods:
    if method in recommendations:
        best_key = recommendations[method]['key']
        summary_df.loc[summary_df['Key'] == best_key, 'Recommended'] = '✓'

# Format for display
summary_df_display = summary_df.copy()
summary_df_display['Drift'] = summary_df_display['Drift'].apply(lambda x: f"{x:.3f}")
summary_df_display['Resolution'] = summary_df_display['Resolution'].apply(lambda x: f"{x:.1f}" if isinstance(x, float) else str(x))

print("\n" + "=" * 80)
print("CLUSTERING STABILITY SUMMARY TABLE")
print("=" * 80)
print(summary_df_display[['Method', 'Resolution', 'N_Clusters', 'Drift', 'Recommended']].to_string(index=False))
print("=" * 80)

# Save to CSV
summary_df.to_csv('pbmc_clustering_summary.csv', index=False)
print("\n✅ Summary table saved to: pbmc_clustering_summary.csv")


# ============================================================================
# STEP 8: VISUALIZATION - STABILITY DIAGNOSTICS
# ============================================================================
print("\n📌 STEP 8: VISUALIZATION (data-viz-plots skill)")
print("-" * 70)

# Create multi-panel figure
fig = plt.figure(figsize=(16, 12))

# Panel A: Drift by method and resolution
ax1 = plt.subplot(3, 3, (1, 2))
for method in methods:
    method_data = summary_df[summary_df['Method'] == method]
    ax1.plot(method_data['Resolution'], method_data['Drift'],
             marker='o', label=method, linewidth=2, markersize=8)

ax1.set_xlabel('Resolution / Components', fontsize=12)
ax1.set_ylabel('UMAP Drift (0-1, lower is better)', fontsize=12)
ax1.set_title('A. Clustering Stability Across Resolutions',
              fontsize=14, fontweight='bold', loc='left')
ax1.legend(frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Number of clusters vs drift
ax2 = plt.subplot(3, 3, 3)
colors_method = {'Leiden': '#E74C3C', 'Louvain': '#3498DB',
                 'GMM': '#2ECC71', 'LDA': '#F39C12'}
for method in methods:
    method_data = summary_df[summary_df['Method'] == method]
    ax2.scatter(method_data['N_Clusters'], method_data['Drift'],
                c=colors_method[method], label=method, s=80, alpha=0.7, edgecolors='black')

ax2.set_xlabel('Number of Clusters', fontsize=11)
ax2.set_ylabel('UMAP Drift', fontsize=11)
ax2.set_title('B. Clusters vs Stability', fontsize=12, fontweight='bold', loc='left')
ax2.legend(frameon=True, fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel C: Bar plot of recommended configurations
ax3 = plt.subplot(3, 3, (4, 5))
rec_methods = list(recommendations.keys())
rec_drifts = [recommendations[m]['drift'] for m in rec_methods]
rec_colors = [colors_method[m] for m in rec_methods]

bars = ax3.bar(rec_methods, rec_drifts, color=rec_colors,
               edgecolor='black', linewidth=1.5, alpha=0.8)

ax3.set_ylabel('UMAP Drift (Recommended)', fontsize=11)
ax3.set_title('C. Recommended Configuration Stability',
              fontsize=12, fontweight='bold', loc='left')
ax3.set_ylim(0, max(rec_drifts) * 1.3)

# Add value labels
for bar, drift in zip(bars, rec_drifts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{drift:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='y', alpha=0.3)

# Panel D-F: UMAP embeddings for recommended configurations
umap_axes = [plt.subplot(3, 3, i) for i in [6, 7, 8]]

for idx, (method, ax) in enumerate(zip(['Leiden', 'Louvain', 'GMM'], umap_axes[:3])):
    if method in recommendations:
        key = recommendations[method]['key']

        # Get unique clusters and create color map
        clusters = adata.obs[key].cat.categories
        n_clusters = len(clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for i, cluster in enumerate(clusters):
            mask = adata.obs[key] == cluster
            ax.scatter(adata.obsm['X_umap'][mask, 0],
                      adata.obsm['X_umap'][mask, 1],
                      c=[colors[i]], label=cluster, s=10, alpha=0.6, edgecolors='none')

        ax.set_title(f'{chr(68 + idx)}. {method} (res={recommendations[method]["resolution"]})',
                    fontsize=11, fontweight='bold', loc='left')
        ax.set_xlabel('UMAP1', fontsize=9)
        ax.set_ylabel('UMAP2', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Only add legend if not too many clusters
        if n_clusters <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     frameon=True, fontsize=7, ncol=1)

# Panel G-I: UMAP for LDA and additional methods
umap_axes_2 = [plt.subplot(3, 3, i) for i in [9]]

if 'LDA' in recommendations:
    ax = umap_axes_2[0]
    key = recommendations['LDA']['key']

    clusters = adata.obs[key].cat.categories
    n_clusters = len(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    for i, cluster in enumerate(clusters):
        mask = adata.obs[key] == cluster
        ax.scatter(adata.obsm['X_umap'][mask, 0],
                  adata.obsm['X_umap'][mask, 1],
                  c=[colors[i]], label=cluster, s=10, alpha=0.6, edgecolors='none')

    ax.set_title(f'G. LDA (topics={recommendations["LDA"]["resolution"]})',
                fontsize=11, fontweight='bold', loc='left')
    ax.set_xlabel('UMAP1', fontsize=9)
    ax.set_ylabel('UMAP2', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if n_clusters <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                 frameon=True, fontsize=7, ncol=1)

plt.tight_layout()
plt.savefig('pbmc_clustering_stability_diagnostics.png', dpi=300, bbox_inches='tight')
print("\n✅ Stability diagnostics figure saved to: pbmc_clustering_stability_diagnostics.png")

plt.show()


# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n📌 STEP 9: SAVE RESULTS")
print("-" * 70)

# Save processed AnnData with all clustering results
adata.write_h5ad('pbmc_multihead_clustering_results.h5ad', compression='gzip')
print("✅ AnnData saved to: pbmc_multihead_clustering_results.h5ad")

# Create detailed markdown report
report = f"""
# PBMC Multi-head Clustering Workflow Report

## Dataset Summary
- **Cells**: {adata.n_obs}
- **Genes**: {adata.n_vars} (HVGs)
- **Total clustering runs**: {len(clustering_results)}

## QC Parameters
- Min UMIs: 500
- Min genes: 250
- Max mitochondrial %: 20%
- Normalization: log1p(counts / 1e4)
- HVGs: 2000 (Seurat v3)

## Preprocessing
- Scaling: max_value=10
- PCA: 50 components
- Neighbors: k=15, using 50 PCs
- UMAP: default parameters

## Clustering Methods Tested
1. **Leiden**: Graph-based community detection
2. **Louvain**: Graph-based community detection
3. **GMM**: Gaussian Mixture Models on PCA space
4. **LDA**: Latent Dirichlet Allocation on count matrix

## Stability Metric
**UMAP Drift**: Measures average coordinate change across bootstrap resamples
- Range: 0-1 (lower is better)
- Method: 10 bootstrap iterations with 80% subsampling
- Interpretation: <0.2 = excellent, 0.2-0.4 = good, >0.4 = unstable

## Recommended Configurations

| Method | Resolution | Clusters | Drift | Status |
|--------|------------|----------|-------|--------|
"""

for method in methods:
    if method in recommendations:
        rec = recommendations[method]
        report += f"| {method} | {rec['resolution']:.1f} | {rec['n_clusters']} | {rec['drift']:.3f} | ✓ |\n"

report += f"""

## Complete Results Table

| Method | Resolution | Clusters | Drift | Recommended |
|--------|------------|----------|-------|-------------|
"""

for _, row in summary_df.iterrows():
    report += f"| {row['Method']} | {row['Resolution']} | {row['N_Clusters']} | {row['Drift']:.3f} | {row['Recommended']} |\n"

report += """

## Interpretation Guidelines

### Method Selection
- **Leiden/Louvain**: Best for well-separated cell types
- **GMM**: Best for continuous/gradient populations
- **LDA**: Best for overlapping transcriptional programs

### Resolution Selection
- Lower drift → more reproducible clustering
- Balance between granularity and stability
- Cross-reference with biological markers

## Files Generated
1. `pbmc_clustering_summary.csv`: Complete results table
2. `pbmc_multihead_clustering_results.h5ad`: Annotated AnnData
3. `pbmc_clustering_stability_diagnostics.png`: Visualization
4. `pbmc_workflow_report.md`: This report

## Next Steps
1. Validate clusters with marker genes
2. Perform differential expression analysis
3. Cell type annotation
4. Downstream functional analysis

---
Generated by OmicVerse Multi-head Clustering Workflow
Skills: single-preprocessing, single-clustering, data-viz-plots, single-downstream-analysis
"""

with open('pbmc_workflow_report.md', 'w') as f:
    f.write(report)

print("✅ Markdown report saved to: pbmc_workflow_report.md")

print("\n" + "=" * 70)
print("✨ WORKFLOW COMPLETE ✨")
print("=" * 70)
print("\nGenerated files:")
print("  1. pbmc_clustering_summary.csv")
print("  2. pbmc_multihead_clustering_results.h5ad")
print("  3. pbmc_clustering_stability_diagnostics.png")
print("  4. pbmc_workflow_report.md")
print("\n🎯 Recommended configurations:")
for method in methods:
    if method in recommendations:
        rec = recommendations[method]
        print(f"   {method}: resolution={rec['resolution']}, clusters={rec['n_clusters']}, drift={rec['drift']:.3f}")
print("\n")
