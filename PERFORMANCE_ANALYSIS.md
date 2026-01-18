# OmicVerse Performance Analysis Report

This document identifies performance anti-patterns, inefficient algorithms, and optimization opportunities in the OmicVerse codebase.

## Executive Summary

After comprehensive analysis of the codebase, the following categories of performance issues were identified:

| Category | Issues Found | Severity | Estimated Impact |
|----------|-------------|----------|------------------|
| DataFrame Iteration Anti-patterns | 25+ instances | HIGH | 10-1000x slowdown |
| Nested Loop Inefficiencies | 15+ instances | HIGH | O(n²) → O(n) possible |
| Memory Inefficiencies | 20+ instances | MEDIUM-HIGH | 20-50% memory savings possible |
| Missing Caching | 10+ instances | MEDIUM | Repeated expensive computations |
| GPU/Parallel Computing Issues | 8+ instances | MEDIUM-HIGH | Significant GPU underutilization |

---

## 1. DataFrame Iteration Anti-patterns (Critical)

### 1.1 `iterrows()` Usage (HIGH SEVERITY)

**Problem:** `iterrows()` is 100-1000x slower than vectorized pandas operations.

#### Location: `omicverse/bulk/_Enrichment.py:101-109`
```python
# CURRENT (SLOW)
for _, row in df.iterrows():
    genes = row['Genes'].split(';')
    for gene in genes:
        new_data.append({
            'Gene': gene,
            term_col_name: row['Term'],
        })
return pd.DataFrame(new_data)
```

**Fix:** Use vectorized `explode()`:
```python
# OPTIMIZED
df['Genes'] = df['Genes'].str.split(';')
result = df.explode('Genes').rename(columns={'Genes': 'Gene'})[['Gene', 'Term']]
```

#### Location: `omicverse/single/_mofa.py:2049-2050, 2711-2716`
```python
# CURRENT (SLOW) - Called multiple times
for _, row in corr_df_filtered.iterrows():
    corr_matrix.loc[row[f'{view1}_feature'], row[f'{view2}_feature']] = abs(row['correlation'])
```

**Fix:** Use vectorized assignment:
```python
# OPTIMIZED
corr_matrix.values[feat1_indices, feat2_indices] = np.abs(corr_df_filtered['correlation'].values)
```

#### Location: `omicverse/single/_cpdb.py:347-360`
```python
# CURRENT (SLOW) - Nested loop with iterrows
for index, row in tqdm(df.iterrows()):
    for col in df.columns[14:]:
        if pd.notna(row[col]):
            source, target = col.split('|')
            # ... more operations
```

**Fix:** Use `pd.melt()` and vectorized string operations.

---

### 1.2 `apply()` with Lambda Functions (MEDIUM SEVERITY)

#### Location: `omicverse/pl/_cpdb.py:540, 546-547`
```python
# CURRENT (SLOW)
df_row['RowGroup'] = df_row['level_0'].apply(lambda x: x.split('|')[0])
df_col['Source'] = df_col['level_1'].apply(lambda x: x.split('|')[0])
df_col['Target'] = df_col['level_1'].apply(lambda x: x.split('|')[1])
```

**Fix:** Use `.str` accessor:
```python
# OPTIMIZED
df_row['RowGroup'] = df_row['level_0'].str.split('|').str[0]
df_col[['Source', 'Target']] = df_col['level_1'].str.split('|', expand=True)
```

#### Location: `omicverse/single/_deg_ct.py:320`
```python
# CURRENT (SLOW)
self.adata_test.obs['stim'] = self.adata_test.obs[self.condition].apply(
    lambda x: 0 if x == self.ctrl_group else 1
)
```

**Fix:** Use vectorized comparison:
```python
# OPTIMIZED
self.adata_test.obs['stim'] = (self.adata_test.obs[self.condition] != self.ctrl_group).astype(int)
```

---

## 2. Nested Loop Inefficiencies (Critical)

### 2.1 Set Creation Inside Loops

#### Location: `omicverse/pp/_connectivity.py:89-94, 108-111`
```python
# CURRENT (SLOW) - Creates set on EVERY iteration
for i, row in enumerate(indices):
    mask[i, row] = True
    for j in row:
        if i not in set(indices[j]):  # O(k) set creation per iteration!
            W[j, i] = W[i, j]
            mask[j, i] = True
```

**Complexity:** O(n·k²) where n=cells, k=neighbors (~15)

**Fix:** Pre-compute sets:
```python
# OPTIMIZED - O(n·k)
indices_sets = [set(row) for row in indices]
for i, row in enumerate(indices):
    mask[i, row] = True
    for j in row:
        if i not in indices_sets[j]:
            W[j, i] = W[i, j]
            mask[j, i] = True
```

**Estimated Speedup:** ~15x for k=15 neighbors

---

### 2.2 Repeated `np.unique()` Calls

#### Location: `omicverse/single/_pyslingshot.py:263`
```python
# CURRENT (SLOW) - np.unique() called for EVERY cell
self.cluster_label_indices = np.array([
    np.where(np.unique(cluster_labels) == label)[0][0]
    for label in cluster_labels
])
```

**Complexity:** O(n·m·log(m)) where n=cells, m=unique labels

**Fix:** Pre-compute unique mapping:
```python
# OPTIMIZED - O(m·log(m) + n)
unique_labels = np.unique(cluster_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
self.cluster_label_indices = np.array([label_to_idx[label] for label in cluster_labels])
```

**Estimated Speedup:** ~100x for 5000 cells

---

### 2.3 Redundant DataFrame Filtering in Loops

#### Location: `omicverse/single/_cellvote.py:235-246`
```python
# CURRENT (SLOW)
for ct in adata.obs["best_clusters"].cat.categories:
    ct_li = []
    for celltype_key in celltype_keys:
        ct1 = (
            adata.obs.loc[adata.obs["best_clusters"] == ct, celltype_key]  # O(n) per iteration
            .value_counts()
            .index[0]
        )
        ct_li.append(ct1)
```

**Fix:** Use groupby:
```python
# OPTIMIZED
grouped = adata.obs.groupby("best_clusters")
for ct in adata.obs["best_clusters"].cat.categories:
    group = grouped.get_group(ct)
    ct_li = [group[key].value_counts().index[0] for key in celltype_keys]
```

---

## 3. Memory Inefficiencies (High Impact)

### 3.1 Array Appending in Loops with `np.r_`

#### Location: `omicverse/single/_tosica.py:283-290`
```python
# CURRENT (SLOW) - Creates new array on EVERY iteration
predict_class = np.empty(shape=0)
pre_class = np.empty(shape=0)
for i in range(len(pre)):
    if torch.max(pre, dim=1)[0][i] >= cutoff:
        predict_class = np.r_[predict_class, torch.max(pre, dim=1)[1][i].numpy()]
    else:
        predict_class = np.r_[predict_class, n_c]
    pre_class = np.r_[pre_class, torch.max(pre, dim=1)[0][i]]
```

**Problems:**
1. `np.r_` creates new array each iteration - O(n²) memory allocations
2. `torch.max()` called 3 times per iteration instead of once

**Fix:** Pre-allocate and vectorize:
```python
# OPTIMIZED
max_probs, max_indices = torch.max(pre, dim=1)
max_probs_np = max_probs.cpu().numpy()
max_indices_np = max_indices.cpu().numpy()

predict_class = np.where(max_probs_np >= cutoff, max_indices_np, n_c)
pre_class = max_probs_np
```

---

### 3.2 Repeated `np.concatenate()` in Loops

#### Location: `omicverse/single/_multimap.py:2889-2891, 2942-2948`
```python
# CURRENT (SLOW)
rows = np.array([])
cols = np.array([])
vals = np.array([])

for i in range(len(Xs)):
    rows = np.concatenate([rows, X_rows + sum(len_Xs[:i])])  # O(n) per iteration
    cols = np.concatenate([cols, X_cols + sum(len_Xs[:i])])
    vals = np.concatenate([vals, X_vals])
```

**Fix:** Use list appending then single concatenation:
```python
# OPTIMIZED
rows_list, cols_list, vals_list = [], [], []
for i in range(len(Xs)):
    rows_list.append(X_rows + sum(len_Xs[:i]))
    cols_list.append(X_cols + sum(len_Xs[:i]))
    vals_list.append(X_vals)

rows = np.concatenate(rows_list)
cols = np.concatenate(cols_list)
vals = np.concatenate(vals_list)
```

---

### 3.3 Float64 When Float32 Suffices

#### Locations: Multiple files
- `omicverse/pp/_preprocess.py:1518-1519, 1549-1550`
- `omicverse/pp/_highly_variable_genes.py:99, 214-215`
- `omicverse/single/_multimap.py:628, 629, 695`

```python
# CURRENT
mean = np.zeros(size, dtype=np.float64)
var = np.zeros(size, dtype=np.float64)
```

**Fix:** Use float32 for 50% memory savings:
```python
# OPTIMIZED
mean = np.zeros(size, dtype=np.float32)
var = np.zeros(size, dtype=np.float32)
```

---

## 4. Missing Caching Opportunities

### 4.1 Repeated String Splitting

#### Location: `omicverse/bulk/_Enrichment.py:94-95`
```python
# CURRENT (SLOW) - split() called twice per row
enrich_res['num'] = [int(i.split('/')[0]) for i in enrich_res['Overlap']]
enrich_res['fraction'] = [int(i.split('/')[0])/int(i.split('/')[1]) for i in enrich_res['Overlap']]
```

**Fix:** Cache split results:
```python
# OPTIMIZED
splits = enrich_res['Overlap'].str.split('/')
enrich_res['num'] = splits.str[0].astype(int)
enrich_res['fraction'] = splits.str[0].astype(int) / splits.str[1].astype(int)
```

---

### 4.2 Uncached API Calls

#### Location: `omicverse/external/gseapy/enrichr.py:238-248`
```python
# CURRENT - No caching
def get_libraries(self):
    lib_url = '%s/%s/datasetStatistics' % (self.ENRICHR_URL, self._organism)
    response = requests.get(lib_url, verify=True)
    # ... returns list
```

**Fix:** Add `@lru_cache`:
```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_libraries(self):
    # ... same implementation
```

---

## 5. GPU/Parallel Computing Issues

### 5.1 Missing `torch.no_grad()` in Inference

#### Location: `omicverse/single/_scdiffusion.py:533, 861`
```python
# CURRENT (SLOW) - Builds unnecessary gradient graph
cell_gen = autoencoder(torch.tensor(arr).cuda(), return_decoded=True).cpu().detach().numpy()
```

**Fix:** Wrap in no_grad context:
```python
# OPTIMIZED
with torch.no_grad():
    cell_gen = autoencoder(torch.tensor(arr).to(device), return_decoded=True).cpu().numpy()
```

---

### 5.2 Small Batch Sizes

#### Location: `omicverse/single/_tosica.py:38-39`
```python
# CURRENT - Severely underutilizes GPU
batch_size: int = 8
```

**Fix:** Increase default for GPU:
```python
# OPTIMIZED
batch_size: int = 64  # Or make device-aware
```

---

### 5.3 Hardcoded `.cuda()` Without Availability Check

#### Location: `omicverse/single/_scdiffusion.py:533`
```python
# CURRENT - Will fail if no GPU
torch.tensor(arr).cuda()
```

**Fix:** Use device variable:
```python
# OPTIMIZED
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.tensor(arr).to(device)
```

---

## 6. Additional Anti-patterns

### 6.1 Iteration Over DataFrame Index

Multiple files iterate over `.index` when vectorized operations would work:

#### Locations:
- `omicverse/bulk/_tcga.py:124, 146, 163, 186, 333`
- `omicverse/single/_mofa.py:131, 231, 2629, 2896, 2924`
- `omicverse/bulk/_Gene_module.py:308`
- `omicverse/space/_spatrio.py:423, 440, 456`

```python
# CURRENT (SLOW)
for i in pd_c.index:
    # row-by-row operations
```

**Fix:** Use vectorized pandas/numpy operations.

---

## Priority Recommendations

### Immediate (High Impact, Easy Fixes)
1. Replace `iterrows()` with vectorized operations in `_Enrichment.py`
2. Pre-compute sets in `_connectivity.py`
3. Fix `np.r_` loops in `_tosica.py`
4. Add `torch.no_grad()` in `_scdiffusion.py`

### Short-term (High Impact, Moderate Effort)
5. Vectorize nested loops in `_mofa.py`
6. Add caching to Enrichr API calls
7. Increase default batch sizes in GPU code
8. Replace `.apply(lambda)` with `.str` accessor operations

### Medium-term (Moderate Impact, Higher Effort)
9. Refactor `_cellvote.py` groupby operations
10. Convert float64 allocations to float32 where appropriate
11. Add device-aware batch sizing in LLM modules
12. Implement persistent caching for BioMart queries

---

## Estimated Performance Improvements

| Fix | Current | Optimized | Improvement |
|-----|---------|-----------|-------------|
| `iterrows()` → vectorized | O(n²) | O(n) | 100-1000x |
| Set pre-computation | O(n·k²) | O(n·k) | ~15x |
| `np.unique()` caching | O(n·m·log m) | O(m·log m) | ~100x |
| `np.r_` → pre-allocation | O(n²) | O(n) | ~n times |
| `torch.no_grad()` | GPU mem + time | Inference only | 2-4x faster |
| Float64 → Float32 | 8 bytes | 4 bytes | 50% memory |

---

## Testing Recommendations

Before implementing fixes:
1. Profile current performance with `cProfile` or `line_profiler`
2. Create benchmarks for affected functions
3. Ensure test coverage for modified code
4. Validate numerical results match after optimization

---

*Report generated: 2026-01-18*
*Analysis scope: omicverse v1.7.9rc1*
