
import h5py
import numpy as np
import pandas as pd
import sys
from optparse import OptionParser
import json
from scipy.sparse import csr_matrix, isspmatrix_csr, issparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Small number for checking integer values in presence of floating point error
EPSILON = 1e-10

# Maximum default size-factor search range
MAX_RANGE = 1e5

def recover_counts(X, mult_value, max_range, log_base=None, chunk_size=1000):
    """
    Given log-normalized gene expression data, recover the raw read/UMI counts by
    inferring the unknown size factors.

    Parameters
    ----------
    X: 
        The log-normalized expression data. This data is assumed to be normalized 
        via X := log(X/S * mult_value + 1)
        
        **IMPORTANT**: This function REQUIRES log-transformed data (e.g., after 
        `scanpy.pp.normalize_total()` followed by `scanpy.pp.log1p()`).
        It will NOT work correctly on linearly normalized data (e.g., data 
        normalized to CPM/TPM without log transformation).
        
        Expected preprocessing workflow:
        1. sc.pp.normalize_total(adata, target_sum=mult_value)
        2. sc.pp.log1p(adata)
        3. ov.pp.recover_counts(adata.X, mult_value, max_range)
        
    max_range:
        Maximum size-factor search range to use in binary search.
    mult_value:
        The multiplicative value used in the normalization. For example, for TPM
        this value is one million. For logT10K, this value is ten thousand.
    log_base:
        The base of the logarithm (None for natural log)

    Returns
    -------
    counts:
        The inferred counts matrix
    size_factors:
        The array of inferred size-factors (i.e., total counts)
        
    Raises
    ------
    ValueError:
        If input data appears to be non-log-transformed (contains very large values
        that would cause numerical overflow)
    """
    # Validate input data to detect non-log-transformed data
    max_val = X.max() if hasattr(X, 'max') else np.max(X.data if issparse(X) else X)
    if max_val > 50:  # Log-transformed data rarely exceeds ~20-30 in practice
        import warnings
        warnings.warn(
            "Input data contains very large values (max={:.2f}), which suggests "
            "it may not be log-transformed. This function requires log-transformed data "
            "(e.g., after scanpy.pp.normalize_total() + scanpy.pp.log1p()). "
            "Using non-log-transformed data will produce invalid results.".format(max_val),
            UserWarning,
            stacklevel=2
        )
    
    # Recover size-factor normalized data
    if issparse(X):
        if log_base is None:
            X = X.expm1() / mult_value
        else:
            X = (X.power(log_base) - 1) / mult_value
    else:
        if log_base is None:
            X = (np.exp(X) - 1) / mult_value
        else:
            X = (np.power(log_base, X) - 1) / mult_value
    # Infer the size factors
    size_factors_all = []
    num_rows = X.shape[0]
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        X_chunk = X[start:end]

        if issparse(X_chunk):
            X_chunk=X_chunk.toarray()
            
        size_factors=[]
        for x in tqdm(X_chunk):
            s = binary_search(x, max_r=max_range)
            size_factors.append(s)
        size_factors_all+=size_factors

    sf = np.array(size_factors_all)
    if issparse(X):
        # 直接利用广播实现逐行边乘
        counts = X.multiply(sf[:, None]).astype(int)
        counts=counts.tocsr()
    else:
        counts = (X * sf[:, None]).astype(int)
    '''
    if issparse(X):
        size_factors_sparse = csr_matrix(np.diag(size_factors_all))
        counts = X.T.dot(size_factors_sparse)
        counts = (counts.T).astype(int)
    else:
        counts = X.T * size_factors
        counts = (counts.T).astype(int)
    '''
    return counts, size_factors_all

def binary_search(vec, min_r=0, max_r=100000):
    """
    For a given gene expression vector corresponding to a single cell
    or gene expression profile, infer the unknown size factor using
    binary search.
    """
    # Find the smallest non-zero expression value. We assume
    # that this value corresponds to a count of one.
    min_nonzero = np.min([x for x in vec if x != 0])

    # Initialize the search bounds
    min_bound = min_r
    max_bound = max_r

    # Run binary search
    while True:
        curr_s = int((max_bound - min_bound) / 2) + min_bound
        cand_count = min_nonzero * curr_s

        if np.abs(cand_count - 1) < EPSILON:
            return curr_s
        elif cand_count > 1: # The size factor is too big
            max_bound = curr_s 
        elif cand_count < 1: # The size factor is too small
            min_bound = curr_s

        # Sometimes the floating point error is higher than our tolerance. This will manifest 
        # in the binary search algorithm getting stuck deciding between two consecutive integers.
        # Instead of relaxing our tolerance, we simply choose the size-factor that leads to a 
        # smaller error between the putative one-count normalized value and the true one-count 
        # normalized value.
        if max_bound - min_bound == 1:
            cand_count_max = min_nonzero * max_bound
            cand_count_min = min_nonzero * min_bound
            diff_max = np.abs(cand_count_max - 1)
            diff_min = np.abs(cand_count_min - 1)
            if diff_max < diff_min:
                return max_bound
            else:
                return min_bound