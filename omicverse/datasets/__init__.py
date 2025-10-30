r"""
Dataset management utilities for omicverse.

This module provides functions to load datasets for single-cell analysis,
following the dynamo-release sample_data.py pattern.

Main functions:
    get_adata: Download and load AnnData from URLs
    download_data: Download files with progress tracking
    pbmc3k: Load PBMC 3k dataset with fallback to mock data
    create_mock_dataset: Generate synthetic datasets for testing
    
Dataset functions:
    - Scanpy-inspired: blobs, burczynski06, moignard15, paul15, pbmc68k_reduced
    - Simulations: toggleswitch, krumsiek11
    - Dynamo datasets: scnt_seq_neuron_splicing, scnt_seq_neuron_labeling
    - Real datasets: zebrafish, dentate_gyrus, bone_marrow, hematopoiesis
    - Special: multi_brain_5k (multiome data)
    
Core features:
    - Robust download with error handling and retry logic
    - Progress tracking with tqdm
    - Support for h5ad and loom formats
    - Mock data generation for testing
    
Examples:
    >>> import omicverse as ov
    >>> 
    >>> # Load PBMC 3k data (with clustering)
    >>> adata = ov.datasets.pbmc3k(processed=True)
    >>> print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    >>> 
    >>> # Load specific datasets
    >>> adata = ov.datasets.hematopoiesis()
    >>> adata = ov.datasets.paul15()  # Myeloid development
    >>> adata = ov.datasets.blobs()   # Synthetic clusters
    >>> 
    >>> # Create mock data
    >>> adata = ov.datasets.create_mock_dataset(
    ...     n_cells=1000, 
    ...     n_cell_types=5,
    ...     with_clustering=True
    ... )
"""

from ._datasets import (
    # Core utilities
    download_data,
    download_data_requests,
    get_adata,
    
    # Main dataset loaders
    pbmc3k,
    create_mock_dataset,
    
    # Scanpy-inspired datasets
    blobs,
    burczynski06,
    moignard15,
    paul15,
    pbmc68k_reduced,
    toggleswitch,
    krumsiek11,
    
    # Placeholder functions
    gillespie,
    hl60,
    nascseq,
    scslamseq,
    scifate,
    cite_seq,
    
    # Real dataset functions
    scnt_seq_neuron_splicing,
    scnt_seq_neuron_labeling,
    zebrafish,
    dentate_gyrus,
    bone_marrow,
    haber,
    hg_forebrain_glutamatergic,
    chromaffin,
    bm,
    pancreatic_endocrinogenesis,
    dentate_gyrus_scvelo,
    sceu_seq_rpe1,
    sceu_seq_organoid,
    hematopoiesis,
    hematopoiesis_raw,
    human_tfs,
    multi_brain_5k,

    decov_bulk_covid_bulk,
    decov_bulk_covid_single,
)