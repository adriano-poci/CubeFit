# -*- coding: utf-8 -*-
r"""
    CubeFit: Spectral Cube Fitting Toolkit
    --------------------------------------

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    High-level CubeFit pipeline and utilities:
    - ModelCube: Fast generation of rebinned synthetic spectra and convolutions
    - ZarrManager: Zarr-based I/O for large, chunked data cubes and HyperCube storage
    - PipelineRunner: Orchestrates full fitting pipeline (Kaczmarz, BCD, batch NNLS)
    - KaczmarzSolver: Parallel, batched, blockwise Kaczmarz NNLS solvers
    - cube_bcd: Parallel block coordinate descent solvers and cluster/mean NNLS init
    - cube_utils: Generic scientific helpers (mean/cluster NNLS init, loss, batching)
    - plotting: Flexible plotting for spectra, fits, and diagnostics
    - logger: Robust CubeFit logging and formatted progress messages
    - parallel: Batch and multiprocessing helpers
    - reference: Reference fit, diagnostics, and comparison routines

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>
"""

# Define __all__ for explicit package API
__all__ = [
    "ModelCube",
    "ZarrManager",
    "PipelineRunner",
    "KaczmarzSolver",
    "comp_loss_for_total",
    "flatten_x",
    "unflatten_x",
    "random_subset_indices",
    "plot_aperture_fit",
    "get_logger",
    "CubeFitLogger",
]
