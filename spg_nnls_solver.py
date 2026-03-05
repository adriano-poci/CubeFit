# -*- coding: utf-8 -*-
r"""
    spg_nnls_solver.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Modestly-MP, chunk-friendly solver aligned to /HyperCube/models chunking.
    `x` is 1-D (length C*P) in/out. Warm start handled by caller; we accept `x0`.
    `orbit_weights` is accepted (optional); free fit if None

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Fixed bug in computing `nprocs`;
        Wrapped entire `solve_global_kaczmarz_cchunk_mp` in `try/except`. 4
            December 2025
v1.1:   Added column-flux scaling bypass (`cp_flux_ref=None`). 5 December 2025
v1.2:   Experimenting with RMSE cap. 11 December 2025
v1.3:   Introduced L2 into Kaczmarz solving to be consistent with NNLS
            initialisation;
        Disabled buggy `w_band` which was implemented incorrectly. 12 December
            2025
v1.4:   Use the seed vector as a numerical prior during the Kaczmarz solving;
        Implement global RMSE evaluation, and keep only globally-best solution;
        Optionally disable seed prior. 13 December 2025
v1.5:   Use RMSE proxy as guard for epoch solution. 15 December 2025
v2.0:   Implemented global Kaczmarz gradient instead of per-tile updates; added
            `_worker_tile_global_grad_band` and
            `solve_global_kaczmarz_global_step_mp`. 16 December 2025
v2.1:   Added tiny-column freeze inside `_worker_tile_global_grad_band`. 18
            December 2025
v2.2:   Consolidated two tiny-column freeze env var names into one;
        Added fairer max-tile rather than bias to brighter spaxels. 25 December
            2025
v2.3:   Stripped global Kaczmarz solver diagnostics to single gradient and NNLS
            constraint. 26 December 2025
v2.4:   Pre-check gradient before each epoch;
        Pre-check RMSE proxy before each epoch to allow for early exit. 28
            December 2025
v2.5:   Implemented backtracking and step-size reduction based on RMSE proxy. 29
            December 2025
v2.6:   Replaced expensive backtracking RMSE evaluations with O(1) quadratic
            coefficients. 30 December 2025
v3.0:   Switched to diagonal-preconditioned Spectral Projected Gradient method
            to replace Kaczmarz updates. 31 December 2025
v3.1:   Added orbit-weight projection step inside SPG loop in 
            `solve_global_kaczmarz_global_step_mp`. 1 January 2026
v3.2:   Properly account for active orbits using component support masks during
            orbit-weight projection;
        Fixed indentation bug in tile loop in
            `solve_global_kaczmarz_global_step_mp`. 3 January 2026
v3.3:   Correctly capped `dx` during SPG step in
            `solve_global_kaczmarz_global_step_mp`;
        Implement Armijo backtracking with cheap RMSE proxy in
            `solve_global_kaczmarz_global_step_mp`;
        Force small `D` to `np.inf` in `solve_global_kaczmarz_global_step_mp`. 5
            January 2026
v3.4:   Implemented SPG to Kaczmarz workflow. 7 January 2026
v3.5:   Fixed indentation bug for Kaczmarz update in
            `solve_global_kaczmarz_global_step_mp`. 8 January 2026
v3.6:   Corrected all diagnostics and solver heuristics for the new scale of the
            LOSVD. 10 January 2026
v3.7:   Apply `dx` to the effective `x`, including inactive orbits, in SPG
            solver `solve_global_kaczmarz_global_step_mp`;
        Compute a proper `rmse_trial` in the Armijo backtracking step in the SPG
            solver `solve_global_kaczmarz_global_step_mp` to determine step
            acceptance --- otherwise shrink `lr`;
        Corrected Kaczmarz block update accumulation in `solve_kaczmarz_nnls`
            by normalising by the number of rows and the number of spaxels. 11
            January 2026
v3.8:   Added additional acceptance guard in SPG solver based on `step_cos`
            in `solve_global_kaczmarz_global_step_mp`;
        Use epoch 1 as a warm-up with softer criteria in SPG solver in
            `solve_global_kaczmarz_global_step_mp`. 12 January 2026
v3.9:   Added variational orbit-mass prior to break rank-1 degeneracy;
        Softened freeze criteria to include `orbit_weights` prior gradient. 16
            January 2026
v3.10:  Universally removed all ad-hoc scalings;
        Compute dynamic Barzilai–Borwein step length in
            `solve_global_kaczmarz_global_step_mp`;
        Include orbit prior completely in the cost function of the solver in
            `solve_global_kaczmarz_global_step_mp`. 25 January 2026
v3.11:  Scale `w_target` to match total mass before applying prior in
            `solve_global_kaczmarz_global_step_mp`;
        Trim final `best_x` with zero-flooring in
            `solve_global_kaczmarz_global_step_mp`. 26 January 2026
v3.12:  Renamed module to represent change in architecture;
        Renamed `solve_global_kaczmarz_global_step_mp` to `solve_global_spg`;
        Added age curvature prior gradient in `solve_global_spg`. 27 January
            2026
v3.13:  Set data-weighted orbit gradient in `solve_global_spg`;
        Add jitter to input seed in `solve_global_spg`;
        Added `diffuse_seed_full_CP` to diffuse NNLS seed;
        Added soft projection to avoid flooring SFH in `solve_global_spg`. 28
            January 2026
v3.14:  Changed soft projection to hard projection in `solve_global_spg`. 29
            January 2026
v3.15:  Removed projection in favour of re-parametrisation in `solve_global_spg`.
            30 January 2026
v3.16:  Replaced orbit prior with Lagrange multiplier for exact adherence to the
            orbit weights within the solver in `solve_global_spg`. 31 January
            2026
v3.17:  Added bookkeeping to determine if components should be zeroed, and remove
            them from the solver in `solve_global_spg`. 3 February 2026
v3.18:  Added column-energy scaling to the gradient and the diagonal
            preconditioner only, not the forward model, in `solve_global_spg`. 4
            February 2026
v3.19:  Reverted to unscaled gradient and preconditioner to avoid instability. 5
            February 2026
v3.20:  Established correct scaling for the seed `x0`. 9 February 2026
v3.21:  Scale `x` amplitude once per epoch in `solve_global_spg`;
        Added anti-flatness heuristic to penalise flat SFH, using
            `orbit_population_variance_grad`;
        Implemented rank-1 projection of the orbit prior in order to avoid
            introducing flat SFH by re-distributing mass among all populations of
            an orbit in `solve_global_spg`. 10 February 2026
v3.22:  Fixed bug when computing `orbit_res` in `solve_global_spg` by using the
            full per-orbit mass `s_full` instead of only active `s`. 11 February
            2026
v3.23:  Implemented Tikhonov (Levenberg–Marquardt–type) damping of the
            diagonal Gauss–Newton preconditioner (`invD`) in `solve_global_spg`.
            13 February 2026
v3.24:  Reverted to v3.22. 3 March 2026
v3.25:  Streamlined `rmse_proxy_subset` to be more efficient and robust, and 
            added progress bar;
        Fixed issues with RMSE lag and incorrect acceptance logic in
            `solve_global_spg`;
        Made population curvature mass-aware;
        Project only after epoch acceptance logic. 4 March 2026
"""

from __future__ import annotations, print_function

import os, sys, traceback
import math, builtins
import time
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List
from contextlib import contextmanager

import multiprocessing as mp
import numpy as np
import h5py
from tqdm.auto import tqdm

from CubeFit.hdf5_manager import open_h5
from CubeFit import cube_utils as cu
from CubeFit.hypercube_builder import read_global_column_energy

# ----------------------------- Config ---------------------------------

@dataclass
class MPConfig:
    epochs: int = 1
    lr: float = 0.25
    project_nonneg: bool = True
    processes: int = 2              # modest parallelism (2–4 recommended)
    blas_threads: int = 8           # per-process BLAS threads
    apply_mask: bool = True
    # HDF5 *dataset* chunk cache (not RDCC): keep local & harmless
    dset_slots: int = 1_000_003
    dset_bytes: int = 256 * 1024**2
    dset_w0: float = 0.90
    s_tile_override: Optional[int] = None
    pixels_per_aperture: int = 256
    max_tiles: Optional[int] = None

# ---------------------- Small pool utilities --------------------------

def _pool_ping() -> int:
    return 1

def _pool_ok(pool, timeout: float = 5.0) -> bool:
    """
    Returns True if a trivial task round-trips within `timeout`.
    If it times out or raises, the pool is considered unhealthy.
    """
    try:
        res = pool.apply_async(_pool_ping)
        return res.get(timeout=timeout) == 1
    except Exception:
        return False

def _worker_init(blas_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np.max((1, blas_threads // 2)))
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# ------------------------------------------------------------------------------

def orbit_mass_prior_grad(x_cp: np.ndarray,
                          w_target: np.ndarray,
                          lam: float) -> np.ndarray:
    """
    Gradient of quadratic orbit-mass prior.

    Penalizes deviation of per-orbit total mass from target w_target.

    Parameters
    ----------
    x_cp : ndarray, shape (C, P)
        Current solution in physical basis.
    w_target : ndarray, shape (C,)
        Target per-orbit mass fractions (normalized).
    lam : float
        Regularization strength.

    Returns
    -------
    grad_cp : ndarray, shape (C, P)
        Gradient contribution to add to data gradient.
    """
    # per-orbit total mass
    s = np.sum(x_cp, axis=1)          # (C,)
    # residual relative to target
    r = s - w_target                  # (C,)
    # gradient: same correction applied to all populations of orbit c
    grad_cp = lam * r[:, None]        # broadcast to (C,P)
    return grad_cp

# ------------------------------------------------------------------------------

def rmse_proxy_subset(
    h5_path,
    x_CP,
    tile_ranges,
    keep_idx,
    inv_cp_flux_ref,
    w_lam_sqrt,
):
    """
    Compute RMSE proxy over a subset of tiles in the SAME weighted space
    used by the gradient (i.e. apply √w_λ if present). Returns sqrt(ssq/n).
    """

    ssq = 0.0
    nres = 0

    x_flat = x_CP.reshape(-1)

    # Only show tqdm if more than 1 tile
    use_bar = len(tile_ranges) > 1

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M  = f["/HyperCube/models"]

        iterator = tile_ranges
        if use_bar:
            iterator = tqdm(
                tile_ranges,
                desc="[RMSE-proxy]",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )

        for (s0, s1) in iterator:

            Y = np.asarray(DC[s0:s1, :], dtype=np.float64)
            if keep_idx is not None:
                Y = Y[:, keep_idx]

            A = np.asarray(M[s0:s1, :, :, :], dtype=np.float64)

            if keep_idx is not None:
                A = A[:, :, :, keep_idx]

            if inv_cp_flux_ref is not None:
                A *= inv_cp_flux_ref[None, :, :, None]

            Sblk, C, P, Lk = A.shape

            # reshape for single BLAS
            A2 = A.transpose(0, 3, 1, 2).reshape(Sblk * Lk, C * P)

            yhat_flat = A2 @ x_flat
            yhat = yhat_flat.reshape(Sblk, Lk)

            R = Y - yhat
            if not np.all(np.isfinite(R)):
                R = np.nan_to_num(R, copy=False)

            if w_lam_sqrt is not None:
                R *= w_lam_sqrt[None, :]
                ssq += float(np.sum(R * R))
            else:
                ssq += float(np.sum(R * R))

            nres += R.size

    return float(np.sqrt(ssq / max(nres, 1)))

# ------------------------------------------------------------------------------

def _worker_tile_grad_band_single_metric(args):
    """
    Minimal, single-metric gradient worker.

    Computes, for a spatial tile [s0:s1] and component band [c0:c1):

        g_band[c,p] = sum_{s,λ} Ã[s,c,p,λ] * R[s,λ]
        D_band[c,p] = sum_{s,λ} Ã[s,c,p,λ]^2

    where:
        Ã[s,c,p,λ] = A[s,c,p,λ] / sqrt(binCounts[s])
        R[s,λ]      = (Y - Ã x)[s,λ]

    """

    (
        h5_path,
        s0, s1,
        keep_idx,
        c0, c1,
        R_tile, # shape (Sblk, Lk), already weighted by 1/sqrt(binCounts)
        dset_slots,
        dset_bytes,
        dset_w0,
    ) = args

    # Ensure contiguous float64 arrays
    Rw = np.asarray(R_tile, dtype=np.float64, order="C")
    # w_s = np.asarray(w_s, dtype=np.float64, order="C")

    Sblk, Lk = Rw.shape
    band_size = c1 - c0

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        try:
            M.id.set_chunk_cache(dset_slots, dset_bytes, dset_w0)
        except Exception:
            pass

        P = M.shape[2]

        g_band = np.zeros((band_size, P), dtype=np.float64)
        D_band = np.zeros((band_size, P), dtype=np.float64)

        # BLAS temp threshold (bytes)
        try:
            blas_temp_max = int(
                os.environ.get(
                    "CUBEFIT_BLAS_TEMP_MAX_BYTES",
                    str(1024 * 1024**2),  # default 1024 MiB
                )
            )
        except Exception:
            blas_temp_max = 64 * 1024**2

        R_flat = Rw.reshape(-1)  # view, no copy

        for bi, c in enumerate(range(c0, c1)):
            # Read model slice
            A = np.asarray(M[s0:s1, c, :, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                A = A[:, :, keep_idx]  # (Sblk, P, Lk)

            # Diagonal term (cheap, no BLAS needed)
            # D[p] = sum_{s,λ} A[s,p,λ]^2
            D_band[bi] = np.sum(A * A, axis=(0, 2))

            # Gradient term
            # g[p] = sum_{s,λ} A[s,p,λ] * R[s,λ]
            # Prefer BLAS if temporary size is acceptable
            temp_bytes = P * Sblk * Lk * 8  # float64

            if temp_bytes <= blas_temp_max and P * Sblk * Lk > 0:
                # BLAS path: reshape to (P, Sblk*Lk) and matvec
                A2 = np.ascontiguousarray(
                    A.transpose(1, 0, 2).reshape(P, -1)
                )
                g_band[bi] = A2 @ R_flat
                del A2
            else:
                # Memory-light fallback
                g_band[bi] = np.tensordot(
                    A, Rw, axes=([0, 2], [0, 1])
                )

    return g_band, D_band

# ------------------------------------------------------------------------------

def _canon_orbit_weights(h5_path: str,
                         orbit_weights,
                         C: int,
                         P: int) -> np.ndarray | None:
    """
    Return a (C,) float64 prior vector for components, or None if unavailable.
    Accepts:
      - orbit_weights == None: try reading '/CompWeights' from HDF5.
      - orbit_weights shape == (C,): use as-is.
      - orbit_weights shape == (C*P,): sum over populations -> (C,).
    Raises if a provided vector has incompatible size.
    """
    w = None
    if orbit_weights is not None:
        w = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")
    else:
        with open_h5(h5_path, role="reader") as f:
            if "/CompWeights" in f:
                w = np.asarray(f["/CompWeights"][...], dtype=np.float64).ravel(order="C")
            else:
                return None  # no prior available

    if w.size == C:
        pass
    elif w.size == C * P:
        w = w.reshape(C, P).sum(axis=1)
    else:
        raise ValueError(f"orbit_weights length {w.size} incompatible with C={C}, P={P}. "
                         f"Expected C or C*P.")
    # normalize to a comparable scale (optional, keeps magnitudes sane)
    s = np.sum(w)
    if np.isfinite(s) and s > 0.0:
        w = w / s
    return w

# ------------------------------------------------------------------------------

def softbox_params_smooth(eq: int, E: int) -> tuple[float, float]:
    """
    Cosine ramp starting at epoch 2 (1-based):
      eq = 1 → (band, step) = (0.30, 0.20)
      eq = E → (band, step) = (0.15, 0.30)
      2..E ramps smoothly between the two.
    """
    eq = int(eq)
    E  = int(max(2, E))

    if eq <= 1:
        return 0.30, 0.20

    # t=0 at eq=2, t=1 at eq=E
    t = np.clip((eq - 2) / max(1, (E - 2)), 0.0, 1.0)
    s = 0.5 - 0.5 * np.cos(np.pi * t)

    band = (1.0 - s) * 0.30 + s * 0.15   # 0.30 → 0.15
    step = (1.0 - s) * 0.20 + s * 0.30   # 0.20 → 0.30
    return float(band), float(step)

# ------------------------------------------------------------------------------

def orbit_age_smoothness_grad(x: np.ndarray) -> np.ndarray:
    """
    Per-orbit age smoothness gradient.
    x has shape (C, P).
    Returns gradient with same shape.
    Penalizes (x[c,p+1] - x[c,p])^2 for each orbit c.
    """
    C, P = x.shape
    g = np.zeros_like(x)

    # forward differences along age axis
    d = x[:, 1:] - x[:, :-1]    # shape (C, P-1)

    # left boundary
    g[:, 0] -= d[:, 0]

    # interior
    if P > 2:
        g[:, 1:-1] += d[:, :-1] - d[:, 1:]

    # right boundary
    g[:, -1] += d[:, -1]

    return g

# ------------------------------------------------------------------------------

def population_age_curvature_grad(
    x_cp: np.ndarray,
    pop_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Population curvature gradient along the AGE axis only.

    Assumes populations are ordered as (metal, age, alpha) and flattened
    in C-order into length P.

    Parameters
    ----------
    x_cp : ndarray, shape (C, P)
        Current solution in physical basis.
    pop_shape : tuple (n_metals, n_ages, n_alphas)
        Population grid shape before flattening.

    Returns
    -------
    grad_cp : ndarray, shape (C, P)
        Curvature gradient to add to SPG gradient.
    """
    if x_cp.ndim != 2:
        raise ValueError("x_cp must have shape (C, P)")

    nM, nA, nZ = (int(v) for v in pop_shape)
    C, P = x_cp.shape
    if nM * nA * nZ != P:
        raise ValueError(
            f"pop_shape {pop_shape} incompatible with P={P}"
        )

    # Reshape to (C, nM, nA, nZ)
    X = x_cp.reshape((C, nM, nA, nZ), order="C")
    grad = np.zeros_like(X)

    if nA > 1:
        # interior ages
        if nA > 2:
            grad[:, :, 1:-1, :] = (
                2.0 * X[:, :, 1:-1, :]
                - X[:, :, 0:-2, :]
                - X[:, :, 2:, :]
            )

        # boundaries (one-sided)
        grad[:, :, 0, :] = X[:, :, 0, :] - X[:, :, 1, :]
        grad[:, :, -1, :] = X[:, :, -1, :] - X[:, :, -2, :]

    # Back to (C, P)
    return grad.reshape((C, P), order="C")

# ------------------------------------------------------------------------------

def orbit_population_variance_grad(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-orbit anti-flatness gradient (encourages non-flat SFH).

    Returns a gradient that points away from the per-orbit mean:
        ~ (x - mean) / (per-orbit-mass + eps)

    Using (x - mean) makes the gradient increase the objective
    when the SFH is *peaked* and decrease it when the SFH becomes flat,
    i.e. it penalizes flatness as intended.
    """
    mean = np.mean(x, axis=1, keepdims=True)  # (C,1)
    s = np.sum(x, axis=1, keepdims=True)      # (C,1)
    s_eps = np.maximum(s, eps)
    # return (x - mean) normalized by per-orbit mass
    return (mean - x) / s_eps

# ------------------------------------------------------------------------------

def diffuse_seed_full_CP(
    seed_cp: np.ndarray,
    sigma_c: float,
    sigma_p: float,
    *,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Diffuse a sparse NNLS seed on the full (C,P) grid using a separable
    Gaussian kernel in orbit (c) and population (p).

    Parameters
    ----------
    seed_cp : ndarray (C,P)
        NNLS seed with zeros in non-sampled locations.
    sigma_c : float
        Gaussian width in orbit index units.
    sigma_p : float
        Gaussian width in population index units.

    Returns
    -------
    seed_support : ndarray (C,P)
        Smooth, positive field encoding proximity to NNLS seed support.
        Normalized per-row to max=1.
    """
    C, P = seed_cp.shape

    # Early exit: no seed information
    if not np.any(seed_cp > 0):
        return np.zeros_like(seed_cp)

    # Orbit kernel
    c_idx = np.arange(C, dtype=np.float64)
    dc = c_idx[:, None] - c_idx[None, :]
    Kc = np.exp(-0.5 * (dc / max(sigma_c, eps)) ** 2)

    # Population kernel
    p_idx = np.arange(P, dtype=np.float64)
    dp = p_idx[:, None] - p_idx[None, :]
    Kp = np.exp(-0.5 * (dp / max(sigma_p, eps)) ** 2)

    # Convolution: Kc @ seed @ Kp
    seed_support = Kc @ seed_cp @ Kp

    # Normalize per orbit so max=1 (scale-free gating)
    row_max = seed_support.max(axis=1, keepdims=True)
    seed_support = np.divide(
        seed_support,
        row_max,
        out=np.zeros_like(seed_support),
        where=row_max > eps,
    )

    return seed_support

# ------------------------------------------------------------------------------

def solve_global_spg(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    tracker: Optional[object] = None,
) -> tuple[np.ndarray, dict]:
    """
    SPG-class global NNLS solver with orbit-weight projection.

    This is a diagonal-preconditioned Spectral Projected Gradient method.
    It converges to the global NNLS optimum (for fixed templates) and
    enforces spaxel-independent orbit weights via a projection step.
    """

    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # Load cube metadata
    # ------------------------------------------------------------
    with open_h5(h5_path, role="reader") as f:
        S, L = map(int, f["/DataCube"].shape)
        _, C, P, Lm = map(int, f["/HyperCube/models"].shape)
        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch")

        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        Lk = int(keep_idx.size) if keep_idx is not None else L

        # read chunking info (but don't keep the dataset handle)
        chunks = f["/HyperCube/models"].chunks
        s_tile = int(chunks[0]) if (chunks and chunks[0]) else 128
        if cfg.s_tile_override is not None:
            s_tile = int(cfg.s_tile_override)

    # ------------------------------------------------------------
    # Load NNLS patch support and KNOWN_ZERO masks
    # ------------------------------------------------------------
    with open_h5(h5_path, role="reader") as f:
        # Persistent KNOWN_ZERO mask (global)
        if "/HyperCube/known_zero_mask" in f:
            known_zero = np.asarray(
                f["/HyperCube/known_zero_mask"][...], dtype=bool
            )
            if known_zero.shape != (C, P):
                raise RuntimeError("known_zero_mask has wrong shape")
        else:
            known_zero = np.zeros((C, P), dtype=bool)

        # Optional NNLS patch metadata
        if "/Seeds/seed_support_mask" in f:
            seed_support_mask = np.asarray(
                f["/Seeds/seed_support_mask"][...], dtype=bool
            )
            if seed_support_mask.shape != (C, P):
                raise RuntimeError("seed_support_mask has wrong shape")
        else:
            seed_support_mask = None

        if "/Seeds/seed_tested_mask" in f:
            seed_tested_mask = np.asarray(
                f["/Seeds/seed_tested_mask"][...], dtype=bool
            )
            if seed_tested_mask.shape != (C, P):
                raise RuntimeError("seed_tested_mask has wrong shape")
        else:
            seed_tested_mask = None
    seed_tested_mask = None
    known_zero = np.zeros((C, P), dtype=bool)

    s_ranges = [(s0, min(S, s0 + s_tile)) for s0 in range(0, S, s_tile)]

    print(
        f"[SPG] S={S}, L={L} (kept {Lk}), C={C}, P={P}, "
        f"s_tile={s_tile}, epochs={cfg.epochs}, lr0={cfg.lr}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Global ||Y|| for trust region (compute once)
    # ------------------------------------------------------------
    Y_glob_norm2 = 0.0
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        for (s0, s1) in s_ranges:
            Yt = np.asarray(DC[s0:s1, :], np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            # guard against NaN/Inf
            if not np.all(np.isfinite(Yt)):
                Yt = np.nan_to_num(
                    Yt, nan=0.0, posinf=0.0, neginf=0.0, copy=False
                )
            Y_glob_norm2 += float(np.sum(Yt * Yt))
        # binCounts = np.asarray(f['/BinCounts'][...], dtype=int)

    Y_glob_norm = float(np.sqrt(Y_glob_norm2))
    print(
        f"[SPG] Global ||Y|| = {Y_glob_norm:.6e}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Orbit-weight prior (spaxel independent)
    # ------------------------------------------------------------
    w_target = None
    # orbit_beta = float(os.environ.get("CUBEFIT_ORBIT_BETA", "0.2"))
    if orbit_weights is not None:
        w_target = _canon_orbit_weights(h5_path, orbit_weights, C=C, P=P)
        print("[SPG] Orbit-weight projection enabled.", flush=True)

    # ------------------------------------------------------------
    # Initialise x
    # ------------------------------------------------------------
    if x0 is None:
        x = np.zeros((C, P), dtype=np.float64)
    else:
        x0 = np.asarray(x0, np.float64).ravel()
        if x0.size != C * P:
            raise ValueError("x0 has wrong size")
        x = x0.reshape(C, P).copy()

    # ------------------------------------------------------------
    # Cheap global amplitude alignment using precomputed energies
    # ------------------------------------------------------------
    if x0 is not None:
        E_cp = read_global_column_energy(h5_path)  # (C,P)
        if E_cp is not None:
            model_power = float(np.sum((x * x) * E_cp))
            if model_power > 0.0:
                gamma = np.sqrt(Y_glob_norm2 / model_power)
                if np.isfinite(gamma) and gamma > 0.0:
                    x *= gamma
                    print(
                        f"[SPG] amplitude aligned from col_energy: "
                        f"gamma={gamma:.3e}",
                        flush=True,
                    )

    # If x0 came from NNLS seed, clear BB history so BB step isn't anchored to old state.
    # This ensures a genuine BB step computed from the first epoch, not a small correction.
    x_prev = None
    g_prev = None

    # Optional small jitter to enable escaping exact null-space: tiny relative perturbation
    if os.environ.get("CUBEFIT_SEED_JITTER", "0") == "1":
        jitter_rel = float(os.environ.get("CUBEFIT_SEED_JITTER_REL", "1e-8"))
        # add tiny positive perturbation proportional to current magnitude (preserve non-negativity)
        scale = np.maximum(1.0, np.linalg.norm(x))  # scale roughly related to global mass
        rng = np.random.default_rng()  # fixed seed for reproducibility
        noise = rng.normal(loc=0.0, scale=jitter_rel * scale, size=x.shape).reshape(x.shape)
        x += np.abs(noise)  # only add positive tiny noise to break ties, keep non-neg
        if cfg.project_nonneg:
            np.maximum(x, 0.0, out=x)
        print(f"[SPG] applied tiny seed jitter rel={jitter_rel}", flush=True)

    if w_target is not None:
        # If seed came from nnls_patch we treat it as shape-only:
        # do NOT trust sum(x) as a total-mass estimator. Instead derive
        # a sensible global mass scale from the data (||Y||) so the orbit
        # prior uses a comparable scale to the gradient objective.
        total_mass_est = float(np.sum(x))
        seed_origin = None
        try:
            with open_h5(h5_path, role="reader") as f:
                if "/Seeds/x0_nnls_patch" in f:
                    seed_origin = f["/Seeds/x0_nnls_patch"].attrs.get("origin", None)
        except Exception:
            seed_origin = None

        if seed_origin and ("nnls_patch" in seed_origin):
            # seed is shape-only: derive scale from data norm, not seed sum
            # Use global Y norm (computed above as Y_glob_norm) if available
            try:
                total_mass_est = builtins.max(1.0, float(Y_glob_norm))
            except NameError:
                # fallback: total data sum
                with open_h5(h5_path, role="reader") as f:
                    DC = f["/DataCube"]
                    total_mass_est = float(np.sqrt(float(np.sum(np.asarray(DC[...], np.float64)**2))))
            # log for traceability
            print(f"[SPG] nnls_patch seed detected: deriving total_mass_est from data = {total_mass_est:.3e}", flush=True)
        else:
            # trusted seed (not nnls_patch) -> use seed mass if positive
            if total_mass_est <= 0.0:
                total_mass_est = builtins.max(1.0, Y_glob_norm)

        w_target = w_target * total_mass_est
        print(f"[SPG] scaled w_target by total_mass_est={total_mass_est:.3e}", flush=True)

    # ------------------------------------------------------------
    # Multiprocessing bands
    # ------------------------------------------------------------
    nprocs_req = builtins.max(1, int(cfg.processes))
    band_size = int(np.ceil(C / nprocs_req))
    bands = []
    c0 = 0
    for _ in range(nprocs_req):
        c1 = builtins.min(C, c0 + band_size)
        if c1 > c0:
            bands.append((c0, c1))
        c0 = c1

    use_pool = len(bands) > 1
    pool = None
    if use_pool:
        ctx = mp.get_context(
            os.environ.get("CUBEFIT_MP_CTX", "forkserver")
        )
        pool = ctx.Pool(
            processes=len(bands),
            initializer=_worker_init,
            initargs=(int(cfg.blas_threads),),
        )

    # ------------------------------------------------------------
    # SPG bookkeeping
    # ------------------------------------------------------------
    eps = float(os.environ.get("CUBEFIT_EPS", "1e-12"))
    lr = float(cfg.lr)

    # --- BB history ---
    alpha_bb = float(cfg.lr)   # initial BB step guess

    best_x = x.copy()
    best_proxy = np.inf

    # --- Active set bookkeeping ---
    active_orbits = np.arange(C, dtype=np.int32)
    min_active = int(os.environ.get("CUBEFIT_MIN_ACTIVE_ORBITS", "8"))

    # ------------------------------------------------------------
    # Population shapes
    # ------------------------------------------------------------
    with open_h5(h5_path, role="reader") as f:
        if "/Templates" not in f:
            raise RuntimeError("Missing /Templates group in HDF5")
        pop_shape = tuple(f["/Templates"].attrs["pop_shape"])

    # Delay orbit-mass projection until amplitude is established
    ORBIT_WARM_EPOCHS = builtins.max(0, builtins.min(2, cfg.epochs - 1))
    # always project final epoch, regardless of `cfg.epochs`
    print(f"[SPG] orbit-weight projection will start at epoch {ORBIT_WARM_EPOCHS + 1}", flush=True)

    # ------------------------------------------------------------
    # Validation controls (two-prong rejection guard)
    # ------------------------------------------------------------
    validate_flag = os.environ.get("CUBEFIT_VALIDATE_PROXY", "1")
    validate_flag = False if validate_flag.lower() in ("0", "false", "no", "off") else True

    try:
        proxy_sample_tiles = int(os.environ.get("CUBEFIT_PROXY_SAMPLE_TILES", "3"))
    except Exception:
        proxy_sample_tiles = 3
    proxy_sample_tiles = max(1, proxy_sample_tiles)

    try:
        rmse_ratio_thresh = float(os.environ.get("CUBEFIT_PROXY_RMSE_RATIO", "1.2"))
    except Exception:
        rmse_ratio_thresh = 1.2

    try:
        data_mult_thresh = float(os.environ.get("CUBEFIT_PROXY_MULT", "10.0"))
    except Exception:
        data_mult_thresh = 10.0

    # ============================================================
    # Main epochs
    # ============================================================
    try:
        for ep in range(cfg.epochs):

            g_tot = np.zeros_like(x)
            D_tot = np.zeros_like(x)
            ssq = 0.0
            nres = 0
            # --- amplitude accumulators ---
            dot_AxY = 0.0
            dot_AxAx = 0.0

            pbar = tqdm(
                total=len(s_ranges),
                desc=f"[SPG] epoch {ep+1}/{cfg.epochs}",
                mininterval=2.0,
                dynamic_ncols=True,
            )
            if ep == 0 and seed_support_mask is not None and w_target is None:
                # only freeze if seed carries amplitude information
                seed_origin = None
                try:
                    with open_h5(h5_path, role="reader") as f:
                        if "/Seeds/x0_nnls_patch" in f:
                            seed_origin = f["/Seeds/x0_nnls_patch"].attrs.get("origin", None)
                except Exception:
                    seed_origin = None

                if seed_origin and ("nnls_patch" not in seed_origin):
                    initial_freeze = seed_tested_mask & (~seed_support_mask)
                    known_zero[initial_freeze] = True
                else:
                    # nnls_patch -> do not freeze; allow SPG to find amplitude
                    print("[SPG] nnls_patch detected: skipping initial freeze of patch-tested zeros", flush=True)

            # ---------------- Gradient accumulation (PARALLEL) ----------------
            for (s0, s1) in s_ranges:

                x_eff = x.copy()
                inactive = np.ones(C, dtype=bool)
                inactive[active_orbits] = False
                x_eff[inactive, :] = 0.0

                with open_h5(h5_path, role="reader") as f:
                    DC = f["/DataCube"]
                    M  = f["/HyperCube/models"]

                    # ----------------------------------------------------------
                    # 1) Read RAW bin flux (no SB conversion)
                    # ----------------------------------------------------------
                    Y = np.asarray(DC[s0:s1, :], dtype=np.float64)  # (Sblk, L)
                    if keep_idx is not None:
                        Y = Y[:, keep_idx]                           # (Sblk, Lk)

                    # ----------------------------------------------------------
                    # 3) Build prediction in the SAME weighted space
                    # ----------------------------------------------------------
                    yhat = np.zeros_like(Y)

                    c_iter = active_orbits
                    for c in c_iter:
                        A = np.asarray(M[s0:s1, c, :, :], dtype=np.float64)
                        if keep_idx is not None:
                            A = A[:, :, keep_idx]

                        yhat += x_eff[c] @ A

                    # --- accumulate exact amplitude statistics ---
                    # These two scalars are sufficient to optimally rescale x
                    dot_AxY += float(np.sum(yhat * Y))
                    dot_AxAx += float(np.sum(yhat * yhat))

                # ------------------------------------------------------------
                # 4) Residual (already weighted)
                # ------------------------------------------------------------
                R = Y - yhat
                ssq += float(np.sum(R * R))
                nres += R.size

                # ---- prepare worker jobs ----
                jobs = []
                band_map = []

                for (c_start, c_stop) in bands:
                    c_sel = np.intersect1d(active_orbits,
                                        np.arange(c_start, c_stop),
                                        assume_unique=True)
                    if c_sel.size == 0:
                        continue

                    c0 = int(c_sel.min())
                    c1 = int(c_sel.max()) + 1

                    jobs.append((
                        h5_path,
                        int(s0), int(s1),
                        keep_idx,
                        c0, c1,
                        R.copy(order="C"),
                        cfg.dset_slots,
                        cfg.dset_bytes,
                        cfg.dset_w0,
                    ))
                    band_map.append((c0, c1))

                # ---- run workers ----
                results = pool.map(_worker_tile_grad_band_single_metric, jobs)

                for (c0, c1), (g_band, D_band) in zip(band_map, results):
                    g_tot[c0:c1] += g_band
                    D_tot[c0:c1] += D_band

                pbar.update(1)

            pbar.close()

            # ---------------- Normalize to mean-squared objective ----------------
            if nres > 0:
                g_tot /= float(nres)
            # Use this data_proxy consistently for any quadratic model / acceptance
            # checks later in this epoch.  It must reflect the residuals computed
            # above (ssq, nres) for the current x, not the previous epoch.
            data_proxy = 0.5 * ssq / builtins.max(nres, 1)

            # ---------------- Assemble gradient (data term) ----------------
            # Data gradient ONLY
            g_data = -g_tot

            # Add data-shape priors only
            g = g_data.copy()

            data_mask = np.isfinite(g_data)
            norm_data = np.linalg.norm(g_data[data_mask]) if np.any(data_mask) else 0.0

            # ---- population curvature prior (scale-aware) ----
            lambda_pop = float(os.environ.get("CUBEFIT_LAMBDA_POP", "0.0"))
            if lambda_pop > 0.0:
                g_pop = population_age_curvature_grad(x, pop_shape=pop_shape)

                pop_mask = np.isfinite(g_pop)
                norm_pop = np.linalg.norm(g_pop[pop_mask]) if np.any(pop_mask) else 0.0

                if norm_pop > 0.0 and norm_data > 0.0:
                    scale_pop = norm_data / norm_pop
                else:
                    scale_pop = 1.0

                # Clamp scaling to avoid runaway amplification
                scale_pop = float(np.clip(scale_pop, 1e-6, 1e6))

                g += lambda_pop * scale_pop * g_pop

                print(
                    f"[SPG-DBG] POP "
                    f"||g_pop||={norm_pop:.3e} "
                    f"scale={scale_pop:.3e} "
                    f"λ_eff={lambda_pop*scale_pop:.3e}",
                    flush=True,
                )
            else:
                g_pop = None
            
            # ---- per-orbit age smoothness prior (amplitude-normalised / per-mass) ----
            lambda_age = float(os.environ.get("CUBEFIT_LAMBDA_AGE", "0.0"))
            if lambda_age > 0.0:
                g_age_raw = orbit_age_smoothness_grad(x)

                # Per-orbit mass normalization
                s = np.sum(x, axis=1)
                eps_age = 1e-12 * builtins.max(np.max(s), 1.0)
                inv_mass = 1.0 / (s + eps_age)

                g_age = inv_mass[:, None] * g_age_raw

                age_mask = np.isfinite(g_age)
                norm_age = np.linalg.norm(g_age[age_mask]) if np.any(age_mask) else 0.0

                if norm_age > 0.0 and norm_data > 0.0:
                    scale_age = norm_data / norm_age
                else:
                    scale_age = 1.0

                scale_age = float(np.clip(scale_age, 1e-6, 1e6))

                g += lambda_age * scale_age * g_age

                print(
                    f"[SPG-DBG] AGE "
                    f"||g_age||={norm_age:.3e} "
                    f"scale={scale_age:.3e} "
                    f"λ_eff={lambda_age*scale_age:.3e}",
                    flush=True,
                )
            else:
                g_age = None

            # ---------------- Build safe diagonal preconditioner ----------------
            D_raw = D_tot.copy()  # per-col denom (may have zeros)
            pos = np.isfinite(D_raw) & (D_raw > 0.0)

            if np.any(pos):
                D_ref = float(np.median(D_raw[pos]))
            else:
                D_ref = 1.0

            # normalize to median scale (scale-free)
            D = D_raw / builtins.max(D_ref, 1e-30)

            # compute absolute floor: env OR data-driven small fraction of D_ref
            abs_zero_env = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-12"))
            data_floor_mul = float(os.environ.get("CUBEFIT_ZERO_COL_DATAFLOOR_MUL", "1e-8"))
            data_floor = data_floor_mul * builtins.max(D_ref, 1.0)
            abs_zero = builtins.max(abs_zero_env, data_floor)

            # positive denominators get at least abs_zero, others set to 0
            D = np.where(pos, np.maximum(D, abs_zero), 0.0)

            # ---------------- Freeze logic (respect priors) ----------------
            freeze = (
                (D <= 0.0)
                | (~np.isfinite(D))
                | known_zero
            )

            if g_pop is not None:
                pop_eps = 1e-12 * builtins.max(np.max(np.abs(g_pop)), 1.0)
                freeze &= (np.abs(g_pop) <= pop_eps)
            if g_age is not None:
                age_eps = 1e-12 * builtins.max(np.max(np.abs(g_age)), 1.0)
                freeze &= (np.abs(g_age) <= age_eps)
                print(f"[SPG-DBG] ||g_age|| = {np.linalg.norm(g_age):.3e}",
                    flush=True)

            if np.any(freeze):
                g = g.copy()
                g[freeze] = 0.0
                D = D.copy()
                D[freeze] = np.inf

            # ---------------- form capped inverse diag (invD) ----------------
            invD = np.zeros_like(D, dtype=np.float64)
            finite_mask = np.isfinite(D) & (D > 0.0)
            invD[finite_mask] = 1.0 / D[finite_mask]
            # cap invD to avoid enormous steps in low-curvature directions
            max_inv_d = float(os.environ.get("CUBEFIT_MAX_INV_D", "1e6"))
            if np.isfinite(max_inv_d) and (max_inv_d > 0.0):
                invD = np.minimum(invD, max_inv_d)

            # ---------------- per-orbit anti-flatness prior (scale-aware, step-anchored) ----
            # Compute raw prior gradient once (name: g_asmooth_raw)
            lambda_asmooth_env = float(os.environ.get("CUBEFIT_LAMBDA_ASMOOTH", "0.0"))
            if lambda_asmooth_env > 0.0:
                # raw prior gradient (scale-free / unitless direction)
                g_asmooth_raw = orbit_population_variance_grad(x)

                # mask + raw norm for diagnostics
                as_mask = np.isfinite(g_asmooth_raw)
                norm_as_raw = np.linalg.norm(g_asmooth_raw[as_mask]) if np.any(as_mask) else 0.0

                # compute data gradient norm (only finite entries)
                data_mask = np.isfinite(g_data)
                gdata_norm = np.linalg.norm(g_data[data_mask]) if np.any(data_mask) else 0.0

                # compute the *effective* step the prior would cause after preconditioner
                # and BB scaling: eff_vec = alpha_bb * invD * g_asmooth_raw
                # (note: invD should already be present; if not, compute a safe approx)
                try:
                    invD_local = invD  # use precomputed invD
                except NameError:
                    # fallback conservative invD (no division by zero)
                    D_tmp = np.copy(D_raw)
                    pos_tmp = np.isfinite(D_tmp) & (D_tmp > 0.0)
                    Dref_tmp = float(np.median(D_tmp[pos_tmp])) if np.any(pos_tmp) else 1.0
                    Dtmp = D_raw / builtins.max(Dref_tmp, 1e-30)
                    abs_zero_env = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-12"))
                    data_floor_mul = float(os.environ.get("CUBEFIT_ZERO_COL_DATAFLOOR_MUL", "1e-8"))
                    data_floor = data_floor_mul * builtins.max(Dref_tmp, 1.0)
                    abs_zero = builtins.max(abs_zero_env, data_floor)
                    Dtmp = np.where(pos_tmp, np.maximum(Dtmp, abs_zero), 0.0)
                    invD_local = np.zeros_like(Dtmp)
                    finite_mask_tmp = np.isfinite(Dtmp) & (Dtmp > 0.0)
                    invD_local[finite_mask_tmp] = 1.0 / Dtmp[finite_mask_tmp]
                    max_inv_d = float(os.environ.get("CUBEFIT_MAX_INV_D", "1e6"))
                    if np.isfinite(max_inv_d) and (max_inv_d > 0.0):
                        invD_local = np.minimum(invD_local, max_inv_d)

                eff_vec = alpha_bb * invD_local * g_asmooth_raw
                eff_mask = np.isfinite(eff_vec)
                eff_norm = np.linalg.norm(eff_vec[eff_mask]) if np.any(eff_mask) else 0.0

                # Compute automatic λ so the prior's effective step is a small fraction of
                # the data gradient (target_ratio is the fractional influence we allow).
                target_ratio = float(os.environ.get("CUBEFIT_ASMOOTH_REL", "1e-3"))
                lambda_auto = 0.0
                if eff_norm > 0.0 and gdata_norm > 0.0:
                    lambda_auto = target_ratio * gdata_norm / eff_norm

                # final λ is the user-specified floor or the auto value, whichever is larger
                lambda_asmooth = float(builtins.max(lambda_asmooth_env, lambda_auto))

                # clamp lambda to avoid catastrophic amplification
                lambda_max = float(os.environ.get("CUBEFIT_ASMOOTH_LAMBDA_MAX", "1e6"))
                lambda_min = float(os.environ.get("CUBEFIT_ASMOOTH_LAMBDA_MIN", "0.0"))
                lambda_asmooth = float(np.clip(lambda_asmooth, lambda_min, lambda_max))

                # apply prior (if non-zero)
                if lambda_asmooth > 0.0:
                    g += lambda_asmooth * g_asmooth_raw

                # diagnostics
                print(
                    f"[SPG-DBG] ASMOOTH ||g_as_raw||={norm_as_raw:.3e} "
                    f"eff_norm={eff_norm:.3e} gdata_norm={gdata_norm:.3e} "
                    f"λ_env={lambda_asmooth_env:.3e} λ_auto={lambda_auto:.3e} "
                    f"λ_use={lambda_asmooth:.3e}",
                    flush=True,
                )
            else:
                g_asmooth_raw = None

            # ----------------------------
            # BB step (diagonal-preconditioned) - same as before
            # ----------------------------
            if (x_prev is not None) and (g_prev is not None):

                s = (x - x_prev).ravel()
                y_vec = (g - g_prev).ravel()

                sy = float(np.dot(s, y_vec))
                yy = float(np.dot(y_vec, y_vec))
                ss = float(np.dot(s, s))

                if sy > 1e-16 and yy > 1e-16:

                    if ep % 2 == 0:
                        # BB1
                        alpha_bb = ss / sy
                    else:
                        # BB2
                        alpha_bb = sy / yy

                else:
                    alpha_bb = lr  # fallback
                print(f"[BB] alpha={alpha_bb:.3e}, sy={sy:.3e}", flush=True)

            # Safeguards
            alpha_min = float(os.environ.get("CUBEFIT_ALPHA_MIN", "1e-6"))
            alpha_max = float(os.environ.get("CUBEFIT_ALPHA_MAX", "1e2"))
            alpha_bb = float(np.clip(alpha_bb, alpha_min, alpha_max))

            
            # ============================================================
            # Two-prong acceptance guard (sample + optional full)
            # ============================================================

            # ---- snapshot BEFORE step ----
            x_before = x.copy()
            x_prev_before = None if x_prev is None else x_prev.copy()
            g_prev_before = None if g_prev is None else g_prev.copy()

            # ---- build deterministic evenly-spaced tile subset ----
            n_tiles = len(s_ranges)
            if proxy_sample_tiles >= n_tiles:
                s_ranges_sample = s_ranges
            else:
                idxs = []
                if proxy_sample_tiles == 1:
                    idxs = [n_tiles // 2]
                else:
                    for i in range(proxy_sample_tiles):
                        idx = int(round(i * (n_tiles - 1) / (proxy_sample_tiles - 1)))
                        idxs.append(idx)
                idxs = sorted(set(np.clip(idxs, 0, n_tiles - 1)))
                s_ranges_sample = [s_ranges[i] for i in idxs]

            # ---- compute sample RMSE BEFORE step ----
            if validate_flag:
                rmse_sample_before = rmse_proxy_subset(
                    h5_path,
                    x_before,
                    s_ranges_sample,
                    keep_idx,
                    None,
                    None,
                )
                data_sample_before = 0.5 * rmse_sample_before**2
            else:
                rmse_sample_before = None
                data_sample_before = None

            # ============================================================
            # Perform SPG step
            # ============================================================

            dx = -alpha_bb * (g * invD)
            step_norm = np.linalg.norm(dx)
            x_norm = np.linalg.norm(x)
            max_rel_step = float(os.environ.get("CUBEFIT_MAX_REL_STEP", "0.5"))
            if step_norm > max_rel_step * builtins.max(1.0, x_norm):
                scale = (max_rel_step * builtins.max(1.0, x_norm)) / step_norm
                dx *= scale
                print(f"[SPG-DBG] Capped SPG step by max_rel_step={max_rel_step} "
                      f"from {step_norm:.3e} to {scale:.3e}",
                      flush=True)
            x += dx
            x = np.maximum(x, 0.0)

            # ============================================================
            # SAMPLE VALIDATION (cheap)
            # ============================================================

            rejected = False

            if validate_flag:

                rmse_sample_after = rmse_proxy_subset(
                    h5_path,
                    x,
                    s_ranges_sample,
                    keep_idx,
                    None,
                    None,
                )
                data_sample_after = 0.5 * rmse_sample_after**2

                if (
                    rmse_sample_after > rmse_sample_before * rmse_ratio_thresh
                    or data_sample_after > data_sample_before * data_mult_thresh
                ):

                    print(
                        "[SPG-VALID] REJECTED (sample) "
                        f"rmse_before={rmse_sample_before:.3e}, "
                        f"rmse_after={rmse_sample_after:.3e}",
                        flush=True,
                    )

                    x = x_before
                    x_prev = x_prev_before
                    g_prev = g_prev_before
                    alpha_bb = max(1e-12, alpha_bb * 0.5)
                    rejected = True

            # ============================================================
            # If rejected, skip remaining acceptance logic
            # ============================================================

            if rejected:
                continue

            print(f"[SPG-DBG] D_ref={D_ref:.3e}, D.min={float(np.min(D)):.3e}, D.max={float(np.max(D)):.3e}", flush=True)
            frac_finite = np.sum(np.isfinite(D)) / float(D.size)
            print(f"[SPG-DBG] fraction finite denominators = {frac_finite:.3f}", flush=True)

            # ============================================================
            # exact global amplitude rescaling (epoch-wise)
            # ============================================================
            if dot_AxAx > 0.0 and dot_AxY > 0.0:
                gamma = dot_AxY / dot_AxAx

                if np.isfinite(gamma) and gamma > 0.0:
                    x *= gamma

            # ---------------------- POST-ACCEPT PROJECTION ---------------------
            # Apply orbit projection only after the trial step has survived
            # validation (so projection cannot be undone by a rejection).
            projection_applied = False
            if w_target is not None and (ep + 1) > ORBIT_WARM_EPOCHS:

                # Determine which orbits are actually movable
                known_zero_orbit = np.all(known_zero, axis=1)      # (C,)
                D_orbit = np.sum(D_tot, axis=1)                   # (C,)
                active = (~known_zero_orbit) & (D_orbit > 0.0)

                if np.any(active):

                    # --- rank-1 orbit-mass projection (mean-only) ---
                    Pdim = x.shape[1]

                    s = np.sum(x[active, :], axis=1)          # (n_active,)
                    w = w_target[active]                      # (n_active,)

                    alpha = np.sum(s) / np.sum(w)
                    s_proj = alpha * w                        # target per-orbit mass

                    # redistribute mass proportionally
                    # (avoid divide-by-zero)
                    ratio = s_proj / np.maximum(s, 1e-30)
                    x[active, :] *= ratio[:, None]

                    # enforce non-negativity after projection
                    np.maximum(x, 0.0, out=x)

                    projection_applied = True

            if projection_applied:
                print(f"[SPG-PROJ] projection applied at epoch {ep+1}", flush=True)

            # ------------------------------------------------------------
            # Orbit mismatch metric (only meaningful after projection)
            # ------------------------------------------------------------
            if w_target is not None and (ep + 1) > ORBIT_WARM_EPOCHS:
                s_all = np.sum(x, axis=1)
                orbit_mis = np.linalg.norm(s_all - w_target)
            else:
                orbit_mis = 0.0

            # --- update BB history ---
            x_prev = x.copy()
            g_prev = g.copy()

            # ------------------------------------------------------------
            # Reconstruct x from s and y
            # ------------------------------------------------------------
            g_norm = np.linalg.norm(g)
            x_norm = np.linalg.norm(x)

            # ----------------- RECOMPUTE DATA PROXY ON POST-UPDATE x -----------
            # Evaluate a true RMSE proxy for the *current* candidate x (after
            # projection / amplitude rescale). This prevents accepting a new
            # best_x when the real objective has exploded.
            #
            # Make this validation optional (set CUBEFIT_VALIDATE_PROXY=0 to
            # disable for speed once you trust the solver).
            validate = os.environ.get("CUBEFIT_VALIDATE_PROXY", "1")
            validate = False if validate.lower() in ("0", "false", "no", "off") else True
            if validate:
                # rmse_proxy_subset returns sqrt(ssq/n) (same convention as used
                # elsewhere). We derive data_proxy = 0.5 * ssq/n = 0.5 * rmse^2
                rmse_curr = rmse_proxy_subset(
                    h5_path,
                    x,           # current candidate (C,P)
                    s_ranges,    # evaluate full cube (can be changed to subset)
                    keep_idx,
                    None,        # inv_cp_flux_ref not used here
                    None,        # w_lam_sqrt not used here
                )
                rmse = float(rmse_curr)
                data_proxy = 0.5 * (rmse_curr ** 2)
            else:
                # Fall back to the (cheap but stale) proxy if validation disabled.
                # Note: 'ssq' currently reflects residuals computed BEFORE the
                # update, so this branch is not recommended unless you turn it
                # off intentionally for performance.
                data_proxy = 0.5 * ssq / builtins.max(nres, 1)

            # ---------------- Active orbit update (CRITICAL) ----------------
            x_row_l1 = np.sum(x, axis=1)
            g_row_inf = np.max(np.abs(g), axis=1)

            eps_mass = 1e-12 * builtins.max(np.sum(x_row_l1), 1.0)
            eps_grad = 1e-10 * builtins.max(np.max(g_row_inf), 1.0)

            new_active = np.nonzero(
                (x_row_l1 > eps_mass) | (g_row_inf > eps_grad)
            )[0]

            if new_active.size < min_active:
                new_active = np.argsort(g_row_inf)[-min_active:]

            active_orbits = new_active.astype(np.int32)

            # ---------------- Diagnostics ----------------
            rmse = float(rmse_curr) if validate else np.sqrt(ssq / max(nres,1))
            pg_norm = np.linalg.norm(g[np.isfinite(g)])
            D_pos = D[np.isfinite(D) & (D < np.inf)]

            # ------------------------------------------------------------
            # Composite acceptance metric
            # ------------------------------------------------------------
            if w_target is not None and (ep + 1) > ORBIT_WARM_EPOCHS:
                # Scale orbit penalty to data term (dimensionless, conservative)
                scale = builtins.max(data_proxy, 1.0)
                beta = float(os.environ.get("CUBEFIT_ORBIT_ACCEPT_BETA", "1e-6"))

                total_proxy = data_proxy + beta * (orbit_mis / scale) ** 2
            else:
                total_proxy = data_proxy
            
            import matplotlib.pyplot as plt
            plt.clf()
            plt.hist(np.log10(x[x > 0].flatten()), bins=50, color='blue', alpha=0.7)
            plt.title(f"Epoch {ep+1} solution histogram")
            plt.xlabel("x value")
            plt.ylabel("Count")
            plt.savefig(f"spg_epoch_{ep+1}_histogram.png")
            plt.close('all')

            # use total_proxy when comparing best_proxy / saving best_x
            if total_proxy < best_proxy and not rejected:
                print(f"[SPG] new best proxy: {total_proxy:.3e} (previous {best_proxy:.3e})", flush=True)
                best_proxy = total_proxy
                best_x = x.copy()

            print(
                f"[SPG-AMP] gamma={gamma:.3e}  "
                f"<Ax,Y>={dot_AxY:.3e}  "
                f"<Ax,Ax>={dot_AxAx:.3e}",
                flush=True,
            )
            print(
                f"[SPG] epoch {ep+1}  "
                f"RMSE={rmse:.4e}  "
                f"||x||={x_norm:.3e}"
            )
            print(
                f"||g||={pg_norm:.3e}  "
                f"total_proxy={total_proxy:.3e}"
            )
            print(
                f"alpha_bb={alpha_bb:.2e}  "
                f"active={active_orbits.size}/{C}",
                flush=True,
            )
            if w_target is not None and (ep + 1) > ORBIT_WARM_EPOCHS:
                s_full = np.sum(x, axis=1)
                orbit_res = np.linalg.norm(s_full - w_target)
            else:
                orbit_res = 0.0
            print(f"[SPG-orbit] ||s - w||={orbit_res:.3e}", flush=True)
        
        th = cu.zero_floor_inplace(best_x, rel_tol=1e-25, abs_tol=0.0)
        print(f"[SPG] zero-floor applied: threshold={th:.3e}", flush=True)

        elapsed = time.perf_counter() - t0
        return best_x, dict(
            epochs=cfg.epochs,
            elapsed_sec=elapsed,
            rmse_proxy_best=float(best_proxy),
            active_orbits=active_orbits.copy(),
            known_zero_mask=known_zero.copy(),
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

# ------------------------------------------------------------------------------

def solve_kaczmarz_nnls(
    h5_path: str,
    x0: np.ndarray,                    # (C,P)
    *,
    active_orbits: np.ndarray | None = None,
    orbit_weights: np.ndarray | None = None,
    orbit_beta: float | None = None,
    max_epochs: int = 10,
    tol_kkt: float = 1e-6,
    shuffle_spaxels: bool = True,
    apply_mask: bool = True,
    use_lambda_weights: bool = True,
    project_nonneg: bool = True,
    tracker=None,
):
    """
    Row-action solver (block-Kaczmarz) for CubeFit.

    This function preserves the original public signature and HDF5 layout
    used elsewhere in the code base:
    - /HyperCube/models -> M with shape (S, C, P, L)
    - /DataCube         -> DC with shape (S, L)

    The implementation replaces the erroneous per-lambda projection with a
    BLAS-friendly block Kaczmarz update (columns of the model block are the
    equations).  The per-chunk update still occurs inside the c-loop to
    preserve the chunked solver semantics.
    """
    # Defensive copy of x
    x = np.asarray(x0, dtype=np.float64, copy=True)  # shape (C, P)

    # Normalize active_orbits default
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        S, Ctot, P, L = map(int, M.shape)
        chunks = getattr(M, "chunks", None)
        s_tile = int(chunks[0]) if (chunks and chunks[0]) else 128
    s_ranges = [(s0, min(S, s0 + s_tile)) for s0 in range(0, S, s_tile)]

    if active_orbits is None:
        active_orbits = np.arange(Ctot, dtype=int)

    # Canonical orbit-weight target (reuse helper present in module)
    if orbit_weights is not None:
        w_target = _canon_orbit_weights(
            h5_path,
            orbit_weights,
            C=Ctot,
            P=P,
        )
        if orbit_beta is None:
            orbit_beta = float(os.environ.get("CUBEFIT_ORBIT_BETA", "0.2"))
    else:
        w_target = None

    # Main loop over epochs and spaxels
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        # Optional mask -> keep_idx
        keep_idx = None
        if "/Mask" in f:
            m = np.asarray(f["/Mask"][...], bool).ravel()
            keep_idx = np.flatnonzero(m)
            Lk = int(keep_idx.size)
        else:
            Lk = int(L)
        
        binCounts = np.asarray(f['/BinCounts'][...], dtype=int)

        # optional lambda sqrt weights
        w_lam_sqrt = None
        if use_lambda_weights:
            if "/LambdaWeights" in f:
                w_full = np.asarray(f["/LambdaWeights"][...], np.float64).ravel()
                w_lam_sqrt = np.sqrt(np.maximum(w_full, 1e-12))
            else:
                # fall back to helper (same source, consistent behavior)
                try:
                    w_full = cu.read_lambda_weights(h5_path)
                    w_lam_sqrt = np.sqrt(np.maximum(w_full, 1e-12))
                except Exception:
                    w_lam_sqrt = None

            if w_lam_sqrt is not None and keep_idx is not None:
                w_lam_sqrt = w_lam_sqrt[keep_idx]

        for epoch in range(int(max_epochs)):

            max_kkt = 0.0

            # ------------------------------------------------------------
            # Jacobi-style Kaczmarz: accumulate updates over spaxels
            # ------------------------------------------------------------
            dx_accum = np.zeros_like(x, dtype=np.float64)
            n_spax = 0

            for (s0, s1) in tqdm(
                s_ranges,
                desc=f"[Kaczmarz] epoch {epoch+1}/{max_epochs}",
                leave=False,
            ):
                # read data block once
                Y = np.asarray(DC[s0:s1, :], dtype=np.float64)
                if keep_idx is not None:
                    Y = Y[:, keep_idx]

                # binCounts for this tile -> single-metric spatial weight w_s = 1/sqrt(binCounts)
                bs_tile = np.asarray(binCounts[s0:s1], dtype=np.float64)
                if np.any(bs_tile <= 0.0):
                    raise RuntimeError(f"Invalid binCounts in tile {s0}:{s1}")
                w_s = (1.0 / np.sqrt(bs_tile)).astype(np.float64)  # (tile_len,)

                # apply the SAME spatial weighting to data rows
                Yw = Y * w_s[:, None]  # (tile_len, Lk)

                # build effective x for residuals (match SPG active set)
                x_eff = x.copy()
                inactive = np.ones(Ctot, dtype=bool)
                inactive[active_orbits] = False
                x_eff[inactive, :] = 0.0

                # apply epoch-level orbit blending to x_eff if requested (same as before)
                if (w_target is not None) and (orbit_beta is not None) and (orbit_beta > 0.0):
                    ai = active_orbits
                    s_cur = np.sum(x_eff[ai, :], axis=1)
                    s_safe = np.maximum(s_cur, 1e-30)
                    s_tgt = (1.0 - orbit_beta) * s_cur + orbit_beta * w_target[ai]
                    x_eff[ai, :] *= (s_tgt / s_safe)[:, None]

                # compute yhat in weighted space (apply w_s to model rows)
                yhat = np.zeros_like(Yw)
                for c in active_orbits:
                    A_blk = np.asarray(M[s0:s1, c], np.float64, order="C")  # (tile_len, P, L)
                    if keep_idx is not None:
                        A_blk = A_blk[:, :, keep_idx]  # (tile_len, P, Lk)

                    # apply SAME spatial weighting to model rows
                    A_blk *= w_s[:, None, None]

                    yhat += x_eff[c] @ A_blk  # (tile_len, Lk)

                # Residual in same weighted space
                R = Yw - yhat

                # apply λ-weights if present (√w_λ) — same convention as SPG
                if w_lam_sqrt is not None:
                    Rw = R * w_lam_sqrt[None, :]
                else:
                    Rw = R

                # accumulate updates over tile: use weighted A (w_s applied) and λ-weights applied below
                for c in active_orbits:
                    A_blk = np.asarray(M[s0:s1, c], np.float64, order="C")
                    if keep_idx is not None:
                        A_blk = A_blk[:, :, keep_idx]  # shape (tile_len, P, Lk)

                    # apply SAME spatial weighting to model block
                    A_blk *= w_s[:, None, None]

                    # apply lambda weights if present
                    if w_lam_sqrt is not None:
                        Aw = A_blk * w_lam_sqrt[None, None, :]
                    else:
                        Aw = A_blk

                    # compute row_norm2 = sum_p Aw^2  -> shape (tile_len, Lk)
                    row_norm2 = np.sum(Aw * Aw, axis=1)
                    # avoid divide-by-zero
                    row_norm2 = np.where(row_norm2 > 0.0, row_norm2, np.inf)

                    # scale = Rw / row_norm2  -> shape (tile_len, Lk)
                    scale = Rw / row_norm2

                    # flatten Aw to shape (tile_len*Lk, P) for BLAS matvec
                    Aw_flat = Aw.transpose(0, 2, 1).reshape(-1, Aw.shape[1])  # (s*Lk, P)
                    scale_flat = scale.ravel()

                    # update = Aw_flat.T @ scale_flat  -> (P,)
                    if scale_flat.size:
                        update = Aw_flat.T @ scale_flat
                        update /= float(scale_flat.size)  # average over contributions
                    else:
                        update = np.zeros(Aw.shape[1], dtype=np.float64)

                    dx_accum[c] += update

                    # KKT diagnostic for this tile (data gradient sign)
                    # g_c = - sum_{s,λ} (Aw_no_lambda ? A_blk : Aw) * Rw
                    # Use A_blk (un-weighted by sqrt(lambda)) so sign corresponds to raw gradient if desired;
                    # here we compute using Aw (with λ) to match the current metric.
                    g_c = -np.einsum("sl,spl->p", Rw, Aw)
                    viol = np.where(
                        x[c] > 0.0,
                        np.abs(g_c),
                        np.maximum(-g_c, 0.0),
                    )
                    max_kkt = max(max_kkt, float(np.max(viol)))

                n_spax += (s1 - s0)

            # ------------------------------------------------------------
            # Optional proximal pull toward orbit target (epoch-level)
            # ------------------------------------------------------------
            if (w_target is not None) and (orbit_beta is not None) and (orbit_beta > 0.0):
                # small regularization toward target orbit weights
                alpha = float(os.environ.get("CUBEFIT_ORBIT_PROX_ALPHA", "1e-3"))
                for c in range(Ctot):
                    dx_accum[c] += -alpha * (x[c] - w_target[c]) * n_spax

            # ------------------------------------------------------------
            # Apply averaged update once per epoch
            # ------------------------------------------------------------
            if n_spax > 0:
                x += dx_accum / n_spax
                if project_nonneg:
                    np.maximum(x, 0.0, out=x)

            # Optionally snapshot x into tracker (non-blocking sidecar).
            if tracker is not None:
                try:
                    # snapshot periodically / on demand
                    tracker.maybe_snapshot_x(x, epoch=epoch + 1, rmse=None, force=False)
                except Exception:
                    # tracker is best-effort: do not fail solver on tracker errors
                    pass

            # epoch end: print and optionally persist
            print(
                f"[Kaczmarz] epoch {epoch+1}/{max_epochs}, KKT_inf={max_kkt:.3e}",
                flush=True,
            )

            # Convergence criterion
            if (tol_kkt is not None) and (max_kkt < tol_kkt):
                print("[Kaczmarz] Converged (KKT residual small).", flush=True)
                break

    return x

# ------------------------------------------------------------------------------

def probe_kaczmarz_tile(
    h5_path: str,
    s0: int | None = None,
    s1: int | None = None,
    c: int | None = None,
    lr: float = 0.25,
    x_source: str = "auto",   # "auto" | "zeros"
    project_nonneg: bool = True,
):
    """
    Single-band probe that mirrors the worker math on one component.
    Uses the same λ-weighting and global energy blend, so scale matches.
    """

    bt_steps   = int(np.max((0, int(os.environ.get("CUBEFIT_BT_STEPS", "3")))))
    bt_factor  = float(os.environ.get("CUBEFIT_BT_FACTOR", "0.5"))
    tau_trust  = float(os.environ.get("CUBEFIT_TRUST_TAU", "0.7"))
    eps        = float(os.environ.get("CUBEFIT_EPS", "1e-12"))
    rel_zero   = float(os.environ.get("CUBEFIT_ZERO_COL_REL", "1e-12"))
    abs_zero   = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-24"))
    tau_global = float(os.environ.get("CUBEFIT_GLOBAL_TAU", "0.5"))
    beta_blend = float(os.environ.get("CUBEFIT_GLOBAL_ENERGY_BLEND", "1e-2"))

    with h5py.File(h5_path, "r") as f:
        M  = f["/HyperCube/models"]  # (S,C,P,L)
        DC = f["/DataCube"]          # (S,L)
        S, Ctot, P, L = map(int, M.shape)
        chunks = M.chunks or (S, 1, P, L)
        S_chunk = int(chunks[0])

        if s0 is None or s1 is None:
            s0 = 0
            s1 = int(np.min((S, S_chunk)))
        if c is None:
            c = int(Ctot // 2)

        keep_idx = None
        if "/Mask" in f:
            m = np.asarray(f["/Mask"][...], bool).ravel()
            keep_idx = np.flatnonzero(m)
        Lk = int(L if keep_idx is None else keep_idx.size)

        # x source
        if x_source == "auto" and "/X_global" in f:
            x1d = np.asarray(f["/X_global"][...], np.float64, order="C")
            x_CP = x1d.reshape(Ctot, P)
        else:
            x_CP = np.zeros((Ctot, P), np.float64)

        # Y (tile), global ||Y||
        Y = np.asarray(DC[s0:s1, :], np.float64, order="C")
        if keep_idx is not None:
            Y = Y[:, keep_idx]  # (Sblk, Lk)
        Sblk = int(s1 - s0)

        Yglob2 = 0.0
        for t0 in range(0, S, S_chunk):
            t1 = int(np.min((S, t0 + S_chunk)))
            Yt = np.asarray(DC[t0:t1, :], np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            Yglob2 += float(np.sum(Yt * Yt))
        Y_glob_norm = float(np.sqrt(Yglob2))

        # yhat (tile) exactly like the solver
        yhat = np.zeros((Sblk, Lk), np.float64)
        for cc in range(Ctot):
            A_cc = np.asarray(M[s0:s1, cc, :, :], np.float32, order="C")
            if keep_idx is not None:
                A_cc = A_cc[:, :, keep_idx]
            xc = x_CP[cc, :].astype(np.float64, copy=False)
            for s in range(Sblk):
                yhat[s, :] += xc @ A_cc[s, :, :]

        R = Y - yhat

        # ---- worker-like band update on component c ----
        A = np.asarray(M[s0:s1, c, :, :], np.float32, order="C")
        if keep_idx is not None:
            A = A[:, :, keep_idx]  # (Sblk, P, Lk)
        cp_flux_ref = cu._ensure_cp_flux_ref(h5_path, keep_idx=None if keep_idx is None else np.arange(L)[keep_idx])
        A = A * (1.0 / cp_flux_ref[int(c), :])[None, :, None]

        # sanitize
        badR = ~np.isfinite(R); R[badR] = 0.0
        badA = ~np.isfinite(A); A[badA] = 0.0

        # λ-weights (mirror main solver)
        lamw_enable = os.environ.get(
            "CUBEFIT_LAMBDA_WEIGHTS_ENABLE", "1"
        ).lower() not in ("0", "false", "no", "off")
        if lamw_enable and "/HyperCube/lambda_weights" in f:
            w_full = np.asarray(f["/HyperCube/lambda_weights"][...],
                                np.float64)
            if keep_idx is not None:
                w_lam_sqrt = np.sqrt(np.maximum(w_full[keep_idx], 1e-6))
            else:
                w_lam_sqrt = np.sqrt(np.maximum(w_full, 1e-6))
        else:
            w_lam_sqrt = None

        # gradient (weighted)
        if w_lam_sqrt is not None:
            A_w = A * w_lam_sqrt[None, None, :]
            Rw  = R * w_lam_sqrt[None, :]
        else:
            A_w = A; Rw = R

        g = np.zeros((P,), np.float64)
        for s in range(Sblk):
            g += A_w[s, :, :].astype(np.float64, copy=False) @ Rw[s, :]

        # local per-column denom (weighted)
        col_denom = np.sum(np.square(A_w, dtype=np.float64), axis=(0, 2))

        # freeze near-zero columns (tile-local)
        med_energy = float(np.median(col_denom[col_denom > 0])) if np.any(col_denom > 0) else 0.0
        tiny_col = np.max((abs_zero, rel_zero * med_energy))
        freeze = col_denom <= tiny_col
        if np.any(freeze):
            g[freeze] = 0.0
            col_denom = np.where(freeze, np.inf, col_denom)

        invD = 1.0 / np.maximum(col_denom, eps)
        dx_c = float(lr) * (g * invD)  # (P,)

        # ΔR for alpha=1 (unweighted)
        R_delta = np.zeros((Sblk, Lk), np.float64)
        for s in range(Sblk):
            R_delta[s, :] -= (
                A[s, :, :].astype(np.float64, copy=False).T @ dx_c
            )

        # trust region (tile, weighted)
        if w_lam_sqrt is not None:
            Rw_delta = R_delta * w_lam_sqrt[None, :]
            rn = float(np.linalg.norm(R * w_lam_sqrt[None, :]))
        else:
            Rw_delta = R_delta
            rn = float(np.linalg.norm(R))
        rd = float(np.linalg.norm(Rw_delta))
        alpha_max = 1.0 if rd == 0.0 else min(1.0, (tau_trust * rn) / rd)

        # backtracking
        alpha = alpha_max
        def _rmse_w(MAT):  # weighted RMSE helper
            if w_lam_sqrt is None:
                return float(np.sqrt(np.mean(MAT * MAT)))
            Z = MAT * w_lam_sqrt[None, :]
            return float(np.sqrt(np.mean(Z * Z)))

        rmse_before = _rmse_w(R)
        rmse_after  = _rmse_w(R + alpha * R_delta)
        if not (rmse_after < rmse_before):
            a = alpha
            for _ in range(bt_steps):
                a *= bt_factor
                if a <= 0.0:
                    break
                rmse_after = _rmse_w(R + a * R_delta)
                if rmse_after < rmse_before:
                    alpha = a
                    break
            else:
                alpha = a

        # global cap
        upd_energy_sq = float(np.sum((dx_c.astype(np.float64) ** 2) * Eg_row))
        if (upd_energy_sq > 0.0) and (Y_glob_norm > 0.0):
            step_norm_global = float(np.sqrt(upd_energy_sq)) * alpha
            cap = float(tau_global * Y_glob_norm)
            if step_norm_global > cap:
                alpha *= float(np.minimum(1.0, cap / np.maximum(1e-12, step_norm_global)))

        dx_c *= alpha
        if project_nonneg:
            over_neg = dx_c < -x_CP[c, :]
            if np.any(over_neg):
                dx_c[over_neg] = -x_CP[c, :][over_neg]
                R_delta.fill(0.0)
                for s in range(Sblk):
                    R_delta[s, :] -= (
                        A[s, :, :].astype(np.float64, copy=False).T @ dx_c
                    )
        else:
            if alpha != 1.0:
                R_delta *= alpha

        R_after = R + R_delta
        yhat_norm = float(np.linalg.norm(yhat))
        yhat_next_norm = float(np.linalg.norm(yhat - R_delta))

        out = {
            "rmse_before": float(np.sqrt(np.mean(R * R))),
            "rmse_after":  float(np.sqrt(np.mean(R_after * R_after))),
            "y_norm":      float(np.linalg.norm(Y)),
            "yhat_norm":   yhat_norm,
            "yhat_next_norm": yhat_next_norm,
            "g_norm":      float(np.linalg.norm(g)),
            "dx_norm":     float(np.linalg.norm(dx_c)),
            "global_upd_norm": float(np.sqrt(np.maximum(0.0, upd_energy_sq)) * alpha),
            "Y_glob_norm": Y_glob_norm,
            "Sblk":        Sblk,
            "Lk":          Lk,
            "c":           int(c),
            "alpha":       float(alpha),
            "frozen_cols": int(np.count_nonzero(freeze)),
        }
        print("[Probe]", out)
        return out

# ------------------------------------------------------------------------------