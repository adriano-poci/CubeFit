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
            initilisation;
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
from CubeFit.hypercube_builder import read_global_column_energy
from CubeFit import cube_utils as cu

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

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M  = f["/HyperCube/models"]

        for (s0, s1) in tile_ranges:
            Y = np.asarray(DC[s0:s1, :], np.float64)
            if keep_idx is not None:
                Y = Y[:, keep_idx]

            yhat = np.zeros_like(Y)
            for c in range(x_CP.shape[0]):
                A = np.asarray(M[s0:s1, c, :, :], np.float64)
                if keep_idx is not None:
                    A = A[:, :, keep_idx]
                if inv_cp_flux_ref is not None:
                    A = A * inv_cp_flux_ref[c, :][None, :, None]

                # x_CP[c,:] shape (P,), A shape (Sblk, P, Lk)
                # tensordot over population dim -> (Sblk, Lk)
                yhat += np.tensordot(A, x_CP[c, :], axes=([1], [0]))

            R = Y - yhat
            if not np.all(np.isfinite(R)):
                R = np.nan_to_num(R, copy=False)

            # --- RMSE proxy must match gradient objective ---
            if w_lam_sqrt is not None:
                Rw = R * w_lam_sqrt[None, :]
                ssq += float(np.sum(Rw * Rw))
            else:
                ssq += float(np.sum(R * R))

            nres += int(R.size)

    return float(np.sqrt(ssq / builtins.max(nres, 1)))

# ------------------------------------------------------------------------------

def project_orbit_ratios(s: np.ndarray, w_target: np.ndarray) -> np.ndarray:
    """
    Project per-orbit masses s onto the ray spanned by w_target:
        s <- alpha * w_target
    where alpha preserves total mass.

    Both s and w_target must be non-negative.
    """
    s_sum = float(np.sum(s))
    w_sum = float(np.sum(w_target))

    if s_sum <= 0.0 or w_sum <= 0.0:
        return s  # nothing sensible to do

    alpha = s_sum / w_sum
    return alpha * w_target

# ---------------------------- Worker ---------------------------------

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
        col_scale, # shape (C, P), column scaling factors
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

            # A = A / col_scale[c][None, :, None]

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
    # Column-energy normalisation
    # ------------------------------------------------------------
    E_global = read_global_column_energy(h5_path)  # shape (C, P)

    # Robust reference scale
    E_ref = np.median(E_global[E_global > 0.0])
    if not np.isfinite(E_ref) or E_ref <= 0.0:
        raise RuntimeError("Invalid global column energy scale")

    # Column scaling factors (sqrt because curvature ~ A^2)
    col_scale = np.sqrt(E_global / E_ref)

    # Safety: avoid divide-by-zero
    col_scale = np.where(col_scale > 0.0, col_scale, 1.0)

    print(
        f"[SPG] Column-energy normalisation enabled: "
        f"E_ref={E_ref:.3e}",
        flush=True,
    )

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
    # Reparameterise x = s * y
    # ------------------------------------------------------------
    eps = 1e-30
    # orbit masses
    s = x.sum(axis=1)        # shape (C,)
    # avoid zero-mass orbits
    s_safe = np.where(s > 0.0, s, 1.0)
    # per-orbit SFHs (simplex)
    y_dist = x / s_safe[:, None]  # shape (C,P)
    # enforce simplex exactly
    y_dist = np.maximum(y_dist, 0.0)
    y_dist /= (y_dist.sum(axis=1, keepdims=True) + eps)
    # reconstruct x exactly
    x = s[:, None] * y_dist


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
        # scale prior to match the current total mass scale in x
        total_mass_est = float(np.sum(x))
        # if x is all zeros at start, use a safe proxy: global ||Y|| or 1.0
        if total_mass_est <= 0.0:
            # fallback: scale to total observed flux (or 1.0)
            total_mass_est = builtins.max(1.0, Y_glob_norm)  # Y_glob_norm computed earlier
        w_target = w_target * total_mass_est
        print(f"[SPG] scaled w_target by total_mass_est={total_mass_est:.3e}", flush=True)
    # ------------------------------------------------------------
    # Augmented Lagrangian state for exact orbit ratios
    # ------------------------------------------------------------
    if w_target is not None:
        u_orbit = np.zeros(C, dtype=np.float64)

        rho0 = float(os.environ.get("CUBEFIT_ORBIT_RHO0", "1e-3"))
        rho_growth = float(os.environ.get("CUBEFIT_ORBIT_RHO_GROWTH", "2.0"))
        rho_max = float(os.environ.get("CUBEFIT_ORBIT_RHO_MAX", "1e6"))
    else:
        u_orbit = None
    # ------------------------------------------------------------
    # Build seed-support field for SPG update gating
    # ------------------------------------------------------------
    seed_support = None
    if x0 is not None:
        x_seed_cp = x.copy()   # x currently holds the NNLS seed
        sigma_c = float(os.environ.get("CUBEFIT_SEED_SIGMA_C", "2.0"))
        sigma_p = float(os.environ.get("CUBEFIT_SEED_SIGMA_P", "2.0"))
        seed_support = diffuse_seed_full_CP(
            x_seed_cp,
            sigma_c=sigma_c,
            sigma_p=sigma_p,
        )
        print(
            f"[SPG] built seed-support (sigma_c={sigma_c}, sigma_p={sigma_p})",
            flush=True,
        )

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

    orbit_prior_active = False


    # ============================================================
    # Main epochs
    # ============================================================
    try:
        for ep in range(cfg.epochs):

            g_tot = np.zeros_like(x)
            D_tot = np.zeros_like(x)
            ssq = 0.0
            nres = 0

            pbar = tqdm(
                total=len(s_ranges),
                desc=f"[SPG] epoch {ep+1}/{cfg.epochs}",
                mininterval=2.0,
                dynamic_ncols=True,
            )
            if ep == 0 and seed_support_mask is not None and w_target is None:
                # Freeze bins that patch tested and found zero,
                # but only if they are not already ACTIVE
                initial_freeze = seed_tested_mask & (~seed_support_mask)
                known_zero[initial_freeze] = True

            # ---------------- Gradient accumulation (PARALLEL) ----------------
            for (s0, s1) in s_ranges:

                x_eff = x.copy()
                inactive = np.ones(C, dtype=bool)
                inactive[active_orbits] = False
                x_eff[inactive, :] = 0.0

                with open_h5(h5_path, role="reader") as f:
                    DC = f["/DataCube"]
                    M  = f["/HyperCube/models"]

                    # ------------------------------------------------------------
                    # 1) Read RAW bin flux (no SB conversion)
                    # ------------------------------------------------------------
                    Y = np.asarray(DC[s0:s1, :], dtype=np.float64)  # (Sblk, L)
                    if keep_idx is not None:
                        Y = Y[:, keep_idx]                           # (Sblk, Lk)

                    # ------------------------------------------------------------
                    # 3) Build prediction in the SAME weighted space
                    # ------------------------------------------------------------
                    yhat = np.zeros_like(Y)

                    c_iter = active_orbits
                    for c in c_iter:
                        A = np.asarray(M[s0:s1, c, :, :], dtype=np.float64)
                        # (Sblk, P, L)
                        if keep_idx is not None:
                            A = A[:, :, keep_idx] # (Sblk, P, Lk)

                        # A = A / col_scale[c][None, :, None]
                        yhat += x_eff[c] @ A

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
                        col_scale,
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
                D_tot /= float(nres)

            # ---------------- Assemble gradient (data term) ----------------
            # Data gradient ONLY
            g_data = -g_tot

            # Add data-shape priors only
            g = g_data.copy()

            # ---- population curvature prior ----
            lambda_pop = float(os.environ.get("CUBEFIT_LAMBDA_POP", "0.0"))
            if lambda_pop > 0.0:
                g_pop = population_age_curvature_grad(x, pop_shape)
                g += lambda_pop * g_pop
            else:
                g_pop = None
            # ---- per-orbit age smoothness prior ----
            lambda_age = float(os.environ.get("CUBEFIT_LAMBDA_AGE", "0.0"))
            if lambda_age > 0.0:
                g_age = orbit_age_smoothness_grad(x)
                g += lambda_age * g_age
            else:
                g_age = None

            # ------------------------------------------------------------
            # Augmented Lagrangian orbit constraint (A1)
            # ------------------------------------------------------------
            if w_target is not None:
                # penalty schedule
                rho_orbit = min(rho0 * (rho_growth ** ep), rho_max)

                # per-orbit mass
                s = np.sum(x, axis=1)          # (C,)
                r = s - w_target               # (C,)

                # gradient contribution (broadcast to populations)
                g_orbit = (u_orbit + rho_orbit * r)[:, None]

                # add to total gradient
                g += g_orbit

            # --- capture raw (pre-freeze) gradient for orbit-mass updates / diagnostics
            # This is the key: compute orbit-level gradient BEFORE any freeze mutates g.
            g_raw = g.copy()
            g_s_data = np.sum(g_raw * y_dist, axis=1)   # (C,)

            # ---------------- Build safe diagonal preconditioner ----------------
            D_raw = D_tot.copy()  # per-col denom (may have zeros)
            # Balance curvature using column energy
            D_raw /= (col_scale ** 2)
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

            # ----------------------------
            # BB step (diagonal-preconditioned) - same as before
            # ----------------------------
            if (x_prev is not None) and (g_prev is not None):
                dx_flat = (x - x_prev).ravel()
                dg_flat = (g - g_prev).ravel()
                sy = float(np.dot(dx_flat, dg_flat))

                if sy > 1e-16:
                    alpha_bb = float(np.dot(dx_flat, dx_flat) / sy)
                    alpha_bb = float(np.clip(alpha_bb, 1e-8, 1e8))
                print(f"[BB] alpha={alpha_bb:.3e}, sy={sy:.3e}", flush=True)

            # ------------------------------------------------------------
            # SPG step in orbit-mass space (s)
            # Use the pre-freeze orbit gradient (g_s_data) so freezing does not
            # zero out the orbit-level descent direction.
            # ------------------------------------------------------------

            # simple preconditioner for s (collapse D)
            D_s = np.sum(D, axis=1)
            D_s = np.where(D_s > 0.0, D_s, np.inf)

            # BB step for s (use g_s_data computed from g_raw)
            if w_target is not None:
                # Pure AL-driven orbit descent (A1)
                r = s - w_target
                ds = -alpha_bb * (u_orbit + rho_orbit * r)
            else:
                ds = -alpha_bb * (g_s_data / D_s)

            # trust region on s
            s_norm = np.linalg.norm(s)
            ds_norm = np.linalg.norm(ds)
            s_cap = (0.005 if ep == 0 else 0.005) * builtins.max(s_norm, 1.0)

            if ds_norm > s_cap and ds_norm > 0:
                ds *= s_cap / ds_norm

            # apply and project (keep non-neg)
            s += ds
            s = np.maximum(s, 0.0)

            # after computing D and D_ref and D_s
            print(f"[SPG-DBG] D_ref={D_ref:.3e}, D.min={float(np.min(D)):.3e}, D.max={float(np.max(D)):.3e}", flush=True)
            print(f"[SPG-DBG] D_s min/max = {float(np.min(D_s)):.3e}/{float(np.max(D_s)):.3e}", flush=True)
            frac_finite = np.sum(np.isfinite(D)) / float(D.size)
            print(f"[SPG-DBG] fraction finite denominators = {frac_finite:.3f}", flush=True)
            print(f"[SPG-DBG] ||w_target||={0.0 if w_target is None else np.linalg.norm(w_target):.3e}, sum(s)={np.sum(s):.3e}", flush=True)
            if w_target is not None:
                # show a few orbit-level values to eyeball scale
                print("[SPG-DBG] sample w_target, s, D_s for first 8 orbits:",
                    np.vstack([w_target[:8], s[:8], D_s[:8]]).T, flush=True)

            # ------------------------------------------------------------
            # Robust projected update of y (orbit-internal SFH)
            #   - per-row preconditioning
            #   - per-orbit L1 step cap
            #   - exact simplex projection
            #   - hard support mask from NNLS seed
            # ------------------------------------------------------------
            lambda_y = float(os.environ.get("CUBEFIT_LAMBDA_Y", "5e-4"))
            if lambda_y > 0.0:
                # Gradient wrt y (holding s fixed)
                g_y = s[:, None] * g_data

                # Remove null direction
                g_y -= (np.sum(g_y, axis=1, keepdims=True) * y_dist)

                # Per-row preconditioning
                D_rowcol = D.reshape(C, P)
                row_scale = np.maximum(D_rowcol.sum(axis=1), eps)

                dy_raw = -lambda_y * (g_y / row_scale[:, None])

                # Per-orbit L1 cap
                max_l1 = float(os.environ.get("CUBEFIT_Y_MAX_L1", "0.05"))
                l1 = np.sum(np.abs(dy_raw), axis=1)
                scale = np.minimum(1.0, max_l1 / (l1 + eps))
                dy = dy_raw * scale[:, None]

                # -------- HARD SUPPORT MASK --------
                row_max = D_rowcol.max(axis=1, keepdims=True)
                support_mask = D_rowcol >= (1e-3 * row_max)

                if seed_support is not None:
                    support_mask |= (seed_support > 1e-3)

                # Apply update ONLY on supported bins
                y_new = y_dist.copy()
                y_new[support_mask] += dy[support_mask]

                # Enforce exact zeros outside support
                y_new[~support_mask] = 0.0

                # Renormalise ONLY over supported bins
                row_sum = y_new.sum(axis=1, keepdims=True)
                y_dist = np.divide(
                    y_new,
                    row_sum,
                    out=np.zeros_like(y_new),
                    where=row_sum > eps,
                )

                # Diagnostics
                mean_support = float(np.mean(support_mask.sum(axis=1)))
                mean_l1 = float(np.mean(np.sum(np.abs(y_dist - y_new), axis=1)))
                print(
                    f"[SPG-y] mean supported bins/orbit={mean_support:.2f}",
                    flush=True,
                )


            # --- update BB history ---
            x_prev = x.copy()
            g_prev = g.copy()

            # ------------------------------------------------------------
            # Reconstruct x from s and y
            # ------------------------------------------------------------
            x = s[:, None] * y_dist
            g_norm = np.linalg.norm(g)
            x_norm = np.linalg.norm(x)

            # ------------------------------------------------------------
            # Augmented Lagrangian multiplier update (A1)
            # ------------------------------------------------------------
            if w_target is not None:
                s = np.sum(x, axis=1)
                u_orbit += rho_orbit * (s - w_target)

                orbit_err = np.linalg.norm(s - w_target) / np.linalg.norm(w_target)
                print(f"[SPG-AL] epoch {ep+1}  ||s-w||/||w|| = {orbit_err:.3e}", flush=True)


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

            # ---------------- Diagnostics (new, meaningful set) ----------------
            rmse = np.sqrt(ssq / builtins.max(nres, 1))
            pg_norm = np.linalg.norm(g[np.isfinite(g)])
            D_pos = D[np.isfinite(D) & (D < np.inf)]

            # compute data proxy (you have ssq normalized earlier)
            data_proxy = 0.5 * ssq / builtins.max(nres, 1)  # or whichever proxy you use for data term
            total_proxy = data_proxy# + f_orbit
            
            import matplotlib.pyplot as plt
            plt.clf()
            plt.hist(np.log10(x[x > 0].flatten()), bins=50, color='blue', alpha=0.7)
            plt.title(f"Epoch {ep+1} solution histogram")
            plt.xlabel("x value")
            plt.ylabel("Count")
            plt.savefig(f"spg_epoch_{ep+1}_histogram.png")
            plt.close('all')

            # use total_proxy when comparing best_proxy / saving best_x
            if total_proxy < best_proxy:
                best_proxy = total_proxy
                best_x = x.copy()

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
            orbit_res = np.linalg.norm(s - w_target) if w_target is not None else 0.0
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

        # --- global energy blend
        E_global = read_global_column_energy(h5_path)  # (C,P)
        Eg_row = np.asarray(E_global[int(c), :], np.float64)  # (P,)
        col_denom = np.maximum(col_denom, float(beta_blend) * Eg_row)

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