# -*- coding: utf-8 -*-
r"""
    block_coord_nnls.py
    Adriano Poci
    University of Oxford
    2026

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
v1.0:   7 March 2026
"""

from __future__ import annotations, print_function

import os, sys, traceback
import math, builtins
import time
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List, Dict
from contextlib import contextmanager
try:
    from scipy.optimize import nnls as _scipy_nnls
    _HAS_SCIPY_NNLS = True
except Exception:
    _HAS_SCIPY_NNLS = False
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import h5py

from CubeFit.hdf5_manager import open_h5
from CubeFit import cube_utils as cu
from CubeFit.hypercube_builder import read_global_column_energy


def _init_worker(blas_threads: int):
    # called once per process; set BLAS env vars
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, blas_threads // 2))

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

# helper fallback projected-gradient for NNLS (small blocks)
def _projected_gradient_nnls(A, b, max_iter=200, tol=1e-8):
    A = np.asarray(A, dtype=np.float64, order="C")
    b = np.asarray(b, dtype=np.float64, order="C")
    m, n = A.shape
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    ATA = A.T @ A
    ATb = A.T @ b
    x = np.zeros((n,), dtype=np.float64)
    L = np.linalg.norm(ATA, 2)
    if not np.isfinite(L) or L <= 0:
        L = 1.0
    step = 1.0 / L
    for k in range(max_iter):
        grad = ATA @ x - ATb
        x_new = x - step * grad
        x_new[x_new < 0] = 0.0
        if np.linalg.norm(x_new - x) < tol * (1.0 + np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    return x

# ------------------------------------------------------------------------------

def _nnls_from_quadratic(
    ATA: np.ndarray,
    ATy: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Solve: min_{z >= 0} 0.5 * z^T ATA z - ATy^T z
    using projected gradient on the quadratic form (ATA small, dense).
    """
    ATA = np.asarray(ATA, dtype=np.float64, order="C")
    ATy = np.asarray(ATy, dtype=np.float64, order="C")
    n = ATA.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    # initial guess
    if x0 is None:
        x = np.zeros((n,), dtype=np.float64)
    else:
        x = np.maximum(0.0, np.asarray(x0, dtype=np.float64).ravel())

    # Lipschitz estimate = spectral norm of ATA (power iteration)
    def _spectral_norm(mat, iters=10):
        v = np.random.default_rng(123456).normal(size=(mat.shape[1],))
        v /= (np.linalg.norm(v) + 1e-30)
        for _ in range(iters):
            v = mat @ v
            nv = np.linalg.norm(v)
            if nv == 0:
                return 0.0
            v /= nv
        # Rayleigh quotient approx
        w = mat @ v
        return float(np.dot(v, w))

    L = _spectral_norm(ATA)
    if not np.isfinite(L) or L <= 0.0:
        L = 1.0
    step = 1.0 / L

    prev_norm = np.linalg.norm(x)
    for k in range(max_iter):
        grad = ATA @ x - ATy
        x_new = x - step * grad
        # project
        np.maximum(x_new, 0.0, out=x_new)
        diff = np.linalg.norm(x_new - x)
        x = x_new
        cur_norm = np.linalg.norm(x)
        if diff <= tol * (1.0 + cur_norm):
            break
        # optionally adjust step with backtracking if divergence occurs (rare)
        if k % 200 == 0 and k > 0:
            # recompute Lipschitz if stagnating
            L = _spectral_norm(ATA)
            if np.isfinite(L) and L > 0:
                step = 1.0 / L
    return x

# ------------------------------------------------------------------------------

def solve_monolithic_nnls(
    h5_path: str,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    orbit_beta: float = 0.0,
    hard_project: bool = False,
    apply_mask: bool = True,
):
    """
    True monolithic NNLS baseline with orbit-weight support.

    Streams tiles, assembles global normal equations ATA, ATy (in raw units),
    optionally applies a soft orbit prior (orbit_beta) as an augmentation to
    ATA/ATy, solves the global NNLS, and optionally applies a hard orbit
    projection so per-orbit totals match the supplied prior.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 dataset (same layout as the rest of this codebase).
    orbit_weights : array-like or None
        Prior per-orbit (C,) or per-column (C*P,) weights. If provided will be
        canonicalized via _canon_orbit_weights().
    orbit_beta : float
        Soft prior strength. If >0, augments ATA/ATy per-orbit with a
        rank-1 penalty in the style used elsewhere in the code.
    hard_project : bool
        If True and orbit_weights provided, apply the hard projection after
        solving so each active orbit's sum matches the prior-scale (preserves
        relative prior proportions).
    apply_mask : bool
        Whether to apply the wavelength mask (consistent with other routines).

    Returns
    -------
    x : ndarray (C, P)
        NNLS solution in original units.
    stats : dict
        Basic diagnostics: l1, nonzero_count, CP.
    """
    # Gather metadata
    with open_h5(h5_path, role="reader") as f:
        S, L = map(int, f["/DataCube"].shape)
        _, C, P, Lm = map(int, f["/HyperCube/models"].shape)
        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch")
        mask = cu._get_mask(f) if apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        chunks = f["/HyperCube/models"].chunks
        s_tile = int(chunks[0]) if (chunks and chunks[0]) else 128

    s_ranges = [(s0, min(S, s0 + s_tile)) for s0 in range(0, S, s_tile)]
    CP = int(C * P)

    # canonicalize orbit_weights (if any)
    w_target = None
    if orbit_weights is not None:
        w_target = _canon_orbit_weights(h5_path, orbit_weights, C=C, P=P)
        # ensure finite and non-negative
        w_target = np.nan_to_num(w_target, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        w_target[w_target < 0.0] = 0.0
        # if prior sums to zero (degenerate) drop it
        if np.sum(w_target) <= 0.0:
            w_target = None

    # allocate global normal equations (streamed assembly)
    ATA = np.zeros((CP, CP), dtype=np.float64)
    ATy = np.zeros((CP,), dtype=np.float64)

    print(f"[MONO] assembling global normal equations CP={CP}", flush=True)
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M  = f["/HyperCube/models"]

        # try to encourage chunk cache like other routines
        try:
            M.id.set_chunk_cache(1_000_003, 256 * 1024**2, 0.90)
        except Exception:
            pass

        for (s0, s1) in tqdm(s_ranges, desc="[MONO] tiles", disable=False):
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]

            Sblk = s1 - s0
            Lk = Yt.shape[1]

            # load model tile: (Sblk, C, P, Lk)
            A_block = np.asarray(M[s0:s1, :, :, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                A_block = A_block[:, :, :, keep_idx]

            # reshape to design (Sblk*Lk, CP)
            A2 = A_block.transpose(0, 3, 1, 2).reshape(Sblk * Lk, CP)
            y_flat = Yt.reshape(-1)

            # accumulate normal equations (note: uses data y, not residual)
            ATA += A2.T @ A2
            ATy += A2.T @ y_flat

            # update D_tot is optional here; the block solver tracks D_tot separately.
            # If you need D_tot for later scaling, compute it separately.

    # Optionally apply soft orbit prior as a rank-1 augmentation per-orbit.
    # We follow the same per-orbit strategy: for orbit cc, for columns idx = cc*P:(cc+1)*P
    if (w_target is not None) and (orbit_beta is not None) and (orbit_beta > 0.0):
        print(f"[MONO] applying soft orbit prior (beta={orbit_beta})", flush=True)
        for cc in range(C):
            idx0 = cc * P
            idx1 = idx0 + P
            # u = ones (P,)
            u = np.ones((P,), dtype=np.float64)
            # ATA_block += 2*beta * (u outer u)
            ATA[idx0:idx1, idx0:idx1] += (2.0 * float(orbit_beta)) * np.outer(u, u)
            # ATy_block += 2*beta * w_cc  (distribute prior mass equally across populations)
            # scale w_target[cc] into per-population vector (uniform across P)
            ATy[idx0:idx1] += (2.0 * float(orbit_beta)) * float(w_target[cc]) * np.ones((P,), dtype=np.float64)

    # small global stabilizer
    ATA += 1e-12 * np.eye(CP, dtype=np.float64)

    # Solve global quadratic NNLS
    print("[MONO] solving global NNLS (projected-gradient on ATA/ATy)...", flush=True)
    x_flat = _nnls_from_quadratic(ATA, ATy, x0=None, max_iter=5000, tol=1e-8)
    x = x_flat.reshape(C, P)

    # If requested, apply the same hard orbit projection used in block solver:
    if (w_target is not None) and hard_project:
        print("[MONO] applying hard orbit projection to match orbit_weights", flush=True)
        known_zero_orbit = None
        # attempt to read known_zero mask if present
        try:
            with open_h5(h5_path, role="reader") as f:
                if "/HyperCube/known_zero_mask" in f:
                    known_zero = np.asarray(f["/HyperCube/known_zero_mask"][...], dtype=bool)
                    known_zero_orbit = np.all(known_zero, axis=1)
                else:
                    known_zero_orbit = np.zeros((C,), dtype=bool)
        except Exception:
            known_zero_orbit = np.zeros((C,), dtype=bool)

        D_orbit = None  # not available here; treat active = ~known_zero_orbit
        active = (~known_zero_orbit) if known_zero_orbit is not None else np.ones((C,), dtype=bool)
        if np.any(active):
            s = np.sum(x[active, :], axis=1)
            w = np.asarray(w_target, dtype=np.float64)[active]
            w_sum = float(np.sum(w)) if np.sum(w) > 0.0 else 1.0
            alpha = float(np.sum(s)) / w_sum if (w_sum > 0.0) else 1.0
            s_proj = alpha * w
            ratio = s_proj / np.maximum(s, 1e-30)
            x[active, :] *= ratio[:, None]
            np.maximum(x, 0.0, out=x)

    # Basic stats
    l1 = float(np.sum(x))
    nonzero = int(np.count_nonzero(x > 0))
    stats = dict(l1=l1, nonzero=nonzero, CP=CP)

    print(f"[MONO] done. l1={l1:.3e} nonzero={nonzero}/{CP}", flush=True)
    return x, stats

# ------------------------------------------------------------------------------

def monolithic_nnls_scipy(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    enforce_orbit_projection: bool = True,
    use_scipy_if_available: bool = True,
):
    """
    Build the full design matrix A and data vector y from the hypercube and
    solve the global NNLS problem x = argmin_{x>=0} ||A x - y||^2.

    Notes
    -----
    - This constructs A in memory. For your typical problem (C*P ~ 1k,
      S*L ~ a few million), this is generally feasible but may require a
      few tens of GB of RAM. The function prints an estimate and raises if
      the estimated bytes exceed 120 GB as a safety guard.
    - If SciPy's `nnls` is available it is used (single RHS). Otherwise the
      function forms ATA/ATy and calls the quadratic NNLS fallback
      `_nnls_from_quadratic`.
    - If `enforce_orbit_projection` is True and `orbit_weights` is provided,
      the solution is post-scaled per-orbit exactly as your epoch-end
      hard projection does (so results are comparable).
    - Returns (x, stats) where x is (C,P) ndarray, stats contains simple
      diagnostics.
    """
    import math

    t0 = time.perf_counter()
    with open_h5(h5_path, role="reader") as f:
        S, L = map(int, f["/DataCube"].shape)
        _, C, P, Lm = map(int, f["/HyperCube/models"].shape)
        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch")
        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None

    Lk = int(keep_idx.size) if keep_idx is not None else int(L)
    rows = int(S) * int(Lk)
    cols = int(C) * int(P)

    # memory estimate and safety guard (bytes)
    est_bytes = rows * cols * np.dtype(np.float64).itemsize + rows * np.dtype(np.float64).itemsize
    print(f"[MONO-NNLS] building A_full with shape ({rows},{cols}), "
          f"estimated memory {est_bytes/1024**3:.2f} GiB", flush=True)
    if est_bytes > 120 * 1024**3:
        raise MemoryError(f"Estimated A+y memory {est_bytes/1024**3:.1f} GiB > 120 GiB; "
                          "aborting to avoid OOM. Reduce problem size or use streaming solver.")

    # allocate
    A_full = np.empty((rows, cols), dtype=np.float64)
    y_full = np.empty((rows,), dtype=np.float64)

    # fill by streaming tiles (identical tiling logic to main solver)
    s_tile = int(getattr(cfg, "s_tile_override", None) or 128)
    s_ranges = [(s0, min(S, s0 + s_tile)) for s0 in range(0, S, s_tile)]

    row_ptr = 0
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        try:
            M.id.set_chunk_cache(cfg.dset_slots, cfg.dset_bytes, cfg.dset_w0)
        except Exception:
            pass

        for (s0, s1) in tqdm(s_ranges, desc="[MONO-NNLS] build tiles", disable=not getattr(cfg, "verbose", True)):
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            Sblk = s1 - s0
            Lk_loc = Yt.shape[1]

            # read full model slice for this tile and reshape to (Sblk*Lk_loc, C*P)
            # M[s0:s1, :, :, :] has shape (Sblk, C, P, L) -> keep_idx -> (Sblk, C, P, Lk)
            M_tile = np.asarray(M[s0:s1, :, :, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                M_tile = M_tile[:, :, :, keep_idx]
            # transpose to (Sblk, Lk, C, P) then reshape to rows x cols
            M_t = M_tile.transpose(0, 3, 1, 2).reshape(Sblk * Lk_loc, C * P)
            nrows_loc = M_t.shape[0]

            # place blocks
            A_full[row_ptr:row_ptr + nrows_loc, :] = M_t
            y_full[row_ptr:row_ptr + nrows_loc] = Yt.reshape(-1)

            row_ptr += nrows_loc

    if row_ptr != rows:
        # should not happen, defensive
        A_full = A_full[:row_ptr, :]
        y_full = y_full[:row_ptr]
        print(f"[MONO-NNLS] Warning: filled rows {row_ptr} != expected {rows}", flush=True)

    # Solve NNLS (prefer SciPy)
    x_flat = None
    if use_scipy_if_available and _HAS_SCIPY_NNLS:
        print("[MONO-NNLS] calling scipy.optimize.nnls on full A,y", flush=True)
        x_flat, rnorm = _scipy_nnls(A_full, y_full)
        x_flat = np.asarray(x_flat, dtype=np.float64).ravel()

        # --- Direct SciPy diagnostics ---
        l1 = float(np.sum(x_flat))
        l2 = float(np.linalg.norm(x_flat))
        nonzero = int(np.count_nonzero(x_flat > 0))
        resid = float(np.linalg.norm(A_full @ x_flat - y_full))

        print(f"[MONO-SCIPY] L1={l1:.6e}", flush=True)
        print(f"[MONO-SCIPY] L2={l2:.6e}", flush=True)
        print(f"[MONO-SCIPY] nonzero={nonzero}/{x_flat.size}", flush=True)
        print(f"[MONO-SCIPY] residual_norm={resid:.6e}", flush=True)
    else:
        # fallback: form ATA / ATy and solve quadratic NNLS
        print("[MONO-NNLS] SciPy nnls unavailable; falling back to ATA/ATy + quad NNLS", flush=True)
        ATA = A_full.T @ A_full
        ATy = A_full.T @ y_full
        x_flat = _nnls_from_quadratic(ATA, ATy, x0=None,
                                      max_iter=4000, tol=1e-8)

    # reshape to (C,P)
    x = x_flat.reshape((C, P)).copy()

    # optional post-solve hard orbit projection (match existing pipeline)
    if enforce_orbit_projection and (orbit_weights is not None):
        w_t = _canon_orbit_weights(h5_path, orbit_weights, C=C, P=P)
        if w_t is not None:
            known_zero_orbit = np.all(np.zeros((C, P), dtype=bool), axis=1)  # no known_zero info here
            # compute per-orbit sums on current x
            s = np.sum(x, axis=1)
            w = np.asarray(w_t, dtype=np.float64)
            w_sum = float(np.sum(w))
            alpha = float(np.sum(s)) / w_sum if (w_sum > 0.0) else 1.0
            s_proj = alpha * w
            ratio = s_proj / np.maximum(s, 1e-30)
            x *= ratio[:, None]
            np.maximum(x, 0.0, out=x)

    elapsed = time.perf_counter() - t0
    stats = dict(
        elapsed_sec=float(elapsed),
        rows=int(rows),
        cols=int(cols),
    )
    print(f"[MONO-NNLS] done in {elapsed:.1f}s; x_sum={float(np.sum(x)):.3e}", flush=True)
    return x, stats

# ------------------------------------------------------------------------------

def _streaming_active_set_nnls_via_streaming_matvec(
    h5_path: str,
    s_ranges: list,
    keep_idx,
    ATy_flat: np.ndarray,
    inv_sqrt_energy_flat: np.ndarray,
    C: int,
    P: int,
    *,
    cfg: MPConfig,
    max_active: int = 1000,
    tol_grad: float = 1e-8,
    max_iter: int = 5000,
):
    """
    TRUE MONOLITHIC streaming active-set NNLS.

    Hybrid parallel:
        - Process-level parallel over tile batches
        - BLAS threads inside each worker
        - Parent reduces partial results

    No per-orbit approximation.
    No stored ATA.
    Exact streaming monolithic math.
    """

    CP = int(C * P)
    S_flat = np.asarray(inv_sqrt_energy_flat, dtype=np.float64).ravel()
    ATy_scaled = S_flat * ATy_flat

    z = np.zeros((CP,), dtype=np.float64)
    active = np.zeros((CP,), dtype=bool)

    # ------------------------------------------------------------
    # Worker: compute ATAz partial for a batch of tiles
    # ------------------------------------------------------------

    def _worker_ATAz(args):
        h5_path, batch, keep_idx, x_cand = args

        from CubeFit.hdf5_manager import open_h5
        import numpy as np

        partial = np.zeros((CP,), dtype=np.float64)

        with open_h5(h5_path, role="reader") as f:
            DC = f["/DataCube"]
            M = f["/HyperCube/models"]

            for (s0, s1) in batch:
                Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
                if keep_idx is not None:
                    Yt = Yt[:, keep_idx]

                Sblk = s1 - s0
                Lk_loc = Yt.shape[1]

                M_tile = np.asarray(M[s0:s1, :, :, :],
                                    dtype=np.float64, order="C")
                if keep_idx is not None:
                    M_tile = M_tile[:, :, :, keep_idx]

                A2 = M_tile.transpose(0, 3, 1, 2).reshape(
                    Sblk * Lk_loc, CP)

                v = A2 @ x_cand
                partial += A2.T @ v

        return partial

    # ------------------------------------------------------------
    # Helper: compute ATAz in parallel
    # ------------------------------------------------------------

    def _compute_ATAz_scaled(z_vec):

        x_cand = S_flat * z_vec

        # --- batching tiles per worker ---
        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        ATAz = np.zeros((CP,), dtype=np.float64)

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(cfg.blas_threads,)
        ) as exe:

            futures = [
                exe.submit(_worker_ATAz,
                           (h5_path, batch, keep_idx, x_cand))
                for batch in batches if len(batch) > 0
            ]

            for f in tqdm(as_completed(futures),
                          total=len(futures),
                          desc="[MONO] ATAz tiles",
                          leave=False):
                ATAz += f.result()

        return S_flat * ATAz

    # ------------------------------------------------------------
    # Active-set loop
    # ------------------------------------------------------------

    for it in range(max_iter):

        ATAz_scaled = _compute_ATAz_scaled(z)
        grad = ATy_scaled - ATAz_scaled

        not_active = np.where(~active)[0]
        if not_active.size == 0:
            break

        gvals = grad[not_active]
        idx = not_active[np.argmax(gvals)]
        if gvals.max() <= tol_grad:
            break

        active[idx] = True

        if np.count_nonzero(active) > max_active:
            active[idx] = False
            break

        # --------------------------------------------------------
        # Reduced solve (parallel tile streaming)
        # --------------------------------------------------------

        active_idx = np.nonzero(active)[0]
        k = len(active_idx)
        S_active = S_flat[active_idx]

        ATA_sub = np.zeros((k, k), dtype=np.float64)
        ATy_sub = np.zeros((k,), dtype=np.float64)

        # tile batching
        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        def _worker_reduced(args):
            h5_path, batch, keep_idx, active_idx, S_active = args
            from CubeFit.hdf5_manager import open_h5
            import numpy as np

            k = len(active_idx)
            ATA_loc = np.zeros((k, k), dtype=np.float64)
            ATy_loc = np.zeros((k,), dtype=np.float64)

            with open_h5(h5_path, role="reader") as f:
                DC = f["/DataCube"]
                M = f["/HyperCube/models"]

                for (s0, s1) in batch:

                    Yt = np.asarray(DC[s0:s1, :],
                                    dtype=np.float64, order="C")
                    if keep_idx is not None:
                        Yt = Yt[:, keep_idx]

                    Sblk = s1 - s0
                    Lk_loc = Yt.shape[1]

                    M_tile = np.asarray(M[s0:s1, :, :, :],
                                        dtype=np.float64, order="C")
                    if keep_idx is not None:
                        M_tile = M_tile[:, :, :, keep_idx]

                    A2 = np.empty((Sblk * Lk_loc, k),
                                  dtype=np.float64)

                    for j, gcol in enumerate(active_idx):
                        cc = gcol // P
                        p = gcol % P
                        A2[:, j] = M_tile[:, cc, p, :].reshape(
                            Sblk * Lk_loc)

                    A2 *= S_active[None, :]

                    ATA_loc += A2.T @ A2
                    ATy_loc += A2.T @ Yt.reshape(-1)

            return ATA_loc, ATy_loc

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(cfg.blas_threads,)
        ) as exe:

            futures = [
                exe.submit(_worker_reduced,
                           (h5_path, batch, keep_idx,
                            active_idx, S_active))
                for batch in batches if len(batch) > 0
            ]

            for f in tqdm(as_completed(futures),
                          total=len(futures),
                          desc="[MONO] reduced tiles",
                          leave=False):
                ATA_loc, ATy_loc = f.result()
                ATA_sub += ATA_loc
                ATy_sub += ATy_loc

        ATA_sub += 1e-12 * np.eye(k)

        try:
            z_sub = np.linalg.solve(ATA_sub, ATy_sub)
        except np.linalg.LinAlgError:
            z_sub = _nnls_from_quadratic(
                ATA_sub, ATy_sub,
                max_iter=2000, tol=1e-8
            )

        z[active_idx] = z_sub
        z[z < 0.0] = 0.0
        active[z == 0.0] = False

    return S_flat * z

# ------------------------------------------------------------------------------

def solve_block_coord_nnls(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    tracker: Optional[object] = None,
    block_size: Optional[int] = None,
    monolithic_max_active: int = 1000,
    ):
    """
    Fused single-pass block-coordinate NNLS solver (BLAS-friendly).

    - Accumulates ATA and ATy for each block during one streaming pass per
      epoch (no repeated HDF5 reads per block).
    - Solves small quadratic NNLS problems per block using a projected-gradient
      solver on the quadratic form ATA/ATy.
    - Supports soft orbit prior (orbit_beta) via augmentation of ATA/ATy.
    - Applies hard rank-1 orbit projection at epoch end (using D_tot).
    """
    t0 = time.perf_counter()

    # --- orbit-solve hyper-parameters (tunable) ---
    _ORBIT_SOLVE = {
        "tiny_ridge": 1e-6,
        "lambda_curv": 1e-2, # smoother SFH inside each orbit
        "max_iter_quad": 4000,
        "tol_quad": 1e-8,
    }

    # ---------------------------- metadata -----------------------------
    with open_h5(h5_path, role="reader") as f:
        S, L = map(int, f["/DataCube"].shape)
        _, C, P, Lm = map(int, f["/HyperCube/models"].shape)
        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch")
        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        chunks = f["/HyperCube/models"].chunks
        s_tile = int(chunks[0]) if (chunks and chunks[0]) else 128
        if cfg.s_tile_override is not None:
            s_tile = int(cfg.s_tile_override)

    Lk_guess = None
    if keep_idx is not None:
        Lk_guess = int(keep_idx.size)

    s_ranges = [(s0, min(S, s0 + s_tile)) for s0 in range(0, S, s_tile)]
    CP = int(C * P)

    # ---------------------------- orbit prior --------------------------
    w_target = None
    if orbit_weights is not None:
        w_target = _canon_orbit_weights(h5_path, orbit_weights, C=C, P=P)

    # ---------------------------- initial x ----------------------------
    if x0 is None:
        x = np.zeros((C, P), dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64).ravel(order="C")
        if x0.size != CP:
            raise ValueError("x0 has wrong size")
        x = x0.reshape(C, P).copy()

    # ---------------------------- block tiling -------------------------
    if block_size is None:
        # heuristic: aim for ~CP / (8 * processes) cols per block (bounded)
        block_size = max(16, int(min(256, max(16, CP // max(1, cfg.processes * 8)))))
    block_size = int(block_size)
    n_blocks = int(math.ceil(CP / block_size))
    blocks = [(i * block_size, min(CP, (i + 1) * block_size)) for i in range(n_blocks)]

    # Precompute block metadata (maps, shapes)
    block_meta = {}
    for bi, (c0, c1) in enumerate(blocks):
        cols = np.arange(c0, c1, dtype=np.int64)
        ncols = cols.size
        # map local_col_idx -> (orbit_cc, p)
        cc_arr = (cols // P).astype(np.int64)
        pp_arr = (cols % P).astype(np.int64)
        # for soft prior we need w_vec per column (if w_target present)
        if w_target is not None:
            w_vec = np.asarray([float(w_target[int(cc)]) for cc in cc_arr], dtype=np.float64)
        else:
            w_vec = None
        block_meta[bi] = dict(
            cols=cols,
            ncols=ncols,
            cc_arr=cc_arr,
            pp_arr=pp_arr,
            w_vec=w_vec,
        )

    verbose = getattr(cfg, "verbose", True)
    print(f"[BC-FUSED] blocks={n_blocks}, block_size={block_size}, processes={cfg.processes}", flush=True)

    # ------------------ persistent masks: known_zero & seed masks ------------------
    # Load once (deterministic) from HDF5, fall back to False if missing.
    with open_h5(h5_path, role="reader") as f:
        # persistent KNOWN_ZERO mask (global)
        if "/HyperCube/known_zero_mask" in f:
            known_zero = np.asarray(f["/HyperCube/known_zero_mask"][...], dtype=bool)
            if known_zero.shape != (C, P):
                raise RuntimeError("known_zero_mask has wrong shape (expected (C,P))")
        else:
            known_zero = np.zeros((C, P), dtype=bool)

        # Optional NNLS patch metadata (used by other helpers)
        if "/Seeds/seed_support_mask" in f:
            seed_support_mask = np.asarray(f["/Seeds/seed_support_mask"][...], dtype=bool)
            if seed_support_mask.shape != (C, P):
                raise RuntimeError("seed_support_mask has wrong shape (expected (C,P))")
        else:
            seed_support_mask = None

        if "/Seeds/seed_tested_mask" in f:
            seed_tested_mask = np.asarray(f["/Seeds/seed_tested_mask"][...], dtype=bool)
            if seed_tested_mask.shape != (C, P):
                raise RuntimeError("seed_tested_mask has wrong shape (expected (C,P))")
        else:
            seed_tested_mask = None
    # -------------------------------------------------------------------------------
    # before epoch loop (after known_zero loaded)
    known_zero_orbit = np.all(known_zero, axis=1)  # shape (C,)

    # pool only used to set BLAS threads per worker if you later parallelize
    ctx = mp.get_context(os.environ.get("CUBEFIT_MP_CTX", "forkserver"))
    pool = ctx.Pool(processes=max(1, int(cfg.processes)), initializer=_worker_init, initargs=(int(cfg.blas_threads),))

    try:
        best_x = x.copy()
        best_proxy = np.inf

        # epoch loop
        for ep in range(cfg.epochs):
            print(f"[BC-FUSED] epoch {ep+1}/{cfg.epochs}", flush=True)

            # D_tot per column (C,P)
            D_tot = np.zeros((C, P), dtype=np.float64)

            # ---------- single streaming pass: build ATy_flat & D_tot -------
            ATy_flat = np.zeros((CP,), dtype=np.float64)
            y_parts = []
            with open_h5(h5_path, role="reader") as f:
                DC = f["/DataCube"]
                M  = f["/HyperCube/models"]
                try:
                    M.id.set_chunk_cache(cfg.dset_slots, cfg.dset_bytes,
                        cfg.dset_w0)
                except Exception:
                    pass

                tile_iter = s_ranges
                if verbose and (len(s_ranges) > 1):
                    tile_iter = tqdm(s_ranges,
                        desc=f"[BC-FUSED] tiles ep{ep+1}",
                        disable=not verbose)

                for (s0, s1) in tile_iter:
                    Yt = np.asarray(DC[s0:s1, :], dtype=np.float64,
                                    order="C")
                    if keep_idx is not None:
                        Yt = Yt[:, keep_idx]
                    Sblk = s1 - s0
                    Lk = Yt.shape[1]

                    # accumulate prediction for diagnostics (warm-start)
                    yhat_tile = np.zeros((Sblk, Lk), dtype=np.float64)
                    # read model tile (Sblk, C, P, Lk)
                    M_tile = np.asarray(M[s0:s1, :, :, :], dtype=np.float64,
                        order="C")
                    if keep_idx is not None:
                        M_tile = M_tile[:, :, :, keep_idx]

                    # compute D_tot contributions (per-column energy)
                    # sum over Sblk and wavelength dims
                    # M_tile: (Sblk, C, P, Lk)
                    D_tot += np.sum(M_tile * M_tile, axis=(0, 3))

                    # build A2_tile once for ATy accumulation
                    A2_tile = M_tile.transpose(0, 3, 1, 2).reshape(
                        Sblk * Lk, CP)
                    y_flat = Yt.reshape(-1)

                    # accumulate ATy (uses data y, not residual)
                    ATy_flat += A2_tile.T @ y_flat

                    # diagnostics (same as before)
                    try:
                        yf_norm = float(np.linalg.norm(y_flat)) if y_flat.size > 0 else 0.0
                        yhatf_norm = float(np.linalg.norm(yhat_tile)) if yhat_tile.size > 0 else 0.0
                        r_norm = float(np.linalg.norm(y_flat - yhat_tile.reshape(-1))) if y_flat.size > 0 else 0.0

                        nnans = int(np.count_nonzero(~np.isfinite(y_flat)))
                        nnans += int(np.count_nonzero(~np.isfinite(yhat_tile)))
                        nnans += int(np.count_nonzero(~np.isfinite(y_flat - yhat_tile.reshape(-1))))

                        print(
                            f"[BC-FUSED][tile s={s0}:{s1}] Sblk={Sblk} Lk={Lk} "
                            f"||y||={yf_norm:.3e} ||yhat||={yhatf_norm:.3e} "
                            f"||r||={r_norm:.3e} nonfinite_vals={nnans}",
                            flush=True
                        )
                    except Exception as _e:
                        print("[BC-FUSED][tile diag] error while computing tile diagnostics:", _e, flush=True)

                    y_parts.append(y_flat)
            # end single streaming pass
            # ------------------------ end streaming --------------------------

            # compute inv_sqrt_energy and scaled targets
            col_energy = D_tot.copy()
            col_energy[col_energy <= 0.0] = 1.0
            inv_sqrt_energy = 1.0 / np.sqrt(col_energy)
            inv_sqrt_energy_flat = inv_sqrt_energy.ravel(order="C")

            # run streaming active-set NNLS (monolithic)
            print("[BC-FUSED][MONO] starting streaming active-set NNLS (mono)", flush=True)
            x_flat_unscaled = _streaming_active_set_nnls_via_streaming_matvec(
                h5_path=h5_path,
                s_ranges=s_ranges,
                keep_idx=keep_idx,
                ATy_flat=ATy_flat,
                inv_sqrt_energy_flat=inv_sqrt_energy_flat,
                C=C,
                P=P,
                cfg=cfg,
                max_active=monolithic_max_active,
                tol_grad=1e-8,
                max_iter=5 * monolithic_max_active,
            )
            x = x_flat_unscaled.reshape(C, P).copy()
            print("[BC-FUSED][MONO] finished streaming active-set NNLS", flush=True)

            # --- DIAG: after all blocks solved (before projection) ---
            try:
                x_flat = x.ravel(order="C")
                x_nonzero = int(np.count_nonzero(x_flat > 0))
                x_sparsity = 1.0 - (x_nonzero / float(x_flat.size)) if x_flat.size else 1.0
                x_l1 = float(np.sum(x_flat))
                x_max = float(np.max(x_flat)) if x_flat.size else 0.0
                x_min = float(np.min(x_flat)) if x_flat.size else 0.0
                D_tot_finite = np.isfinite(D_tot)
                dpos = float(np.sum(D_tot[D_tot_finite]))
                print(
                    f"[BC-FUSED][after-blocks] x_l1={x_l1:.3e} max={x_max:.3e} min={x_min:.3e} "
                    f"nonzero={x_nonzero}/{x_flat.size} sparsity={x_sparsity:.3f} D_tot_sum={dpos:.3e}",
                    flush=True
                )

                # check for non-finite in x
                nbad = int(np.count_nonzero(~np.isfinite(x_flat)))
                if nbad:
                    print(f"[BC-FUSED][after-blocks] WARNING: non-finite entries in x: {nbad}", flush=True)

            except Exception as _e:
                print("[BC-FUSED][after-blocks] diag error:", _e, flush=True)
            # --- end after-blocks diagnostics ---

            # -------------------- HARD ORBIT PROJECTION (epoch-end) --------------------
            projection_applied = False
            if w_target is not None:
                known_zero_orbit = np.all(known_zero, axis=1)
                D_orbit = np.sum(D_tot, axis=1)
                active = (~known_zero_orbit) & (D_orbit > 0.0)
                if np.any(active):
                    s = np.sum(x[active, :], axis=1)
                    w = np.asarray(w_target, dtype=np.float64)[active]
                    w_sum = float(np.sum(w))
                    alpha = float(np.sum(s)) / w_sum if (w_sum > 0.0) else 1.0
                    s_proj = alpha * w
                    ratio = s_proj / np.maximum(s, 1e-30)
                    x[active, :] *= ratio[:, None]
                    np.maximum(x, 0.0, out=x)
                    projection_applied = True

            # -------------------- epoch diagnostics & best-x --------------------
            # Compute simple RMSE proxy (full scan cheap relative to earlier streaming)
            rmse_curr = float("nan")
            ssq = 0.0
            nres = 0
            with open_h5(h5_path, role="reader") as f:
                DC = f["/DataCube"]
                for (s0, s1) in s_ranges:
                    Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
                    if keep_idx is not None:
                        Yt = Yt[:, keep_idx]
                    # build current yhat_tile for diagnostics
                    yhat_tile = np.zeros_like(Yt)
                    for cc in range(C):
                        if known_zero_orbit[cc]:
                            continue
                        A_cc = np.asarray(f["/HyperCube/models"][s0:s1, cc, :, :], dtype=np.float64, order="C")
                        if keep_idx is not None:
                            A_cc = A_cc[:, :, keep_idx]
                        yhat_tile += x[cc] @ A_cc
                    R = Yt - yhat_tile
                    if not np.all(np.isfinite(R)):
                        R = np.nan_to_num(R, copy=False)
                    ssq += float(np.sum(R * R))
                    nres += R.size
            if nres > 0:
                rmse_curr = float(np.sqrt(ssq / max(1, nres)))
                data_proxy = 0.5 * (rmse_curr ** 2)
            else:
                data_proxy = np.inf


            # --- DIAG: epoch summary (timings & numerical health) ---
            try:
                epoch_elapsed = time.perf_counter() - t0
                # D_tot stats per-orbit
                D_tot_flat = D_tot.ravel(order="C")
                D_finite = D_tot_flat[np.isfinite(D_tot_flat) & (D_tot_flat > 0)]
                D_median = float(np.median(D_finite)) if D_finite.size else 0.0
                D_min = float(np.min(D_finite)) if D_finite.size else 0.0
                D_max = float(np.max(D_finite)) if D_finite.size else 0.0

                x_flat = best_x.ravel(order="C")
                x_nonfinite = int(np.count_nonzero(~np.isfinite(x_flat)))
                x_nonzero = int(np.count_nonzero(x_flat > 0))
                x_sum = float(np.sum(x_flat))
                x_norm = float(np.linalg.norm(x_flat)) if x_flat.size else 0.0

                rmse_str = f"{rmse_curr:.3e}" if np.isfinite(rmse_curr) else "nan"

                print("=== [BC-FUSED][epoch-summary] ===", flush=True)
                print(f"[BC-FUSED][epoch-summary] epoch={ep+1}/{cfg.epochs} elapsed_total={epoch_elapsed:.1f}s", flush=True)
                print(f"[BC-FUSED][epoch-summary] data_proxy={data_proxy:.3e} rmse={rmse_str:.3e}", flush=True)
                print(f"[BC-FUSED][epoch-summary] best_proxy={best_proxy:.3e}", flush=True)
                print(
                    f"[BC-FUSED][epoch-summary] x_sum={x_sum:.3e} x_norm={x_norm:.3e} nonzero={x_nonzero}/{x_flat.size} nonfinite={x_nonfinite}",
                    flush=True
                )
                print(
                    f"[BC-FUSED][epoch-summary] D_tot diag med/min/max = {D_median:.3e} / {D_min:.3e} / {D_max:.3e}",
                    flush=True
                )

                # sanity checks that likely indicate major numerical problems
                if x_nonzero == 0:
                    print("[BC-FUSED][epoch-summary] ALERT: x is entirely zero after epoch!", flush=True)
                if x_nonfinite > 0:
                    print("[BC-FUSED][epoch-summary] ALERT: non-finite elements in x!", flush=True)
                if not np.isfinite(data_proxy) or data_proxy <= 0.0:
                    print("[BC-FUSED][epoch-summary] ALERT: data_proxy non-finite or <= 0.0", flush=True)

                # show top contributors (per-orbit totals)
                try:
                    s_full = np.sum(x, axis=1)  # per-orbit
                    top_idx = np.argsort(s_full)[::-1][:10]
                    top_vals = s_full[top_idx]
                    print("[BC-FUSED][epoch-summary] top orbits (index:mass): " + ", ".join(
                        [f"{int(i)}:{v:.3e}" for i, v in zip(top_idx.tolist(), top_vals.tolist())]
                    ), flush=True)
                except Exception:
                    pass

                print("=== [BC-FUSED][epoch-summary] end ===", flush=True)
            except Exception as _e:
                print("[BC-FUSED][epoch-summary] error:", _e, flush=True)
            # --- end epoch summary diagnostics ---

            if data_proxy < best_proxy:
                best_proxy = float(data_proxy)
                best_x = x.copy()
                print(f"[BC-FUSED] new best proxy {best_proxy:.3e} at epoch {ep+1}", flush=True)

        # done epochs
        th = cu.zero_floor_inplace(best_x, rel_tol=1e-25, abs_tol=0.0)
        elapsed = time.perf_counter() - t0
        stats = dict(
            epochs=int(cfg.epochs),
            elapsed_sec=elapsed,
            rmse_proxy_best=float(best_proxy),
            active_orbits=np.arange(C, dtype=np.int32),
            known_zero_mask=known_zero.copy(),
        )
        return best_x, stats

    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass

# ------------------------------------------------------------------------------