# -*- coding: utf-8 -*-
r"""
    streaming_nnls.py
    Adriano Poci
    University of Oxford
    2026

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Implementation of a TRUE MONOLITHIC streaming active-set NNLS solver for the CubeFit problem, with hybrid parallelism (process-level over tile batches + BLAS threads inside workers).

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   7 March 2026
v1.1:   Implemented streaming active-set Lawson-Hanson NNLS;
        Added rank-1 orbit penalisation with `orbit_beta` parameter instead of
            hard projection. 13 March 2026
v1.2:   Apply the orbit prior in the `z`-space of the solver in
            `_streaming_active_set_nnls_via_streaming_matvec`;
        Let orbit prior also influence active set promotion, not just data
            gradient in `_streaming_active_set_nnls_via_streaming_matvec`;
        Screen for temporarily-negative active-set entries and drop them if they
            are substantially negative (numerical noise tolerance) in 
            `_streaming_active_set_nnls_via_streaming_matvec`. 14 March 2026
v1.3:   Compute global `alpha_fixed` in 
            `_streaming_active_set_nnls_via_streaming_matvec` so that the orbit
            prior scale is predictable;
        Added soft exit where new column is randomly promoted to test local 
            optima, up to `explore_budget` times in 
            `_streaming_active_set_nnls_via_streaming_matvec`. 15 March 2026
"""

from __future__ import annotations, print_function

import os, sys, traceback
import math, builtins
import time
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List, Dict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import h5py
try:
    from scipy.optimize import nnls as _scipy_nnls
    _HAS_SCIPY_NNLS = True
except Exception:
    _HAS_SCIPY_NNLS = False

from CubeFit.hdf5_manager import open_h5
from CubeFit import cube_utils as cu
from CubeFit.hypercube_builder import read_global_column_energy


def _init_worker(blas_threads: int):
    # called once per process; set BLAS env vars
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, blas_threads // 2))

# ------------------------------------------------------------------------------

def _worker_reduced(
    h5_path: str,
    batch: list,
    keep_idx,
    active_idx: np.ndarray,
    S_active: np.ndarray,
    CP: int,
    P: int,
):
    """
    Worker to compute partial ATA_sub and ATy_sub for a batch of tiles,
    given the active indices and their scaling S_active.

    Memory-efficient, exact-equivalence implementation that does NOT
    form the large (Sblk*Lk, k) A2 matrix.  Uses tensor contractions to
    compute the same sums:

        ATA_loc[i,j] += sum_{s,lambda} (S_i * M_{s,ci,pi,lambda})
                              * (S_j * M_{s,cj,pj,lambda})

        ATy_loc[i] += sum_{s,lambda} (S_i * M_{s,ci,pi,lambda}) * Yt[s,lambda]

    Inputs
    ------
    - active_idx: 1D int array of global column indices (length k)
    - S_active : 1D float array of length k (per-active-column scaling)
    - P : number of populations per orbit (used for cc/p decode)
    """
    k = int(active_idx.size)
    ATA_loc = np.zeros((k, k), dtype=np.float64)
    ATy_loc = np.zeros((k,), dtype=np.float64)

    # decode orbit & population indices for active columns
    cc_arr = (active_idx // P).astype(np.int64)
    pp_arr = (active_idx % P).astype(np.int64)

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]         # kept for shape reference; Yt read below
        M = f["/HyperCube/models"]

        for (s0, s1) in batch:
            # read data tile and model tile
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]           # shape (Sblk, Lk)
            Sblk = s1 - s0
            Lk_loc = Yt.shape[1]

            M_tile = np.asarray(M[s0:s1, :, :, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                M_tile = M_tile[:, :, :, keep_idx]   # shape (Sblk, C, P, Lk_loc)

            # Extract the k active column slices into shape (Sblk, k, Lk_loc)
            # M_sub[s, i, lambda] = M_tile[s, cc_arr[i], pp_arr[i], lambda]
            # This is the minimal temporary we need (k typically much smaller than CP)
            M_sub = M_tile[:, cc_arr, pp_arr, :]  # shape (Sblk, k, Lk_loc)

            # --- compute inner products over s and lambda ---
            # ATA_unscaled_loc[i,j] = sum_{s,lambda} M_sub[s,i,lambda] * M_sub[s,j,lambda]
            # Use einsum for clarity and numerical stability / memory locality.
            # result shape (k,k)
            ATA_unscaled_loc = np.einsum("s i l, s j l -> i j", M_sub, M_sub, optimize=True)

            # scale by S_active outer product: ATA_loc += (S_i * S_j) * ATA_unscaled_loc
            outerS = (S_active[:, None] * S_active[None, :])
            ATA_loc += outerS * ATA_unscaled_loc

            # --- compute ATy contributions ---
            # ATy_unscaled_loc[i] = sum_{s,lambda} M_sub[s,i,lambda] * Yt[s,lambda]
            # shape (k,)
            ATy_unscaled_loc = np.einsum("s i l, s l -> i", M_sub, Yt, optimize=True)

            # scale by S_active: ATy_loc += S_i * ATy_unscaled_loc[i]
            ATy_loc += (S_active * ATy_unscaled_loc)

            # (optional consistency check -- enable for one debug run only)
            # If you want to validate exact equivalence to former A2-based code,
            # enable the block below (it will allocate A2 for the tile and compare).
            #
            # if False:
            #     # build A2 (old approach) for a single-tile verification
            #     A2_chk = np.empty((Sblk * Lk_loc, k), dtype=np.float64)
            #     for j, gcol in enumerate(active_idx):
            #         cc = int(gcol // P)
            #         p = int(gcol % P)
            #         A2_chk[:, j] = M_tile[:, cc, p, :].reshape(Sblk * Lk_loc)
            #     A2_chk *= S_active[None, :]
            #     ATA_chk = A2_chk.T @ A2_chk
            #     ATy_chk = A2_chk.T @ Yt.reshape(-1)
            #     # tolerance checks
            #     err_ATA = np.max(np.abs(ATA_loc - ATA_chk))
            #     err_ATy = np.max(np.abs(ATy_loc - ATy_chk))
            #     if (err_ATA > 1e-10) or (err_ATy > 1e-10):
            #         raise RuntimeError(f"_worker_reduced verification failed: err_ATA={err_ATA}, err_ATy={err_ATy}")

    return ATA_loc, ATy_loc

# ------------------------------------------------------------------------------

def _worker_ATAz(
    h5_path: str,
    batch: list,
    keep_idx,
    z: np.ndarray,
    CP: int,
    S_flat: np.ndarray,
):

    partial = np.zeros((CP,), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]

        for (s0, s1) in batch:

            M_tile = np.asarray(M[s0:s1, :, :, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                M_tile = M_tile[:, :, :, keep_idx]

            Sblk, C, P, Lk = M_tile.shape

            # reshape vectors
            z_s = (S_flat * z).reshape(C, P)

            # compute v = A (S z)
            v = np.zeros((Sblk, Lk), dtype=np.float64)

            for cc in range(C):
                v += np.tensordot(
                    z_s[cc],
                    M_tile[:, cc, :, :],
                    axes=(0, 1)
                )

            # compute g = A^T v
            g = np.zeros((C, P), dtype=np.float64)

            for cc in range(C):
                g[cc] = np.tensordot(
                    M_tile[:, cc, :, :],
                    v,
                    axes=([0, 2], [0, 1])
                )

            # apply column scaling
            partial += (S_flat.reshape(C, P) * g).reshape(-1)

    return partial

# ------------------------------------------------------------------------------

def _worker_ATAz_from_tuple(args):
    """
    Thin wrapper to allow executor.map with tuple-packed arguments.
    Must be top-level to be pickleable.
    """
    return _worker_ATAz(*args)

# ------------------------------------------------------------------------------

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
    orbit_beta: float = 0.0 # strength of rank-1 orbit penalisation (if > 0)
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
    # normalize to a comparable scale
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

def _streaming_active_set_nnls_via_streaming_matvec(
    h5_path: str,
    s_ranges: list,
    keep_idx,
    ATy_flat: np.ndarray,
    inv_sqrt_energy_flat: np.ndarray,
    C: int,
    P: int,
    *,
    executor,
    cfg: MPConfig,
    orbit_weights: Optional[np.ndarray] = None,
    orbit_beta_eff: float = 0.0,
    max_active: int = 1000,
    tol_grad: float = 1e-8,
    max_iter: int = 5000,
):
    """
    TRUE MONOLITHIC streaming active-set NNLS.

    Hybrid parallel:
        - Process-level parallel over tile batches (executor)
        - BLAS threads inside each worker (initializer/_init_worker)
        - Parent reduces partial results

    Assumes executor is a ProcessPoolExecutor created once by caller.
    """

    negative_grad_count = 0
    explore_budget = 2

    CP = int(C * P)
    S_flat = np.asarray(inv_sqrt_energy_flat, dtype=np.float64).ravel()
    ATy_scaled = S_flat * ATy_flat

    # --- orbit prior setup ---
    w_target = None
    if orbit_weights is not None:
        w_target = _canon_orbit_weights(h5_path, orbit_weights, C=C, P=P)
    orbit_beta = float(orbit_beta_eff)

    z = np.zeros((CP,), dtype=np.float64)
    active = np.zeros((CP,), dtype=bool)

    # ------------------------------------------------------------
    # Helper: compute ATAz in parallel using provided executor
    # ------------------------------------------------------------
    def _compute_ATAz_scaled(z_vec):
        # here z_vec is the solver variable (unscaled).
        # Workers will apply column-scaling S_flat directly to A2.
        x_cand = np.asarray(z_vec, dtype=np.float64)

        # batch s_ranges across workers
        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        ATAz = np.zeros((CP,), dtype=np.float64)

        # use spawn context for HDF5 safety
        ctx = mp.get_context("spawn")

        # submit tasks to the (persistent) executor if provided; otherwise
        # create a temporary pool (but prefer passing a persistent executor).
        # Here we assume executor argument was passed into outer function.
        worker_args = [
            (h5_path, batch, keep_idx, x_cand, CP, S_flat)
            for batch in batches if len(batch) > 0
        ]
        # Using executor.map (streamed) is fine — it returns results in submission order
        it = executor.map(_worker_ATAz_from_tuple, worker_args)

        for r in tqdm(it, total=len(worker_args), desc="[MONO] ATAz tiles", leave=False):
            try:
                ATAz += r
            except Exception as e:
                print("[ERROR] worker returned exception:", e, flush=True)
                raise

        # Note: workers already applied S_flat to columns, so ATAz is the
        # correctly scaled A^T A z in the scaled-column convention.
        return ATAz

    # ------------------------------------------------------------
    # Active-set outer loop
    # ------------------------------------------------------------
    for it in range(max_iter):
        ATAz_scaled = _compute_ATAz_scaled(z)
        grad = ATy_scaled - ATAz_scaled

        # ---- inject prior-gradient into the promotion gradient (so prior can
        # ---- encourage activation of columns that help meet orbit targets)
        # ---- Prior influence on promotion gradient (revised alpha + clipping)
        if (w_target is not None) and (orbit_beta > 0.0):
            # Compute alpha from current x (x = S * z) so prior redistributes
            # existing mass rather than creating mass from ATy magnitude.
                # use fixed alpha computed once (not the running x-sum)
            alpha_global = float(np.sum(S_flat * z))

            # Build prior gradient in z-space:
            # d/dz 0.5*beta*(S·z_local - alpha*w_cc)^2 = beta * (S·z_local - alpha*w_cc) * S
            g_prior = np.zeros_like(grad)
            # Loop per-orbit (cheap relative to expensive matvecs)
            for cc in range(C):
                base = cc * P
                idxs = np.arange(base, base + P, dtype=np.int64)
                S_local = S_flat[idxs]
                z_local = z[idxs]
                w_cc = float(w_target[cc]) if (w_target is not None) else 0.0
                resid = float(np.dot(S_local, z_local) - alpha_global * w_cc)
                if not np.isfinite(resid):
                    continue
                s_norm = float(np.sum(S_local * S_local)) + 1e-30
                g_prior[idxs] = (orbit_beta / s_norm) * resid * S_local

            # Prevent prior from overwhelming data gradient:
            # compute robust norms, allow prior up to `clip_factor` times data gradient norm
            clip_factor = 8.0
            g_prior_norm = np.linalg.norm(g_prior)
            g_data_norm = np.linalg.norm(grad) + 1e-30
            if g_prior_norm > clip_factor * g_data_norm:
                # scale down prior proportionally
                g_prior *= (clip_factor * g_data_norm) / (g_prior_norm + 1e-30)

            # Add prior gradient to promotion gradient
            grad = grad + g_prior

        # diagnostic logging for the active-set iteration
        max_grad_overall = float(np.max(grad)) if grad.size else 0.0
        avg_grad = float(np.mean(grad)) if grad.size else 0.0
        n_active = int(np.count_nonzero(active))
        print(f"[MONO][iter {it+1}] max_grad={max_grad_overall:.3e} avg_grad={avg_grad:.3e} active={n_active}", flush=True)

        not_active = np.where(~active)[0]
        if not_active.size == 0:
            break

        gvals = grad[not_active]
        if gvals.size == 0:
            break

        # ---------- promotion: penalise columns with very large S to avoid tiny-D bias ----------
        # not_active and gvals already defined above
        if gvals.size == 0:
            break
        # per-candidate scaling (S_flat is in scope; larger S -> originally favoured)
        S_not = S_flat[not_active]
        # robust normalization
        S_med = np.median(S_not) + 1e-30
        S_norm = S_not / S_med
        penalty_strength = 0.5
        adj_score = gvals / (1.0 + penalty_strength * (S_norm - 1.0))
        # if many candidates have very similar adj_score, optionally prefer the one with
        # larger raw gvals among the top-k (break ties toward raw gradient)
        topk = 5
        if adj_score.size > topk:
            top_idxs = np.argsort(adj_score)[-topk:]
            # pick among topk the index with largest raw gvals (stable tie-break)
            pick_local = top_idxs[np.argmax(gvals[top_idxs])]
        else:
            pick_local = int(np.argmax(adj_score))
        imax = int(pick_local)
        idx = int(not_active[imax])
        max_g = float(gvals[imax])

        if max_g <= tol_grad:

            negative_grad_count += 1

            if negative_grad_count <= explore_budget and not_active.size > 0:
                # choose randomly among the least negative gradients
                k = min(5, gvals.size)  # candidate pool
                top = np.argsort(gvals)[-k:]   # least negative
                pick_local = int(np.random.choice(top))
                idx = int(not_active[pick_local])

                print(f"[MONO][explore] promoting column {idx} despite negative gradient", flush=True)

                active[idx] = True
                continue

            # exploration budget exhausted → normal termination
            break

        # promote
        active[idx] = True
        if int(np.count_nonzero(active)) > int(max_active):
            # abort promotion if would exceed allowed size
            active[idx] = False
            break

        # --------------------------------------------------------
        # Reduced solve (assemble ATA_sub & ATy_sub) in parallel
        # --------------------------------------------------------
        # gather active indices and their scale factors
        active_idx = np.nonzero(active)[0].astype(np.int64)
        k = int(active_idx.size)
        if k == 0:
            continue

        S_active = S_flat[active_idx]

        # zero accumulators
        ATA_sub = np.zeros((k, k), dtype=np.float64)
        ATy_sub = np.zeros((k,), dtype=np.float64)

        # prepare batches (round-robin)
        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        # submit reduced-assembly jobs to persistent executor
        futures = [
            executor.submit(
                _worker_reduced,
                h5_path,
                batch,
                keep_idx,
                active_idx,
                S_active,
                CP,
                P,
            )
            for batch in batches if len(batch) > 0
        ]

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="[MONO] reduced tiles",
            leave=False,
        ):
            ATA_loc, ATy_loc = f.result()
            ATA_sub += ATA_loc
            ATy_sub += ATy_loc

        # ------------------ REDUCED-SOLVE DIAGNOSTICS (parent) -----------------------
        # active_idx (k,), S_active (k,) are already defined above
        try:
            # S_active summary
            Sact = S_active
            print(f"[DIAG] S_active stats: min/median/max: {float(np.min(Sact)):.4e}/{float(np.median(Sact)):.4e}/{float(np.max(Sact)):.4e}", flush=True)

            # Basic norms of assembled scaled ATA / ATy
            ATA_norm = float(np.linalg.norm(ATA_sub))
            ATy_norm = float(np.linalg.norm(ATy_sub))
            ATA_diag = np.diag(ATA_sub)[:min(6, ATA_sub.shape[0])]
            print(f"[DIAG] ATA_sub shape, ATy_sub[0:6]: {ATA_sub.shape}, {ATy_sub[:6].tolist()}", flush=True)
            print(f"[DIAG] ATA_sub norm / ATy_sub norm: {ATA_norm:.4e} / {ATy_norm:.4e}", flush=True)
            print(f"[DIAG] ATA_sub diag (first 6): {ATA_diag.tolist()}", flush=True)

            # Attempt to infer unscaled ATA/ATy (undo S scaling):
            # ATA_sub_scaled = (S_active * S_active^T) * ATA_unscaled  (since you scaled columns)
            # So we can approximate ATA_unscaled = ATA_sub / (S_i * S_j)
            outerS = (Sact[:, None] * Sact[None, :])
            # avoid divide-by-zero
            outerS_safe = outerS.copy()
            outerS_safe[outerS_safe == 0.0] = 1.0
            ATA_unscaled_est = ATA_sub / outerS_safe

            # similarly ATy_unscaled_est = ATy_sub / S_active
            Svec_safe = Sact.copy()
            Svec_safe[Svec_safe == 0.0] = 1.0
            ATy_unscaled_est = ATy_sub / Svec_safe

            ATA_unscaled_norm = float(np.linalg.norm(ATA_unscaled_est))
            ATy_unscaled_norm = float(np.linalg.norm(ATy_unscaled_est))
            print(f"[DIAG] inferred unscaled ATA/ATy norms: {ATA_unscaled_norm:.4e} / {ATy_unscaled_norm:.4e}", flush=True)

            # Condition number of the scaled reduced normal (small k so cheap)
            try:
                eigs = np.linalg.eigvalsh(ATA_sub)
                emin = float(np.min(eigs))
                emax = float(np.max(eigs))
                cond = float(emax / max(1e-30, emin))
                print(f"[DIAG] ATA_sub eigmin/emax/cond: {emin:.4e}/{emax:.4e}/{cond:.4e}", flush=True)
            except Exception:
                pass

            # Quick "what-if": solve unscaled system (debug only) and report solution norms
            try:
                ridge = 1e-12 * np.eye(ATA_unscaled_est.shape[0], dtype=np.float64)
                z_unscaled = np.linalg.solve(ATA_unscaled_est + ridge, ATy_unscaled_est)
                print(f"[DIAG] z_unscaled stats: l1, l2, max: {float(np.sum(z_unscaled)):.4e}, {float(np.linalg.norm(z_unscaled)):.4e}, {float(np.max(z_unscaled)):.4e}", flush=True)
            except Exception as _e:
                print(f"[DIAG] solving unscaled diagnostic failed: {_e}", flush=True)

        except Exception as _e:
            print(f"[DIAG] reduced-diagnostics error: {_e}", flush=True)
        # ---------------------------------------------------------------------------

        # ---- Stabilised reduced solve: adaptive ridge then SVD-truncated fallback ----
        # k = ATA_sub.shape[0]
        # adaptive ridge proportional to median diagonal (robust to scale)
        diag = np.diag(ATA_sub)
        diag_med = float(np.median(diag)) if diag.size else 0.0
        # pick a relative ridge (tunable); higher if condition is bad
        ridge_rel = 1e-4
        ridge = max(1e-12, ridge_rel * max(1.0, diag_med))
        ATA_sub += ridge * np.eye(k, dtype=np.float64)

        # ----------------------------------------------------
        # Robust orbit prior (scale-independent)
        # ----------------------------------------------------
        if (w_target is not None) and (orbit_beta > 0.0):

            # group active variables by orbit
            orbit_groups = {}
            for local_i, gcol in enumerate(active_idx):
                cc = int(gcol // P)
                orbit_groups.setdefault(cc, []).append(local_i)

            # fixed total mass scale
            alpha = float(np.sum(S_flat * z))

            for cc, idx_list in orbit_groups.items():

                idx = np.array(idx_list, dtype=np.int64)
                if idx.size == 0:
                    continue

                S_local = S_active[idx]
                w_cc = float(w_target[cc])

                # --- normalization to remove dependence on C, P, and S scaling ---
                s_norm = float(np.sum(S_local * S_local)) + 1e-30
                beta_eff = orbit_beta / s_norm

                # apply rank-1 prior
                ATA_sub[np.ix_(idx, idx)] += beta_eff * np.outer(S_local, S_local)
                ATy_sub[idx] += beta_eff * (alpha * w_cc) * S_local

                # diagnostics
                try:
                    print(
                        f"[DIAG][orbit_prior_norm] orbit={cc} cols={len(idx)} "
                        f"s_norm={s_norm:.3e} beta_eff={beta_eff:.3e} "
                        f"target_mass={(alpha*w_cc):.3e}",
                        flush=True,
                    )
                except Exception:
                    pass

        z_sub = None
        try:
            z_sub = np.linalg.solve(ATA_sub, ATy_sub)
        except np.linalg.LinAlgError:
            # SVD truncated pseudo-inverse fallback: robust for near-singular ATA_sub
            U, svals, Vt = np.linalg.svd(ATA_sub, full_matrices=False)
            smax = svals[0] if svals.size else 1.0
            s_thresh = max(1e-12, 1e-6 * smax)
            s_inv = np.array([1.0/s if s > s_thresh else 0.0 for s in svals], dtype=np.float64)
            z_sub = Vt.T @ (s_inv * (U.T @ ATy_sub))

        # final non-negativity enforcement (keep small negatives for hysteresis handling)
        # but we will not aggressively remove here — see Patch C
        z_sub = np.maximum(z_sub, -1e-12)

        # write back, enforce non-negativity + active-set adjustments
        for ii, gcol in enumerate(active_idx):
            z[int(gcol)] = float(z_sub[ii])

        # Lawson–Hanson style step-back: only drop columns that are substantially
        # negative (numerical noise tolerance), otherwise clip to small positive
        # and keep active to allow subsequent refinement.
        scale = np.max(np.abs(z)) + 1e-30
        _eps_drop = 1e-10 * scale
        neg_idx = np.where(z < -_eps_drop)[0]
        if neg_idx.size > 0:
            # zero them and remove from active
            z[neg_idx] = 0.0
            active[neg_idx] = False
        # clip tiny negatives
        tiny_neg = np.where((z >= -_eps_drop) & (z < 0.0))[0]
        if tiny_neg.size > 0:
            z[tiny_neg] = 0.0

    return S_flat * z

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

def solve_streaming_nnls(
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

    # -------------------------------------------------------------------------------
    # before epoch loop (after known_zero loaded)
    known_zero_orbit = np.all(known_zero, axis=1)  # shape (C,)

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
                        f"||y||={yf_norm:.3e} " + \
                        [f"||yhat||={yhatf_norm:.3e} " if yhatf_norm > 0.0 else ""][0] + \
                        f"||r||={r_norm:.3e} nonfinite_vals={nnans}",
                        flush=True
                    )
                except Exception as _e:
                    print("[BC-FUSED][tile diag] error while computing tile diagnostics:", _e, flush=True)

                y_parts.append(y_flat)
        # end single streaming pass
        # ------------------------ end streaming --------------------------

        # ------------------------------------------------------------------
        # compute inv_sqrt_energy and scaled targets (per-tile average energy)
        # Use average column energy per tile (not the summed D_tot) so the
        # worker tile matrices and the global scaling agree.
        # ------------------------------------------------------------------
        col_energy_sum = D_tot.copy()     # D_tot currently holds summed energy
        n_tiles = max(1, len(s_ranges))   # number of tiles used in streaming pass

        # Convert summed energy -> per-tile average energy
        col_energy = col_energy_sum / float(n_tiles)

        # ensure we only use positive entries for median calculation
        pos_mask = (col_energy > 0.0)
        if np.any(pos_mask):
            energy_med = float(np.median(col_energy[pos_mask]))
        else:
            energy_med = 1.0

        # Robust floor: clamp tiny energies to a fraction of the median
        # (tunable; conservative default = 1% of median)
        floor_frac = 1e-2
        energy_floor = max(1e-30, energy_med * floor_frac)
        col_energy = np.where(col_energy > energy_floor, col_energy, energy_floor)

        # Compute raw S (1/sqrt(energy))
        S_temp = 1.0 / np.sqrt(col_energy)

        # Optional cap on extreme S to avoid a tiny set of columns dominating.
        # Cap relative to median(S). Tunable; default allows up to 8x median.
        try:
            S_median = float(np.median(S_temp))
        except Exception:
            S_median = 1.0
        max_S_factor = 8.0
        S_cap = max_S_factor * max(1.0, S_median)
        S_temp = np.where(S_temp <= S_cap, S_temp, S_cap)

        # Final inv_sqrt_energy used by the solver
        inv_sqrt_energy = S_temp
        inv_sqrt_energy_flat = inv_sqrt_energy.ravel(order="C")


        beta_eff = 0.0
        # ------------------------------------------------------------------
        # Orbit prior strength: pass-through cfg.orbit_beta (no extra scaling).
        # The prior block later normalizes by sum(S_local^2) so cfg.orbit_beta is
        # now scale-stable and interpretable (tune cfg.orbit_beta directly).
        # ------------------------------------------------------------------
        if orbit_weights is not None and cfg.orbit_beta > 0.0:
            beta_eff = float(cfg.orbit_beta)
            print(f"[DIAG] orbit prior : cfg.orbit_beta = {beta_eff:.4e}",
                flush=True)

        # DIAGNOSTIC: report S statistics (helps find extreme scalings)
        S_sample = inv_sqrt_energy_flat
        try:
            S_min = float(np.min(S_sample))
            S_p50 = float(np.median(S_sample))
            S_p90 = float(np.percentile(S_sample, 90.0))
            S_p99 = float(np.percentile(S_sample, 99.0))
            S_max = float(np.max(S_sample))
            print(f"[DIAG] inv_sqrt_energy after per-tile avg + floor/cap: min/med/p90/p99/max = {S_min:.4e}/{S_p50:.4e}/{S_p90:.4e}/{S_p99:.4e}/{S_max:.4e}", flush=True)
        except Exception as _e:
            print(f"[DIAG] error printing S stats: {_e}", flush=True)

        # run streaming active-set NNLS (monolithic) using persistent executor
        print("[BC-FUSED][MONO] starting streaming active-set NNLS (mono)", flush=True)

        # --------------------------
        # Create persistent executor
        # --------------------------
        n_workers = max(1, int(cfg.processes))
        # Use spawn to be safe with HDF5 + forking
        ctx = mp.get_context("spawn")

        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(cfg.blas_threads,),
        )

        # DIAGNOSTICS: place immediately before the monolithic call
        with open_h5(h5_path, role="reader") as f:
            # small sample: first tile only (fast)
            s0, s1 = s_ranges[0]
            DC = f["/DataCube"]
            M  = f["/HyperCube/models"]
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            M_tile = np.asarray(M[s0:s1, :, :, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                M_tile = M_tile[:, :, :, keep_idx]

        # 1) D_tot sanity
        D_sample = np.sum(M_tile * M_tile, axis=(0, 3))
        print(f"[DIAG] sample D_tot (per-column) stats: min/max/median = {D_sample.min():.4e}/{D_sample.max():.4e}/{np.median(D_sample):.4e}", flush=True)

        # 2) ATy from streaming vs manual for the same tile
        CP = int(C * P)
        A2_tile = M_tile.transpose(0, 3, 1, 2).reshape((s1 - s0) * Yt.shape[1], CP)
        ATy_tile_stream = A2_tile.T @ Yt.reshape(-1)
        ATy_tile_manual = np.zeros_like(ATy_tile_stream)
        # compute by summing per-column norm & dot to detect transpose/reshape errors
        for col in range(min(10, ATy_tile_stream.size)):
            ATy_tile_manual[col] = np.dot(A2_tile[:, col], Yt.reshape(-1))
        print(f"[DIAG] ATy_tile difference (first 10 cols) maxabs = {float(np.max(np.abs(ATy_tile_stream[:10] - ATy_tile_manual[:10]))):.4e}", flush=True)

        # 3) compare scaled vs unscaled reduced-worker ATA/ATy for a tiny active set
        # pick first k columns as "active"
        k = min(6, CP)
        active_idx = np.arange(k, dtype=np.int64)
        S_flat = np.asarray(inv_sqrt_energy_flat, dtype=np.float64).ravel()
        S_active = S_flat[active_idx]
        # Build scaled A2 like reduced worker
        A2_small = np.empty((A2_tile.shape[0], k), dtype=np.float64)
        for j, gcol in enumerate(active_idx):
            cc = int(gcol // P)
            p = int(gcol % P)
            A2_small[:, j] = M_tile[:, cc, p, :].reshape(-1)
        A2_small *= S_active[None, :]
        ATA_sub_local = A2_small.T @ A2_small
        ATy_sub_local = A2_small.T @ Yt.reshape(-1)
        print(f"[DIAG] ATA_sub_local shape, ATy_sub_local[0:6]: {ATA_sub_local.shape} {ATy_sub_local[:6]}", flush=True)

        try:
            x_flat_unscaled = _streaming_active_set_nnls_via_streaming_matvec(
                h5_path=h5_path,
                s_ranges=s_ranges,
                keep_idx=keep_idx,
                ATy_flat=ATy_flat,
                inv_sqrt_energy_flat=inv_sqrt_energy_flat,
                C=C,
                P=P,
                executor=executor,
                cfg=cfg,
                orbit_weights=orbit_weights,
                orbit_beta_eff=beta_eff,
                max_active=monolithic_max_active,
                tol_grad=1e-8,
                max_iter=5 * monolithic_max_active,
            )
            x = x_flat_unscaled.reshape(C, P).copy()
            print("[BC-FUSED][MONO] finished streaming active-set NNLS", flush=True)
        finally:
            try:
                # shut down worker pool once per monolithic solve
                executor.shutdown(wait=True)
            except Exception:
                pass

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

        if data_proxy < best_proxy:
            best_proxy = float(data_proxy)
            best_x = x.copy()
            print(f"[BC-FUSED] new best proxy {best_proxy:.3e} at epoch {ep+1}", flush=True)

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

            print("=== [BC-FUSED][epoch-summary] ===", flush=True)
            print(f"[BC-FUSED][epoch-summary] epoch={ep+1}/{cfg.epochs} elapsed_total={epoch_elapsed:.1f}s", flush=True)
            if np.isfinite(rmse_curr):
                print(f"[BC-FUSED][epoch-summary] data_proxy={data_proxy:.3e} rmse={rmse_curr:.3e}", flush=True)
            else:
                print(f"[BC-FUSED][epoch-summary] data_proxy={data_proxy:.3e} rmse=nan", flush=True)
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
            # --- orbit_weights residual diagnostic (if prior provided) -----
            try:
                if (w_target is not None) and (np.sum(w_target) > 0.0):
                    # s_full: per-orbit sums of current x (shape (C,))
                    w = np.asarray(w_target, dtype=np.float64)
                    w_sum = float(np.sum(w))
                    # same alpha as projection: scale target to current total mass
                    alpha = float(np.sum(s_full)) / max(1e-30, w_sum)
                    s_proj = alpha * w

                    # residuals (un-normalised) and fractional relative to s_proj
                    r = s_full - s_proj
                    eps = 1e-30
                    frac = r / np.maximum(np.abs(s_proj), eps)

                    # summary norms
                    l1 = float(np.sum(np.abs(r)))
                    l2 = float(np.linalg.norm(r))
                    maxabs = float(np.max(np.abs(r))) if r.size else 0.0
                    mean_frac = float(np.median(frac)) if frac.size else 0.0

                    # top offenders by absolute residual
                    nshow = min(8, r.size)
                    idx_sort = np.argsort(np.abs(r))[::-1][:nshow]
                    offenders = ", ".join(
                        [f"{int(i)}:{r[i]:+.3e}({frac[i]:+.2%})" for i in idx_sort]
                    )

                    print("[DIAG][orbit_weights] residuals: L1={:.3e} L2={:.3e} "
                        "max_abs={:.3e} median_frac={:+.2%}".format(
                            l1, l2, maxabs, mean_frac), flush=True)
                    print("[DIAG][orbit_weights] top offenders (idx:resid(frac)): "
                        + offenders, flush=True)
                else:
                    # no prior available
                    print("[DIAG][orbit_weights] no w_target present; skipping "
                        "orbit-residual diagnostic.", flush=True)
            except Exception as _e:
                print("[DIAG][orbit_weights] diagnostic failed:", _e, flush=True)

            print("=== [BC-FUSED][epoch-summary] end ===", flush=True)
        except Exception as _e:
            print("[BC-FUSED][epoch-summary] error:", _e, flush=True)
        # --- end epoch summary diagnostics ---

    # done epochs
    th = cu.zero_floor_inplace(best_x, rel_tol=1e-25, abs_tol=0.0)
    elapsed = time.perf_counter() - t0
    stats = dict(
        epochs=int(cfg.epochs),
        elapsed_sec=elapsed,
        rmse_proxy_best=float(best_proxy),
        known_zero_mask=known_zero.copy(),
    )
    return best_x, stats

# ------------------------------------------------------------------------------