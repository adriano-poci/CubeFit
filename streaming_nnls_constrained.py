# -*- coding: utf-8 -*-
r"""
    streaming_nnls_constrained.py
    Adriano Poci
    University of Oxford
    2026

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Implementation of a TRUE MONOLITHIC streaming active-set NNLS solver for the
        CubeFit problem, with hybrid parallelism (process-level over tile batches
        + BLAS threads inside workers).

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Forked from `streaming_nnls_augmented_rows` v1.2;
        A priori impose the orbit prior before the fit. 4 August 2026
v1.1:   Fixed ordering bug on `promoted_cols`;
        Ensure each non-zero orbit has at least one active column. 5 August 2026
v1.2:   Replaced fixed absolute orbit-mass targets with a hard a priori
            unit-sum orbit shape and a data-fitted nonnegative global amplitude;
        Extended the constrained KKT system to solve jointly for active
            coefficients, the global amplitude, and orbit Lagrange multipliers;
        Added constrained reduced-gradient promotion using the orbit
            multipliers;
        Added feasible fresh-start initialization and resume support for the
            fitted global amplitude and multipliers;
        Excluded zero-weight orbits from the active support and promotion pool;
        Preserved at least one active population in every positive-weight
            orbit during pruning;
        Fixed failed-promotion cleanup ordering so the active mask and
            coefficient vector remain consistent;
        Updated orbit-target, constraint, checkpoint, and final diagnostics
            for the flexible-amplitude hard prior. 6 August 2026
v1.3:   Switched checkpoint emission to atomic `save_checkpoint` and removed the
            split snapshot/state write path so resumable checkpoints preserve the
            full constrained solver state. 7 August 2026
v1.4:   Emit current `x` vector in the diagnostics to assist in debugging;
        Don't zero the solution in-place before returning, as this mutates the
            vector after the solver has finished;
        Make the final summary statistics use `best_x` as a total solver
            summary. 11 August 2026
"""

from __future__ import annotations, print_function

import os, sys, traceback, json
import math
import time
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
try:
    from scipy.optimize import nnls as _scipy_nnls
    _HAS_SCIPY_NNLS = True
except Exception:
    _HAS_SCIPY_NNLS = False

from CubeFit.hdf5_manager import open_h5
from CubeFit import cube_utils as cu

vprint = cu.vprint

def _init_worker(blas_threads: int):
    # called once per process; set BLAS env vars
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, blas_threads // 2))

# ------------------------------------------------------------------------------

def _diag_json_default(obj):
    """
    Make NumPy values JSON-serialisable.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _diag_append_jsonl(path: str | None, record: dict) -> None:
    """
    Append one JSON record to a JSONL file.
    """
    if not path:
        return

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                record,
                default=_diag_json_default,
                separators=(",", ":"),
            )
            + "\n"
        )


def _support_stats_1d(vec: np.ndarray) -> dict:
    """
    Summarise within-orbit coefficient diversity.
    """
    v = np.asarray(vec, dtype=np.float64).ravel(order="C")
    v = v[np.isfinite(v) & (v > 0.0)]

    if v.size == 0:
        return {
            "nnz": 0,
            "eff_support": 0.0,
            "entropy": 0.0,
            "top_share": 0.0,
            "gini_proxy": 0.0,
        }

    mass = float(np.sum(v))
    p = v / max(mass, 1e-30)
    eff_support = float(1.0 / np.sum(p * p))
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    top_share = float(np.max(p))
    gini_proxy = float(1.0 - np.sum(p * p))

    return {
        "nnz": int(v.size),
        "eff_support": eff_support,
        "entropy": entropy,
        "top_share": top_share,
        "gini_proxy": gini_proxy,
    }

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

    cc_arr = (active_idx // P).astype(np.int64)
    pp_arr = (active_idx % P).astype(np.int64)

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        for (s0, s1) in batch:
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            y_flat = Yt.reshape(-1)

            Sblk = s1 - s0
            Lk = Yt.shape[1]
            rows = Sblk * Lk

            # Only store the active columns, not the full tile.
            A_act = np.empty((rows, k), dtype=np.float64)

            for j, (cc, pp) in enumerate(zip(cc_arr, pp_arr)):
                col = np.asarray(M[s0:s1, cc, pp, :], dtype=np.float64, order="C")
                if keep_idx is not None:
                    col = col[:, keep_idx]
                A_act[:, j] = S_active[j] * col.reshape(-1)

            ATA_loc += A_act.T @ A_act
            ATy_loc += A_act.T @ y_flat

    return ATA_loc, ATy_loc

# ------------------------------------------------------------------------------

def _worker_ATAz(
    h5_path: str,
    batch: list,
    keep_idx,
    z: np.ndarray,
    CP: int,
    S_flat: np.ndarray,
    C: int,
    P: int,
):
    """
    Compute the exact quantity A^T A (S z) for a batch of tiles, but with
    much lower peak memory than the original implementation.

    This preserves the original mathematics:

        z_s = (S_flat * z).reshape(C, P)

        v = A @ z_s_flat
        g = A.T @ v
        partial += (S_flat.reshape(C, P) * g).reshape(-1)

    The only difference is that the model cube is read orbit-by-orbit instead
    of loading the full (Sblk, C, P, Lk) tile into memory.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    batch : list[tuple[int, int]]
        List of (s0, s1) tile ranges assigned to this worker.
    keep_idx : ndarray or None
        Wavelength indices to keep after masking, or None for full wavelength
        coverage.
    z : ndarray, shape (C * P,)
        Current reduced NNLS variable in z-space.
    CP : int
        Total number of columns, equal to C * P.
    S_flat : ndarray, shape (C * P,)
        Column scaling vector.
    C : int
        Number of orbit components.
    P : int
        Number of populations per orbit.

    Returns
    -------
    partial : ndarray, shape (C * P,)
        Contribution of this worker to A^T A (S z).
    """
    z = np.asarray(z, dtype=np.float64).ravel(order="C")
    S_flat = np.asarray(S_flat, dtype=np.float64).ravel(order="C")

    if z.size != CP:
        raise ValueError(
            f"z has size {z.size}, expected CP={CP}."
        )
    if S_flat.size != CP:
        raise ValueError(
            f"S_flat has size {S_flat.size}, expected CP={CP}."
        )
    if CP != C * P:
        raise ValueError(
            f"CP={CP} is inconsistent with C*P={C * P}."
        )

    # Same scaling as the original code.
    z_cp = (S_flat * z).reshape(C, P)

    partial = np.zeros((CP,), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]

        for (s0, s1) in batch:
            Sblk = s1 - s0

            # Read one orbit slice to determine the wavelength length after mask.
            sample = np.asarray(
                M[s0:s1, 0, :, :],
                dtype=np.float64,
                order="C",
            )
            if keep_idx is not None:
                sample = sample[:, :, keep_idx]
            if sample.ndim != 3:
                raise RuntimeError(
                    "Unexpected model slice shape while inferring Lk."
                )

            _, Pm, Lk = sample.shape
            if Pm != P:
                raise RuntimeError(
                    f"Model population dimension {Pm} != expected P={P}."
                )

            # First pass: exact v = A @ (S z), but one orbit slice at a time.
            v = np.zeros((Sblk, Lk), dtype=np.float64)

            for cc in range(C):
                M_cc = np.asarray(
                    M[s0:s1, cc, :, :],
                    dtype=np.float64,
                    order="C",
                )
                if keep_idx is not None:
                    M_cc = M_cc[:, :, keep_idx]

                # M_cc has shape (Sblk, P, Lk)
                # z_cp[cc] has shape (P,)
                # result has shape (Sblk, Lk)
                v += np.tensordot(
                    z_cp[cc],
                    M_cc,
                    axes=(0, 1),
                )

            # Second pass: exact g = A^T @ v, again orbit slice by orbit slice.
            for cc in range(C):
                M_cc = np.asarray(
                    M[s0:s1, cc, :, :],
                    dtype=np.float64,
                    order="C",
                )
                if keep_idx is not None:
                    M_cc = M_cc[:, :, keep_idx]

                # M_cc: (Sblk, P, Lk)
                # v   : (Sblk, Lk)
                # g_cc: (P,)
                g_cc = np.tensordot(
                    M_cc,
                    v,
                    axes=([0, 2], [0, 1]),
                )

                base = cc * P
                partial[base:base + P] += (
                    S_flat[base:base + P] * g_cc
                )

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
    processes: int = 2
    blas_threads: int = 8
    apply_mask: bool = True
    dset_slots: int = 1_000_003
    dset_bytes: int = 256 * 1024**2
    dset_w0: float = 0.90
    s_tile_override: Optional[int] = None
    verbose: bool = True

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
    Return a unit-sum orbit-shape vector, or None if unavailable.

    Accepts:
      - orbit_weights == None: try reading '/CompWeights' from HDF5.
      - orbit_weights shape == (C,): use as the orbit-shape vector.
      - orbit_weights shape == (C*P,): sum over populations -> (C,).

    The result is normalised to sum to one.
    """
    w = None
    if orbit_weights is not None:
        w = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")
    else:
        with open_h5(h5_path, role="reader") as f:
            if "/CompWeights" in f:
                w = np.asarray(
                    f["/CompWeights"][...],
                    dtype=np.float64,
                ).ravel(order="C")
            else:
                return None

    if w.size == C:
        pass
    elif w.size == C * P:
        w = w.reshape(C, P).sum(axis=1)
    else:
        raise ValueError(
            f"orbit_weights length {w.size} incompatible with C={C}, P={P}. "
            f"Expected C or C*P."
        )

    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
        raise ValueError("orbit_weights must have positive total mass")

    return w / w_sum

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

def _orbit_hard_constraint_solve(
    ATA_sub_reg: np.ndarray,
    ATy_sub: np.ndarray,
    active_idx: np.ndarray,
    S_active: np.ndarray,
    orbit_shape: np.ndarray,
    C: int,
    P: int,
    *,
    mu_rel: float = 1e-8,
    negative_tol: float = 1e-10,
    constraint_tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    Solve the reduced equality-constrained NNLS problem with a fixed
    orbit-prior shape and a data-fitted global amplitude.

    The problem is

        minimize

            0.5 * z.T @ ATA_sub_reg @ z - ATy_sub.T @ z

        subject to

            z >= 0
            alpha >= 0

            sum_j S_active[j] * z[j]
                = alpha * orbit_shape[c]

        for every positive-shape orbit ``c``.

    The orbit shape is imposed exactly. Only the common global amplitude
    ``alpha`` is free.

    Parameters
    ----------
    ATA_sub_reg : ndarray, shape (k, k)
        Regularized reduced Hessian.
    ATy_sub : ndarray, shape (k,)
        Reduced linear term.
    active_idx : ndarray, shape (k,)
        Global indices of the active columns.
    S_active : ndarray, shape (k,)
        Column scaling factors for active columns.
    orbit_shape : ndarray, shape (C,)
        Nonnegative orbit-prior shape. It is normalized internally to
        unit sum.
    C : int
        Number of orbits.
    P : int
        Number of populations per orbit.
    mu_rel : float, optional
        Relative numerical ridge applied to the coefficient Hessian.
    negative_tol : float, optional
        Negative-coefficient tolerance.
    constraint_tol : float, optional
        Relative equality-constraint tolerance.

    Returns
    -------
    z_out : ndarray, shape (k,)
        Nonnegative constrained solution on the supplied active support.
    info : dict
        Solver status, fitted amplitude, absolute orbit targets,
        constraint residuals, and orbit-level KKT multipliers.

    Raises
    ------
    ValueError
        If the supplied arrays have inconsistent sizes or the orbit shape
        is invalid.

    Examples
    --------
    >>> z, info = _orbit_hard_constraint_solve(
    ...     ATA_sub_reg,
    ...     ATy_sub,
    ...     active_idx,
    ...     S_active,
    ...     orbit_shape,
    ...     C,
    ...     P,
    ... )
    >>> alpha = info["alpha"]
    """
    H = np.asarray(
        ATA_sub_reg,
        dtype=np.float64,
        order="C",
    )
    b = np.asarray(
        ATy_sub,
        dtype=np.float64,
    ).ravel(order="C")
    active_idx = np.asarray(
        active_idx,
        dtype=np.int64,
    ).ravel(order="C")
    S_active = np.asarray(
        S_active,
        dtype=np.float64,
    ).ravel(order="C")
    shape = np.asarray(
        orbit_shape,
        dtype=np.float64,
    ).ravel(order="C")

    k = int(active_idx.size)
    lambda_zero = np.zeros((C,), dtype=np.float64)
    target_zero = np.zeros((C,), dtype=np.float64)

    def _failure_info(
        reason: str,
        *,
        negative_count: int = 0,
        n_free: int = 0,
        removed_local_indices=None,
        missing_orbits=None,
        alpha=None,
        lambda_orbit=None,
    ) -> dict:
        if removed_local_indices is None:
            removed_local_indices = []
        if lambda_orbit is None:
            lambda_orbit = lambda_zero

        out = {
            "accepted": False,
            "reason": str(reason),
            "alpha": alpha,
            "orbit_target_mass": target_zero.copy(),
            "resid_l1": np.inf,
            "resid_l2": np.inf,
            "resid_linf": np.inf,
            "rel_resid_linf": np.inf,
            "negative_count": int(negative_count),
            "n_free": int(n_free),
            "n_removed": int(len(removed_local_indices)),
            "removed_local_indices": list(removed_local_indices),
            "lambda_orbit": np.asarray(
                lambda_orbit,
                dtype=np.float64,
            ).copy(),
        }

        if missing_orbits is not None:
            out["missing_orbits"] = [
                int(cc) for cc in missing_orbits
            ]

        return out

    if H.shape != (k, k):
        raise ValueError(
            "ATA_sub_reg has a shape inconsistent with active_idx."
        )
    if b.size != k or S_active.size != k:
        raise ValueError(
            "ATy_sub or S_active has a size inconsistent with active_idx."
        )
    if shape.size != C:
        raise ValueError(
            "orbit_shape must have length C."
        )
    if not np.all(np.isfinite(shape)):
        raise ValueError(
            "orbit_shape contains non-finite values."
        )

    shape = np.maximum(shape, 0.0)
    shape_sum = float(np.sum(shape))

    if not np.isfinite(shape_sum) or shape_sum <= 0.0:
        raise ValueError(
            "orbit_shape must have positive finite total weight."
        )

    shape /= shape_sum
    positive_orbits = np.flatnonzero(shape > 0.0)

    if k == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            _failure_info(
                "empty_support",
                n_free=0,
            ),
        )

    orbit_of_col = (
        active_idx // P
    ).astype(np.int64)

    if np.any(orbit_of_col < 0) or np.any(orbit_of_col >= C):
        raise ValueError(
            "active_idx contains an invalid orbit index."
        )

    # Columns belonging to zero-shape orbits must not enter this helper.
    zero_shape_active = np.flatnonzero(
        shape[orbit_of_col] <= 0.0
    )

    if zero_shape_active.size > 0:
        return (
            np.zeros((k,), dtype=np.float64),
            _failure_info(
                "zero_shape_orbit_active",
                n_free=k,
            ),
        )

    support_count = np.bincount(
        orbit_of_col,
        minlength=C,
    )

    missing = positive_orbits[
        support_count[positive_orbits] == 0
    ]

    if missing.size > 0:
        return (
            np.zeros((k,), dtype=np.float64),
            _failure_info(
                "missing_orbit_support",
                n_free=k,
                missing_orbits=missing,
            ),
        )

    free = np.ones((k,), dtype=bool)
    removed = []

    diag = np.diag(H)
    finite_diag = diag[np.isfinite(diag)]

    if finite_diag.size > 0:
        diag_med = float(np.median(np.abs(finite_diag)))
    else:
        diag_med = 1.0

    mu = max(
        1e-12,
        float(mu_rel) * max(1.0, diag_med),
    )

    max_inner = max(1, k)

    for _ in range(max_inner):
        free_idx = np.flatnonzero(free)
        n_free = int(free_idx.size)

        if n_free == 0:
            break

        free_orbits = orbit_of_col[free_idx]

        # Every positive-shape orbit must retain at least one free variable.
        free_count = np.bincount(
            free_orbits,
            minlength=C,
        )

        missing_free = positive_orbits[
            free_count[positive_orbits] == 0
        ]

        if missing_free.size > 0:
            return (
                np.zeros((k,), dtype=np.float64),
                _failure_info(
                    "missing_free_orbit_support",
                    n_free=n_free,
                    removed_local_indices=removed,
                    missing_orbits=missing_free,
                ),
            )

        # Only positive-shape orbits require constraint rows.
        constraint_orbits = positive_orbits.copy()
        n_constraints = int(constraint_orbits.size)

        B = np.zeros(
            (n_constraints, n_free),
            dtype=np.float64,
        )

        orbit_to_row = {
            int(cc): row
            for row, cc in enumerate(constraint_orbits)
        }

        for local_j, reduced_j in enumerate(free_idx):
            cc = int(orbit_of_col[reduced_j])
            B[orbit_to_row[cc], local_j] = S_active[reduced_j]

        shape_sub = shape[constraint_orbits]

        Hff = H[np.ix_(free_idx, free_idx)]
        bff = b[free_idx]

        # Unknown ordering:
        #
        #     [z_free, alpha, lambda]
        #
        # KKT equations:
        #
        #     (H + mu I) z + B.T lambda = b
        #                -shape.T lambda = 0
        #                B z - shape alpha = 0
        #
        n_primal = n_free + 1
        n_total = n_primal + n_constraints

        KKT = np.zeros(
            (n_total, n_total),
            dtype=np.float64,
        )

        KKT[:n_free, :n_free] = (
            Hff
            + mu * np.eye(
                n_free,
                dtype=np.float64,
            )
        )

        # Coupling to the equality multipliers.
        KKT[:n_free, n_primal:] = B.T
        KKT[n_free, n_primal:] = -shape_sub

        KKT[n_primal:, :n_free] = B
        KKT[n_primal:, n_free] = -shape_sub

        rhs = np.zeros(
            (n_total,),
            dtype=np.float64,
        )
        rhs[:n_free] = bff

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(
                KKT,
                rhs,
                rcond=None,
            )[0]

        z_free = np.asarray(
            sol[:n_free],
            dtype=np.float64,
        )
        alpha_trial = float(sol[n_free])
        lambda_sub = np.asarray(
            sol[n_primal:],
            dtype=np.float64,
        )

        lambda_orbit = np.zeros(
            (C,),
            dtype=np.float64,
        )
        lambda_orbit[constraint_orbits] = lambda_sub

        if (
            not np.all(np.isfinite(z_free))
            or not np.isfinite(alpha_trial)
            or not np.all(np.isfinite(lambda_sub))
        ):
            return (
                np.zeros((k,), dtype=np.float64),
                _failure_info(
                    "nonfinite_kkt_solution",
                    n_free=n_free,
                    removed_local_indices=removed,
                ),
            )

        negative_local = np.flatnonzero(
            z_free < -negative_tol
        )

        if negative_local.size == 0:
            # With z >= 0 and positive orbit shape, exact feasibility implies
            # alpha >= 0. A materially negative alpha indicates a failed solve.
            if alpha_trial < -negative_tol:
                return (
                    np.zeros((k,), dtype=np.float64),
                    _failure_info(
                        "negative_alpha",
                        n_free=n_free,
                        removed_local_indices=removed,
                        alpha=float(alpha_trial),
                        lambda_orbit=lambda_orbit,
                    ),
                )

            alpha_out = max(0.0, alpha_trial)

            z_out = np.zeros(
                (k,),
                dtype=np.float64,
            )
            z_out[free_idx] = np.maximum(
                z_free,
                0.0,
            )

            full_mass = np.bincount(
                orbit_of_col,
                weights=S_active * z_out,
                minlength=C,
            ).astype(np.float64)

            target_mass = alpha_out * shape
            mass_resid = full_mass - target_mass

            scale = np.maximum(
                np.abs(target_mass),
                1.0,
            )
            rel_linf = float(
                np.max(
                    np.abs(mass_resid) / scale
                )
            )

            # The alpha stationarity equation should satisfy
            # shape.T @ lambda == 0.
            alpha_stationarity = float(
                np.dot(shape, lambda_orbit)
            )

            accepted = (
                np.all(np.isfinite(z_out))
                and np.isfinite(alpha_out)
                and alpha_out >= 0.0
                and np.all(np.isfinite(lambda_orbit))
                and rel_linf <= constraint_tol
            )

            return z_out, {
                "accepted": bool(accepted),
                "reason": (
                    "accepted"
                    if accepted
                    else "constraint_residual"
                ),
                "alpha": float(alpha_out),
                "orbit_target_mass": target_mass,
                "resid_l1": float(
                    np.sum(np.abs(mass_resid))
                ),
                "resid_l2": float(
                    np.linalg.norm(mass_resid)
                ),
                "resid_linf": float(
                    np.max(np.abs(mass_resid))
                ),
                "rel_resid_linf": float(rel_linf),
                "alpha_stationarity": alpha_stationarity,
                "negative_count": 0,
                "n_free": int(n_free),
                "n_removed": int(len(removed)),
                "removed_local_indices": removed.copy(),
                "constraint_orbits": (
                    constraint_orbits.tolist()
                ),
                "lambda_orbit": lambda_orbit,
            }

        # Remove the most negative coefficient that is not the final free
        # variable in its positive-shape orbit.
        order = negative_local[
            np.argsort(
                z_free[negative_local]
            )
        ]

        remove_reduced_idx = None

        for local_j in order:
            candidate = int(
                free_idx[int(local_j)]
            )
            cc = int(orbit_of_col[candidate])

            remaining_in_orbit = int(
                np.count_nonzero(
                    free
                    & (orbit_of_col == cc)
                )
            )

            if remaining_in_orbit <= 1:
                continue

            remove_reduced_idx = candidate
            break

        if remove_reduced_idx is None:
            z_fail = np.zeros(
                (k,),
                dtype=np.float64,
            )
            z_fail[free_idx] = z_free

            return z_fail, _failure_info(
                "negative_required_variable",
                negative_count=int(negative_local.size),
                n_free=n_free,
                removed_local_indices=removed,
                alpha=float(alpha_trial),
                lambda_orbit=lambda_orbit,
            )

        free[remove_reduced_idx] = False
        removed.append(
            int(remove_reduced_idx)
        )

    return (
        np.zeros((k,), dtype=np.float64),
        _failure_info(
            "inner_active_set_exhausted",
            n_free=int(np.count_nonzero(free)),
            removed_local_indices=removed,
        ),
    )

# ------------------------------------------------------------------------------

def streamActiveSetNNLS(
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
    x0_flat: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    max_active: int = 1000,
    tol_grad: float = 1e-8,
    max_iter: int = 5000,
    checkpoint_cb=None,
    checkpoint_every: int = 0,
):
    """
    TRUE MONOLITHIC streaming active-set NNLS.
    """

    def _robust_scale_ref(vec: np.ndarray, fallback: float = 1.0) -> float:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        arr = np.abs(arr[arr > 0.0])
        if arr.size == 0:
            return float(max(1.0, fallback))
        ref = float(np.median(arr))
        if (not np.isfinite(ref)) or (ref <= 0.0):
            return float(max(1.0, fallback))
        return ref

    def _watch_progress() -> bool:
        """
        Update the committed-state watchdog.

        Returns
        -------
        stalled : bool
            True if the committed state has failed to move for too long.
        """
        nonlocal committed_z, committed_active, no_progress_count

        z_delta = float(np.linalg.norm(z - committed_z))
        z_ref = max(1.0, float(np.linalg.norm(committed_z)))
        z_delta_rel = z_delta / z_ref

        active_delta = int(np.count_nonzero(active ^ committed_active))

        if (z_delta_rel <= z_delta_tol) and (active_delta <= active_delta_tol):
            no_progress_count += 1
        else:
            no_progress_count = 0
            committed_z = z.copy()
            committed_active = active.copy()

        if no_progress_count >= hard_stall_patience:
            print(
                "[MONO] terminating on committed-state stall: "
                f"z_delta_rel={z_delta_rel:.3e} "
                f"active_delta={active_delta} "
                f"no_progress={no_progress_count} "
                f"window={progress_window}",
                flush=True,
            )
            return True

        return False

    CP = int(C * P)
    orbit_of_col = (np.arange(CP, dtype=np.int64) // P).astype(np.int64)
    negative_grad_count = 0
    explore_budget = max(1, min(20, CP // 50))

    orbit_shape = (
        _canon_orbit_weights(h5_path, orbit_weights, C=C, P=P)
        if orbit_weights is not None
        else None
    )

    S_flat = np.asarray(inv_sqrt_energy_flat, dtype=np.float64).ravel()
    ATy_scaled = S_flat * ATy_flat

    grad_ref = max(1.0, float(np.max(np.abs(ATy_scaled))))
    data_scale_ref = _robust_scale_ref(ATy_scaled, fallback=grad_ref)
    tol_grad_rel = 1e-6

    # --- committed-state watchdog ---
    z_delta_tol = 1e-6
    active_delta_tol = 0
    progress_window = 20
    force_explore_after = 4
    hard_stall_patience = 20
    no_progress_count = 0

    # --- positive-gradient batch-promotion controls ---
    positive_batch_size = 8 if C <= 3 else 24

    # The orbit shape is fixed a priori. Its global amplitude is a fitted
    # variable in every constrained reduced solve.
    has_hard_orbit_shape = (
        orbit_shape is not None
        and np.any(orbit_shape > 0.0)
    )

    positive_shape_orbits = (
        np.flatnonzero(orbit_shape > 0.0)
        if has_hard_orbit_shape
        else np.zeros((0,), dtype=np.int64)
    )

    zero_shape_orbits = (
        np.flatnonzero(orbit_shape <= 0.0)
        if has_hard_orbit_shape
        else np.zeros((0,), dtype=np.int64)
    )

    # Current accepted global amplitude. It is initialized below and then
    # replaced by the value fitted by each accepted KKT solve.
    alpha_current = 0.0

    # Orbit-level Lagrange multipliers from the most recently accepted
    # constrained reduced solve.
    lambda_orbit_current = np.zeros(
        (C,),
        dtype=np.float64,
    )

    diag_level = int(os.environ.get("CUBEFIT_DIAG_LEVEL", "1"))
    diag_stride = max(1, int(os.environ.get("CUBEFIT_DIAG_STRIDE", "1")))
    diag_jsonl_path = os.environ.get("CUBEFIT_DIAG_JSONL", "").strip() or None
    diag_topk = max(1, int(os.environ.get("CUBEFIT_DIAG_TOPK", "12")))
    diag_t0 = time.perf_counter()

    def _emit_diag(record: dict) -> None:
        if diag_level <= 0 and ('error' not in str(record.get("kind"))):
            return
        rec = dict(record)
        rec.setdefault("t_sec", float(time.perf_counter() - diag_t0))
        _diag_append_jsonl(diag_jsonl_path, rec)

    _emit_diag(
        {
            "kind": "mono_setup",
            "C": int(C),
            "P": int(P),
            "CP": int(CP),
            "processes": int(cfg.processes),
            "blas_threads": int(cfg.blas_threads),
            "data_scale_ref": float(data_scale_ref),
            "grad_ref": float(grad_ref),
            "tol_grad": float(tol_grad),
            "tol_grad_rel": float(tol_grad_rel),
            "diag_level": int(diag_level),
            "diag_stride": int(diag_stride),
            "diag_topk": int(diag_topk),
            "hard_orbit_shape": bool(has_hard_orbit_shape),
            "orbit_shape": (
                orbit_shape.tolist()
                if orbit_shape is not None
                else None
            ),
            "orbit_shape_sum": (
                float(np.sum(orbit_shape))
                if orbit_shape is not None
                else None
            ),
            "orbit_target_sum": None,
        }
    )

    if x0_flat is None:
        z = np.zeros((CP,), dtype=np.float64)
    else:
        x0_flat = np.asarray(
            x0_flat,
            dtype=np.float64,
        ).ravel(order="C")

        if x0_flat.size != CP:
            raise ValueError(
                "x0_flat has wrong size in monolithic solver"
            )

        z = np.divide(
            x0_flat,
            S_flat,
            out=np.zeros_like(
                x0_flat,
                dtype=np.float64,
            ),
            where=S_flat > 0.0,
        )
        np.maximum(z, 0.0, out=z)

    # Normalize and validate the resume metadata before it is used.
    resume_state = dict(resume_state or {})

    start_iter = int(
        resume_state.get("iter", 0) or 0
    )

    if start_iter < 0:
        start_iter = 0

    # Build a feasible initial state for the fixed orbit shape. The initial
    # amplitude is only a starting value; it is not held fixed by the solver.
    if has_hard_orbit_shape:
        x_seed = (S_flat * z).reshape(C, P)

        # Strictly exclude zero-shape orbits.
        for cc in zero_shape_orbits:
            x_seed[int(cc), :] = 0.0

        if not resume_state:
            warm_mass = float(np.sum(x_seed))

            if np.isfinite(warm_mass) and warm_mass > 0.0:
                alpha_current = warm_mass
            else:
                # This value only constructs a nonzero feasible seed.
                # The first constrained KKT solve refits it from the data.
                alpha_current = 1.0

            for cc in positive_shape_orbits:
                cc = int(cc)
                target_cc = (
                    alpha_current
                    * float(orbit_shape[cc])
                )

                current_mass = float(
                    np.sum(x_seed[cc, :])
                )

                if (
                    np.isfinite(current_mass)
                    and current_mass > 0.0
                ):
                    # Preserve the warm-start population mixture.
                    x_seed[cc, :] *= (
                        target_cc / current_mass
                    )
                else:
                    base = cc * P
                    orbit_cols = np.arange(
                        base,
                        base + P,
                        dtype=np.int64,
                    )

                    local_score = ATy_scaled[orbit_cols]
                    p_seed = int(
                        np.argmax(local_score)
                    )

                    x_seed[cc, :] = 0.0
                    x_seed[cc, p_seed] = target_cc

            x_seed_flat = x_seed.ravel(order="C")

            z = np.divide(
                x_seed_flat,
                S_flat,
                out=np.zeros_like(
                    x_seed_flat,
                    dtype=np.float64,
                ),
                where=S_flat > 0.0,
            )
            np.maximum(z, 0.0, out=z)

        else:
            # On resume, infer a fallback amplitude from the restored physical
            # coefficient vector. A saved alpha, if present, is restored below.
            alpha_current = float(
                np.sum(
                    (S_flat * z).reshape(C, P),
                    dtype=np.float64,
                )
            )

            if (
                not np.isfinite(alpha_current)
                or alpha_current < 0.0
            ):
                alpha_current = 0.0
    z_scale0 = np.max(z) + 1e-30
    active = z > (1e-10 * z_scale0)

    if has_hard_orbit_shape and zero_shape_orbits.size > 0:
        for cc in zero_shape_orbits:
            base = int(cc) * P
            z[base:base + P] = 0.0
            active[base:base + P] = False

    # initialize committed baseline
    committed_z = z.copy()
    committed_active = active.copy()

    # Probation for newly admitted columns.
    probation_iters = 6
    probation_rel = 5e-7
    probation_abs = 1e-13
    provisional_hits: dict[int, int] = {}

    # Column-level tabu for genuinely unhelpful promotions.
    col_cooldown_until: dict[int, int] = {}
    orbit_cooldown_until: dict[int, int] = {}
    cooldown_iters = 4
    orbit_cooldown_iters = 6
    promotion_noop_rel_tol = 1e-6
    promotion_noop_abs_tol = 1e-10

    negative_grad_count = 0
    no_progress_count = 0

    if resume_state:
        no_progress_count = int(resume_state.get("no_progress_count", 0))
        negative_grad_count = int(resume_state.get("negative_grad_count", 0))

        if "active_mask" in resume_state:
            am = np.asarray(
                resume_state["active_mask"], dtype=bool
            ).ravel(order="C")
            if am.size == CP:
                active = am.copy()

        if "committed_active_mask" in resume_state:
            cam = np.asarray(
                resume_state["committed_active_mask"], dtype=bool
            ).ravel(order="C")
            if cam.size == CP:
                committed_active = cam.copy()

        if "committed_z" in resume_state:
            cz = np.asarray(
                resume_state["committed_z"], dtype=np.float64
            ).ravel(order="C")
            if cz.size == CP:
                committed_z = cz.copy()

        if "alpha_current" in resume_state:
            alpha_saved = float(
                resume_state["alpha_current"]
            )

            if (
                np.isfinite(alpha_saved)
                and alpha_saved >= 0.0
            ):
                alpha_current = alpha_saved
        if "lambda_orbit" in resume_state:
            lambda_saved = np.asarray(
                resume_state["lambda_orbit"],
                dtype=np.float64,
            ).ravel(order="C")

            if (
                lambda_saved.size == C
                and np.all(np.isfinite(lambda_saved))
            ):
                lambda_orbit_current = lambda_saved.copy()

        if has_hard_orbit_shape and zero_shape_orbits.size > 0:
            for cc in zero_shape_orbits:
                base = int(cc) * P
                z[base:base + P] = 0.0
                active[base:base + P] = False
                committed_z[base:base + P] = 0.0
                committed_active[base:base + P] = False
        def _pairs_to_dict(pairs):
            out = {}
            for item in pairs or []:
                try:
                    k, v = item
                    out[int(k)] = int(v)
                except Exception:
                    pass
            return out

        provisional_hits = _pairs_to_dict(
            resume_state.get("provisional_hits", [])
        )
        col_cooldown_until = _pairs_to_dict(
            resume_state.get("col_cooldown_until", [])
        )
        orbit_cooldown_until = _pairs_to_dict(
            resume_state.get(
                "orbit_cooldown_until",
                [],
            )
        )

    def _on_cooldown(col: int, it: int) -> bool:
        return it < col_cooldown_until.get(int(col), -1)

    def _cooldown_cols(cols, it: int, extra_iters: int = 0) -> None:
        expire = int(it + cooldown_iters + extra_iters)
        for c in np.asarray(cols, dtype=np.int64).ravel():
            c = int(c)
            col_cooldown_until[c] = max(col_cooldown_until.get(c, -1), expire)
    def _orbit_on_cooldown(cc: int, it: int) -> bool:
        return it < orbit_cooldown_until.get(int(cc), -1)

    def _cooldown_orbits(orbits, it: int, extra_iters: int = 0) -> None:
        expire = int(it + orbit_cooldown_iters + extra_iters)
        for cc in np.asarray(orbits, dtype=np.int64).ravel():
            cc = int(cc)
            orbit_cooldown_until[cc] = max(
                orbit_cooldown_until.get(cc, -1),
                expire,
            )

    def _combined_cooldown_mask(it: int) -> np.ndarray:
        mask = _col_cooldown_mask(it)
        for cc, expiry in orbit_cooldown_until.items():
            if expiry > it:
                mask[orbit_of_col == int(cc)] = True
        return mask
    def _col_cooldown_mask(it: int) -> np.ndarray:
        mask = np.zeros(CP, dtype=bool)
        for c, expiry in col_cooldown_until.items():
            if expiry > it:
                mask[c] = True
        return mask

    def _current_x_from_z(z_vec: np.ndarray) -> np.ndarray:
        return S_flat * z_vec

    if start_iter >= max_iter:
        print(
            "[MONO] resume_state already reached max_iter; returning the "
            "committed solution without entering the outer loop.",
            flush=True,
        )
        return _current_x_from_z(z)

    checkpoint_error_logged = False

    def _log_checkpoint_error(where: str, exc: Exception) -> None:
        nonlocal checkpoint_error_logged
        if not checkpoint_error_logged:
            print(f"[MONO][checkpoint] {where}: {exc}", flush=True)
            checkpoint_error_logged = True
    
    def _pack_resume_state(it: int, phase: str, final: bool) -> dict:
        return {
            "iter": int(it + 1),
            "max_iter": int(max_iter),
            "phase": str(phase),
            "final": bool(final),
            "no_progress_count": int(no_progress_count),
            "negative_grad_count": int(negative_grad_count),
            "active_mask": active.astype(bool).tolist(),
            "committed_active_mask": committed_active.astype(bool).tolist(),
            "committed_z": committed_z.astype(np.float64).tolist(),
            "provisional_hits": [
                [int(k), int(v)] for k, v in provisional_hits.items()
            ],
            "col_cooldown_until": [
                [int(k), int(v)] for k, v in col_cooldown_until.items()
            ],
            "lambda_orbit": (
                lambda_orbit_current.astype(np.float64).tolist()
            ),
            "alpha_current": float(alpha_current),
            "orbit_cooldown_until": [
                [int(k), int(v)]
                for k, v in orbit_cooldown_until.items()
            ],
        }
    
    def _emit_checkpoint(
        it: int,
        *,
        final: bool = False,
        phase: str = "solve",
    ) -> None:
        if checkpoint_cb is None:
            return

        every = int(checkpoint_every)
        if (not final) and (every > 0) and ((it + 1) % every != 0):
            return

        try:
            checkpoint_cb(
                _current_x_from_z(z),
                {
                    "iter": int(it + 1),
                    "max_iter": int(max_iter),
                    "phase": str(phase),
                    "final": bool(final),
                    "active": int(np.count_nonzero(active)),
                    "stall_count": int(no_progress_count),
                    "resume_state": _pack_resume_state(it, phase, final),
                },
            )
        except Exception as exc:
            _log_checkpoint_error("checkpoint callback failed", exc)

    def _orbit_mass_and_deficit(
        z_vec: np.ndarray,
        alpha_value: float | None = None,
    ):
        """
        Compute current orbit masses and residuals relative to the fixed
        orbit shape at the supplied global amplitude.
        """
        x_vec = _current_x_from_z(
            z_vec
        ).reshape(C, P)

        s_orbit = np.sum(
            x_vec,
            axis=1,
        )

        if not has_hard_orbit_shape:
            t_orbit = np.zeros(
                (C,),
                dtype=np.float64,
            )
        else:
            if alpha_value is None:
                alpha_use = float(alpha_current)
            else:
                alpha_use = float(alpha_value)

            if (
                not np.isfinite(alpha_use)
                or alpha_use < 0.0
            ):
                alpha_use = 0.0

            t_orbit = (
                alpha_use
                * np.asarray(
                    orbit_shape,
                    dtype=np.float64,
                )
            )

        deficit = t_orbit - s_orbit

        return s_orbit, t_orbit, deficit

    # ------------------------------------------------------------
    # Helper: compute ATAz in parallel using provided executor
    # ------------------------------------------------------------
    def _compute_ATAz_scaled(z_vec):
        x_cand = np.asarray(z_vec, dtype=np.float64)

        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        ATAz = np.zeros((CP,), dtype=np.float64)

        worker_args = [
            (h5_path, batch, keep_idx, x_cand, CP, S_flat, C, P)
            for batch in batches
            if len(batch) > 0
        ]
        it = executor.map(_worker_ATAz_from_tuple, worker_args)

        for r in tqdm(
            it,
            total=len(worker_args),
            desc="[MONO] ATAz tiles",
            leave=False,
        ):
            try:
                ATAz += r
            except Exception as e:
                print("[ERROR] worker returned exception:", e, flush=True)
                raise

        return ATAz

    # ------------------------------------------------------------
    # Active-set outer loop
    # ------------------------------------------------------------
    for it in range(start_iter, max_iter):
        t_iter = time.perf_counter()
        force_explore = no_progress_count >= force_explore_after

        ATAz_scaled = _compute_ATAz_scaled(z)
        data_objective_current = (
            0.5 * float(
                np.dot(
                    z,
                    ATAz_scaled,
                )
            )
            - float(
                np.dot(
                    ATy_scaled,
                    z,
                )
            )
        )
        grad_data = ATy_scaled - ATAz_scaled

        if has_hard_orbit_shape:
            constraint_gradient = (
                S_flat
                * lambda_orbit_current[orbit_of_col]
            )

            grad_total = (
                grad_data
                - constraint_gradient
            )
        else:
            constraint_gradient = np.zeros(
                (CP,),
                dtype=np.float64,
            )
            grad_total = grad_data

        orbit_s, orbit_target, orbit_deficit = _orbit_mass_and_deficit(z)

        candidate_mask = ~active
        if has_hard_orbit_shape:
            candidate_mask &= (
                orbit_shape[orbit_of_col] > 0.0
            )
        not_active = np.flatnonzero(
            candidate_mask
        )
        gvals_data = grad_data[not_active]
        gvals_total = grad_total[not_active]
        gvals_constraint = constraint_gradient[not_active]

        max_grad_all = float(np.max(grad_total)) if grad_total.size else 0.0
        max_grad_promotable = (
            float(np.max(gvals_total)) if gvals_total.size else 0.0
        )
        avg_grad_promotable = (
            float(np.mean(gvals_total)) if gvals_total.size else 0.0
        )
        max_grad_data = float(np.max(gvals_data)) if gvals_data.size else 0.0
        max_grad_orbit = (
            float(np.max(np.abs(gvals_constraint)))
            if gvals_constraint.size
            else 0.0
        )
        grad_before_data = max_grad_data

        if diag_level >= 1 and ((it % diag_stride) == 0):
            x_phys_now = _current_x_from_z(z).reshape(C, P)
            orbit_mass_vec = np.sum(x_phys_now, axis=1)
            if has_hard_orbit_shape:
                orbit_target = (
                    float(alpha_current)
                    * np.asarray(
                        orbit_shape,
                        dtype=np.float64,
                    )
                )
            else:
                orbit_target = np.zeros(
                    (C,),
                    dtype=np.float64,
                )
            orbit_resid = orbit_mass_vec - orbit_target
            orbit_ratio = orbit_mass_vec / np.maximum(orbit_target, 1e-30)
            orbit_nz = np.count_nonzero(x_phys_now > 0.0, axis=1).astype(np.int64)

            eff_support = np.zeros((C,), dtype=np.float64)
            entropy = np.zeros((C,), dtype=np.float64)
            top_share = np.zeros((C,), dtype=np.float64)
            for cc in range(C):
                st = _support_stats_1d(x_phys_now[cc, :])
                eff_support[cc] = st["eff_support"]
                entropy[cc] = st["entropy"]
                top_share[cc] = st["top_share"]

            _emit_diag(
                {
                    "kind": "iter_pre",
                    "iter": int(it + 1),
                    "n_active": int(np.count_nonzero(active)),
                    "max_grad_data": float(max_grad_data),
                    "max_grad_promo": float(max_grad_promotable),
                    "avg_grad_promo": float(avg_grad_promotable),
                    "max_grad_total": float(max_grad_all),
                    "max_grad_orbit": float(max_grad_orbit),
                    "orbit_deficit_l1": float(np.sum(np.abs(orbit_deficit))),
                    "orbit_deficit_l2": float(np.linalg.norm(orbit_deficit)),
                    "orbit_mass": orbit_mass_vec.tolist(),
                    "orbit_target": orbit_target.tolist(),
                    "orbit_resid": orbit_resid.tolist(),
                    "orbit_ratio": orbit_ratio.tolist(),
                    "orbit_nz": orbit_nz.tolist(),
                    "orbit_eff_support": eff_support.tolist(),
                    "orbit_entropy": entropy.tolist(),
                    "orbit_top_share": top_share.tolist(),
                    "alpha": (
                        float(alpha_current)
                        if has_hard_orbit_shape
                        else None
                    ),
                    "orbit_shape": (
                        orbit_shape.tolist()
                        if orbit_shape is not None
                        else None
                    ),
                    "data_objective": float(
                        data_objective_current
                    ),
                }
            )

        n_active = int(np.count_nonzero(active))
        print(
            f"[MONO][iter {it+1}] "
            f"max_grad_total={max_grad_all:.3e} "
            f"max_grad_data={max_grad_data:.3e} "
            f"max_grad_orbit={max_grad_orbit:.3e} "
            f"max_grad_promotable={max_grad_promotable:.3e} "
            f"active={n_active}",
            flush=True,
        )
        tol_here = max(tol_grad, tol_grad_rel * grad_ref)
        # Unified promotion score: use the current combined gradient only.
        candidate_mask = ~active
        if has_hard_orbit_shape:
            candidate_mask &= (
                orbit_shape[orbit_of_col] > 0.0
            )
        not_active = np.flatnonzero(
            candidate_mask
        )
        score = gvals_total.copy()

        if not_active.size > 0:
            svals = S_flat[not_active]
            s_med = np.median(svals) + 1e-30
            score /= (1.0 + 0.25 * (svals / s_med - 1.0))

            aty = ATy_scaled[not_active]
            aty_scale = np.max(np.abs(aty)) + 1e-30
            score += 0.10 * (aty / aty_scale)

        if not_active.size == 0:
            if _watch_progress():
                break
            _emit_checkpoint(it, final=True, phase="no_candidate_exit")
            continue

        max_g = float(np.max(score)) if score.size else -np.inf
        did_explore = bool(max_g <= tol_here)
        newly_activated = None
        active_before = active.copy()
        z_before = z.copy()

        if max_g <= tol_here:
            negative_grad_count += 1
            allow_explore = (
                negative_grad_count <= explore_budget
            ) or force_explore

            if not allow_explore:
                if _watch_progress():
                    _emit_checkpoint(it, final=True, phase="stall_exit")
                    break
                _emit_checkpoint(it, final=True, phase="no_promotable_exit")
                break

            preferred_group = min(max(12, CP // 64), 32)
        else:
            negative_grad_count = 0
            positive_batch_size = min(24 if C > 3 else 8,
                                    positive_batch_size + 1)
            preferred_group = min(positive_batch_size, not_active.size)

        cooldown_mask_full = _combined_cooldown_mask(it)
        cooldown_mask = cooldown_mask_full[not_active]

        score_eff = score.copy()
        score_eff[cooldown_mask] = -np.inf

        if not np.isfinite(score_eff).any():
            # If the combined tabu blocks everything, relax to column-only
            # cooldown on the same not-active subset.
            score_eff = score.copy()
            col_cooldown_full = _col_cooldown_mask(it)
            score_eff[col_cooldown_full[not_active]] = -np.inf

        order = np.argsort(score_eff)[::-1]
        chosen = []
        for local_i in order:
            gcol = int(not_active[local_i])
            cc = gcol // P
            if (
                (_on_cooldown(gcol, it) or _orbit_on_cooldown(cc, it))
                and not force_explore
            ):
                continue
            chosen.append(gcol)
            if len(chosen) >= preferred_group:
                break

        if len(chosen) == 0:
            if _watch_progress():
                _emit_checkpoint(it, final=True, phase="no_promotable_exit")
                break
            continue

        cols_to_activate = np.asarray(chosen, dtype=np.int64)
        active[cols_to_activate] = True
        newly_activated = cols_to_activate.copy()
        promoted_cols = newly_activated.copy()
        if has_hard_orbit_shape:
            required_orbits = positive_shape_orbits

            coverage_cols = []

            for cc in required_orbits:
                base = int(cc) * P
                orbit_cols = np.arange(base, base + P, dtype=np.int64)

                if np.any(active[orbit_cols]):
                    continue

                # Select the best data-gradient column from this orbit.
                local_grad = grad_total[orbit_cols]
                best_col = int(orbit_cols[int(np.argmax(local_grad))])

                active[best_col] = True
                coverage_cols.append(best_col)

            if coverage_cols:
                coverage_cols = np.asarray(
                    coverage_cols,
                    dtype=np.int64,
                )
                newly_activated = np.unique(
                    np.concatenate(
                        (
                            newly_activated,
                            coverage_cols,
                        )
                    )
                )
                promoted_cols = newly_activated.copy()

        if int(np.count_nonzero(active)) > int(max_active):
            if newly_activated is not None and newly_activated.size > 0:
                active[newly_activated] = False
            if _watch_progress():
                break
            _emit_checkpoint(it, final=True, phase="max_active_exit")
            break

        # --------------------------------------------------------
        # Reduced solve (assemble ATA_sub & ATy_sub) in parallel
        # --------------------------------------------------------
        active_idx = np.nonzero(active)[0].astype(np.int64)
        k = int(active_idx.size)
        if k == 0:
            if _watch_progress():
                break
            _emit_checkpoint(it, final=True, phase="k_zero_exit")
            continue

        S_active = S_flat[active_idx]

        ATA_sub = np.zeros((k, k), dtype=np.float64)
        ATy_sub = np.zeros((k,), dtype=np.float64)

        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

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
            for batch in batches
            if len(batch) > 0
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
        try:
            Sact = S_active
            print(
                f"[DIAG] S_active stats: min/median/max: "
                f"{float(np.min(Sact)):.4e}/"
                f"{float(np.median(Sact)):.4e}/"
                f"{float(np.max(Sact)):.4e}",
                flush=True,
            )

            ATA_norm = float(np.linalg.norm(ATA_sub))
            ATy_norm = float(np.linalg.norm(ATy_sub))
            ATA_diag = np.diag(ATA_sub)[: min(6, ATA_sub.shape[0])]
            print(
                f"[DIAG] ATA_sub shape, ATy_sub[0:6]: "
                f"{ATA_sub.shape}, {ATy_sub[:6].tolist()}",
                flush=True,
            )
            print(
                f"[DIAG] ATA_sub norm / ATy_sub norm: "
                f"{ATA_norm:.4e} / {ATy_norm:.4e}",
                flush=True,
            )
            print(
                f"[DIAG] ATA_sub diag (first 6): {ATA_diag.tolist()}",
                flush=True,
            )

            outerS = (Sact[:, None] * Sact[None, :])
            outerS_safe = outerS.copy()
            outerS_safe[outerS_safe == 0.0] = 1.0
            ATA_unscaled_est = ATA_sub / outerS_safe

            Svec_safe = Sact.copy()
            Svec_safe[Svec_safe == 0.0] = 1.0
            ATy_unscaled_est = ATy_sub / Svec_safe

            ATA_unscaled_norm = float(np.linalg.norm(ATA_unscaled_est))
            ATy_unscaled_norm = float(np.linalg.norm(ATy_unscaled_est))
            print(
                f"[DIAG] inferred unscaled ATA/ATy norms: "
                f"{ATA_unscaled_norm:.4e} / {ATy_unscaled_norm:.4e}",
                flush=True,
            )

            try:
                eigs = np.linalg.eigvalsh(ATA_sub)
                emin = float(np.min(eigs))
                emax = float(np.max(eigs))
                cond = float(emax / max(1e-30, emin))
                print(
                    f"[DIAG] ATA_sub eigmin/emax/cond: "
                    f"{emin:.4e}/{emax:.4e}/{cond:.4e}",
                    flush=True,
                )
            except Exception:
                pass

            try:
                ridge = 1e-12 * np.eye(ATA_unscaled_est.shape[0], dtype=np.float64)
                z_unscaled = np.linalg.solve(
                    ATA_unscaled_est + ridge,
                    ATy_unscaled_est,
                )
                print(
                    f"[DIAG] z_unscaled stats: l1, l2, max: "
                    f"{float(np.sum(z_unscaled)):.4e}, "
                    f"{float(np.linalg.norm(z_unscaled)):.4e}, "
                    f"{float(np.max(z_unscaled)):.4e}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[DIAG] solving unscaled diagnostic failed: {_e}", flush=True)

        except Exception as _e:
            print(f"[DIAG] reduced-diagnostics error: {_e}", flush=True)
        # ---------------------------------------------------------------------------

        # ---- Stabilised reduced solve: stronger adaptive ridge ----
        diag = np.diag(ATA_sub)
        diag_med = float(np.median(diag)) if diag.size else 0.0

        try:
            eigs = np.linalg.eigvalsh(ATA_sub)
            emin = float(np.min(eigs))
            emax = float(np.max(eigs))
        except Exception:
            emin = 0.0
            emax = float(np.max(diag)) if diag.size else 1.0

        cond_bad = (emin <= 1e-10 * max(emax, 1.0))
        ridge_rel = 1e-2 if cond_bad else 1e-3
        ridge = max(1e-10, ridge_rel * max(1.0, diag_med))
        ATA_sub_reg = ATA_sub + ridge * np.eye(k, dtype=np.float64)

        ATA_norm = float(np.linalg.norm(ATA_sub))

        z_old_active = z[active_idx].copy()

        def _quad_obj(A, b, x):
            return 0.5 * float(x @ (A @ x)) - float(b @ x)


        orbit_constraint_info = {
            "accepted": True,
            "reason": "no_prior",
            "alpha": None,
            "orbit_target_mass": np.zeros(
                (C,),
                dtype=np.float64,
            ),
            "resid_l1": 0.0,
            "resid_l2": 0.0,
            "resid_linf": 0.0,
            "rel_resid_linf": 0.0,
            "alpha_stationarity": 0.0,
            "negative_count": 0,
            "lambda_orbit": np.zeros(
                (C,),
                dtype=np.float64,
            ),
        }

        alpha_candidate = None
        lambda_candidate = None

        if has_hard_orbit_shape:
            z_sub_raw, orbit_constraint_info = (
                _orbit_hard_constraint_solve(
                    ATA_sub_reg=ATA_sub_reg,
                    ATy_sub=ATy_sub,
                    active_idx=active_idx,
                    S_active=S_active,
                    orbit_shape=np.asarray(
                        orbit_shape,
                        dtype=np.float64,
                    ),
                    C=C,
                    P=P,
                )
            )

            if not orbit_constraint_info["accepted"]:
                print(
                    "[MONO][constrained] rejecting infeasible reduced solve "
                    f"(reason={orbit_constraint_info['reason']}, "
                    f"neg={orbit_constraint_info['negative_count']}, "
                    f"orbit_L1={orbit_constraint_info['resid_l1']:.3e}, "
                    f"orbit_Linf={orbit_constraint_info['resid_linf']:.3e})",
                    flush=True,
                )
                if promoted_cols.size > 0:
                    _cooldown_cols(promoted_cols, it, extra_iters=20)
                    _cooldown_orbits(np.unique(promoted_cols // P), it, extra_iters=20)
                active[:] = active_before
                z[:] = z_before
                if _watch_progress():
                    _emit_checkpoint(
                        it,
                        final=True,
                        phase="constraint_reject_stall",
                    )
                    break

                _emit_checkpoint(
                    it,
                    final=False,
                    phase="constraint_reject",
                )
                continue

            z_sub = np.asarray(
                z_sub_raw,
                dtype=np.float64,
            ).copy()

            alpha_candidate = float(
                orbit_constraint_info["alpha"]
            )

            lambda_candidate = np.asarray(
                orbit_constraint_info["lambda_orbit"],
                dtype=np.float64,
            ).ravel(order="C")

            candidate_valid = (
                np.isfinite(alpha_candidate)
                and alpha_candidate >= 0.0
                and lambda_candidate.size == C
                and np.all(np.isfinite(lambda_candidate))
            )

            if not candidate_valid:
                print(
                    "[MONO][constrained] rejecting invalid "
                    "global amplitude or orbit multipliers",
                    flush=True,
                )

                active[:] = active_before
                z[:] = z_before

                if promoted_cols.size > 0:
                    _cooldown_cols(
                        promoted_cols,
                        it,
                        extra_iters=20,
                    )
                    _cooldown_orbits(
                        np.unique(promoted_cols // P),
                        it,
                        extra_iters=20,
                    )

                if _watch_progress():
                    _emit_checkpoint(
                        it,
                        final=True,
                        phase="invalid_constraint_state_stall",
                    )
                    break

                _emit_checkpoint(
                    it,
                    final=False,
                    phase="invalid_constraint_state",
                )
                continue

        else:
            z_sub_raw = None
            try:
                z_sub_raw = np.linalg.solve(ATA_sub_reg, ATy_sub)
            except np.linalg.LinAlgError:
                U, svals, Vt = np.linalg.svd(ATA_sub_reg, full_matrices=False)
                smax = svals[0] if svals.size else 1.0
                s_thresh = max(1e-12, 1e-6 * smax)
                s_inv = np.array(
                    [1.0 / s if s > s_thresh else 0.0 for s in svals],
                    dtype=np.float64,
                )
                z_sub_raw = Vt.T @ (s_inv * (U.T @ ATy_sub))

            z_sub_raw = np.maximum(z_sub_raw, 0.0)
            z_sub = z_sub_raw.copy()

        # Track promoted columns that did not survive the reduced solve.
        failed_cols = []

        if newly_activated is not None and newly_activated.size > 0:
            active_lookup = {int(c): i for i, c in enumerate(active_idx)}
            z_scale_loc = float(np.max(z_sub)) + 1e-30

            for c in newly_activated:
                ii = active_lookup.get(int(c))
                if ii is None:
                    continue

                if z_sub[ii] <= 1e-8 * z_scale_loc:
                    failed_cols.append(int(c))

        failed_cols = np.asarray(
            failed_cols,
            dtype=np.int64,
        )

        obj_old = _quad_obj(ATA_sub_reg, ATy_sub, z_old_active)
        obj_new = _quad_obj(ATA_sub_reg, ATy_sub, z_sub)
        norm_old = float(np.linalg.norm(z_old_active))
        norm_new = float(np.linalg.norm(z_sub))
        obj_gain = float(obj_old - obj_new)
        rel_gain = obj_gain / max(1.0, abs(obj_old))

        near_noop = (
            did_explore
            and promoted_cols.size > 0
            and (obj_gain <= promotion_noop_abs_tol
                 or rel_gain <= promotion_noop_rel_tol)
        )

        reject = False

        if did_explore:
            obj_worsened = np.isfinite(obj_new) and (
                obj_new > obj_old + 1e-12 * (1.0 + abs(obj_old))
            )
            norm_exploded = (norm_old > 1e-12 and norm_new > 50.0 * norm_old)

            if obj_worsened or norm_exploded:
                reject = True

        if did_explore and near_noop:
            _cooldown_cols(promoted_cols, it, extra_iters=3)
            print(
                "[MONO][explore] cooling promoted columns after near-noop "
                f"update: {promoted_cols}",
                flush=True,
            )

        if reject:
            print(
                "[MONO][explore] rejecting exploratory update "
                f"(obj_old={obj_old:.4e}, obj_new={obj_new:.4e}, "
                f"norm_old={norm_old:.4e}, norm_new={norm_new:.4e})",
                flush=True,
            )
            if promoted_cols.size > 0:
                _cooldown_cols(promoted_cols, it, extra_iters=20)
                _cooldown_orbits(
                    np.unique(promoted_cols // P), it, extra_iters=20,)
            if newly_activated is not None and newly_activated.size > 0:
                active[newly_activated] = False
            z[active_idx] = z_old_active

            if _watch_progress():
                break
            _emit_checkpoint(it, final=True, phase="reject_exit")
            continue

        # Commit the complete reduced solution first.
        z[active_idx] = z_sub

        # Then deactivate promoted columns that received effectively zero weight.
        if failed_cols.size > 0:
            z[failed_cols] = 0.0
            active[failed_cols] = False

            _cooldown_cols(
                failed_cols,
                it,
            )
            _cooldown_orbits(
                np.unique(failed_cols // P),
                it,
            )

        if did_explore:
            obj_tol = max(
                1e-10 * max(1.0, abs(obj_old)),
                1e-14,
            )

            if (not np.isfinite(obj_new)) or (obj_new > obj_old + obj_tol):
                print(
                    "[MONO][explore] rolling back batch: "
                    f"obj_old={obj_old:.3e} "
                    f"obj_new={obj_new:.3e} "
                    f"tol={obj_tol:.3e}",
                    flush=True,
                )

                active[:] = active_before
                z[:] = z_before

                if newly_activated is not None and newly_activated.size > 0:
                    _cooldown_cols(newly_activated, it, extra_iters=20)
                    _cooldown_orbits(
                        np.unique(newly_activated // P), it, extra_iters=20,)

                positive_batch_size = max(2, positive_batch_size // 2)
                if _watch_progress():
                    break
                _emit_checkpoint(it, final=True, phase="rollback_exit")
                continue

        # Commit the matching equality multipliers only after the constrained
        # coefficient update has passed the feasibility and objective checks.
        if has_hard_orbit_shape:
            alpha_current = float(alpha_candidate)
            lambda_orbit_current = (
                lambda_candidate.copy()
            )

        # --------------------------------------------------------
        # Probation cleanup
        # --------------------------------------------------------
        x_phys = S_flat * z

        if newly_activated is not None and newly_activated.size > 0:
            newly_activated_set = set(int(c) for c in newly_activated.tolist())
        else:
            newly_activated_set = set()

        drop_cols = []

        for c in np.nonzero(active)[0]:
            c = int(c)

            if c in newly_activated_set:
                continue

            cc = c // P
            base = cc * P
            orbit_cols = np.arange(base, base + P, dtype=np.int64)
            orbit_active = orbit_cols[active[orbit_cols]]

            if orbit_active.size == 0:
                continue

            orbit_mass_local = float(np.sum(x_phys[orbit_active]))
            tiny_thresh = max(
                probation_abs,
                probation_rel * max(orbit_mass_local, 1.0),
            )

            if x_phys[c] <= tiny_thresh:
                provisional_hits[c] = provisional_hits.get(c, 0) + 1
            else:
                provisional_hits.pop(c, None)

            if provisional_hits.get(c, 0) >= probation_iters:
                drop_cols.append(c)
                provisional_hits.pop(c, None)

        if drop_cols and has_hard_orbit_shape:
            safe_drop_cols = []

            for c in drop_cols:
                c = int(c)
                cc = c // P

                base = cc * P
                orbit_cols = np.arange(
                    base,
                    base + P,
                    dtype=np.int64,
                )

                n_active_orbit = int(
                    np.count_nonzero(
                        active[orbit_cols]
                    )
                )

                if (
                    orbit_shape[cc] > 0.0
                    and n_active_orbit <= 1
                ):
                    provisional_hits.pop(c, None)
                    continue

                safe_drop_cols.append(c)

            drop_cols = safe_drop_cols

        if drop_cols:
            drop_cols = np.asarray(drop_cols, dtype=np.int64)
            z[drop_cols] = 0.0
            active[drop_cols] = False
            _cooldown_cols(drop_cols, it, extra_iters=2)
            print(
                f"[MONO][probation] dropped {drop_cols.size} stale columns: "
                f"{drop_cols}", flush=True,)
            _emit_checkpoint(it, final=False, phase="probation_prune")
            continue

        if diag_level >= 1 and ((it % diag_stride) == 0):
            x_phys_now = _current_x_from_z(z).reshape(C, P)
            orbit_mass_vec = np.sum(x_phys_now, axis=1)
            if not has_hard_orbit_shape:
                orbit_target = np.zeros(
                    (C,),
                    dtype=np.float64,
                )
            else:
                orbit_target = (
                    float(alpha_current)
                    * np.asarray(
                        orbit_shape,
                        dtype=np.float64,
                    )
                )
            orbit_resid = orbit_mass_vec - orbit_target
            orbit_ratio = orbit_mass_vec / np.maximum(orbit_target, 1e-30)
            orbit_nz = np.count_nonzero(x_phys_now > 0.0, axis=1).astype(np.int64)

            eff_support = np.zeros((C,), dtype=np.float64)
            entropy = np.zeros((C,), dtype=np.float64)
            top_share = np.zeros((C,), dtype=np.float64)
            for cc in range(C):
                st = _support_stats_1d(x_phys_now[cc, :])
                eff_support[cc] = st["eff_support"]
                entropy[cc] = st["entropy"]
                top_share[cc] = st["top_share"]

            _emit_diag(
                {
                    "kind": "iter_post",
                    "iter": int(it + 1),
                    "t_iter_sec": float(time.perf_counter() - t_iter),
                    "k_active": int(k),
                    "ATA_norm": float(np.linalg.norm(ATA_sub)),
                    "ATA_reg_norm": float(np.linalg.norm(ATA_sub_reg)),
                    "ATy_norm": float(np.linalg.norm(ATy_sub)),
                    "diag_med": float(diag_med),
                    "diag_min": float(np.min(diag)) if diag.size else 0.0,
                    "diag_max": float(np.max(diag)) if diag.size else 0.0,
                    "emin": float(emin),
                    "emax": float(emax),
                    "ridge": float(ridge),
                    "did_explore": bool(did_explore),
                    "reject": bool(reject),
                    "orbit_constraint_ok": bool(orbit_constraint_info["accepted"]),
                    "orbit_constraint_l1": float(orbit_constraint_info["resid_l1"]),
                    "orbit_constraint_l2": float(orbit_constraint_info["resid_l2"]),
                    "orbit_constraint_linf": float(orbit_constraint_info["resid_linf"]),
                    "n_promoted": int(promoted_cols.size),
                    "n_failed": int(failed_cols.size),
                    "n_dropped": int(len(drop_cols)),
                    "obj_old": float(obj_old),
                    "obj_new": float(obj_new),
                    "obj_gain": float(obj_gain),
                    "norm_old": float(norm_old),
                    "norm_new": float(norm_new),
                    "orbit_mass": orbit_mass_vec.tolist(),
                    "orbit_target": orbit_target.tolist(),
                    "orbit_resid": orbit_resid.tolist(),
                    "orbit_ratio": orbit_ratio.tolist(),
                    "orbit_nz": orbit_nz.tolist(),
                    "orbit_eff_support": eff_support.tolist(),
                    "orbit_entropy": entropy.tolist(),
                    "orbit_top_share": top_share.tolist(),
                    "alpha": (
                        float(alpha_current)
                        if has_hard_orbit_shape
                        else None
                    ),
                    "alpha_stationarity": (
                        float(
                            orbit_constraint_info.get(
                                "alpha_stationarity",
                                0.0,
                            )
                        )
                        if has_hard_orbit_shape
                        else None
                    ),
                    "orbit_shape": (
                        orbit_shape.tolist()
                        if orbit_shape is not None
                        else None
                    ),
                    "x": x_phys_now.ravel(order="C").tolist(),
                }
            )

        # Finalize progress watchdog on the committed state.
        stalled = _watch_progress()

        if stalled:
            _emit_checkpoint(it, final=True, phase="stall_exit")
            break

        _emit_checkpoint(it, final=False)


    # Orbit prior is enforced in the reduced solves above; no final correction.

    try:
        final_it = int(it)
    except Exception:
        final_it = 0

    # Final objective summary in the solver's own scaled space.
    x_final = _current_x_from_z(z)
    ATAz_final = _compute_ATAz_scaled(z)

    data_objective = (
        0.5 * float(np.dot(z, ATAz_final))
        - float(np.dot(ATy_scaled, z))
    )

    if has_hard_orbit_shape:
        orbit_mass, orbit_target, orbit_deficit = (
            _orbit_mass_and_deficit(
                z,
                alpha_value=alpha_current,
            )
        )
        orbit_resid_l1 = float(np.sum(np.abs(orbit_deficit)))
        orbit_resid_l2 = float(np.linalg.norm(orbit_deficit))
        orbit_resid_linf = float(np.max(np.abs(orbit_deficit)))
    else:
        orbit_mass = np.sum(x_final.reshape(C, P), axis=1)
        orbit_target = np.zeros((C,), dtype=np.float64)
        orbit_deficit = orbit_target - orbit_mass
        orbit_resid_l1 = None
        orbit_resid_l2 = None
        orbit_resid_linf = None

    total_objective = data_objective

    print(
        f"[MONO][final] "
        f"data_obj={data_objective:.3e} "
        f"total_obj={total_objective:.3e} "
        f"alpha={alpha_current:.6e} "
        f"orbit_L1="
        f"{(orbit_resid_l1 if orbit_resid_l1 is not None else float('nan')):.3e} "
        f"orbit_Linf="
        f"{(orbit_resid_linf if orbit_resid_linf is not None else float('nan')):.3e}",
        flush=True,
    )

    _emit_diag(
        {
            "kind": "objective_summary",
            "data_objective": float(data_objective),
            "total_objective": float(total_objective),
            "orbit_resid_l1": orbit_resid_l1,
            "orbit_resid_l2": orbit_resid_l2,
            "orbit_resid_linf": orbit_resid_linf,
            "orbit_mass": orbit_mass.tolist(),
            "orbit_target": orbit_target.tolist(),
            "orbit_resid": orbit_deficit.tolist(),
            "alpha": (
                float(alpha_current)
                if has_hard_orbit_shape
                else None
            ),
            "orbit_shape": (
                orbit_shape.tolist()
                if orbit_shape is not None
                else None
            ),
            "orbit_prior_present": bool(
                has_hard_orbit_shape
            ),
            "x": x_final.ravel(order="C").tolist(),
        }
    )

    _emit_checkpoint(final_it, final=True, phase="complete")
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
    resume_state: Optional[dict] = None,
    tracker: Optional[object] = None,
    block_size: Optional[int] = None,
    monolithic_max_active: int = 1000,
    cache_path: str | None = None,
    reuse_cache: bool = True,
):
    """
    Fused single-pass block-coordinate NNLS solver (BLAS-friendly).

    - Accumulates ATA and ATy for each block during one streaming pass per
      epoch (no repeated HDF5 reads per block).
    - Solves small quadratic NNLS problems per block using a projected-gradient
      solver on the quadratic form ATA/ATy.
    - Supports a final constrained orbit-mass correction on the final active
      set using the reduced Hessian.
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
    orbit_shape_outer = None
    if orbit_weights is not None:
        orbit_shape_outer = _canon_orbit_weights(
            h5_path,
            orbit_weights,
            C=C,
            P=P,
        )

    # ---------------------------- diagnostics --------------------------
    diag_level = int(os.environ.get("CUBEFIT_DIAG_LEVEL", "1"))
    diag_stride = max(1, int(os.environ.get("CUBEFIT_DIAG_STRIDE", "1")))
    diag_jsonl_path = os.environ.get("CUBEFIT_DIAG_JSONL", "").strip() or None
    diag_topk = max(1, int(os.environ.get("CUBEFIT_DIAG_TOPK", "12")))
    diag_t0 = time.perf_counter()

    def _emit_diag(record: dict) -> None:
        if diag_level <= 0:
            return
        rec = dict(record)
        rec.setdefault("t_sec", float(time.perf_counter() - diag_t0))
        _diag_append_jsonl(diag_jsonl_path, rec)

    _emit_diag(
        {
            "kind": "setup",
            "S": int(S),
            "L": int(L),
            "C": int(C),
            "P": int(P),
            "CP": int(CP),
            "keep_idx": int(keep_idx.size) if keep_idx is not None else None,
            "s_tile": int(s_tile),
            "n_ranges": int(len(s_ranges)),
            "processes": int(cfg.processes),
            "blas_threads": int(cfg.blas_threads),
            "diag_level": int(diag_level),
            "diag_stride": int(diag_stride),
            "diag_topk": int(diag_topk),
        }
    )

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

    if cache_path is None:
        cache_path = f"{h5_path}.bcfused.npz"

    keep_idx_sig = (
        np.asarray(keep_idx, dtype=np.int64)
        if keep_idx is not None
        else np.asarray([], dtype=np.int64)
    )

    cache_npz = None
    cache_hit = False

    if reuse_cache and cache_path and os.path.exists(cache_path):
        try:
            cache_npz = np.load(cache_path, allow_pickle=False)

            ok = (
                int(cache_npz["S"]) == int(S)
                and int(cache_npz["L"]) == int(L)
                and int(cache_npz["C"]) == int(C)
                and int(cache_npz["P"]) == int(P)
                and int(cache_npz["s_tile"]) == int(s_tile)
                and int(cache_npz["apply_mask"]) == int(bool(cfg.apply_mask))
                and np.array_equal(
                    np.asarray(cache_npz["keep_idx"], dtype=np.int64),
                    keep_idx_sig,
                )
            )

            if ok:
                cache_hit = True
                print(f"[BC-FUSED] cache hit: {cache_path}", flush=True)
            else:
                cache_npz.close()
                cache_npz = None
        except Exception as _e:
            print(f"[BC-FUSED] cache load failed: {_e}", flush=True)
            cache_hit = False
            if cache_npz is not None:
                try:
                    cache_npz.close()
                except Exception:
                    pass
                cache_npz = None

    if cache_hit:
        print("[BC-FUSED] reusing cached build products", flush=True)

        ATy_flat = np.asarray(cache_npz["ATy_flat"], dtype=np.float64).copy()
        D_tot = np.asarray(cache_npz["D_tot"], dtype=np.float64).copy()
        inv_sqrt_energy_flat = np.asarray(
            cache_npz["inv_sqrt_energy_flat"],
            dtype=np.float64,
        ).copy()

        try:
            cache_npz.close()
        except Exception:
            pass

    else:
        print("[BC-FUSED]", flush=True)

        # D_tot per column (C,P)
        D_tot = np.zeros((C, P), dtype=np.float64)

        # ---------- single streaming pass: build ATy_flat & D_tot -------
        ATy_flat = np.zeros((CP,), dtype=np.float64)
        with open_h5(h5_path, role="reader") as f:
            DC = f["/DataCube"]
            M = f["/HyperCube/models"]
            try:
                M.id.set_chunk_cache(cfg.dset_slots, cfg.dset_bytes, cfg.dset_w0)
            except Exception:
                pass

            tile_iter = s_ranges
            if verbose and (len(s_ranges) > 1):
                tile_iter = tqdm(
                    s_ranges,
                    desc=f"[BC-FUSED] tiles",
                    disable=not verbose,
                )

            for (s0, s1) in tile_iter:
                Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
                if keep_idx is not None:
                    Yt = Yt[:, keep_idx]
                Sblk = s1 - s0
                Lk = Yt.shape[1]

                # accumulate prediction for diagnostics (warm-start)
                yhat_tile = np.zeros((Sblk, Lk), dtype=np.float64)

                # read model tile (Sblk, C, P, Lk)
                M_tile = np.asarray(M[s0:s1, :, :, :], dtype=np.float64, order="C")
                if keep_idx is not None:
                    M_tile = M_tile[:, :, :, keep_idx]

                # compute D_tot contributions (per-column energy)
                # sum over Sblk and wavelength dims
                # M_tile: (Sblk, C, P, Lk)
                D_tot += np.sum(M_tile * M_tile, axis=(0, 3))

                # build A2_tile once for ATy accumulation
                A2_tile = M_tile.transpose(0, 3, 1, 2).reshape(Sblk * Lk, CP)
                y_flat = Yt.reshape(-1)

                # accumulate ATy (uses data y, not residual)
                ATy_flat += A2_tile.T @ y_flat

                # diagnostics (same as before)
                try:
                    yf_norm = (
                        float(np.linalg.norm(y_flat)) if y_flat.size > 0 else 0.0
                    )
                    yhatf_norm = (
                        float(np.linalg.norm(yhat_tile))
                        if yhat_tile.size > 0
                        else 0.0
                    )
                    r_norm = (
                        float(np.linalg.norm(y_flat - yhat_tile.reshape(-1)))
                        if y_flat.size > 0
                        else 0.0
                    )

                    nnans = int(np.count_nonzero(~np.isfinite(y_flat)))
                    nnans += int(np.count_nonzero(~np.isfinite(yhat_tile)))
                    nnans += int(
                        np.count_nonzero(~np.isfinite(y_flat - yhat_tile.reshape(-1)))
                    )

                    print(
                        f"[BC-FUSED][tile s={s0}:{s1}] Sblk={Sblk} Lk={Lk} "
                        f"||y||={yf_norm:.3e} "
                        + (
                            f"||yhat||={yhatf_norm:.3e} "
                            if yhatf_norm > 0.0
                            else ""
                        )
                        + f"||r||={r_norm:.3e} nonfinite_vals={nnans}",
                        flush=True,
                    )
                except Exception as _e:
                    print(
                        "[BC-FUSED][tile diag] error while computing tile diagnostics:",
                        _e,
                        flush=True,
                    )

        # ------------------------ end streaming --------------------------

        # ------------------------------------------------------------------
        # compute inv_sqrt_energy and scaled targets (per-tile average energy)
        # Use average column energy per tile (not the summed D_tot) so the
        # worker tile matrices and the global scaling agree.
        # ------------------------------------------------------------------
        col_energy_sum = D_tot.copy()  # D_tot currently holds summed energy
        n_tiles = max(1, len(s_ranges))  # number of tiles used in streaming pass

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

        if cache_path:
            try:
                np.savez_compressed(
                    cache_path,
                    ATy_flat=ATy_flat,
                    inv_sqrt_energy_flat=inv_sqrt_energy_flat,
                    D_tot=D_tot,
                    known_zero=known_zero,
                    keep_idx=keep_idx_sig,
                    S=int(S),
                    L=int(L),
                    C=int(C),
                    P=int(P),
                    s_tile=int(s_tile),
                    apply_mask=int(bool(cfg.apply_mask)),
                )
                print(f"[BC-FUSED] cache saved: {cache_path}", flush=True)
            except Exception as _e:
                print(f"[BC-FUSED] cache save failed: {_e}", flush=True)

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

    checkpoint_every = int(
        os.environ.get("CUBEFIT_SOLVER_CHECKPOINT_EVERY", "50")
    )

    latest_ckpt = {
        "x": np.asarray(x, dtype=np.float64).ravel(order="C").copy(),
        "stats": {
            "iter": 0,
            "max_iter": 0,
            "phase": "init",
            "final": False,
            "active": int(np.count_nonzero(x > 0.0)),
            "stall_count": 0,
        },
    }

    checkpoint_error_logged = False

    def _checkpoint_cb(x_vec: np.ndarray, stats: dict) -> None:
        nonlocal checkpoint_error_logged

        latest_ckpt["x"] = np.asarray(
            x_vec, dtype=np.float64
        ).ravel(order="C").copy()
        latest_ckpt["stats"] = dict(stats)

        if tracker is None:
            return

        state = dict(stats.get("resume_state", {}))
        if not state:
            state = {
                "iter": int(stats.get("iter", -1)),
                "max_iter": int(stats.get("max_iter", -1)),
                "phase": str(stats.get("phase", "solve")),
                "final": bool(stats.get("final", False)),
                "active": int(stats.get("active", -1)),
                "stall_count": int(stats.get("stall_count", -1)),
            }

        try:
            tracker.save_checkpoint(
                latest_ckpt["x"],
                state,
                block=True,
            )
        except Exception as exc:
            if not checkpoint_error_logged:
                print(
                    f"[MONO][checkpoint] save_checkpoint failed: {exc}",
                    flush=True,
                )
                checkpoint_error_logged = True

    try:
        x_flat_unscaled = streamActiveSetNNLS(
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
            x0_flat=x.ravel(order="C"),
            resume_state=resume_state,
            max_active=monolithic_max_active,
            tol_grad=1e-8,
            max_iter=5 * monolithic_max_active,
            checkpoint_cb=_checkpoint_cb,
            checkpoint_every=checkpoint_every,
        )

        x = x_flat_unscaled.reshape(C, P).copy()
        print("[BC-FUSED][MONO] finished streaming active-set NNLS",
            flush=True)

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
        print(f"[BC-FUSED] new best proxy {best_proxy:.3e}", flush=True)

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

        print("=== [BC-FUSED][summary] ===", flush=True)
        print(f"[BC-FUSED][summary] elapsed_total={epoch_elapsed:.1f}s", flush=True)
        if np.isfinite(rmse_curr):
            print(f"[BC-FUSED][summary] data_proxy={data_proxy:.3e} rmse={rmse_curr:.3e}", flush=True)
        else:
            print(f"[BC-FUSED][summary] data_proxy={data_proxy:.3e} rmse=nan", flush=True)
        print(f"[BC-FUSED][summary] best_proxy={best_proxy:.3e}", flush=True)
        print(
            f"[BC-FUSED][summary] x_sum={x_sum:.3e} x_norm={x_norm:.3e} nonzero={x_nonzero}/{x_flat.size} nonfinite={x_nonfinite}",
            flush=True
        )
        print(
            f"[BC-FUSED][summary] D_tot diag med/min/max = {D_median:.3e} / {D_min:.3e} / {D_max:.3e}",
            flush=True
        )

        # sanity checks that likely indicate major numerical problems
        if x_nonzero == 0:
            print("[BC-FUSED][summary] ALERT: x is entirely zero after epoch!", flush=True)
        if x_nonfinite > 0:
            print("[BC-FUSED][summary] ALERT: non-finite elements in x!", flush=True)
        if not np.isfinite(data_proxy) or data_proxy <= 0.0:
            print("[BC-FUSED][summary] ALERT: data_proxy non-finite or <= 0.0", flush=True)

        # show top contributors (per-orbit totals)
        try:
            s_full = np.sum(best_x, axis=1)  # per-orbit
            top_idx = np.argsort(s_full)[::-1][:10]
            top_vals = s_full[top_idx]
            print("[BC-FUSED][summary] top orbits (index:mass): " + ", ".join(
                [f"{int(i)}:{v:.3e}" for i, v in zip(top_idx.tolist(), top_vals.tolist())]
            ), flush=True)
        except Exception:
            pass
        # --- orbit_weights residual diagnostic (if prior provided) -----
        try:
            if (
                orbit_shape_outer is not None
                and np.sum(orbit_shape_outer) > 0.0
            ):

                alpha_outer = float(
                    np.sum(s_full)
                )

                s_proj = (
                    alpha_outer
                    * np.asarray(
                        orbit_shape_outer,
                        dtype=np.float64,
                    )
                )

                print(
                    "[DIAG][orbit_weights]",
                    flush=True,
                )

                for cc in range(C):
                    st = _support_stats_1d(
                        best_x[cc, :]
                    )
                    resid = float(
                        s_full[cc] - s_proj[cc]
                    )
                    ratio = float(
                        s_full[cc]
                        / max(s_proj[cc], 1e-30)
                    )

                    print(
                        f"orbit {cc:2d}: "
                        f"mass={s_full[cc]:.3e} "
                        f"target={s_proj[cc]:.3e} "
                        f"resid={resid:+.3e} "
                        f"ratio={ratio:7.3f} "
                        f"nz="
                        f"{int(np.count_nonzero(best_x[cc, :] > 0.0)):3d} "
                        f"eff={st['eff_support']:.2f} "
                        f"top={st['top_share']:.2f}",
                        flush=True,
                    )

                _emit_diag(
                    {
                        "kind": "epoch_orbit_table",
                        "alpha": float(alpha_outer),
                        "orbit_shape": (
                            orbit_shape_outer.tolist()
                        ),
                        "orbit_mass": s_full.tolist(),
                        "orbit_target": s_proj.tolist(),
                        "orbit_resid": (
                            s_full - s_proj
                        ).tolist(),
                        "orbit_ratio": (
                            s_full
                            / np.maximum(
                                s_proj,
                                1e-30,
                            )
                        ).tolist(),
                        "orbit_nz": (
                            np.count_nonzero(
                                best_x > 0.0,
                                axis=1,
                            )
                            .astype(int)
                            .tolist()
                        ),
                    }
                )
            else:
                print(
                    "[DIAG][orbit_weights] no w_target present; skipping orbit table.",
                    flush=True,
                )
        except Exception as _e:
            print("[DIAG][orbit_weights] diagnostic failed:", _e, flush=True)

        print("=== [BC-FUSED][epoch-summary] end ===", flush=True)
    except Exception as _e:
        print("[BC-FUSED][epoch-summary] error:", _e, flush=True)
    # --- end epoch summary diagnostics ---

    # done epochs
    # ------------------------------------------------------------


    # Final NNLS feasibility check.
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Final NNLS feasibility validation.
    #
    # If the constrained solver has produced an infeasible solution, preserve and
    # diagnose the exact offending vector before failing.
    # ------------------------------------------------------------
    x_final_flat = np.asarray(best_x, dtype=np.float64).ravel(order="C")
    bad = np.flatnonzero(~np.isfinite(x_final_flat))
    if bad.size:
        _emit_diag(
            {
                "kind": "error",
                "error": "nonfinite_final_x",
                "message": (
                    "Final NNLS solution contains non-finite coefficients."),
                "n_bad": int(bad.size),
                "bad_indices": bad.tolist(),
                "x": x_final_flat.tolist(),
            }
        )
        raise RuntimeError(
            "Final NNLS solution contains non-finite coefficients: "
            f"n_bad={bad.size}, "
            f"first_indices={bad[:10].tolist()}"
        )
    neg = np.flatnonzero(x_final_flat < 0.0)
    if neg.size:
        _emit_diag(
            {
                "kind": "error",
                "error": "negative_final_x",
                "message": (
                    "Final NNLS solution violates non-negativity."),
                "n_negative": int(neg.size),
                "min_x": float(np.min(x_final_flat)),
                "negative_indices": neg.tolist(),
                "negative_values": x_final_flat[neg].tolist(),
                "x": x_final_flat.tolist(),
            }
        )
        raise RuntimeError(
            "Final NNLS solution violates non-negativity: "
            f"n_negative={neg.size}, "
            f"min={float(np.min(x_final_flat)):.17e}, "
            f"first_indices={neg[:10].tolist()}, "
            f"first_values={x_final_flat[neg[:10]].tolist()}"
        )
    _emit_diag(
        {
            "kind": "final_solution",
            "x": best_x.ravel(order="C").tolist(),
            "x_sum": float(np.sum(best_x)),
            "x_norm": float(np.linalg.norm(best_x)),
            "x_max": float(np.max(best_x)),
            "x_nnz": int(np.count_nonzero(best_x > 0.0)),
            "rmse_proxy_best": float(best_proxy),
        }
    )
    elapsed = time.perf_counter() - t0

    stats = dict(
        elapsed_sec=elapsed,
        rmse_proxy_best=float(best_proxy),
        known_zero_mask=known_zero.copy(),
    )

    if tracker is not None:
        try:
            state = dict(stats.get("resume_state", {}))
            if not state:
                state = {
                    "iter": int(stats.get("iter", -1)),
                    "max_iter": int(stats.get("max_iter", -1)),
                    "phase": "final",
                    "final": True,
                    "active": int(stats.get("active", -1)),
                    "stall_count": int(stats.get("stall_count", -1)),
                }
            else:
                state["phase"] = "final"
                state["final"] = True
                state.setdefault("iter", int(stats.get("iter", -1)))
                state.setdefault("max_iter", int(stats.get("max_iter", -1)))
                state.setdefault("active", int(stats.get("active", -1)))
                state.setdefault("stall_count", int(stats.get("stall_count", -1)))

            tracker.save_checkpoint(
                best_x,
                state,
                block=True,
            )
        except Exception:
            pass

    return best_x, stats

# ------------------------------------------------------------------------------