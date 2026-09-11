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
v1.5:   Expanded exit diagnostics to include the reason, category, and
            convergence status of the solver termination. 12 August 2026
v1.6:   Dramatically expanded diagnostics for noop columns in
            `streamActiveSetNNLS`;
        Added explicit constrained KKT convergence diagnostics and termination
            in `streamActiveSetNNLS`;
        Distinguished active-positive stationarity from inactive-column KKT
            violations in `streamActiveSetNNLS`;
        Added detailed promotion-attempt and promotion-outcome diagnostics in
            `streamActiveSetNNLS`;
        Tightened near-noop detection to require negligible objective and
            solution-vector changes in `streamActiveSetNNLS`;
        Made checkpointing iteration-based and removed legacy epoch semantics in
            `streamActiveSetNNLS`;
        Made the monolithic solver the authoritative source of final
            checkpoint state in `streamActiveSetNNLS` and `solve_streaming_nnls`;
        Identified post-solve support pruning as requiring a constrained
            re-solve before the resulting state can be considered committed in
            `streamActiveSetNNLS`;
        No longer mutate `z` when rejecting failed promotions, restoring the
            previous feasible solver state instead in `streamActiveSetNNLS`;
        Removed probation pruning to prevent post-solve support changes from
            violating the hard orbit constraints in `streamActiveSetNNLS`. 26
            August 2026.
v1.7:   Removed residual probation-pruning state and bookkeeping so the
            implementation matches the committed-state semantics documented
            in v1.6;
        Added constrained KKT initialization for x-only warm starts, re-solving
            the supplied support before the first outer iteration to establish
            mutually consistent coefficients, global amplitude, orbit
            multipliers, and numerical ridge;
        Added the committed numerical ridge to checkpoint/resume state and to
            the full-space reduced gradient used for KKT convergence tests;
        Extended the constrained reduced NNLS active-set solve with dual
            feasibility checks and re-entry of fixed-zero variables that
            violate the KKT conditions;
        Added the inner KKT ridge and fixed-variable dual residual to
            constrained-solve diagnostics;
        Expanded outer KKT convergence to distinguish positive-active
            stationarity, zero-active dual feasibility, and inactive-column
            dual feasibility;
        Made KKT termination independent of diagnostic emission and moved
            promotion-score construction outside diagnostic conditionals;
        Separated mathematical promotion eligibility from heuristic candidate
            ranking, using the constrained reduced gradient for KKT decisions
            and the promotion score only to rank eligible columns;
        Preserved the KKT-violation filter when relaxing promotion cooldowns;
        Tightened failed-promotion zero thresholds to avoid rejecting small
            but valid newly active coefficients;
        Commit the fitted amplitude, orbit multipliers, and complete numerical
            ridge atomically with each accepted constrained solution
        Improved code formatting. 7 September 2026
v1.8:   Added unified outer-iteration progress accounting and persistent
            unresolved-dual stall detection to prevent repeated failed
            promotion cycles;
        Restricted normal promotions to genuine KKT-violating columns and
            removed obsolete forced-coverage promotion;
        Made failed and rejected promotions restore the previous feasible
            solver state before progress accounting;
        Enforced `known_zero_mask` as a hard solver constraint during
            initialization, resume, and inactive-column promotion, with
            explicit infeasibility detection for fully masked positive-weight
            orbits;
        Made the monolithic solution authoritative on return;
        Retained the numerical ridge as part of the solved regularized
            objective, with the committed total ridge used consistently in
            the reduced solve, checkpoint state, and outer KKT convergence
            test;
        Added adjustable `regularisation_scale` throughout the solver pathway.
            10 September 2026
v1.9:   Allow `regularisation_scale` to be zero to disable stabilisation ridge;
        Updated `monolithic_nnls_scipy` to have the same API as
            `solve_streaming_nnls` and solve the same mathematics;
        Added adjacent function `monolithicNNLS` computing a genuine
            monolithic non-negative least-squares solution. 11 September 2026
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
    from scipy.optimize import Bounds, LinearConstraint, minimize
    _HAS_SCIPY_OPTIMIZE = True
except Exception:
    _HAS_SCIPY_OPTIMIZE = False

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
    t_read = 0.0
    t_forward = 0.0
    t_transpose = 0.0

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

        if int(M.shape[2]) != P:
            raise RuntimeError(
                f"Model population dimension {M.shape[2]} != expected P={P}.")

        Lk = int(keep_idx.size) if keep_idx is not None else int(M.shape[3])

        for s0, s1 in batch:
            Sblk = s1 - s0

            # First pass: exact v = A @ (S z), one orbit slice at a time.
            v = np.zeros((Sblk, Lk), dtype=np.float64)

            for cc in range(C):
                t0 = time.perf_counter()
                M_cc = M[s0:s1, cc, :, :][...]
                if keep_idx is not None:
                    M_cc = M_cc[:, :, keep_idx]
                M_cc = np.asarray(M_cc, dtype=np.float64, order="C")
                t_read += time.perf_counter() - t0

                # M_cc has shape (Sblk, P, Lk)
                # z_cp[cc] has shape (P,)
                # result has shape (Sblk, Lk)
                t0 = time.perf_counter()
                v += np.tensordot(z_cp[cc], M_cc, axes=(0, 1))
                t_forward += time.perf_counter() - t0
                del M_cc

            # Second pass: exact g = A^T @ v, again orbit slice by orbit slice.
            for cc in range(C):
                t0 = time.perf_counter()
                M_cc = M[s0:s1, cc, :, :][...]
                if keep_idx is not None:
                    M_cc = M_cc[:, :, keep_idx]
                M_cc = np.asarray(M_cc, dtype=np.float64, order="C")
                t_read += time.perf_counter() - t0

                # M_cc: (Sblk, P, Lk)
                # v   : (Sblk, Lk)
                # g_cc: (P,)
                t0 = time.perf_counter()
                g_cc = np.tensordot(M_cc, v, axes=([0, 2], [0, 1]))
                t_transpose += time.perf_counter() - t0
                del M_cc

                base = cc * P
                partial[base:base + P] += (S_flat[base:base + P] * g_cc)

    print(
        f"[ATAz worker {os.getpid()}] read={t_read:.1f}s "
        f"forward={t_forward:.1f}s transpose={t_transpose:.1f}s",
        flush=True)
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

def _get_known_zero_mask(h5_path: str, C: int, P: int) -> np.ndarray:
    """
    Return the persistent hard-exclusion mask for the solver.

    Parameters
    ----------
    h5_path : str
        Path to the CubeFit HDF5 file.
    C : int
        Number of orbit components.
    P : int
        Number of population columns per orbit.

    Returns
    -------
    known_zero : ndarray, shape (C, P)
        Boolean mask. True entries are fixed to zero by the solver.

    Raises
    ------
    RuntimeError
        If a stored mask has a shape inconsistent with ``(C, P)``.

    Examples
    --------
    >>> known_zero = _get_known_zero_mask(h5_path, C=3, P=360)
    """
    with open_h5(h5_path, role="reader") as f:
        if "/HyperCube/known_zero_mask" not in f:
            return np.zeros((C, P), dtype=bool)

        known_zero = np.asarray(
            f["/HyperCube/known_zero_mask"][...], dtype=bool)

    if known_zero.shape != (C, P):
        raise RuntimeError(
            f"known_zero_mask has shape {known_zero.shape}; "
            f"expected {(C, P)}.")

    return known_zero

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
    regularisation_scale: float = 1.0,
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
    regularisation_scale : float, optional
        Multiplicative scale applied to the numerical regularisation.
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
    H = np.asarray(ATA_sub_reg, dtype=np.float64, order="C")
    b = np.asarray(ATy_sub, dtype=np.float64).ravel(order="C")
    active_idx = np.asarray(active_idx, dtype=np.int64).ravel(order="C")
    S_active = np.asarray(S_active, dtype=np.float64).ravel(order="C")
    shape = np.asarray(orbit_shape, dtype=np.float64).ravel(order="C")

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
            "lambda_orbit": np.asarray(lambda_orbit, dtype=np.float64).copy(),
        }

        if missing_orbits is not None:
            out["missing_orbits"] = [int(cc) for cc in missing_orbits]

        return out

    if H.shape != (k, k):
        raise ValueError(
            "ATA_sub_reg has a shape inconsistent with active_idx.")
    if b.size != k or S_active.size != k:
        raise ValueError(
            "ATy_sub or S_active has a size inconsistent with active_idx.")
    if shape.size != C:
        raise ValueError(
            "orbit_shape must have length C.")
    if not np.all(np.isfinite(shape)):
        raise ValueError(
            "orbit_shape contains non-finite values.")

    shape = np.maximum(shape, 0.0)
    shape_sum = float(np.sum(shape))

    if not np.isfinite(shape_sum) or shape_sum <= 0.0:
        raise ValueError(
            "orbit_shape must have positive finite total weight.")

    shape /= shape_sum
    positive_orbits = np.flatnonzero(shape > 0.0)

    if k == 0:
        return (np.zeros((0,), dtype=np.float64),
            _failure_info("empty_support", n_free=0))

    orbit_of_col = (active_idx // P).astype(np.int64)

    if np.any(orbit_of_col < 0) or np.any(orbit_of_col >= C):
        raise ValueError(
            "active_idx contains an invalid orbit index.")

    # Columns belonging to zero-shape orbits must not enter this helper.
    zero_shape_active = np.flatnonzero(shape[orbit_of_col] <= 0.0)

    if zero_shape_active.size > 0:
        return (np.zeros((k,), dtype=np.float64),
            _failure_info("zero_shape_orbit_active", n_free=k))

    support_count = np.bincount(orbit_of_col, minlength=C,)

    missing = positive_orbits[support_count[positive_orbits] == 0]

    if missing.size > 0:
        return (np.zeros((k,), dtype=np.float64),
            _failure_info("missing_orbit_support", n_free=k,
                missing_orbits=missing))

    free = np.ones((k,), dtype=bool)
    removed = []

    diag = np.diag(H)
    finite_diag = diag[np.isfinite(diag)]

    if finite_diag.size > 0:
        diag_med = float(np.median(np.abs(finite_diag)))
    else:
        diag_med = 1.0

    mu = (0.0 if regularisation_scale == 0.0 else max(1e-12,
        regularisation_scale * float(mu_rel) * max(1.0, diag_med)))

    dual_scale = max(1.0, float(np.max(np.abs(b))))
    dual_tol = max(1e-10, 1e-10 * dual_scale)
    max_inner = max(10, 4 * k)

    for _ in range(max_inner):
        free_idx = np.flatnonzero(free)
        n_free = int(free_idx.size)

        if n_free == 0:
            break

        free_orbits = orbit_of_col[free_idx]

        # Every positive-shape orbit must retain at least one free variable.
        free_count = np.bincount(free_orbits, minlength=C)

        missing_free = positive_orbits[free_count[positive_orbits] == 0]

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

        B = np.zeros((n_constraints, n_free), dtype=np.float64)

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
        lambda_sub = np.asarray(sol[n_primal:], dtype=np.float64)

        lambda_orbit = np.zeros((C,), dtype=np.float64)
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

            z_out = np.zeros((k,), dtype=np.float64)
            z_out[free_idx] = np.maximum(z_free, 0.0)

            full_mass = np.bincount(orbit_of_col,
                weights=S_active * z_out, minlength=C).astype(np.float64)

            target_mass = alpha_out * shape
            mass_resid = full_mass - target_mass

            scale = np.maximum(np.abs(target_mass), 1.0)
            rel_linf = float(np.max(np.abs(mass_resid) / scale))

            # The alpha stationarity equation should satisfy
            # shape.T @ lambda == 0.
            alpha_stationarity = float(np.dot(shape, lambda_orbit))
            dual_grad = b - (H @ z_out) - (mu * z_out) - S_active * \
                lambda_orbit[orbit_of_col]

            fixed = ~free
            dual_violators = np.flatnonzero(
                fixed & (dual_grad > dual_tol))

            if dual_violators.size > 0:
                reenter = int(
                    dual_violators[np.argmax(dual_grad[dual_violators])])
                free[reenter] = True

                if reenter in removed:
                    removed.remove(reenter)

                continue

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
                "mu": float(mu),
                "dual_linf": float(
                    np.max(np.maximum(dual_grad[~free], 0.0))
                ) if np.any(~free) else 0.0,
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
        order = negative_local[np.argsort(z_free[negative_local])]

        remove_reduced_idx = None

        for local_j in order:
            candidate = int(free_idx[int(local_j)])
            cc = int(orbit_of_col[candidate])

            remaining_in_orbit = int(np.count_nonzero(free
                    & (orbit_of_col == cc)))

            if remaining_in_orbit <= 1:
                continue

            remove_reduced_idx = candidate
            break

        if remove_reduced_idx is None:
            z_fail = np.zeros((k,), dtype=np.float64)
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
    known_zero_mask: Optional[np.ndarray] = None,
    x0_flat: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    max_active: int = 1000,
    tol_grad: float = 1e-8,
    max_iter: int = 5000,
    regularisation_scale: float = 1.0,
    checkpoint_cb=None,
    checkpoint_every: int = 0,
):
    """
    Monolithic streaming active-set NNLS.
    """

    regularisation_scale = float(regularisation_scale)

    if (not np.isfinite(regularisation_scale)) or (regularisation_scale < 0.0):
        raise ValueError(
            "regularisation_scale must be nonnegative and finite.")

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

    CP = int(C * P)
    orbit_of_col = (np.arange(CP, dtype=np.int64) // P).astype(np.int64)
    if known_zero_mask is None:
        known_zero_flat = np.zeros((CP,), dtype=bool)
    else:
        known_zero_flat = np.asarray(known_zero_mask, dtype=bool
            ).ravel(order="C")

        if known_zero_flat.size != CP:
            raise ValueError(
                "known_zero_mask has wrong size in monolithic solver")
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

    # --- outer-iteration progress watchdog ---
    z_delta_tol = 1e-8
    active_delta_tol = 0
    hard_stall_patience = 12
    no_progress_count = 0
    dual_stall_count = 0
    last_dual_col = -1
    last_dual_value = np.nan
    dual_stall_patience = 8

    # --- positive-gradient batch-promotion controls ---
    positive_batch_size = 8 if C <= 3 else 24

    # The orbit shape is fixed a priori. Its global amplitude is a fitted
    # variable in every constrained reduced solve.
    has_hard_orbit_shape = (
        orbit_shape is not None
        and np.any(orbit_shape > 0.0)
    )

    positive_shape_orbits = (np.flatnonzero(orbit_shape > 0.0)
        if has_hard_orbit_shape else np.zeros((0,), dtype=np.int64))

    zero_shape_orbits = (np.flatnonzero(orbit_shape <= 0.0)
        if has_hard_orbit_shape else np.zeros((0,), dtype=np.int64))
    if has_hard_orbit_shape:
        known_zero_cp = known_zero_flat.reshape(C, P)
        blocked_orbits = np.flatnonzero(np.all(known_zero_cp, axis=1)
            & (orbit_shape > 0.0))

        if blocked_orbits.size > 0:
            raise ValueError(
                "known_zero_mask makes the hard orbit constraint "
                "infeasible for positive-weight orbits "
                f"{blocked_orbits.tolist()}"
            )

    # Current accepted global amplitude. It is initialized below and then
    # replaced by the value fitted by each accepted KKT solve.
    alpha_current = 0.0

    # Orbit-level Lagrange multipliers from the most recently accepted
    # constrained reduced solve.
    lambda_orbit_current = np.zeros((C,), dtype=np.float64)
    ridge_current = 0.0

    diag_level = int(os.environ.get("CUBEFIT_DIAG_LEVEL", "1"))
    diag_stride = max(1, int(os.environ.get("CUBEFIT_DIAG_STRIDE", "1")))
    diag_jsonl_path = os.environ.get("CUBEFIT_DIAG_JSONL", "").strip() or None
    diag_topk = max(1, int(os.environ.get("CUBEFIT_DIAG_TOPK", "12")))
    diag_t0 = time.perf_counter()

    # Normalize and validate the resume metadata before it is used.
    resume_state = dict(resume_state or {})
    if resume_state:
        saved_regularisation_scale = float(
            resume_state.get("regularisation_scale", 1.0))

        if not np.isclose(saved_regularisation_scale, regularisation_scale,
            rtol=1e-12, atol=0.0):
            raise ValueError(
                "Cannot full-resume with a different regularisation_scale: "
                f"checkpoint={saved_regularisation_scale:.6e}, "
                f"requested={regularisation_scale:.6e}. "
                "Use warm_start='saved_x' instead.")
    needs_kkt_initialise = bool(x0_flat is not None and not resume_state)

    start_iter = int(resume_state.get("iter", 0) or 0)

    if start_iter < 0:
        start_iter = 0

    def _emit_diag(record: dict) -> None:
        if diag_level <= 0 and ('error' not in str(record.get("kind"))):
            return
        rec = dict(record)
        rec.setdefault("t_sec", float(time.perf_counter() - diag_t0))
        _diag_append_jsonl(diag_jsonl_path, rec)

    _emit_diag(
        {
            "kind": "mono_setup",
            "source": "streamActiveSetNNLS",
            "C": int(C),
            "P": int(P),
            "CP": int(CP),
            "processes": int(cfg.processes),
            "blas_threads": int(cfg.blas_threads),
            "data_scale_ref": float(data_scale_ref),
            "grad_ref": float(grad_ref),
            "tol_grad": float(tol_grad),
            "tol_grad_rel": float(tol_grad_rel),
            "regularisation_scale": float(regularisation_scale),
            "diag_level": int(diag_level),
            "diag_stride": int(diag_stride),
            "diag_topk": int(diag_topk),
            "hard_orbit_shape": bool(has_hard_orbit_shape),
            "orbit_shape": (orbit_shape.tolist() if orbit_shape is not None
                else None),
            "orbit_shape_sum": (float(np.sum(orbit_shape))
                if orbit_shape is not None else None),
            "orbit_target_sum": None,
            "max_iter": int(max_iter),
            "start_iter": int(start_iter),
            "max_active": int(max_active),
            "explore_budget": int(explore_budget),
            "hard_stall_patience": int(hard_stall_patience),
            "z_delta_tol": float(z_delta_tol),
            "active_delta_tol": int(active_delta_tol),
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

        z = np.divide(x0_flat, S_flat,
            out=np.zeros_like(x0_flat, dtype=np.float64),
            where=S_flat > 0.0)
        np.maximum(z, 0.0, out=z)

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
                target_cc = alpha_current * float(orbit_shape[cc])

                current_mass = float(np.sum(x_seed[cc, :]))

                if (
                    np.isfinite(current_mass)
                    and current_mass > 0.0
                ):
                    # Preserve the warm-start population mixture.
                    x_seed[cc, :] *= (target_cc / current_mass)
                else:
                    base = cc * P
                    orbit_cols = np.arange(base, base + P, dtype=np.int64)

                    allowed_p = np.flatnonzero(~known_zero_flat[orbit_cols])

                    if allowed_p.size == 0:
                        raise ValueError(
                            f"Orbit {cc} has no allowed population columns.")

                    local_score = ATy_scaled[base + allowed_p]
                    p_seed = int(allowed_p[int(np.argmax(local_score))])

                    x_seed[cc, :] = 0.0
                    x_seed[cc, p_seed] = target_cc

            x_seed_flat = x_seed.ravel(order="C")

            z = np.divide(x_seed_flat, S_flat,
                out=np.zeros_like(x_seed_flat, dtype=np.float64),
                where=S_flat > 0.0)
            np.maximum(z, 0.0, out=z)

        else:
            # On resume, infer a fallback amplitude from the restored physical
            # coefficient vector. A saved alpha, if present, is restored below.
            alpha_current = float(np.sum((S_flat * z).reshape(C, P),
                dtype=np.float64))

            if (
                not np.isfinite(alpha_current)
                or alpha_current < 0.0
            ):
                alpha_current = 0.0
    z[known_zero_flat] = 0.0
    z_scale0 = np.max(z) + 1e-30
    active = z > (1e-10 * z_scale0)
    active[known_zero_flat] = False

    if has_hard_orbit_shape and zero_shape_orbits.size > 0:
        for cc in zero_shape_orbits:
            base = int(cc) * P
            z[base:base + P] = 0.0
            active[base:base + P] = False

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
        dual_stall_count = int(resume_state.get("dual_stall_count", 0))
        last_dual_col = int(resume_state.get("last_dual_col", -1))
        last_dual_value = float(
            resume_state.get("last_dual_value", np.nan))
        negative_grad_count = int(resume_state.get("negative_grad_count", 0))

        if "active_mask" in resume_state:
            am = np.asarray(
                resume_state["active_mask"], dtype=bool
            ).ravel(order="C")
            if am.size == CP:
                active = am.copy()

        if "alpha_current" in resume_state:
            alpha_saved = float(
                resume_state["alpha_current"]
            )

            if (
                np.isfinite(alpha_saved)
                and alpha_saved >= 0.0
            ):
                alpha_current = alpha_saved
        if "ridge_current" in resume_state:
            ridge_saved = float(resume_state["ridge_current"])
            if np.isfinite(ridge_saved) and ridge_saved >= 0.0:
                ridge_current = ridge_saved
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
        def _pairs_to_dict(pairs):
            out = {}
            for item in pairs or []:
                try:
                    k, v = item
                    out[int(k)] = int(v)
                except Exception:
                    pass
            return out

        col_cooldown_until = _pairs_to_dict(
            resume_state.get("col_cooldown_until", [])
        )
        orbit_cooldown_until = _pairs_to_dict(
            resume_state.get(
                "orbit_cooldown_until",
                [],
            )
        )
    z[known_zero_flat] = 0.0
    active[known_zero_flat] = False

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
            "col_cooldown_until": [
                [int(k), int(v)] for k, v in col_cooldown_until.items()],
            "lambda_orbit": lambda_orbit_current.astype(np.float64).tolist(),
            "alpha_current": float(alpha_current),
            "ridge_current": float(ridge_current),
            "regularisation_scale": float(regularisation_scale),
            "orbit_cooldown_until": [[int(k), int(v)]
                for k, v in orbit_cooldown_until.items()],
            "stop_reason": str(stop_reason),
            "stop_category": str(stop_category),
            "stop_converged": bool(stop_converged),
            "stop_details": dict(stop_details),
            "dual_stall_count": int(dual_stall_count),
            "last_dual_col": int(last_dual_col),
            "last_dual_value": float(last_dual_value),
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
    def _assemble_reduced(active_idx):
        """Assemble reduced normal equations on the supplied support."""
        active_idx = np.asarray(active_idx, dtype=np.int64)
        k = int(active_idx.size)
        S_active = S_flat[active_idx]

        ATA_sub = np.zeros((k, k), dtype=np.float64)
        ATy_sub = np.zeros(k, dtype=np.float64)

        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        futures = [
            executor.submit(
                _worker_reduced, h5_path, batch, keep_idx, active_idx,
                S_active, CP, P)
            for batch in batches if batch
        ]

        for future in as_completed(futures):
            ATA_loc, ATy_loc = future.result()
            ATA_sub += ATA_loc
            ATy_sub += ATy_loc

        return ATA_sub, ATy_sub, S_active
    def _solve_reduced_active(active_idx):
        """Assemble and solve the constrained reduced problem."""
        active_idx = np.asarray(active_idx, dtype=np.int64,)
        k_loc = int(active_idx.size)

        if k_loc == 0:
            raise RuntimeError(
                "Cannot solve reduced problem with empty active support.")

        ATA_loc, ATy_loc, S_loc = _assemble_reduced(active_idx)

        diag_loc = np.diag(ATA_loc)
        diag_med_loc = (float(np.median(diag_loc)) if diag_loc.size
            else 0.0)

        try:
            eigs_loc = np.linalg.eigvalsh(ATA_loc)
            emin_loc = float(np.min(eigs_loc))
            emax_loc = float(np.max(eigs_loc))
        except Exception:
            emin_loc = 0.0
            emax_loc = (float(np.max(diag_loc)) if diag_loc.size else 1.0)

        cond_bad_loc = emin_loc <= 1e-10 * max(emax_loc, 1.0)
        ridge_rel_loc = regularisation_scale * (
            1e-2 if cond_bad_loc else 1e-3)
        ridge_loc = (0.0 if regularisation_scale == 0.0 else
            max(1e-10, ridge_rel_loc * max(1.0, diag_med_loc)))

        ATA_reg_loc = (ATA_loc + ridge_loc * np.eye(k_loc, dtype=np.float64))

        if has_hard_orbit_shape:
            z_raw_loc, info_loc = _orbit_hard_constraint_solve(
                ATA_sub_reg=ATA_reg_loc, ATy_sub=ATy_loc,
                active_idx=active_idx, S_active=S_loc,
                orbit_shape=np.asarray(orbit_shape, dtype=np.float64),
                C=C, P=P, regularisation_scale=regularisation_scale)

            if not info_loc["accepted"]:
                return (
                    active_idx, None, None, None, None, None, None, None)

            z_loc = np.asarray(z_raw_loc, dtype=np.float64).copy()
            alpha_loc = float(info_loc["alpha"])
            lambda_loc = np.asarray(info_loc["lambda_orbit"], dtype=np.float64
            ).ravel(order="C")
        else:
            try:
                z_loc = np.linalg.solve(ATA_reg_loc, ATy_loc)
            except np.linalg.LinAlgError:
                U, svals, Vt = np.linalg.svd(ATA_reg_loc, full_matrices=False)
                smax = svals[0] if svals.size else 1.0
                s_thresh = max(1e-12, 1e-6 * smax)
                s_inv = np.array([1.0 / s if s > s_thresh else 0.0
                        for s in svals], dtype=np.float64,)
                z_loc = Vt.T @ (s_inv * (U.T @ ATy_loc))

            z_loc = np.maximum(z_loc, 0.0)
            alpha_loc = None
            lambda_loc = None
            info_loc = {
                "accepted": True,
                "mu": 0.0,
                "lambda_orbit": np.zeros((C,), dtype=np.float64),
            }

        return (
            active_idx,
            z_loc,
            info_loc,
            alpha_loc,
            lambda_loc,
            ATA_reg_loc,
            ATy_loc,
            S_loc,
        )
    # --------------------------------------------------------------------------
    if needs_kkt_initialise and has_hard_orbit_shape:
        active_idx = np.flatnonzero(active).astype(np.int64)

        if active_idx.size == 0:
            raise RuntimeError(
                "Cannot initialise constrained KKT state from empty x support.")

        ATA_init, ATy_init, S_init = _assemble_reduced(active_idx)
        diag_init = np.diag(ATA_init)
        diag_med_init = (
            float(np.median(diag_init)) if diag_init.size else 0.0)

        try:
            eigs_init = np.linalg.eigvalsh(ATA_init)
            emin_init = float(np.min(eigs_init))
            emax_init = float(np.max(eigs_init))
        except Exception:
            emin_init = 0.0
            emax_init = (
                float(np.max(diag_init)) if diag_init.size else 1.0)

        cond_bad = emin_init <= 1e-10 * max(emax_init, 1.0)
        ridge_rel = regularisation_scale * (1e-2 if cond_bad else 1e-3)
        ridge = (0.0 if regularisation_scale == 0.0
            else max(1e-10, ridge_rel * max(1.0, diag_med_init)))

        ATA_init_reg = ATA_init + ridge * np.eye(
            active_idx.size, dtype=np.float64)

        z_init, info_init = _orbit_hard_constraint_solve(
            ATA_sub_reg=ATA_init_reg,
            ATy_sub=ATy_init,
            active_idx=active_idx,
            S_active=S_init,
            orbit_shape=np.asarray(orbit_shape, dtype=np.float64),
            C=C,
            P=P,
            regularisation_scale=regularisation_scale,
        )

        if not info_init["accepted"]:
            raise RuntimeError(
                "Initial constrained KKT solve failed: "
                f"{info_init['reason']}.")

        z[:] = 0.0
        z[active_idx] = z_init
        alpha_current = float(info_init["alpha"])
        lambda_orbit_current = np.asarray(
            info_init["lambda_orbit"], dtype=np.float64).copy()
        ridge_current = float(ridge + info_init.get("mu", 0.0))

        print(
            "[MONO][init] established constrained KKT state: "
            f"active={active_idx.size} alpha={alpha_current:.6e} "
            f"ridge={ridge_current:.6e}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Outer-loop termination diagnostics
    # ------------------------------------------------------------
    stop_reason = "max_iter"
    stop_category = "limit"
    stop_details = {}
    stop_iter = None
    stop_converged = False

    def _set_stop(
        reason: str,
        category: str,
        it: int,
        *,
        converged: bool = False,
        **details,
    ) -> None:
        """
        Record and report the reason for terminating the outer solver.

        Parameters
        ----------
        reason : str
            Machine-readable termination reason.
        category : str
            Broad termination category, such as ``"converged"``,
            ``"limit"``, ``"stall"``, ``"active_set"``, or
            ``"numerical"``.
        it : int
            Zero-based outer iteration index.
        converged : bool, optional
            Whether the termination represents solver convergence.
        **details
            Additional diagnostic values associated with the exit.

        Returns
        -------
        None

        Raises
        ------
        None

        Examples
        --------
        >>> _set_stop(
        ...     "max_active",
        ...     "active_set",
        ...     it,
        ...     active=982,
        ...     requested=32,
        ...     max_active=1000,
        ... )
        """
        nonlocal stop_reason
        nonlocal stop_category
        nonlocal stop_details
        nonlocal stop_iter
        nonlocal stop_converged

        stop_reason = str(reason)
        stop_category = str(category)
        stop_details = dict(details)
        stop_iter = int(it)
        stop_converged = bool(converged)

        detail_text = " ".join(
            f"{key}={value}"
            for key, value in stop_details.items()
        )

        print(
            f"[MONO][STOP] reason={stop_reason} "
            f"category={stop_category} "
            f"converged={stop_converged} "
            f"iter={it + 1}/{max_iter}"
            + (f" {detail_text}" if detail_text else ""),
            flush=True,
        )

        _emit_diag(
            {
                "kind": "termination",
                "reason": stop_reason,
                "category": stop_category,
                "converged": stop_converged,
                "iter": int(it + 1),
                "iter_zero_based": int(it),
                "start_iter": int(start_iter),
                "max_iter": int(max_iter),
                "n_active": int(np.count_nonzero(active)),
                "max_active": int(max_active),
                "negative_grad_count": int(negative_grad_count),
                "explore_budget": int(explore_budget),
                "no_progress_count": int(no_progress_count),
                "hard_stall_patience": int(hard_stall_patience),
                "alpha": (
                    float(alpha_current)
                    if has_hard_orbit_shape
                    else None
                ),
                "details": stop_details,
            }
        )

    def _finish_iteration(
        it: int,
        phase: str,
        z_start: np.ndarray,
        active_start: np.ndarray,
        *,
        max_grad_dual: float,
        tol_here: float,
        best_dual_col: int,
        best_dual_value: float,
    ) -> bool:
        """
        Account for progress exactly once at the end of an outer iteration.

        Returns
        -------
        bool
            True when the outer solver should terminate.
        """
        nonlocal no_progress_count
        nonlocal dual_stall_count, last_dual_col, last_dual_value

        z_delta = float(np.linalg.norm(z - z_start))
        z_ref = max(1.0, float(np.linalg.norm(z_start)))
        z_delta_rel = z_delta / z_ref
        active_delta = int(np.count_nonzero(active ^ active_start))

        unchanged = (
            z_delta_rel <= z_delta_tol
            and active_delta <= active_delta_tol
        )

        if unchanged:
            no_progress_count += 1
        else:
            no_progress_count = 0

        same_dual = (
            best_dual_col >= 0
            and best_dual_col == last_dual_col
            and np.isfinite(best_dual_value)
            and np.isfinite(last_dual_value)
            and abs(best_dual_value - last_dual_value)
                <= 1e-8 * max(1.0, abs(last_dual_value))
        )

        if same_dual and best_dual_value > tol_here:
            dual_stall_count += 1
        else:
            dual_stall_count = 0

        last_dual_col = int(best_dual_col)
        last_dual_value = float(best_dual_value)

        _emit_diag({
            "kind": "iter_progress",
            "source": "streamActiveSetNNLS",
            "iter": int(it + 1),
            "phase": str(phase),
            "z_delta_rel": float(z_delta_rel),
            "active_delta": int(active_delta),
            "no_progress_count": int(no_progress_count),
            "hard_stall_patience": int(hard_stall_patience),
            "best_dual_col": int(best_dual_col),
            "best_dual_value": float(best_dual_value),
            "dual_stall_count": int(dual_stall_count),
            "dual_stall_patience": int(dual_stall_patience),
            "max_grad_dual": float(max_grad_dual),
            "tol_here": float(tol_here),
        })

        if dual_stall_count >= dual_stall_patience:
            _set_stop(
                "unresolved_dual_stall",
                "stall",
                it,
                converged=False,
                best_dual_col=int(best_dual_col),
                best_dual_value=float(best_dual_value),
                dual_stall_count=int(dual_stall_count),
                tol_here=float(tol_here),
                n_active=int(np.count_nonzero(active)),
            )
            _emit_checkpoint(
                it, final=True, phase="unresolved_dual_stall")
            return True

        if no_progress_count < hard_stall_patience:
            _emit_checkpoint(it, final=False, phase=phase)
            return False

        _set_stop(
            "outer_iteration_stall",
            "stall",
            it,
            converged=False,
            phase=str(phase),
            z_delta_rel=float(z_delta_rel),
            active_delta=int(active_delta),
            no_progress_count=int(no_progress_count),
            max_grad_dual=float(max_grad_dual),
            tol_here=float(tol_here),
            n_active=int(np.count_nonzero(active)),
        )
        _emit_checkpoint(it, final=True, phase="outer_stall")
        return True
    # ------------------------------------------------------------
    # Active-set outer loop
    # ------------------------------------------------------------
    for it in range(start_iter, max_iter):
        t_iter = time.perf_counter()
        z_iter_start = z.copy()
        active_iter_start = active.copy()

        ATAz_scaled = _compute_ATAz_scaled(z)
        data_objective_current = (0.5 * float(np.dot(z, ATAz_scaled))
            - float(np.dot(ATy_scaled, z)))
        grad_data = ATy_scaled - ATAz_scaled
        grad_ridge = -float(ridge_current) * z

        if has_hard_orbit_shape:
            constraint_gradient = S_flat * lambda_orbit_current[orbit_of_col]
            grad_total = grad_data + grad_ridge - constraint_gradient
        else:
            constraint_gradient = np.zeros(CP, dtype=np.float64)
            grad_total = grad_data + grad_ridge
        
        orbit_s, orbit_target, orbit_deficit = _orbit_mass_and_deficit(z)

        candidate_mask = ~active
        candidate_mask &= ~known_zero_flat
        if has_hard_orbit_shape:
            candidate_mask &= (orbit_shape[orbit_of_col] > 0.0)
        not_active = np.flatnonzero(candidate_mask)
        gvals_data = grad_data[not_active]
        gvals_total = grad_total[not_active]
        gvals_constraint = constraint_gradient[not_active]

        max_grad_all = float(np.max(grad_total)) if grad_total.size else 0.0
        max_grad_promotable = (float(np.max(gvals_total)) if gvals_total.size
            else 0.0)
        avg_grad_promotable = (float(np.mean(gvals_total)) if gvals_total.size
            else 0.0)
        max_grad_data = float(np.max(gvals_data)) if gvals_data.size else 0.0
        max_grad_orbit = (float(np.max(np.abs(gvals_constraint)))
            if gvals_constraint.size else 0.0)

        tol_here = max(tol_grad, tol_grad_rel * grad_ref)

        # Check the full KKT conditions before attempting any promotion.
        z_scale = max(1.0, float(np.max(np.abs(z))))
        positive_tol = 1e-12 * z_scale

        positive_active = active & (z > positive_tol)
        zero_active = active & ~positive_active
        inactive_free = candidate_mask

        max_grad_active = (
            float(np.max(np.abs(grad_total[positive_active])))
            if np.any(positive_active) else 0.0)

        max_grad_zero_active = (
            float(np.max(grad_total[zero_active]))
            if np.any(zero_active) else -np.inf)

        max_grad_inactive = (
            float(np.max(grad_total[inactive_free]))
            if np.any(inactive_free) else -np.inf)

        max_grad_dual = max(
            max_grad_zero_active, max_grad_inactive, 0.0)

        active_ok = max_grad_active <= tol_here
        dual_ok = max_grad_dual <= tol_here
        kkt_converged = active_ok and dual_ok
        best_dual_col = (int(not_active[np.argmax(gvals_total)])
            if gvals_total.size else -1)
        best_dual_value = (float(np.max(gvals_total))
            if gvals_total.size else -np.inf)

        if kkt_converged:
            _set_stop(
                "kkt_converged",
                "converged",
                it,
                converged=True,
                max_grad_active=float(max_grad_active),
                max_grad_dual=float(max_grad_dual),
                tol_here=float(tol_here),
                n_active=int(np.count_nonzero(active)),
            )
            _emit_checkpoint(it, final=True, phase="kkt_converged")
            break

        if dual_ok and not active_ok:
            _set_stop(
                "active_stationarity_failure",
                "numerical",
                it,
                converged=False,
                max_grad_active=float(max_grad_active),
                max_grad_dual=float(max_grad_dual),
                ridge=float(ridge_current),
                tol_here=float(tol_here),
            )
            _emit_checkpoint(
                it, final=True, phase="active_stationarity_failure")
            break

        if diag_level >= 1 and ((it % diag_stride) == 0):
            x_phys_now = _current_x_from_z(z).reshape(C, P)
            orbit_mass_vec = np.sum(x_phys_now, axis=1)
            if has_hard_orbit_shape:
                orbit_target = float(alpha_current) * np.asarray(orbit_shape,
                    dtype=np.float64)
            else:
                orbit_target = np.zeros((C,), dtype=np.float64,)
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

            best_inactive = (
                int(not_active[int(np.argmax(gvals_total))])
                if not_active.size
                else -1
            )

            best_inactive_grad = (
                float(np.max(gvals_total))
                if gvals_total.size
                else -np.inf
            )

            _emit_diag(
                {
                    "kind": "iter_pre",
                    "source": "streamActiveSetNNLS",
                    "iter": int(it + 1),
                    "n_active": int(np.count_nonzero(active)),
                    "max_grad_data": float(max_grad_data),
                    "max_grad_promo": float(max_grad_promotable),
                    "avg_grad_promo": float(avg_grad_promotable),
                    "max_grad_total": float(max_grad_all),
                    "max_grad_orbit": float(max_grad_orbit),
                    "max_grad_zero_active": float(max_grad_zero_active),
                    "max_grad_dual": float(max_grad_dual),
                    "active_ok": bool(active_ok),
                    "dual_ok": bool(dual_ok),
                    "tol_here": float(tol_here),
                    "ridge_current": float(ridge_current),
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
                    "data_objective": float(data_objective_current),
                    "n_candidates": int(not_active.size),
                    "n_cooldown": int(
                        np.count_nonzero(_combined_cooldown_mask(it)[not_active])
                    ),
                    "n_col_cooldown": int(
                        np.count_nonzero(_col_cooldown_mask(it)[not_active])
                    ),
                    "n_orbit_cooldown": int(
                        np.count_nonzero(
                            _combined_cooldown_mask(it)[not_active]
                            & ~_col_cooldown_mask(it)[not_active]
                        )
                    ),
                    "best_inactive_col": int(best_inactive),
                    "best_inactive_grad": float(best_inactive_grad),
                    "best_inactive_orbit": (
                        int(best_inactive // P)
                        if best_inactive >= 0
                        else -1
                    ),
                    "max_grad_active": float(max_grad_active),
                    "max_grad_inactive": float(max_grad_inactive),
                    "kkt_converged": bool(kkt_converged),
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

            # Unified promotion score: use the current combined gradient only.
            candidate_mask = ~active
            candidate_mask &= ~known_zero_flat
            if has_hard_orbit_shape:
                candidate_mask &= (orbit_shape[orbit_of_col] > 0.0)

            not_active = np.flatnonzero(candidate_mask)
        
        # Ranking score only; this is not a KKT quantity.
        score = gvals_total.copy()

        if not_active.size > 0:
            svals = S_flat[not_active]
            s_med = np.median(svals) + 1e-30
            score /= (1.0 + 0.25 * (svals / s_med - 1.0))

            aty = ATy_scaled[not_active]
            aty_scale = np.max(np.abs(aty)) + 1e-30
            score += 0.10 * (aty / aty_scale)

        if not_active.size == 0:
            if _finish_iteration(
                it, "no_candidate", z_iter_start, active_iter_start,
                max_grad_dual=max_grad_dual, tol_here=tol_here,
                best_dual_col=best_dual_col, best_dual_value=best_dual_value):
                break
            continue

        max_g = float(np.max(score)) if score.size else -np.inf
        has_inactive_violation = max_grad_promotable > tol_here
        did_explore = not has_inactive_violation
        newly_activated = None
        active_before = active.copy()
        z_before = z.copy()

        if did_explore:
            negative_grad_count += 1
            allow_explore = negative_grad_count <= explore_budget

            if not allow_explore:
                _set_stop(
                    "exploration_budget_exhausted",
                    "active_set",
                    it,
                    converged=False,
                    max_grad_total=float(max_grad_all),
                    max_grad_promotable=float(max_grad_promotable),
                    max_score=float(max_g),
                    tol_here=float(tol_here),
                    negative_grad_count=int(negative_grad_count),
                    explore_budget=int(explore_budget),
                )
                _emit_checkpoint(
                    it, final=True, phase="no_promotable_exit")
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

        if not did_explore:
            score_eff[gvals_total <= tol_here] = -np.inf

        if not np.isfinite(score_eff).any():
            # If the combined tabu blocks everything, relax to column-only
            # cooldown on the same not-active subset.
            score_eff = score.copy()
            col_cooldown_full = _col_cooldown_mask(it)
            score_eff[col_cooldown_full[not_active]] = -np.inf

            if not did_explore:
                score_eff[gvals_total <= tol_here] = -np.inf
        finite_score = np.isfinite(score_eff)
        max_grad_eligible = (float(np.max(gvals_total[finite_score]))
            if np.any(finite_score) else -np.inf)
        max_score_eligible = (float(np.max(score_eff[finite_score]))
            if np.any(finite_score) else -np.inf)

        order = np.argsort(score_eff)[::-1]
        chosen = []

        for local_i in order:
            if not np.isfinite(score_eff[local_i]):
                break

            if not did_explore and gvals_total[local_i] <= tol_here:
                continue

            gcol = int(not_active[local_i])
            cc = gcol // P

            if _on_cooldown(gcol, it) or _orbit_on_cooldown(cc, it):
                continue

            chosen.append(gcol)
            if len(chosen) >= preferred_group:
                break
        
        if len(chosen) == 0:
            if _finish_iteration(
                it, "no_eligible_promotions", z_iter_start, active_iter_start,
                max_grad_dual=max_grad_dual, tol_here=tol_here,
                best_dual_col=best_dual_col, best_dual_value=best_dual_value):
                break
            continue

        cols_to_activate = np.asarray(chosen, dtype=np.int64)
        active[cols_to_activate] = True
        newly_activated = cols_to_activate.copy()
        promoted_cols = newly_activated.copy()
        n_candidates = int(not_active.size)
        n_cooldown = int(np.count_nonzero(cooldown_mask))
        n_col_cooldown = int(np.count_nonzero(
            _col_cooldown_mask(it)[not_active]))
        n_orbit_cooldown = int(np.count_nonzero(
            _combined_cooldown_mask(it)[not_active] & \
            ~_col_cooldown_mask(it)[not_active]))
        n_eligible = int(np.count_nonzero(np.isfinite(score_eff)))

        candidate_order = np.argsort(gvals_total)[::-1]

        diag_candidates = []
        for local_i in candidate_order[:diag_topk]:
            c = int(not_active[local_i])
            diag_candidates.append({
                "col": c,
                "orbit": int(c // P),
                "grad_total": float(gvals_total[local_i]),
                "score": float(score[local_i]),
                "cooldown": bool(cooldown_mask[local_i]),
                "col_cooldown": bool(
                    _col_cooldown_mask(it)[not_active][local_i]
                ),
            })

        promoted_scores = []
        for c in promoted_cols:
            local = np.flatnonzero(not_active == int(c))
            if local.size == 0:
                continue

            j = int(local[0])
            promoted_scores.append({
                "col": int(c),
                "orbit": int(c // P),
                "grad_total": float(gvals_total[j]),
                "score": float(score[j]),
                "cooldown": bool(cooldown_mask[j]),
            })

        n_active_after = int(np.count_nonzero(active))

        if n_active_after > int(max_active):
            n_requested = (int(newly_activated.size) if newly_activated is not
                None else 0)
            n_active_before = int(np.count_nonzero(active_before))
            n_available = max(0, int(max_active) - n_active_before)

            # Roll back this promotion before terminating.
            if (
                newly_activated is not None
                and newly_activated.size > 0
            ):
                active[newly_activated] = False

            n_active_restored = int(np.count_nonzero(active))

            _set_stop(
                "max_active_promotion_overflow",
                "active_set",
                it,
                converged=False,
                active_before=n_active_before,
                requested=n_requested,
                available=n_available,
                active_after_provisional=n_active_after,
                active_after_rollback=n_active_restored,
                max_active=int(max_active),
                overflow=max(
                    0,
                    n_active_after - int(max_active),
                ),
                max_grad_total=float(max_grad_all),
                max_grad_promotable=float(max_grad_promotable),
                tol_here=float(tol_here),
                did_explore=bool(did_explore),
            )

            _emit_checkpoint(
                it,
                final=True,
                phase="max_active_exit",
            )
            break

        _emit_diag(
            {
                "kind": "promotion_attempt",
                "source": "streamActiveSetNNLS",
                "iter": int(it + 1),
                "did_explore": bool(did_explore),
                "negative_grad_count": int(negative_grad_count),
                "tol_here": float(tol_here),
                "max_grad_total": float(max_grad_all),
                "max_grad_promotable": float(max_grad_promotable),
                "avg_grad_promotable": float(avg_grad_promotable),
                "max_score": float(max_g),
                "n_active_before": int(
                    np.count_nonzero(active_before)
                ),
                "n_candidates": int(n_candidates),
                "n_cooldown": int(n_cooldown),
                "n_col_cooldown": int(n_col_cooldown),
                "n_orbit_cooldown": int(n_orbit_cooldown),
                "n_eligible": int(n_eligible),
                "preferred_group": int(preferred_group),
                "n_promoted": int(promoted_cols.size),
                "candidate_topk": diag_candidates,
                "promoted": promoted_scores,
                "max_grad_eligible": float(max_grad_eligible),
                "max_score_eligible": float(max_score_eligible),
            }
        )

        # --------------------------------------------------------
        # Reduced solve (assemble ATA_sub & ATy_sub) in parallel
        # --------------------------------------------------------
        active_idx = np.nonzero(active)[0].astype(np.int64)
        k = int(active_idx.size)
        if k == 0:
            if _finish_iteration(
                it, "empty_active_set", z_iter_start, active_iter_start,
                max_grad_dual=max_grad_dual, tol_here=tol_here,
                best_dual_col=best_dual_col, best_dual_value=best_dual_value):
                break
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
        ridge_rel = regularisation_scale * (1e-2 if cond_bad else 1e-3)
        ridge = (0.0 if regularisation_scale == 0.0
            else max(1e-10, ridge_rel * max(1.0, diag_med)))
        ATA_sub_reg = ATA_sub + ridge * np.eye(k, dtype=np.float64)

        ATA_norm = float(np.linalg.norm(ATA_sub))

        z_old_active = z[active_idx].copy()

        def _quad_obj(A, b, x):
            return 0.5 * float(x @ (A @ x)) - float(b @ x)


        orbit_constraint_info = {
            "accepted": True,
            "reason": "no_prior",
            "alpha": None,
            "orbit_target_mass": np.zeros((C,), dtype=np.float64),
            "resid_l1": 0.0,
            "resid_l2": 0.0,
            "resid_linf": 0.0,
            "rel_resid_linf": 0.0,
            "alpha_stationarity": 0.0,
            "negative_count": 0,
            "lambda_orbit": np.zeros((C,), dtype=np.float64),
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
                    orbit_shape=np.asarray(orbit_shape, dtype=np.float64),
                    C=C,
                    P=P,
                    regularisation_scale=regularisation_scale,
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

                if _finish_iteration(
                    it, "constraint_reject", z_iter_start, active_iter_start,
                    max_grad_dual=max_grad_dual, tol_here=tol_here,
                best_dual_col=best_dual_col, best_dual_value=best_dual_value):
                    break
                continue

            z_sub = np.asarray(z_sub_raw, dtype=np.float64).copy()

            alpha_candidate = float(orbit_constraint_info["alpha"])

            lambda_candidate = np.asarray(orbit_constraint_info["lambda_orbit"],
                dtype=np.float64).ravel(order="C")

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
                    _cooldown_cols(promoted_cols, it, extra_iters=20)
                    _cooldown_orbits(np.unique(promoted_cols // P), it,
                        extra_iters=20)

                if _finish_iteration(
                    it, "invalid_constraint_state", z_iter_start,
                        active_iter_start, max_grad_dual=max_grad_dual,
                        tol_here=tol_here, best_dual_col=best_dual_col,
                        best_dual_value=best_dual_value):
                    break
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

                if z_sub[ii] <= 1e-12 * z_scale_loc:
                    failed_cols.append(int(c))

        failed_cols = np.asarray(failed_cols, dtype=np.int64)
        failed_cols_initial = failed_cols.copy()
        n_failed_initial = int(failed_cols_initial.size)
        surviving_promoted = promoted_cols[
            ~np.isin(promoted_cols, failed_cols_initial)]

        mu_inner = float(orbit_constraint_info.get("mu", 0.0))
        H_eval = ATA_sub_reg + mu_inner * np.eye(
            k, dtype=np.float64)
        obj_old = _quad_obj(H_eval, ATy_sub, z_old_active)
        obj_new = _quad_obj(H_eval, ATy_sub, z_sub)
        norm_old = float(np.linalg.norm(z_old_active))
        norm_new = float(np.linalg.norm(z_sub))
        obj_gain = float(obj_old - obj_new)
        rel_gain = obj_gain / max(1.0, abs(obj_old))

        z_step = float(np.linalg.norm(z_sub - z_old_active))
        z_step_rel = z_step / max(1.0, norm_old)

        promoted_mask = np.zeros(k, dtype=bool)
        if promoted_cols.size > 0:
            promoted_set = set(int(c) for c in promoted_cols)
            promoted_mask = np.asarray(
                [int(c) in promoted_set for c in active_idx],
                dtype=bool,
            )

        promoted_z_old = z_old_active[promoted_mask] if np.any(promoted_mask) \
            else np.zeros(0, dtype=np.float64)
        promoted_z_new = z_sub[promoted_mask] if np.any(promoted_mask) \
            else np.zeros(0, dtype=np.float64)

        promoted_z_norm = float(np.linalg.norm(promoted_z_new - promoted_z_old)
            ) if promoted_z_new.size else 0.0

        promoted_new_max = float(np.max(promoted_z_new)) if promoted_z_new.size\
            else 0.0

        promoted_new_l1 = float(np.sum(promoted_z_new)) if promoted_z_new.size \
            else 0.0

        failed_frac = float(failed_cols.size) / float(promoted_cols.size) \
            if promoted_cols.size else 0.0

        near_noop = did_explore and promoted_cols.size > 0 \
            and obj_gain <= promotion_noop_abs_tol \
            and rel_gain <= promotion_noop_rel_tol \
            and z_step_rel <= 1e-8

        z_scale_loc = float(np.max(z_sub)) + 1e-30

        newly_positive = int(np.count_nonzero(
            promoted_z_new > 1e-12 * z_scale_loc)) if promoted_z_new.size \
            else 0

        _emit_diag(
            {
                "kind": "promotion_outcome",
                "source": "streamActiveSetNNLS",
                "iter": int(it + 1),
                "n_active_before": int(
                    np.count_nonzero(active_before)
                ),
                "k_reduced": int(k),
                "n_promoted": int(promoted_cols.size),
                "n_failed": int(n_failed_initial),
                "n_survived": int(surviving_promoted.size),
                "failed_fraction": float(failed_frac),
                "promoted_new_positive": int(newly_positive),
                "promoted_z_norm": float(promoted_z_norm),
                "promoted_z_max": float(promoted_new_max),
                "promoted_z_l1": float(promoted_new_l1),
                "obj_old": float(obj_old),
                "obj_new": float(obj_new),
                "obj_gain": float(obj_gain),
                "rel_gain": float(rel_gain),
                "z_step": float(z_step),
                "z_step_rel": float(z_step_rel),
                "norm_old": float(norm_old),
                "norm_new": float(norm_new),
                "near_noop": bool(near_noop),
                "noop_abs_tol": float(
                    promotion_noop_abs_tol
                ),
                "noop_rel_tol": float(
                    promotion_noop_rel_tol
                ),
                "failed_cols": failed_cols_initial.tolist(),
                "surviving_promoted": surviving_promoted.tolist(),
                "failed_re_resolved": bool(
                    n_failed_initial > 0 and surviving_promoted.size > 0),
            }
        )

        reject = False

        if did_explore:
            obj_worsened = np.isfinite(obj_new) and (
                obj_new > obj_old + 1e-12 * (1.0 + abs(obj_old)))
            norm_exploded = (norm_old > 1e-12 and norm_new > 50.0 * norm_old)

            if obj_worsened or norm_exploded:
                reject = True

        if did_explore and near_noop:
            _cooldown_cols(promoted_cols, it, extra_iters=3)

            print(
                "[MONO][explore] near-noop: "
                f"obj_gain={obj_gain:.6e} "
                f"rel_gain={rel_gain:.6e} "
                f"z_step_rel={z_step_rel:.6e} "
                f"promoted={promoted_cols.size} "
                f"failed={failed_cols.size} "
                f"promoted_z_l1={promoted_new_l1:.6e} "
                f"promoted_z_max={promoted_new_max:.6e} "
                f"max_grad_promotable={max_grad_promotable:.6e} "
                f"max_score={max_g:.6e}",
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

            if _finish_iteration(
                it, "exploratory_reject", z_iter_start, active_iter_start,
                max_grad_dual=max_grad_dual, tol_here=tol_here,
                best_dual_col=best_dual_col, best_dual_value=best_dual_value):
                break
            continue

        # Commit the constrained reduced solution.
        z[active_idx] = z_sub

        # Failed promotions are removed from the active support, but the
        # resulting state must be re-solved before it can be committed as a
        # valid constrained solution.
        if failed_cols.size > 0:

            _cooldown_cols(failed_cols, it)

            if surviving_promoted.size == 0:
                active[:] = active_before
                z[:] = z_before

                if _finish_iteration(
                    it, "failed_promotion_rollback", z_iter_start,
                    active_iter_start, max_grad_dual=max_grad_dual,
                    tol_here=tol_here, best_dual_col=best_dual_col,
                    best_dual_value=best_dual_value):
                    break
                continue

            candidate_active = np.unique(np.concatenate((
                np.flatnonzero(active_before), surviving_promoted))
            ).astype(np.int64, copy=False)

            (active_retry, z_retry, info_retry, alpha_retry, lambda_retry,
                ATA_retry_reg, ATy_retry, S_retry
                ) = _solve_reduced_active(candidate_active)

            if (z_retry is None
                or info_retry is None
                or not info_retry["accepted"]):
                active[:] = active_before
                z[:] = z_before

                _cooldown_orbits(
                    np.unique(failed_cols // P), it, extra_iters=20)

                if _finish_iteration(
                    it, "failed_promotion_reresolve_reject", z_iter_start,
                    active_iter_start, max_grad_dual=max_grad_dual,
                    tol_here=tol_here, best_dual_col=best_dual_col,
                    best_dual_value=best_dual_value):
                    break
                continue

            active[:] = False
            active[active_retry] = True
            z[:] = 0.0
            z[active_retry] = z_retry

            active_idx = active_retry
            k = int(active_idx.size)
            z_sub = z_retry
            S_active = S_retry
            ATA_sub_reg = ATA_retry_reg

            if has_hard_orbit_shape:
                orbit_constraint_info = info_retry
                alpha_candidate = float(alpha_retry)
                lambda_candidate = lambda_retry.copy()

            mu_inner = float(orbit_constraint_info.get("mu", 0.0))
            H_eval = ATA_sub_reg + mu_inner * np.eye(
                k, dtype=np.float64)

            z_old_active = z_before[active_idx].copy()
            obj_old = _quad_obj(H_eval, ATy_retry, z_old_active)
            obj_new = _quad_obj(H_eval, ATy_retry, z_sub)
            norm_old = float(np.linalg.norm(z_old_active))
            norm_new = float(np.linalg.norm(z_sub))
            obj_gain = float(obj_old - obj_new)
            rel_gain = obj_gain / max(1.0, abs(obj_old))
            z_step = float(np.linalg.norm(z_sub - z_old_active))
            z_step_rel = z_step / max(1.0, norm_old)
            promoted_mask = np.isin(active_idx, promoted_cols)
            promoted_z_old = z_old_active[promoted_mask]
            promoted_z_new = z_sub[promoted_mask]
            promoted_z_norm = float(np.linalg.norm(
                promoted_z_new - promoted_z_old))
            promoted_new_max = (float(np.max(promoted_z_new))
                if promoted_z_new.size else 0.0)
            promoted_new_l1 = float(np.sum(promoted_z_new))
            z_scale_loc = float(np.max(z_sub)) + 1e-30
            newly_positive = int(np.count_nonzero(
                promoted_z_new > 1e-12 * z_scale_loc))

            near_noop = (
                did_explore
                and promoted_cols.size > 0
                and obj_gain <= promotion_noop_abs_tol
                and rel_gain <= promotion_noop_rel_tol
                and z_step_rel <= 1e-8
            )

            promoted_cols = surviving_promoted
            failed_cols = np.zeros((0,), dtype=np.int64)

        if did_explore:
            obj_tol = max(1e-10 * max(1.0, abs(obj_old)), 1e-14)

            if (not np.isfinite(obj_new)) or (obj_new > obj_old + obj_tol):
                print(
                    "[MONO][explore] rolling back batch: "
                    f"obj_old={obj_old:.3e} "
                    f"obj_new={obj_new:.3e} "
                    f"tol={obj_tol:.3e}",
                    flush=True)

                active[:] = active_before
                z[:] = z_before

                if newly_activated is not None and newly_activated.size > 0:
                    _cooldown_cols(newly_activated, it, extra_iters=20)
                    _cooldown_orbits(
                        np.unique(newly_activated // P), it, extra_iters=20,)

                positive_batch_size = max(2, positive_batch_size // 2)
                if _finish_iteration(it, "exploratory_rollback", z_iter_start,
                    active_iter_start, max_grad_dual=max_grad_dual,
                    tol_here=tol_here, best_dual_col=best_dual_col,
                    best_dual_value=best_dual_value):
                    break
                continue

        # Commit the matching equality multipliers only after the constrained
        # coefficient update has passed the feasibility and objective checks.
        if has_hard_orbit_shape:
            alpha_current = float(alpha_candidate)
            lambda_orbit_current = lambda_candidate.copy()
            ridge_current = float(
                ridge + orbit_constraint_info.get("mu", 0.0))
        else:
            ridge_current = float(ridge)

        if diag_level >= 1 and ((it % diag_stride) == 0):
            x_phys_now = _current_x_from_z(z).reshape(C, P)
            orbit_mass_vec = np.sum(x_phys_now, axis=1)
            if not has_hard_orbit_shape:
                orbit_target = np.zeros((C,), dtype=np.float64)
            else:
                orbit_target = float(alpha_current) * np.asarray(orbit_shape,
                    dtype=np.float64)
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
                    "source": "streamActiveSetNNLS",
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
                    "ridge": float(ridge_current),
                    "did_explore": bool(did_explore),
                    "reject": bool(reject),
                    "orbit_constraint_ok": bool(orbit_constraint_info["accepted"]),
                    "orbit_constraint_l1": float(orbit_constraint_info["resid_l1"]),
                    "orbit_constraint_l2": float(orbit_constraint_info["resid_l2"]),
                    "orbit_constraint_linf": float(orbit_constraint_info["resid_linf"]),
                    "n_promoted": int(promoted_cols.size),
                    "n_failed": int(failed_cols.size),
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
                    "alpha": (float(alpha_current) if has_hard_orbit_shape
                        else None),
                    "alpha_stationarity": (float(orbit_constraint_info.get(
                                "alpha_stationarity", 0.0))
                        if has_hard_orbit_shape else None),
                    "orbit_shape": (orbit_shape.tolist()
                        if orbit_shape is not None else None),
                    "x": x_phys_now.ravel(order="C").tolist(),
                }
            )

        if _finish_iteration(
            it, "accepted", z_iter_start, active_iter_start,
            max_grad_dual=max_grad_dual, tol_here=tol_here,
            best_dual_col=best_dual_col, best_dual_value=best_dual_value):
            break


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
        orbit_mass, orbit_target, orbit_deficit = _orbit_mass_and_deficit(z,
            alpha_value=alpha_current)
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
        f"stop={stop_reason} "
        f"category={stop_category} "
        f"converged={stop_converged} "
        f"iter="
        f"{((stop_iter + 1) if stop_iter is not None else 0)}/"
        f"{max_iter} "
        f"active={int(np.count_nonzero(active))}/"
        f"{int(max_active)} "
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
            "source": "streamActiveSetNNLS",
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
            "stop_reason": str(stop_reason),
            "stop_category": str(stop_category),
            "stop_converged": bool(stop_converged),
            "stop_iter": (
                int(stop_iter + 1)
                if stop_iter is not None
                else None
            ),
            "max_iter": int(max_iter),
            "n_active_final": int(
                np.count_nonzero(active)
            ),
            "max_active": int(max_active),
            "stop_details": dict(stop_details),
        }
    )
    _emit_diag(
        {
            "kind": "final_solution",
            "source": "streamActiveSetNNLS",
            "iter": int(final_it + 1),
            "x": x_final.ravel(order="C").tolist(),
            "x_sum": float(np.sum(x_final)),
            "x_norm": float(np.linalg.norm(x_final)),
            "x_max": float(np.max(x_final)),
            "x_nnz": int(np.count_nonzero(x_final > 0.0)),
            "alpha": (
                float(alpha_current)
                if has_hard_orbit_shape
                else None
            ),
            "orbit_mass": orbit_mass.tolist(),
            "orbit_target": orbit_target.tolist(),
            "orbit_resid": orbit_deficit.tolist(),
            "orbit_resid_linf": float(
                orbit_resid_linf
            ) if orbit_resid_linf is not None else None,
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
    x0: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    tracker=None,
    monolithic_max_active: int = 2000,
    regularisation_scale: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-9,
):
    """
    Solve the full CubeFit problem as a dense constrained NNLS reference.

    This solver assembles the complete scaled normal equations and solves the
    same nonnegative problem as ``streamActiveSetNNLS``, including the fixed
    orbit shape, fitted global amplitude, known-zero exclusions, and numerical
    ridge. It is intended as a small-C reference calculation.

    Parameters
    ----------
    h5_path : str
        Path to the CubeFit HDF5 file.
    cfg : MPConfig
        Parallelism and HDF5 configuration.
    orbit_weights : ndarray, optional
        Orbit weights defining the fixed orbit shape.
    x0 : ndarray, optional
        Physical-space starting solution with shape ``(C, P)``. The dense
        reference solve does not depend on this starting point.
    resume_state : dict, optional
        Streaming-solver resume state. Full-state resume is not supported.
    tracker : optional
        Accepted for API compatibility with ``solve_streaming_nnls``.
    monolithic_max_active : int, optional
        Accepted for API compatibility. The dense reference uses all allowed
        columns and therefore has no outer active-set size limit.
    regularisation_scale : float, optional
        Multiplicative scale applied to the numerical ridge. Zero disables
        the ridge.
    max_iter : int, optional
        Maximum iterations for the unconstrained fallback NNLS solve.
    tol : float, optional
        Convergence tolerance for the unconstrained fallback NNLS solve.

    Returns
    -------
    x : ndarray
        Physical-space solution with shape ``(C, P)``.
    stats : dict
        Reference-solver diagnostics.

    Raises
    ------
    RuntimeError
        If the dense constrained solve fails.
    ValueError
        If the inputs or hard orbit constraints are inconsistent.

    Examples
    --------
    >>> x, stats = monolithic_nnls_scipy(
    ...     h5_path, cfg, orbit_weights=orbit_weights,
    ...     regularisation_scale=0.0)
    """
    if resume_state:
        raise ValueError(
            "monolithic_nnls_scipy does not support full-state resume; "
            "use zeros or saved_x.")

    del tracker, monolithic_max_active, x0

    regularisation_scale = float(regularisation_scale)
    if (not np.isfinite(regularisation_scale)
            or regularisation_scale < 0.0):
        raise ValueError(
            "regularisation_scale must be nonnegative and finite.")

    t0 = time.perf_counter()

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]
        S, L = map(int, DC.shape)
        _, C, P, Lm = map(int, M.shape)

        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch.")

        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        model_chunks = M.chunks

    CP = int(C * P)
    known_zero = _get_known_zero_mask(h5_path, C, P)
    known_zero_flat = known_zero.ravel(order="C")

    orbit_shape = _canon_orbit_weights(
        h5_path, orbit_weights, C=C, P=P)
    has_hard_orbit_shape = (
        orbit_shape is not None and np.any(orbit_shape > 0.0))

    if has_hard_orbit_shape:
        orbit_shape = np.asarray(orbit_shape, dtype=np.float64)
        blocked = np.flatnonzero(
            np.all(known_zero, axis=1) & (orbit_shape > 0.0))

        if blocked.size:
            raise ValueError(
                "known_zero_mask makes the hard orbit constraint "
                f"infeasible for positive-weight orbits {blocked.tolist()}.")

    Lk = int(keep_idx.size) if keep_idx is not None else L
    s_tile = int(
        cfg.s_tile_override
        or (model_chunks[0] if model_chunks and model_chunks[0] else 128))
    s_ranges = [
        (s0, min(S, s0 + s_tile)) for s0 in range(0, S, s_tile)]

    # Match the streaming solver's per-column energy normalization.
    D_tot = np.zeros((C, P), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]

        for s0, s1 in tqdm(
                s_ranges, desc="[REF-NNLS] scaling",
                disable=not getattr(cfg, "verbose", True)):
            M_tile = np.asarray(
                M[s0:s1, :, :, :], dtype=np.float64, order="C")

            if keep_idx is not None:
                M_tile = M_tile[:, :, :, keep_idx]

            D_tot += np.sum(M_tile * M_tile, axis=(0, 3))

    n_tiles = max(1, len(s_ranges))
    col_energy = D_tot / float(n_tiles)
    positive = col_energy > 0.0

    energy_med = (
        float(np.median(col_energy[positive])) if np.any(positive) else 1.0)
    energy_floor = max(1e-30, energy_med * 1e-2)
    col_energy = np.maximum(col_energy, energy_floor)

    S_temp = 1.0 / np.sqrt(col_energy)
    S_median = float(np.median(S_temp))
    S_cap = 8.0 * max(1.0, S_median)
    S_temp = np.minimum(S_temp, S_cap)
    S_flat = S_temp.ravel(order="C")

    if has_hard_orbit_shape:
        orbit_of_col = np.arange(CP, dtype=np.int64) // P
        allowed = ~known_zero_flat
        allowed &= orbit_shape[orbit_of_col] > 0.0
    else:
        allowed = ~known_zero_flat

    free_idx = np.flatnonzero(allowed).astype(np.int64)

    if free_idx.size == 0:
        raise RuntimeError(
            "Dense reference solve has no free coefficients.")

    n_free = int(free_idx.size)
    S_active = S_flat[free_idx]

    # Assemble the complete scaled normal equations using the same worker as
    # the streaming solver.
    ATA = np.zeros((n_free, n_free), dtype=np.float64)
    ATy = np.zeros((n_free,), dtype=np.float64)

    n_workers = max(1, int(cfg.processes))
    batches = [[] for _ in range(n_workers)]
    for i, tile in enumerate(s_ranges):
        batches[i % n_workers].append(tile)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx,
            initializer=_init_worker,
            initargs=(cfg.blas_threads,)) as executor:
        futures = [
            executor.submit(
                _worker_reduced, h5_path, batch, keep_idx, free_idx,
                S_active, CP, P)
            for batch in batches if batch]

        for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="[REF-NNLS] normal equations",
                disable=not getattr(cfg, "verbose", True)):
            ATA_loc, ATy_loc = future.result()
            ATA += ATA_loc
            ATy += ATy_loc

    diag = np.diag(ATA)
    diag_med = float(np.median(diag)) if diag.size else 0.0

    try:
        eigs = np.linalg.eigvalsh(ATA)
        emin = float(np.min(eigs))
        emax = float(np.max(eigs))
    except Exception:
        emin = 0.0
        emax = float(np.max(diag)) if diag.size else 1.0

    cond_bad = emin <= 1e-10 * max(emax, 1.0)
    ridge_rel = regularisation_scale * (
        1e-2 if cond_bad else 1e-3)
    ridge = (0.0 if regularisation_scale == 0.0 else
        max(1e-10, ridge_rel * max(1.0, diag_med)))

    H = ATA + ridge * np.eye(n_free, dtype=np.float64)

    if has_hard_orbit_shape:
        z_free, ref_info = _orbit_hard_constraint_solve(
            ATA_sub_reg=H,
            ATy_sub=ATy,
            active_idx=free_idx,
            S_active=S_active,
            orbit_shape=np.asarray(orbit_shape, dtype=np.float64),
            C=C,
            P=P,
            regularisation_scale=regularisation_scale)

        if not ref_info["accepted"]:
            raise RuntimeError(
                "Dense reference constrained NNLS failed: "
                f"{ref_info['reason']}")

        z_free = np.asarray(z_free, dtype=np.float64)
        alpha = float(ref_info["alpha"])
        mu = float(ref_info.get("mu", 0.0))
        ridge_total = float(ridge + mu)

        orbit_mass = np.bincount(
            free_idx // P, weights=S_active * z_free,
            minlength=C).astype(np.float64)
        orbit_target = alpha * orbit_shape
        orbit_resid = orbit_mass - orbit_target
        orbit_l1 = float(np.sum(np.abs(orbit_resid)))
        orbit_linf = float(np.max(np.abs(orbit_resid)))
        constraint_violation = float(ref_info["resid_linf"])
        solve_message = "dense constrained active-set KKT solve converged"
        solve_iterations = int(ref_info.get("n_free", n_free))
    else:
        mu = 0.0
        ridge_total = float(ridge)
        z_free = _nnls_from_quadratic(
            H, ATy, max_iter=max_iter, tol=tol)
        z_free = np.asarray(z_free, dtype=np.float64)

        alpha = None
        orbit_mass = np.bincount(
            free_idx // P, weights=S_active * z_free,
            minlength=C).astype(np.float64)
        orbit_target = None
        orbit_resid = None
        orbit_l1 = None
        orbit_linf = None
        constraint_violation = 0.0
        solve_message = "dense projected-gradient NNLS solve completed"
        solve_iterations = None

    np.maximum(z_free, 0.0, out=z_free)

    z_full = np.zeros((CP,), dtype=np.float64)
    z_full[free_idx] = z_free

    x_flat = S_flat * z_full
    x_flat[known_zero_flat] = 0.0
    x = x_flat.reshape(C, P)

    data_objective = (
        0.5 * float(z_free @ (ATA @ z_free))
        - float(ATy @ z_free))
    total_objective = (
        data_objective
        + 0.5 * ridge_total * float(np.dot(z_free, z_free)))

    z_scale = max(1.0, float(np.max(np.abs(z_free))))
    positive_tol = 1e-12 * z_scale
    positive = z_free > positive_tol

    if has_hard_orbit_shape:
        lambda_orbit = np.asarray(
            ref_info["lambda_orbit"], dtype=np.float64)
        grad = (
            ATy - ATA @ z_free - ridge_total * z_free
            - S_active * lambda_orbit[free_idx // P])
        max_grad_active = (
            float(np.max(np.abs(grad[positive])))
            if np.any(positive) else 0.0)
        max_grad_inactive = (
            float(np.max(grad[~positive]))
            if np.any(~positive) else -np.inf)
        max_grad_dual = max(max_grad_inactive, 0.0)
    else:
        grad = ATy - ATA @ z_free - ridge_total * z_free
        max_grad_active = (
            float(np.max(np.abs(grad[positive])))
            if np.any(positive) else 0.0)
        max_grad_inactive = (
            float(np.max(grad[~positive]))
            if np.any(~positive) else -np.inf)
        max_grad_dual = max(max_grad_inactive, 0.0)

    elapsed = float(time.perf_counter() - t0)

    stats = {
        "elapsed_sec": elapsed,
        "rows": int(S * Lk),
        "cols": int(CP),
        "n_free": int(n_free),
        "n_active": int(np.count_nonzero(positive)),
        "x_nnz": int(np.count_nonzero(x > 0.0)),
        "x_sum": float(np.sum(x)),
        "x_norm": float(np.linalg.norm(x)),
        "x_max": float(np.max(x)),
        "data_objective": float(data_objective),
        "total_objective": float(total_objective),
        "ridge": float(ridge),
        "mu": float(mu),
        "ridge_total": float(ridge_total),
        "regularisation_scale": float(regularisation_scale),
        "alpha": alpha,
        "orbit_mass": orbit_mass,
        "orbit_target": orbit_target,
        "orbit_resid": orbit_resid,
        "orbit_resid_l1": orbit_l1,
        "orbit_resid_linf": orbit_linf,
        "constraint_violation": float(constraint_violation),
        "max_grad_active": float(max_grad_active),
        "max_grad_inactive": float(max_grad_inactive),
        "max_grad_dual": float(max_grad_dual),
        "success": True,
        "status": 0,
        "message": solve_message,
        "iterations": solve_iterations,
        "known_zero_mask": known_zero.copy(),
    }

    print(
        f"[REF-NNLS] done in {elapsed:.1f}s "
        f"active={stats['n_active']}/{CP} "
        f"x_sum={stats['x_sum']:.6e} "
        f"data_obj={data_objective:.6e} "
        f"ridge={ridge_total:.6e} "
        f"dual={max_grad_dual:.3e}",
        flush=True)

    return x, stats

# ------------------------------------------------------------------------------

def monolithicNNLS(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    tracker=None,
    monolithic_max_active: int = 2000,
    regularisation_scale: float = 0.0,
    max_iter: int = 2000,
    tol: float = 1e-9,
):
    """
    Solve the literal dense CubeFit constrained least-squares problem.

    This reference solver explicitly constructs the physical-space design
    matrix ``A`` and data vector ``b`` and minimizes

        0.5 * ||A x - b||**2 + 0.5 * ridge * ||x||**2

    subject to non-negativity and, when supplied, the fixed orbit-shape
    constraint

        sum_p x[c, p] = alpha * orbit_shape[c].

    Unlike ``solve_streaming_nnls``, this function does not use normal
    equations, column scaling, streamed ATA products, reduced active sets, or
    the custom constrained KKT solver. It is intended only as an independent
    small-C reference calculation.

    Parameters
    ----------
    h5_path : str
        Path to the CubeFit HDF5 file.
    cfg : MPConfig
        Solver configuration. ``apply_mask`` controls wavelength masking.
    orbit_weights : ndarray, optional
        Orbit weights defining the fixed unit-sum orbit shape.
    x0 : ndarray, optional
        Physical-space initial solution with shape ``(C, P)``.
    resume_state : dict, optional
        Accepted for API compatibility. Full-state resume is not supported.
    tracker : optional
        Accepted for API compatibility and ignored.
    monolithic_max_active : int, optional
        Accepted for API compatibility and ignored.
    regularisation_scale : float, optional
        Physical-space L2 ridge strength. Zero gives the literal unregularized
        constrained least-squares problem.
    max_iter : int, optional
        Maximum number of optimizer iterations.
    tol : float, optional
        Optimizer convergence tolerance.

    Returns
    -------
    x : ndarray, shape (C, P)
        Physical-space nonnegative solution.
    stats : dict
        Reference-solver diagnostics.

    Raises
    ------
    RuntimeError
        If SciPy is unavailable, the dense matrix is too large, or the
        constrained optimization fails.
    ValueError
        If inputs or hard orbit constraints are inconsistent.

    Examples
    --------
    >>> x, stats = monolithicNNLS(
    ...     h5_path, cfg, orbit_weights=orbit_weights,
    ...     regularisation_scale=0.0)
    """
    if not _HAS_SCIPY_OPTIMIZE:
        raise RuntimeError(
            "SciPy optimize is required for monolithicNNLS.")

    if resume_state:
        raise ValueError(
            "monolithicNNLS does not support full-state resume; "
            "use zeros or saved_x.")

    del tracker, monolithic_max_active

    regularisation_scale = float(regularisation_scale)
    if (not np.isfinite(regularisation_scale)
            or regularisation_scale < 0.0):
        raise ValueError(
            "regularisation_scale must be nonnegative and finite.")

    t0 = time.perf_counter()

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        if DC.ndim != 2 or M.ndim != 4:
            raise RuntimeError(
                "Expected /DataCube (S,L) and /HyperCube/models (S,C,P,L).")

        S, L = map(int, DC.shape)
        Sm, C, P, Lm = map(int, M.shape)

        if Sm != S or Lm != L:
            raise RuntimeError(
                "Model and data dimensions are inconsistent.")

        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None

    CP = int(C * P)
    Lk = int(keep_idx.size) if keep_idx is not None else L
    n_rows = int(S * Lk)

    known_zero = _get_known_zero_mask(h5_path, C, P)
    known_zero_flat = known_zero.ravel(order="C")

    orbit_shape = _canon_orbit_weights(
        h5_path, orbit_weights, C=C, P=P)
    has_hard_orbit_shape = (
        orbit_shape is not None and np.any(orbit_shape > 0.0))

    if has_hard_orbit_shape:
        orbit_shape = np.asarray(orbit_shape, dtype=np.float64)
        positive_orbits = np.flatnonzero(orbit_shape > 0.0)
        blocked = np.flatnonzero(
            np.all(known_zero, axis=1) & (orbit_shape > 0.0))

        if blocked.size:
            raise ValueError(
                "known_zero_mask makes the hard orbit constraint "
                f"infeasible for positive-weight orbits {blocked.tolist()}.")
    else:
        positive_orbits = np.arange(C, dtype=np.int64)

    orbit_of_col = np.arange(CP, dtype=np.int64) // P
    allowed = ~known_zero_flat

    if has_hard_orbit_shape:
        allowed &= orbit_shape[orbit_of_col] > 0.0

    free_idx = np.flatnonzero(allowed).astype(np.int64)
    n_free = int(free_idx.size)

    if n_free == 0:
        raise RuntimeError(
            "monolithicNNLS has no allowed coefficient columns.")

    # This is intentionally expensive: construct the literal design matrix.
    required_bytes = int(n_rows * n_free * np.dtype(np.float64).itemsize)
    required_gib = required_bytes / 1024.0**3

    print(
        f"[MONOLITHIC] constructing A={n_rows}x{n_free} "
        f"({required_gib:.2f} GiB float64)",
        flush=True)

    A = np.empty((n_rows, n_free), dtype=np.float64)
    b = np.empty((n_rows,), dtype=np.float64)

    free_c = free_idx // P
    free_p = free_idx % P

    model_chunks = None
    with open_h5(h5_path, role="reader") as f:
        model_chunks = f["/HyperCube/models"].chunks

    s_tile = int(
        cfg.s_tile_override
        or (model_chunks[0] if model_chunks and model_chunks[0] else 128))

    row0 = 0

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        for s0 in tqdm(
                range(0, S, s_tile), desc="[MONOLITHIC] building A"):
            s1 = min(S, s0 + s_tile)
            Y = np.asarray(
                DC[s0:s1, :], dtype=np.float64, order="C")

            if keep_idx is not None:
                Y = Y[:, keep_idx]

            n_tile_rows = int(Y.size)
            row1 = row0 + n_tile_rows
            b[row0:row1] = Y.reshape(-1)

            for j, (cc, pp) in enumerate(zip(free_c, free_p)):
                col = np.asarray(
                    M[s0:s1, cc, pp, :], dtype=np.float64, order="C")

                if keep_idx is not None:
                    col = col[:, keep_idx]

                A[row0:row1, j] = col.reshape(-1)

            row0 = row1

    if row0 != n_rows:
        raise RuntimeError(
            f"Design-matrix row count mismatch: {row0} != {n_rows}.")

    if not np.all(np.isfinite(A)):
        raise RuntimeError(
            "Literal design matrix contains non-finite values.")

    if not np.all(np.isfinite(b)):
        raise RuntimeError(
            "Literal data vector contains non-finite values.")

    # The reference ridge is deliberately defined in physical x space. For an
    # exact comparison to the unregularized streaming problem, use scale=0.
    col_energy = np.sum(A * A, axis=0)
    positive_energy = col_energy[
        np.isfinite(col_energy) & (col_energy > 0.0)]
    energy_ref = (
        float(np.median(positive_energy))
        if positive_energy.size else 1.0)
    ridge = regularisation_scale * 1e-2 * max(1.0, energy_ref)

    if regularisation_scale == 0.0:
        ridge = 0.0

    if has_hard_orbit_shape:
        n_constraints = int(positive_orbits.size)
        n_var = n_free + 1
        B = np.zeros((n_constraints, n_var), dtype=np.float64)
        orbit_to_row = {
            int(cc): row for row, cc in enumerate(positive_orbits)}

        for j, cc in enumerate(free_c):
            B[orbit_to_row[int(cc)], j] = 1.0

        B[:, n_free] = -orbit_shape[positive_orbits]
        zero_constraint = np.zeros(n_constraints, dtype=np.float64)
        constraints = [
            LinearConstraint(B, zero_constraint, zero_constraint)]
    else:
        n_var = n_free
        B = None
        constraints = []

    bounds = Bounds(
        np.zeros(n_var, dtype=np.float64),
        np.full(n_var, np.inf, dtype=np.float64))

    def _objective(u):
        x_free = u[:n_free]
        resid = A @ x_free - b
        return (0.5 * float(np.dot(resid, resid))
            + 0.5 * ridge * float(np.dot(x_free, x_free)))

    def _gradient(u):
        x_free = u[:n_free]
        resid = A @ x_free - b
        grad = np.zeros((n_var,), dtype=np.float64)
        grad[:n_free] = A.T @ resid + ridge * x_free
        return grad

    u0 = np.zeros((n_var,), dtype=np.float64)

    if x0 is not None:
        x0_flat = np.asarray(
            x0, dtype=np.float64).ravel(order="C")

        if x0_flat.size != CP:
            raise ValueError(
                f"x0 has size {x0_flat.size}; expected {CP}.")

        x0_flat = np.maximum(x0_flat, 0.0)
        x0_flat[known_zero_flat] = 0.0
        x0_free = x0_flat[free_idx].copy()
    else:
        x0_free = np.zeros((n_free,), dtype=np.float64)

    if has_hard_orbit_shape:
        free_orbits = free_idx // P

        if np.any(x0_free > 0.0):
            alpha0 = float(np.sum(x0_free))
        else:
            # Obtain a data-informed feasible seed without solving the
            # optimization problem.
            alpha0 = 1.0

        for cc in positive_orbits:
            cc = int(cc)
            local = np.flatnonzero(free_orbits == cc)

            if local.size == 0:
                raise ValueError(
                    f"Orbit {cc} has no allowed population columns.")

            target = alpha0 * float(orbit_shape[cc])
            current = float(np.sum(x0_free[local]))

            if current > 0.0:
                x0_free[local] *= target / current
            else:
                # Choose the column most correlated with the literal data.
                corr = A[:, local].T @ b
                best = int(local[int(np.argmax(corr))])
                x0_free[local] = 0.0
                x0_free[best] = target

        u0[:n_free] = x0_free
        u0[n_free] = alpha0
    else:
        u0[:n_free] = x0_free

    print(
        f"[MONOLITHIC] solving constrained least squares: "
        f"rows={n_rows} free={n_free} ridge={ridge:.6e}",
        flush=True)

    result = minimize(_objective, u0, jac=_gradient, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol),
            "disp": bool(getattr(cfg, "verbose", True))})

    x_free = np.maximum(
        np.asarray(result.x[:n_free], dtype=np.float64), 0.0)

    x_flat = np.zeros((CP,), dtype=np.float64)
    x_flat[free_idx] = x_free
    x_flat[known_zero_flat] = 0.0
    x = x_flat.reshape(C, P)

    resid = A @ x_free - b
    residual_ss = float(np.dot(resid, resid))
    rmse = float(np.sqrt(residual_ss / max(1, n_rows)))
    data_objective = 0.5 * residual_ss
    ridge_objective = 0.5 * ridge * float(np.dot(x_free, x_free))
    total_objective = data_objective + ridge_objective

    if has_hard_orbit_shape:
        alpha = float(result.x[n_free])
        orbit_mass = np.sum(x, axis=1)
        orbit_target = alpha * orbit_shape
        orbit_resid = orbit_mass - orbit_target
        orbit_l1 = float(np.sum(np.abs(orbit_resid)))
        orbit_linf = float(np.max(np.abs(orbit_resid)))
        constraint_violation = float(np.max(np.abs(B @ result.x)))
    else:
        alpha = None
        orbit_mass = np.sum(x, axis=1)
        orbit_target = None
        orbit_resid = None
        orbit_l1 = None
        orbit_linf = None
        constraint_violation = 0.0

    x_scale = max(1.0, float(np.max(np.abs(x_free))))
    positive_tol = 1e-12 * x_scale
    positive = x_free > positive_tol

    elapsed = float(time.perf_counter() - t0)

    stats = {
        "elapsed_sec": elapsed,
        "rows": int(n_rows),
        "cols": int(CP),
        "n_free": int(n_free),
        "n_active": int(np.count_nonzero(positive)),
        "x_nnz": int(np.count_nonzero(x > 0.0)),
        "x_sum": float(np.sum(x)),
        "x_norm": float(np.linalg.norm(x)),
        "x_max": float(np.max(x)),
        "rmse": float(rmse),
        "residual_ss": float(residual_ss),
        "data_objective": float(data_objective),
        "ridge_objective": float(ridge_objective),
        "total_objective": float(total_objective),
        "ridge": float(ridge),
        "regularisation_scale": float(regularisation_scale),
        "alpha": alpha,
        "orbit_mass": orbit_mass,
        "orbit_target": orbit_target,
        "orbit_resid": orbit_resid,
        "orbit_resid_l1": orbit_l1,
        "orbit_resid_linf": orbit_linf,
        "constraint_violation": float(constraint_violation),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "known_zero_mask": known_zero.copy(),
    }

    print(
        f"[MONOLITHIC] done in {elapsed:.1f}s "
        f"success={result.success} active={stats['n_active']}/{CP} "
        f"x_sum={stats['x_sum']:.6e} rmse={rmse:.6e} "
        f"orbit_Linf={constraint_violation:.3e}",
        flush=True)

    if not result.success:
        raise RuntimeError(
            "Literal monolithic constrained NNLS failed: "
            f"{result.message}")

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
    regularisation_scale: float = 1.0,
    cache_path: str | None = None,
    reuse_cache: bool = True,
):
    """
    Fused single-pass block-coordinate NNLS solver (BLAS-friendly).

    - Accumulates ATA and ATy for each block during one streaming pass (no repeated HDF5 reads per block).
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
            "source": "solve_streaming_nnls",
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

    # ------------------ persistent masks: known_zero & seed masks --------------
    known_zero = _get_known_zero_mask(h5_path, C, P)
    known_zero_orbit = np.all(known_zero, axis=1)

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
                    yf_norm = (float(np.linalg.norm(y_flat)) if y_flat.size > 0
                        else 0.0)
                    yhatf_norm = (float(np.linalg.norm(yhat_tile)) if
                        yhat_tile.size > 0 else 0.0)
                    r_norm = (
                        float(np.linalg.norm(y_flat - yhat_tile.reshape(-1)))
                        if y_flat.size > 0 else 0.0)

                    nnans = int(np.count_nonzero(~np.isfinite(y_flat)))
                    nnans += int(np.count_nonzero(~np.isfinite(yhat_tile)))
                    nnans += int(np.count_nonzero(
                        ~np.isfinite(y_flat - yhat_tile.reshape(-1))))

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

    executor = ProcessPoolExecutor(max_workers=n_workers,
        mp_context=ctx, initializer=_init_worker, initargs=(cfg.blas_threads,))
    # `initargs` must be one-lement tuple

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
            tracker.save_checkpoint(latest_ckpt["x"], state, block=True)
        except Exception as exc:
            if not checkpoint_error_logged:
                print(f"[MONO][checkpoint] save_checkpoint failed: {exc}",
                    flush=True)
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
            known_zero_mask=known_zero,
            x0_flat=x.ravel(order="C"),
            resume_state=resume_state,
            max_active=monolithic_max_active,
            tol_grad=1e-8,
            max_iter=5 * monolithic_max_active,
            regularisation_scale=regularisation_scale,
            checkpoint_cb=_checkpoint_cb,
            checkpoint_every=checkpoint_every,
        )

        x = x_flat_unscaled.reshape(C, P).copy()
        print("[BC-FUSED][MONO] finished streaming active-set NNLS",
            flush=True)

        best_x = x.copy()
        best_proxy = np.inf

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

    # -------------------- final diagnostics & best-x --------------------
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

    # --- DIAG: fianl summary (timings & numerical health) ---
    try:
        elapsed = time.perf_counter() - t0
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
        print(f"[BC-FUSED][summary] elapsed_total={elapsed:.1f}s", flush=True)
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
            print("[BC-FUSED][summary] ALERT: x is entirely zero!", flush=True)
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
                        "kind": "final_orbit_table",
                        "source": "solve_streaming_nnls",
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

        print("=== [BC-FUSED][final-summary] end ===", flush=True)
    except Exception as _e:
        print("[BC-FUSED][final-summary] error:", _e, flush=True)
    # --- end final summary diagnostics ---

    # final diagnostics
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
                "source": "solve_streaming_nnls",
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
                "source": "solve_streaming_nnls",
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
    elapsed = time.perf_counter() - t0

    stats = dict(
        elapsed_sec=elapsed,
        rmse_proxy_best=float(best_proxy),
        regularisation_scale=float(regularisation_scale),
        known_zero_mask=known_zero.copy(),
    )

    return best_x, stats

# ------------------------------------------------------------------------------