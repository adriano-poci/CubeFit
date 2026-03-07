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

def _worker_block_nnls(
    args: Tuple[str, int, int, int, Tuple[Tuple[int,int],...],
                Optional[np.ndarray], Optional[np.ndarray],
                int, int, float, int, int]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Worker that builds a single block's design matrix by streaming tiles,
    forms RHS (b_rhs = r + A_block x_block_old) and solves NNLS for the
    block. Returns (z_block, cols_block, block_idx).
    `args` contains:
       (h5_path, block_idx, c0, c1, s_ranges, keep_idx, dset_slots,
        dset_bytes, dset_w0, C, P)
    Note: this top-level function is picklable for multiprocessing.
    """
    (h5_path, block_idx, c0, c1, s_ranges, keep_idx,
     dset_slots, dset_bytes, dset_w0, C, P) = args

    # Open HDF5 locally (each worker)
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]
        try:
            M.id.set_chunk_cache(dset_slots, dset_bytes, dset_w0)
        except Exception:
            pass

        # Determine kept wavelengths
        L = int(DC.shape[1])
        Lk = int(np.flatnonzero(cu._get_mask(f)).size) if ('/Mask' in f and keep_idx is None) else (int(keep_idx.size) if keep_idx is not None else L)
        # Build list of global column indices for this block
        cols = np.arange(c0, c1, dtype=np.int32)
        ncols = cols.size

        # Build block A as (S_total * Lk, ncols)
        rows_list = []
        y_list = []
        for (s0, s1) in s_ranges:
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            rows = s1 - s0
            # For this tile, stack A_tile columns contiguous in col axis
            # A_tile shape (rows, ncols, Lk) -> transpose -> (rows*Lk, ncols)
            A_tile_cols = np.empty((rows * (Yt.shape[1]), ncols), dtype=np.float64)
            for j, gc in enumerate(cols):
                # read M[s0:s1, gc // P, gc % P, :]
                cc = int(gc // P)
                pp = int(gc % P)
                A_slice = np.asarray(M[s0:s1, cc, pp, :], dtype=np.float64, order="C")  # shape (rows, Lk)
                if keep_idx is not None and A_slice.ndim == 2:
                    # already restricted during read above if needed
                    pass
                A_tile_cols[:, j] = A_slice.reshape(-1)
            rows_list.append(A_tile_cols)
            y_list.append(Yt.reshape(-1))
        A_block = np.vstack(rows_list) if rows_list else np.zeros((0, ncols), dtype=np.float64)
        y_flat = np.concatenate(y_list) if y_list else np.zeros((0,), dtype=np.float64)

    # Solve NNLS for block given current x_block_old. We can't access the
    # global x here; instead main process should pass b_rhs. But to keep the
    # worker self-contained (and avoid sending huge rhs), we solve the pure
    # NNLS for y_flat ~ A_block z (equivalent to a seed solve). To apply the
    # correct block-coordinate update in the main process, we instead return
    # A_block and let main process form b_rhs and solve. However returning
    # A_block is heavy (big pickles). To avoid that, we proceed with this
    # pattern: workers assemble A_block and y_flat and compute ATA, ATy and
    # return them to the master. The master then computes the block NNLS.
    # This keeps per-worker I/O local and reduces pickling size (ATA small).
    # Compute ATA and ATy
    ATA = A_block.T @ A_block  # (ncols, ncols)
    ATy = A_block.T @ y_flat   # (ncols,)

    return ATA, ATy, int(block_idx)

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

def solve_block_coord_nnls(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    tracker: Optional[object] = None,
    block_size: Optional[int] = None,
    orbit_beta: float = 0.0,   # soft prior strength (per-request)
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

    # pool only used to set BLAS threads per worker if you later parallelize
    ctx = mp.get_context(os.environ.get("CUBEFIT_MP_CTX", "forkserver"))
    pool = ctx.Pool(processes=max(1, int(cfg.processes)), initializer=_worker_init, initargs=(int(cfg.blas_threads),))

    try:
        best_x = x.copy()
        best_proxy = np.inf

        # epoch loop
        for ep in range(cfg.epochs):
            print(f"[BC-FUSED] epoch {ep+1}/{cfg.epochs}", flush=True)

            # Prepare accumulators for this epoch
            # D_tot per column (C,P)
            D_tot = np.zeros((C, P), dtype=np.float64)
            # ATA and ATy per block
            ATA_blocks = {bi: np.zeros((meta["ncols"], meta["ncols"]), dtype=np.float64)
                          for bi, meta in block_meta.items()}
            ATy_blocks = {bi: np.zeros((meta["ncols"],), dtype=np.float64)
                          for bi, meta in block_meta.items()}

            # Build a quick lookup: for each orbit cc, which blocks & local indices
            # orbit_block_map[cc] = list of (bi, local_col_indices, p_indices)
            orbit_block_map = {cc: [] for cc in range(C)}
            for bi, meta in block_meta.items():
                cols = meta["cols"]
                # for each unique orbit in this block, collect local indices and p's
                unique_ccs = {}
                for local_j, gc in enumerate(cols):
                    cc = int(gc // P)
                    p = int(gc % P)
                    if cc not in unique_ccs:
                        unique_ccs[cc] = {"local": [], "p": []}
                    unique_ccs[cc]["local"].append(local_j)
                    unique_ccs[cc]["p"].append(p)
                for cc, d in unique_ccs.items():
                    orbit_block_map[cc].append((bi, np.asarray(d["local"], dtype=np.int64), np.asarray(d["p"], dtype=np.int64)))

            # We'll stream tiles once: compute yhat, D_tot, and update ATA/ATy
            y_parts = []
            with open_h5(h5_path, role="reader") as f:
                DC = f["/DataCube"]
                M  = f["/HyperCube/models"]

                try:
                    M.id.set_chunk_cache(cfg.dset_slots, cfg.dset_bytes, cfg.dset_w0)
                except Exception:
                    pass

                # tile loop
                tile_iter = s_ranges
                if verbose and (len(s_ranges) > 1):
                    tile_iter = tqdm(s_ranges, desc=f"[BC-FUSED] tiles ep{ep+1}", disable=not verbose)

                for (s0, s1) in tile_iter:
                    Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")
                    if keep_idx is not None:
                        Yt = Yt[:, keep_idx]
                    Sblk = s1 - s0
                    Lk = Yt.shape[1]
                    yhat_tile = np.zeros((Sblk, Lk), dtype=np.float64)

                    # cache A_cc for this tile for reuse (only for cc that appear in any block)
                    A_cache = {}

                    # 1) build yhat_tile & D_tot incrementally
                    for cc in range(C):
                        # skip if entire orbit marked known-zero for speed
                        if np.all(known_zero[cc, :]):
                            continue

                        A_cc = np.asarray(M[s0:s1, cc, :, :], dtype=np.float64, order="C")  # (Sblk, P, L)
                        if keep_idx is not None:
                            A_cc = A_cc[:, :, keep_idx]  # (Sblk, P, Lk)

                        # accumulate prediction
                        # x[cc] shape (P,), A_cc shape (Sblk, P, Lk) -> x[cc] @ A_cc -> (Sblk, Lk)
                        yhat_tile += x[cc] @ A_cc

                        # accumulate curvature
                        D_tot[cc] += np.sum(A_cc * A_cc, axis=(0, 2))

                        # store slices only if this orbit has columns in at least one block
                        if orbit_block_map.get(cc):
                            A_cache[cc] = A_cc  # keep reference for block contributions

                    # residual for this tile (flattened)
                    y_flat = Yt.reshape(-1)
                    yhat_flat = yhat_tile.reshape(-1)
                    r_flat = y_flat - yhat_flat

                    # 2) update ATA/ATy for each block using cached A_cache (no reread)
                    # For each cached orbit, push contributions to all blocks that include that orbit
                    for cc, A_cc in A_cache.items():
                        # A_cc: (Sblk, P, Lk) -> reshape columns per population to flat rows
                        # We'll extract only p's that belong to block local columns
                        for (bi, local_idx_arr, p_idx_arr) in orbit_block_map.get(cc, ()):
                            # build A_tile_cols with shape (Sblk*Lk, n_local)
                            # carefully handle if p_idx_arr empty
                            if local_idx_arr.size == 0:
                                continue
                            # Extract columns for populations p_idx_arr
                            # A_cc[:, p_idx_arr, :] -> shape (Sblk, n_local, Lk)
                            sub = A_cc[:, p_idx_arr, :]  # (Sblk, n_local, Lk)
                            # reshape to (Sblk*Lk, n_local)
                            A_tile = sub.transpose(1, 0, 2).reshape(sub.shape[1], -1).T  # (Sblk*Lk, n_local)
                            # update ATA and ATy
                            ATA_blocks[bi][np.ix_(local_idx_arr, local_idx_arr)] += (A_tile.T @ A_tile)
                            ATy_blocks[bi][local_idx_arr] += (A_tile.T @ r_flat)

                    # keep y parts if you need full y vector for diagnostics (optional)
                    y_parts.append(y_flat)

            # end tile-streaming

            # Build full flattened y and residual norm if needed
            if len(y_parts) > 0:
                y_full = np.concatenate(y_parts)
                # compute global residual norm from ATA/ATy & x if desired later
            else:
                y_full = np.zeros((0,), dtype=np.float64)

            # -------------------- per-block solves ----------------------
            # Solve small quadratic NNLS using ATA_blocks / ATy_blocks (with soft prior augmentation)
            # We treat blocks sequentially and update x in a Jacobi fashion (each block uses residual
            # from start-of-epoch prediction).
            x_flat = x.ravel(order="C")
            # optional block progress bar
            block_iter = list(block_meta.items())
            if verbose:
                block_iter = tqdm(block_iter,
                    desc=f"[BC-FUSED] solve blocks ep{ep+1}",
                    disable=not verbose)

            for bi, meta in block_iter:
                ATA = ATA_blocks[bi]
                ATy = ATy_blocks[bi].copy()  # (ncols,)

                # Soft-orbit augmentation (if requested)
                if (orbit_beta is not None) and (orbit_beta > 0.0) and (meta["w_vec"] is not None):
                    # For block columns, ones vector u (length ncols) with ones per column
                    u = np.ones((meta["ncols"],), dtype=np.float64)
                    # ATA += 2 * orbit_beta * (u u^T)
                    ATA = ATA + (2.0 * float(orbit_beta)) * np.outer(u, u)
                    # ATy += 2 * orbit_beta * w_vec
                    ATy = ATy + (2.0 * float(orbit_beta)) * meta["w_vec"]

                # initial x for block: current x values
                x_block_init = x_flat[meta["cols"] - 0]  # slice (works because cols are contiguous)
                # solve quadratic NNLS
                try:
                    z_block = _nnls_from_quadratic(ATA, ATy, x0=x_block_init, max_iter=2000, tol=1e-8)
                except Exception:
                    # robust fallback small iterations
                    z_block = _nnls_from_quadratic(ATA, ATy, x0=x_block_init, max_iter=500, tol=1e-6)

                # place result into x
                for j_local, gc in enumerate(meta["cols"]):
                    cc = int(gc // P)
                    pp = int(gc % P)
                    x[cc, pp] = float(z_block[j_local])

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
                        if np.all(known_zero[cc, :]):
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