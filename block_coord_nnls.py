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

                    # --- DIAG: tile-level numeric checks & timings ---
                    try:
                        # basic shapes and norms
                        yf_norm = float(np.linalg.norm(y_flat)) if y_flat.size > 0 else 0.0
                        yhatf_norm = float(np.linalg.norm(yhat_flat)) if yhat_flat.size > 0 else 0.0
                        r_norm = float(np.linalg.norm(r_flat)) if r_flat.size > 0 else 0.0

                        nnans = int(np.count_nonzero(~np.isfinite(y_flat)))
                        nnans += int(np.count_nonzero(~np.isfinite(yhat_flat)))
                        nnans += int(np.count_nonzero(~np.isfinite(r_flat)))

                        print(
                            f"[BC-FUSED][tile s={s0}:{s1}] Sblk={Sblk} Lk={Lk} "
                            f"||y||={yf_norm:.3e} ||yhat||={yhatf_norm:.3e} ||r||={r_norm:.3e} "
                            f"nonfinite_vals={nnans}", flush=True
                        )

                        # quick histogram-ish checks (coarse)
                        if r_flat.size > 0:
                            r_max = float(np.max(r_flat))
                            r_min = float(np.min(r_flat))
                            r_pos_frac = float(np.count_nonzero(r_flat > 0) / max(1, r_flat.size))
                            print(
                                f"[BC-FUSED][tile s={s0}:{s1}] r_min={r_min:.3e} r_max={r_max:.3e} "
                                f"r_pos_frac={r_pos_frac:.3f}", flush=True
                            )

                    except Exception as _e:
                        print("[BC-FUSED][tile diag] error while computing tile diagnostics:", _e, flush=True)
                    # --- end tile diagnostics ---

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
                            # === explicit A_tile construction (less error-prone) ===
                            # sub: (Sblk, n_local, Lk)
                            # We want rows = Sblk * Lk, cols = n_local
                            # Do: transpose to (Sblk, Lk, n_local) then reshape
                            sub = A_cc[:, p_idx_arr, :]  # (Sblk, n_local, Lk)
                            # transpose axes to (Sblk, Lk, n_local)
                            sub_t = sub.transpose(0, 2, 1)            # (Sblk, Lk, n_local)
                            # then flatten first two dims -> (Sblk*Lk, n_local)
                            A_tile = sub_t.reshape(sub_t.shape[0] * sub_t.shape[1], sub_t.shape[2])

                            # === deep sanity check (dump tiny sample and fail-fast) ===
                            # sample a few entries
                            sample_idx = np.arange(min(5, y_flat.size))
                            print(f"[BC-FUSED][DEBUG-SAMPLE] tile s={s0}:{s1} Sblk={Sblk} Lk={Lk} "
                                f"||y||={np.linalg.norm(y_flat):.3e} y_sample={y_flat[sample_idx]!r}", flush=True)

                            # check dot product signs between first column of A_tile and y_flat
                            if A_tile.shape[0] > 0 and A_tile.shape[1] > 0:
                                col0 = A_tile[:, 0]
                                dot0 = float(np.dot(col0, y_flat))
                                dot0_r = float(np.dot(col0, (y_flat - (col0 * (dot0 / (np.dot(col0, col0) + 1e-30))))))
                                print(f"[BC-FUSED][DEBUG-SAMPLE] dot(col0,y)={dot0:.3e} dot0_resid_like={dot0_r:.3e} "
                                    f"col0_norm={np.linalg.norm(col0):.3e}", flush=True)
                            # abort early to inspect values if ATy will end up all negative
                            ATy_tile_test = A_tile.T @ y_flat
                            if np.all(ATy_tile_test <= 0.0):
                                # print short diagnostic and raise to inspect environment (or keep running but noisy)
                                print(f"[BC-FUSED][FATAL-DEBUG] ATy_tile_test all <= 0 (sample first 6): "
                                    f"{ATy_tile_test[:6]!r}", flush=True)
                                # Optionally raise to stop run and inspect; comment out raise if you prefer continuation
                                # raise RuntimeError("ATy all non-positive in diagnostic check")

                            # update ATA and ATy
                            # === explicit ATA and ATy from DATA (y_flat), not residual ===
                            # Build ATA contribution for this block
                            ATA_blocks[bi][np.ix_(local_idx_arr, local_idx_arr)] += (A_tile.T @ A_tile)

                            # Build ATy contribution *explicitly from data y_flat*
                            # (very important: this must be A^T @ y, not A^T @ r)
                            ATy_tile = A_tile.T @ y_flat    # shape (n_local,)
                            ATy_blocks[bi][local_idx_arr] += ATy_tile

                            # quick guard / assertion to detect accidental residual usage:
                            # if global tile residual is strongly negative, it is expected,
                            # but ATy computed from y_flat should not be all-negative of huge mag.
                            _ifpos = np.count_nonzero(ATy_tile > 0)
                            _ifneg = np.count_nonzero(ATy_tile < 0)
                            _ifzero = ATy_tile.size - (_ifpos + _ifneg)
                            if _ifpos == 0 and _ifneg > 0:
                                print(f"[BC-FUSED][WARN][tile s={s0}:{s1} bi={bi}] ATy_tile all <=0 "
                                    f"(+={_ifpos} -={_ifneg} 0={_ifzero}) sample_ATy[0:3]={ATy_tile[:3]!r}", flush=True)
                            # ================================================================

                    # keep y parts if you need full y vector for diagnostics (optional)
                    y_parts.append(y_flat)

            # end tile-streaming

            # ---------------- COLUMN NORMALIZATION ----------------
            # Build per-column energy normalization
            col_energy = D_tot.copy()  # shape (C,P)

            # Avoid division by zero
            col_energy[col_energy <= 0.0] = 1.0

            inv_sqrt_energy = 1.0 / np.sqrt(col_energy)

            # Normalize ATA and ATy blocks
            for bi, meta in block_meta.items():
                cols = meta["cols"]
                # map flattened indices to (C,P)
                cc_arr = meta["cc_arr"]
                pp_arr = meta["pp_arr"]

                scale = inv_sqrt_energy[cc_arr, pp_arr]  # shape (ncols,)

                # Scale ATA: S A S
                ATA_blocks[bi] = (ATA_blocks[bi] * scale[:, None]) * scale[None, :]

                # Scale ATy: S ATy
                ATy_blocks[bi] = ATy_blocks[bi] * scale

            # --- DIAG: summary after tile streaming (epoch-level) ---
            try:
                # quick global ATA/ATy sanity summary across blocks
                nblocks = len(ATA_blocks)
                tot_cols = sum([meta["ncols"] for _, meta in block_meta.items()])
                # per-block summaries
                mm = []
                atys_pos = 0
                atys_neg = 0
                atys_zero = 0
                ata_diag_meds = []
                ata_diag_mins = []
                ata_diag_maxs = []
                ata_nonfinite_counts = 0
                aty_nonfinite_counts = 0
                for bi, meta in block_meta.items():
                    ATA = ATA_blocks[bi]
                    ATy = ATy_blocks[bi]
                    # diag stats
                    diag = np.diag(ATA)
                    ata_diag_meds.append(float(np.median(np.abs(diag))) if diag.size else 0.0)
                    ata_diag_mins.append(float(np.min(diag)) if diag.size else 0.0)
                    ata_diag_maxs.append(float(np.max(diag)) if diag.size else 0.0)
                    # ATy sign counts
                    if ATy.size > 0:
                        atys_pos += int(np.count_nonzero(ATy > 0))
                        atys_neg += int(np.count_nonzero(ATy < 0))
                        atys_zero += int(np.count_nonzero(ATy == 0))
                    ata_nonfinite_counts += int(np.count_nonzero(~np.isfinite(ATA)))
                    aty_nonfinite_counts += int(np.count_nonzero(~np.isfinite(ATy)))

                print(
                    "[BC-FUSED][stream-summary] blocks=%d tot_cols=%d ata_nonfinite=%d aty_nonfinite=%d"
                    % (nblocks, tot_cols, ata_nonfinite_counts, aty_nonfinite_counts),
                    flush=True
                )
                if len(ata_diag_meds):
                    print(
                        "[BC-FUSED][stream-summary] ATA diag med/min/max = %.3e / %.3e / %.3e"
                        % (float(np.median(ata_diag_meds)), float(np.min(ata_diag_mins)), float(np.max(ata_diag_maxs))),
                        flush=True
                    )
                print(
                    f"[BC-FUSED][stream-summary] ATy sign counts: +={atys_pos} -={atys_neg} 0={atys_zero}",
                    flush=True
                )
            except Exception as _e:
                print("[BC-FUSED][stream-summary] diag error:", _e, flush=True)
            # --- end stream summary diagnostics ---

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
                # Always define x_block_old at block start (prevents UnboundLocalError)
                x_block_old = x_flat[meta["cols"]]
                ATA = ATA_blocks[bi]
                ATy = ATy_blocks[bi].copy()  # (ncols,)

                # --- DIAG: per-block pre-solve diagnostics ---
                try:
                    ncols = meta["ncols"]
                    diag = np.diag(ATA) if ncols > 0 else np.array([], dtype=float)
                    diag_med = float(np.median(np.abs(diag))) if diag.size else 0.0
                    diag_min = float(np.min(diag)) if diag.size else 0.0
                    diag_max = float(np.max(diag)) if diag.size else 0.0
                    aty_max = float(np.max(ATy)) if ATy.size else 0.0
                    aty_min = float(np.min(ATy)) if ATy.size else 0.0
                    aty_sum = float(np.sum(ATy)) if ATy.size else 0.0
                    aty_pos = int(np.count_nonzero(ATy > 0)) if ATy.size else 0
                    aty_neg = int(np.count_nonzero(ATy < 0)) if ATy.size else 0
                    ata_nonfinite = int(np.count_nonzero(~np.isfinite(ATA)))
                    aty_nonfinite = int(np.count_nonzero(~np.isfinite(ATy)))

                    # simple diag-based condition estimate (cheap)
                    cond_est = np.inf
                    if diag.size and np.median(np.abs(diag)) > 0:
                        cond_est = float(np.max(np.abs(diag)) / float(np.median(np.abs(diag))))

                    print(
                        f"[BC-FUSED][block {bi}] cols={ncols} diag_med={diag_med:.3e} "
                        f"diag_min={diag_min:.3e} diag_max={diag_max:.3e} cond_est={cond_est:.3e}", flush=True
                    )
                    print(
                        f"[BC-FUSED][block {bi}] ATy: max={aty_max:.3e} min={aty_min:.3e} sum={aty_sum:.3e} "
                        f"+count={aty_pos} -count={aty_neg} nonfinite(ATA)={ata_nonfinite} nonfinite(ATy)={aty_nonfinite}",
                        flush=True
                    )

                    # quick guard: warn if ATy is nonpositive (zero optimal) or ATA all tiny
                    if ATy.size and np.all(ATy <= 0):
                        print(f"[BC-FUSED][block {bi}] WARNING: ATy <= 0 for all cols -> zero is KKT candidate", flush=True)
                    if diag.size and float(np.median(np.abs(diag))) <= 0.0:
                        print(f"[BC-FUSED][block {bi}] WARNING: ATA diagonal median is zero; block may be rank-def.", flush=True)

                except Exception as _e:
                    print(f"[BC-FUSED][block {bi}] pre-solve diag error:", _e, flush=True)
                # --- end per-block pre-solve diagnostics ---

                # ------------------ Column-normalize + ridge ------------------
                # Soft-orbit augmentation (if requested)
                if (orbit_beta is not None) and (orbit_beta > 0.0) and (meta["w_vec"] is not None):
                    u = np.ones((meta["ncols"],), dtype=np.float64)
                    ATA = ATA + (2.0 * float(orbit_beta)) * np.outer(u, u)
                    ATy = ATy + (2.0 * float(orbit_beta)) * meta["w_vec"]

                # initial x for block: current x values
                x_block_init = x_flat[meta["cols"] - 0]

                # ---------------- CLEAN QUADRATIC SOLVE ----------------
                # ATA and ATy are already globally column-normalized.
                # Solve directly with strong ridge regularization.

                ncol = ATA.shape[0]
                if ncol == 0:
                    z_block = np.zeros((0,), dtype=np.float64)
                else:
                    # Strong Tikhonov regularization (this is not optional)
                    lambda_l2 = 1e-2   # strong stabilizer
                    ATA_reg = ATA + lambda_l2 * np.eye(ncol, dtype=np.float64)

                    # -------- Age curvature regularization ----------
                    # Penalize (x_{p+1} - 2x_p + x_{p-1})^2 along population axis

                    lambda_curv = 1e-2   # strong smoothing

                    if lambda_curv > 0.0:
                        # Build 1D second-difference operator per orbit within this block
                        # This assumes populations contiguous in P ordering
                        for j_local, gc in enumerate(meta["cols"]):
                            cc = int(gc // P)
                            pp = int(gc % P)

                            if 0 < pp < (P - 1):
                                # Add curvature penalty to diagonal
                                ATA_reg[j_local, j_local] += 2.0 * lambda_curv
                            else:
                                ATA_reg[j_local, j_local] += lambda_curv
                    # -------------------------------------------------

                    # Solve NNLS quadratic
                    try:
                        z_block = _nnls_from_quadratic(
                            ATA_reg,
                            ATy,
                            x0=x_block_old,
                            max_iter=4000,
                            tol=1e-10
                        )
                    except Exception:
                        z_block = _projected_gradient_nnls(
                            ATA_reg,
                            ATy,
                            max_iter=2000,
                            tol=1e-8
                        )
                # ---------------------------------------------------------

                # quick diagnostics per block
                if verbose:
                    print(f"[BC-FUSED][block {bi}] post-solve l1={np.sum(np.abs(z_block)):.3e} "
                          f"max={np.max(z_block):.3e} nonzero={int(np.count_nonzero(z_block))}/{z_block.size}", flush=True)

                for j_local, gc in enumerate(meta["cols"]):
                    cc = int(gc // P)
                    pp = int(gc % P)
                    x[cc, pp] = float(z_block[j_local])

                delta = z_block - x_block_old
                if np.any(delta != 0.0):
                    # Update residual approximation via A_delta
                    # (optional advanced; skip for now if expensive)
                    pass

                # --- DIAG: post-solve block checks (delta & sparsity) ---
                try:
                    x_block_new = np.asarray(z_block, dtype=np.float64)
                    x_block_old = x_block_init
                    delta = np.linalg.norm(x_block_new - x_block_old) if x_block_new.size else 0.0
                    x_block_l1 = float(np.sum(x_block_new))
                    nonzero = int(np.count_nonzero(x_block_new > 0))
                    print(
                        f"[BC-FUSED][block {bi}] post-solve delta_norm={delta:.3e} l1={x_block_l1:.3e} "
                        f"nonzero={nonzero}/{x_block_new.size}", flush=True
                    )
                except Exception as _e:
                    print(f"[BC-FUSED][block {bi}] post-solve diag error:", _e, flush=True)
                # --- end post-solve diagnostics ---

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

                print("=== [BC-FUSED][epoch-summary] ===", flush=True)
                print(f"[BC-FUSED][epoch-summary] epoch={ep+1}/{cfg.epochs} elapsed_total={epoch_elapsed:.1f}s", flush=True)
                print(f"[BC-FUSED][epoch-summary] data_proxy={data_proxy:.3e} rmse={rmse_curr:.3e}", flush=True)
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