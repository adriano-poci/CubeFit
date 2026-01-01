# -*- coding: utf-8 -*-
r"""
    kaczmarz_solver_cchunk_mp_nnls.py
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
"""

from __future__ import annotations, print_function

import os, sys, traceback
import math
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

# ------------------------- Norm mode guard ----------------------------

def _assert_norm_mode(h5_path: str, expect: Optional[str] = None) -> str:
    r"""
    Read /HyperCube/norm.mode and optionally assert a specific mode.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file.
    expect : str, optional
        If given ('data' or 'model'), raise if the stored mode differs.

    Returns
    -------
    str
        The stored normalization mode ('data' or 'model').

    Exceptions
    ----------
    RuntimeError
        If the attribute is missing or does not match `expect`.
    """
    with open_h5(h5_path, role="reader") as f:
        g = f["/HyperCube"]
        mode = str(g.attrs.get("norm.mode", "model")).lower()
    if expect and mode != expect:
        raise RuntimeError(
            f"Hypercube is in norm.mode='{mode}', but solver expects "
            f"'{expect}'. Convert first with convert_hypercube_norm(...)."
        )
    if mode not in ("data", "model"):
        raise RuntimeError(f"Unexpected norm.mode='{mode}'.")
    return mode

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

# ---------------------- Optional shift diagnostic ---------------------

def _xcorr_int_shift(a: np.ndarray, b: np.ndarray) -> int:
    """
    Integer-lag that best aligns b to a by full cross correlation.
    Positive → b is to the right.
    """
    aa = np.asarray(a, np.float64).ravel()
    bb = np.asarray(b, np.float64).ravel()
    n = int(aa.size)
    fa = np.fft.rfft(aa, n=2*n)
    fb = np.fft.rfft(bb, n=2*n)
    cc = np.fft.irfft(fa * np.conj(fb))
    j = int(np.argmax(cc))
    if j > n:
        j -= 2*n
    return j

# ------------------------------------------------------------------------------

def _choose_tiles_fair_spread(
    s_ranges: list[tuple[int, int]],
    k: int,
    seed: int = 12345,
) -> list[tuple[int, int]]:
    """
    Pick k tile ranges with a roughly uniform spread over the spatial index.

    This avoids brightness bias: it sorts by s0 and stratifies the list into k bins,
    selecting one tile per bin (random within each bin).

    Parameters
    ----------
    s_ranges
        List of (s0, s1) tile ranges.
    k
        Number of tiles to select.
    seed
        RNG seed.

    Returns
    -------
    list[tuple[int, int]]
        Selected tile ranges (k or fewer if s_ranges is smaller).
    """
    if k <= 0 or not s_ranges:
        return []
    if k >= len(s_ranges):
        return list(s_ranges)

    s_sorted = sorted(s_ranges, key=lambda t: int(t[0]))
    n = len(s_sorted)
    k = min(k, n)

    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n, k + 1, dtype=int)

    out: list[tuple[int, int]] = []
    for i in range(k):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            idx = a
        else:
            idx = int(rng.integers(a, b))
        out.append(s_sorted[idx])

    return out

# ------------------------------------------------------------------------------

def _safe_scalar_rmse(val: float, label: str) -> float:
    """
    Ensure a scalar RMSE used for epoch comparison is finite.

    If 'val' is NaN, Inf, or negative, print a warning and return +inf
    so that this value will never be considered an improvement over a
    finite best RMSE.
    """
    if not np.isfinite(val) or val < 0.0:
        print(
            f"[Kaczmarz-MP] WARNING: {label} RMSE={val!r} is non-finite "
            f"or negative; treating as +inf.",
            flush=True,
        )
        return float("inf")
    return float(val)

# ------------------------------------------------------------------------------

def rmse_proxy_subset(
    h5_path,
    x_CP,
    tile_ranges,
    keep_idx,
    inv_cp_flux_ref,
    w_lam_sqrt,
):
    ssq = 0.0
    n = 0

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M  = f["/HyperCube/models"]

        for (s0, s1) in tile_ranges:
            Y = np.asarray(DC[s0:s1, :], np.float64)
            if keep_idx is not None:
                Y = Y[:, keep_idx]

            yhat = np.zeros_like(Y)
            for c in range(x_CP.shape[0]):
                A = np.asarray(M[s0:s1, c, :, :], np.float32)
                if keep_idx is not None:
                    A = A[:, :, keep_idx]
                if inv_cp_flux_ref is not None:
                    A = A * inv_cp_flux_ref[c, :][None, :, None]

                yhat += np.tensordot(x_CP[c, :], A, axes=(0, 1))

            R = Y - yhat
            if w_lam_sqrt is not None:
                R = R * w_lam_sqrt

            ssq += float(np.sum(R * R))
            n   += int(R.size)

    return np.sqrt(ssq / max(n, 1))

# ------------------------------------------------------------------------------

def _tiny_col_freeze_inplace(col_denom, grad, rel_zero, abs_zero):
    """
    Freeze numerically tiny / unsupported columns for a given tile-band.

    Parameters
    ----------
    col_denom : (P,) float array
        Per-column denominator computed on the SAME weighted/scaled A used for
        the gradient (e.g. after cp_flux_ref scaling and λ-weighting).
    grad : (P,) float array
        Per-column gradient/numerator (same space as col_denom).
    rel_zero : float
        Relative threshold multiplier applied to median positive denom.
    abs_zero : float
        Absolute floor threshold.
    """
    # robust median on strictly-positive finite entries
    good = np.isfinite(col_denom) & (col_denom > 0.0)
    if np.any(good):
        med_energy = float(np.median(col_denom[good]))
    else:
        med_energy = 0.0

    tiny_col = float(max(abs_zero, rel_zero * med_energy))

    # freeze non-finite or tiny denom
    freeze = (~np.isfinite(col_denom)) | (col_denom <= tiny_col)
    if np.any(freeze):
        grad[freeze] = 0.0
        col_denom[freeze] = np.inf

    return freeze, tiny_col, med_energy

# ---------------------------- Worker ---------------------------------

def _worker_tile_global_grad_band(args):
    r"""
    Compute global Kaczmarz-style gradient contributions for one spatial
    tile and one contiguous band of components.

    This worker does **not** update x or R. It only returns the
    band-local contributions to the global gradient and diagonal
    preconditioner:

        g_band[c,p]  = sum_{s,λ} (√w_λ A_{s,c,p,λ}) * (√w_λ R_{s,λ})
        D_band[c,p]  = sum_{s,λ} (√w_λ A_{s,c,p,λ})^2,

    where R = Y - ŷ is the residual for the current epoch's global x_CP,
    and A already includes any column-flux normalization in the "model"
    basis.

    Additionally, this worker can apply a tile-local "tiny-column freeze"
    safety guard: columns with numerically tiny denom are zeroed in the
    numerator and excluded from the denom accumulator for this tile. This
    prevents large/noisy updates later when the coordinator forms
    invD = 1/max(D, eps).

    Control knobs (env)
    -------------------
    CUBEFIT_ZERO_COL_FREEZE : {0,1} (default 1)
        Enable/disable tile-local freeze.
    CUBEFIT_ZERO_COL_REL : float (default 1e-12)
    CUBEFIT_ZERO_COL_ABS : float (default 1e-24)

    Returns
    -------
    g_band : ndarray, shape (band_size, P), float64
    D_band : ndarray, shape (band_size, P), float64
    """
    (h5_path, s0, s1, keep_idx,
     c_start, c_stop,
     x_band,
     R_tile,
     w_lam_sqrt,
     inv_ref_band,
     dset_slots, dset_bytes, dset_w0) = args

    eps = float(os.environ.get("CUBEFIT_EPS", "1e-12"))

    freeze_enable = os.environ.get(
        "CUBEFIT_ZERO_COL_FREEZE", "1"
    ).lower() not in ("0", "false", "no", "off")
    rel_zero = float(os.environ.get("CUBEFIT_ZERO_COL_REL", "1e-12"))
    abs_zero = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-24"))

    # Shapes
    R_tile = np.asarray(R_tile, dtype=np.float64, order="C")
    Sblk, Lk = R_tile.shape

    # λ-weights (optional)
    if w_lam_sqrt is not None:
        wvec = np.asarray(w_lam_sqrt, np.float64).ravel()
        if wvec.size != Lk:
            raise RuntimeError(
                f"_worker_tile_global_grad_band: w_lam_sqrt length "
                f"{wvec.size} != Lk={Lk}"
            )
        Rw = R_tile * wvec[None, :]
    else:
        wvec = None
        Rw = R_tile

    band_size, P = x_band.shape

    g_band = np.zeros((band_size, P), dtype=np.float64)
    D_band = np.zeros((band_size, P), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]  # (S, C, P, L)
        try:
            M.id.set_chunk_cache(dset_slots, dset_bytes, dset_w0)
        except Exception:
            pass

        for bi, c in enumerate(range(c_start, c_stop)):
            A = np.asarray(
                M[s0:s1, c, :, :], dtype=np.float32, order="C"
            )  # (Sblk, P, L_full)
            if keep_idx is not None:
                A = A[:, :, keep_idx]  # (Sblk, P, Lk)

            # Apply column-flux normalization if present
            if inv_ref_band is not None:
                inv_ref = np.asarray(
                    inv_ref_band[bi, :], dtype=np.float64, copy=False
                )  # (P,)
                A = A * inv_ref[None, :, None]

            # Promote to float64 for safe dot products
            A = A.astype(np.float64, copy=False)

            if wvec is not None:
                A_w = A * wvec[None, None, :]  # (Sblk, P, Lk)
            else:
                A_w = A

            # g_row[p] = Σ_{s,λ} A_w[s,p,λ] * Rw[s,λ]
            g_row = np.tensordot(A_w, Rw, axes=([0, 2], [0, 1]))

            # D_row[p] = Σ_{s,λ} (A_w[s,p,λ])^2
            D_row = np.sum(np.square(A_w, dtype=np.float64), axis=(0, 2))

            # ---- Tiny-column freeze (tile-local) -------------------------
            # Use the existing helper. It sets denom[frozen]=inf; for a
            # global accumulator we instead want to *exclude* this tile’s
            # contribution, so we revert denom[frozen] -> 0 before storing.
            if freeze_enable:
                freeze, _, _ = _tiny_col_freeze_inplace(
                    D_row, g_row, rel_zero, abs_zero
                )
                if np.any(freeze):
                    D_row[freeze] = 0.0

            g_band[bi, :] = g_row
            D_band[bi, :] = D_row

    # Numerical guards
    if not np.all(np.isfinite(g_band)):
        g_band = np.nan_to_num(
            g_band, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )

    # For D: keep zeros (they are meaningful), but sanitize NaN/Inf.
    if not np.all(np.isfinite(D_band)):
        D_band = np.nan_to_num(
            D_band, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )

    # Ensure no negative denom from numerical noise
    np.maximum(D_band, 0.0, out=D_band)

    # Optional hard floor (only if you later invert without g-masking);
    # here we do NOT force a floor, because g has been frozen where denom
    # is tiny/invalid.
    if not np.isfinite(eps) or eps <= 0.0:
        eps = 1e-12

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

def solve_global_kaczmarz_global_step_mp(
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
        DC = f["/DataCube"]
        M  = f["/HyperCube/models"]

        S, L = map(int, DC.shape)
        _, C, P, Lm = map(int, M.shape)
        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch")

        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        Lk = int(keep_idx.size) if keep_idx is not None else L

        s_tile = int(M.chunks[0]) if (M.chunks and M.chunks[0] > 0) else 128
        if cfg.s_tile_override is not None:
            s_tile = int(cfg.s_tile_override)

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

    Y_glob_norm = float(np.sqrt(Y_glob_norm2))
    print(
        f"[SPG] Global ||Y|| = {Y_glob_norm:.6e}",
        flush=True,
    )

    # ------------------------------------------------------------
    # λ-weights (optional)
    # ------------------------------------------------------------
    w_lam_sqrt = None
    if os.environ.get("CUBEFIT_LAMBDA_WEIGHTS_ENABLE", "1").lower() not in (
        "0", "false", "no", "off"
    ):
        try:
            w_full = cu.read_lambda_weights(h5_path)
            w_use = w_full[keep_idx] if keep_idx is not None else w_full
            w_lam_sqrt = np.sqrt(np.maximum(w_use, 1e-6)).astype(np.float64)
            print("[SPG] λ-weights enabled.", flush=True)
        except Exception:
            w_lam_sqrt = None
            print("[SPG] λ-weights unavailable; unweighted LS.", flush=True)

    # ------------------------------------------------------------
    # Global column energy (for preconditioning + orbit projection)
    # ------------------------------------------------------------
    E_global = read_global_column_energy(h5_path)  # (C,P)

    # ------------------------------------------------------------
    # Orbit-weight prior (spaxel independent)
    # ------------------------------------------------------------
    w_target = None
    orbit_beta = float(os.environ.get("CUBEFIT_ORBIT_BETA", "0.2"))
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

    if cfg.project_nonneg:
        np.maximum(x, 0.0, out=x)

    # ------------------------------------------------------------
    # Multiprocessing bands
    # ------------------------------------------------------------
    nprocs_req = max(1, int(cfg.processes))
    band_size = int(np.ceil(C / nprocs_req))
    bands = []
    c0 = 0
    for _ in range(nprocs_req):
        c1 = min(C, c0 + band_size)
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

    x_prev = None
    g_prev = None

    best_x = x.copy()
    best_proxy = np.inf

    # --- Active set bookkeeping ---
    active_orbits = np.arange(C, dtype=np.int32)   # start fully active
    min_active = int(os.environ.get("CUBEFIT_MIN_ACTIVE_ORBITS", "8"))


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

            # ---------------- Gradient accumulation ----------------
            for (s0, s1) in s_ranges:

                x_eff = x.copy()
                inactive = np.ones(C, dtype=bool)
                inactive[active_orbits] = False
                x_eff[inactive, :] = 0.0

                with open_h5(h5_path, role="reader") as f:
                    DC = f["/DataCube"]
                    M  = f["/HyperCube/models"]

                    Y = np.asarray(DC[s0:s1, :], np.float64)
                    if keep_idx is not None:
                        Y = Y[:, keep_idx]

                    # exact model prediction (ACTIVE ORBITS ONLY)
                    yhat = np.zeros_like(Y)
                    for c in active_orbits:
                        A = np.asarray(M[s0:s1, c, :, :], np.float32)
                        if keep_idx is not None:
                            A = A[:, :, keep_idx]
                        yhat += np.tensordot(x_eff[c], A, axes=(0, 1))

                    R = Y - yhat
                    if not np.all(np.isfinite(R)):
                        R = np.nan_to_num(R, copy=False)

                ssq += float(np.sum(R * R))
                nres += int(R.size)

                # ---- Worker jobs: FULL CONTIGUOUS BANDS (unchanged) ----
                jobs = []
                for (c_start, c_stop) in bands:
                    jobs.append(
                        (
                            h5_path,
                            int(s0),
                            int(s1),
                            keep_idx,
                            int(c_start),
                            int(c_stop),
                            x_eff[c_start:c_stop].copy(order="C"),
                            R.copy(order="C"),
                            w_lam_sqrt,
                            None,
                            cfg.dset_slots,
                            cfg.dset_bytes,
                            cfg.dset_w0,
                        )
                    )

                if use_pool:
                    results = pool.map(_worker_tile_global_grad_band, jobs)
                else:
                    results = [_worker_tile_global_grad_band(j) for j in jobs]

                # ---- Aggregate results (UNCHANGED, SAFE) ----
                for (c_start, c_stop), (g_band, D_band) in zip(bands, results):
                    g_tot[c_start:c_stop] += g_band
                    D_tot[c_start:c_stop] += D_band

                pbar.update(1)

            pbar.close()

            # ---------------- SPG step ----------------
            rmse_proxy = np.sqrt(ssq / max(nres, 1))
            if rmse_proxy < best_proxy:
                best_proxy = rmse_proxy
                best_x = x.copy()

            # diagonal preconditioner
            abs_zero = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-24"))
            denom_floor_frac = float(os.environ.get("CUBEFIT_DENOM_FLOOR_FRAC",
                "1e-6"))
            D_pos = D_tot[D_tot > 0.0]
            D_scale = np.percentile(D_pos, 90.0) if D_pos.size else 1.0
            D_floor = max(abs_zero, denom_floor_frac * D_scale)
            D = np.maximum(D_tot, D_floor)

            g = -g_tot  # gradient is negative residual correlation

            # BB spectral step length
            if x_prev is not None and g_prev is not None:
                s = x - x_prev
                y = g - g_prev
                sy = float(np.sum(s * y))
                if sy > 0.0:
                    lr = float(np.sum(s * s) / sy)
                    lr = np.clip(lr, 1e-6, cfg.lr)
                else:
                    lr = cfg.lr
            
            # after computing D
            D_min = np.min(D)
            lr_eff_max = 0.1 * D_min    # conservative, safe
            lr = np.min([lr, lr_eff_max])

            # gradient descent step
            x_prev = x.copy()
            g_prev = g.copy()

            dx = -lr * (g / D)
            step_dot_grad = float(np.vdot(dx, g))
            quad_pred = float(
                np.vdot(g, dx) + 0.5 * np.vdot(dx, D * dx)
            )
            D_flat = D.ravel()
            print("[SPG] Step diagnostics:", flush=True)
            print(
                f"  lr={lr:.3e}, step·grad={step_dot_grad:.3e}, "
                f"quad_pred={quad_pred:.3e}"
            )
            Dpercs = np.percentile(D_flat, [0.1, 1.0, 10.0, 50.0, 90.0, 99.0, 99.9])
            print(f"  D percentiles: 0.1%={Dpercs[0]:.3e}, 1%={Dpercs[1]:.3e}, "
                  f"10%={Dpercs[2]:.3e}, 50%={Dpercs[3]:.3e}, "
                  f"90%={Dpercs[4]:.3e}, 99%={Dpercs[5]:.3e}, "
                  f"99.9%={Dpercs[6]:.3e}")
            print(
                f"  minD={np.min(D_flat):.3e}, medianD={np.median(D_flat):.3e},"
                f" maxD={np.max(D_flat):.3e}")

            if np.vdot(dx, g) > 0.0: # prevent gradient ascent
                dx = -dx

            # ==================================================================
            # Safeguards on step size
            # ==================================================================

            # Safeguard against huge steps
            x_norm  = np.linalg.norm(x)

            # --- adaptive max_frac computed from solver diagnostics ---
            # Inputs available cheaply in your loop:
            #   lr        : current learning rate / step multiplier (scalar)
            #   D         : preconditioner array or small-sample percentiles (you already compute percentiles)
            #   grad      : gradient array (you already have it to compute dx)
            #   x         : current solution vector
            #   dx        : proposed update (already computed, sign-corrected)

            # Parameters (tunable but safe defaults)
            beta = 3.0 # headroom multiplier (>=1). 3 is conservative but responsive.
            min_frac = 1e-8 # absolute lower bound to avoid exact zero cap
            max_frac_cap = 0.5 # absolute upper bound on fractional change per epoch
            eps_x = 1e-12 # prevents division by zero for median(x)

            # VERY cheap, O(1) scalar computations:
            # Prefer using median statistics over mean for robustness on heavy-tailed data.
            medianD = Dpercs[3]
            median_abs_grad = np.median(np.abs(g))
            median_abs_x = np.median(np.abs(x))

            # Estimate typical per-element step magnitude (based on lr * D * grad)
            typical_step = lr * medianD * median_abs_grad

            # Convert to a typical fractional change relative to median(|x|)
            typical_frac = typical_step / max(median_abs_x, eps_x)

            # Proposed adaptive max_frac
            adaptive_max_frac = beta * typical_frac

            # Clip to safe bounds
            adaptive_max_frac = float(np.clip(adaptive_max_frac, min_frac, max_frac_cap))

            # --- Now enforce trust region on dx (L2 or linf variant, choose one) ---
            # Option A (L2 norm)
            dx_norm = np.linalg.norm(dx)      # one scalar
            x_scale = np.max([x_norm, eps_x * np.sqrt(x.size)])  # keep same scaling style if used elsewhere
            if dx_norm > adaptive_max_frac * x_scale:
                dx *= (adaptive_max_frac * x_scale / dx_norm)

            # Option B (Linfty / elementwise) - cheaper if x is huge (avoid full L2)
            # dx_max = np.max(np.abs(dx))
            # x_max  = max(np.max(np.abs(x)), eps_x)
            # if dx_max > adaptive_max_frac * x_max:
            #     dx *= (adaptive_max_frac * x_max / dx_max)

            print(f"  adaptive_max_frac={adaptive_max_frac:.3e}", flush=True)
            print(f"  ||x||={x_norm:.3e}", flush=True)
            print(f"  medianD={medianD:.3e}, median|grad|={median_abs_grad:.3e}, "
                  f"median|x|={median_abs_x:.3e}", flush=True)
            print(f"  ||dx|| before cap={dx_norm:.3e}", flush=True)
            print(f"  ||dx|| after cap={np.linalg.norm(dx):.3e}", flush=True)

            # ---- GLOBAL TRUST REGION ----
            tau_global = float(os.environ.get("CUBEFIT_GLOBAL_TAU", "0.5"))
            if E_global is not None and Y_glob_norm > 0.0:
                step_energy = float(np.sum((dx * dx) * E_global))
                cap = float(tau_global * Y_glob_norm)

                if step_energy > cap * cap:
                    scale = cap / max(np.sqrt(step_energy), 1.0e-12)
                    dx *= scale

            x += dx
            # projection: nonnegativity
            if cfg.project_nonneg:
                np.maximum(x, 0.0, out=x)

            # projection: orbit-weight prior
            if w_target is not None and orbit_beta > 0.0:
                cu.project_to_component_weights(
                    x,
                    w_target,
                    E_cp=E_global,
                    beta=orbit_beta,
                    minw=1e-12,
                )

            # ------------------------------------------------------------
            # Update active orbit set (NNLS-safe)
            # ------------------------------------------------------------
            x_row_l1 = np.sum(x, axis=1)
            g_row_inf = np.max(np.abs(g), axis=1)

            x_scale = max(float(np.sum(x_row_l1)), 1.0)
            g_scale = max(float(np.max(g_row_inf)), 1.0)

            eps_mass = 1e-12 * x_scale
            eps_grad = 1e-10 * g_scale

            new_active = np.nonzero(
                (x_row_l1 > eps_mass) | (g_row_inf > eps_grad)
            )[0]

            # Always keep a minimum number to avoid pathological empty sets
            if new_active.size < min_active:
                topg = np.argsort(g_row_inf)[-min_active:]
                new_active = np.unique(np.concatenate([new_active, topg]))

            active_orbits = new_active.astype(np.int32)

            print(
                f"[SPG] active orbits: {active_orbits.size}/{C}",
                flush=True,
            )

            if tracker is not None:
                tracker.maybe_snapshot_x(
                    x, epoch=ep + 1, rmse=rmse_proxy, force=True
                )

            print(
                f"[SPG] epoch {ep+1} RMSE(proxy)={rmse_proxy:.4e} "
                f"lr={lr:.3e} ||x||={np.linalg.norm(x):.3e}",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        return best_x.ravel(order="C"), dict(
            epochs=cfg.epochs,
            elapsed_sec=elapsed,
            rmse_proxy_best=float(best_proxy),
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

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
