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

def _compute_global_rmse(
    h5_path: str,
    x_CP: np.ndarray,
    *,
    s_ranges: list[tuple[int, int]],
    keep_idx: np.ndarray | None,
    w_lam_sqrt: np.ndarray | None,
    cp_flux_ref: np.ndarray | None,
    inv_cp_flux_ref: np.ndarray | None,
    dset_slots: int,
    dset_bytes: int,
    dset_w0: float,
    weighted: bool = True,
    show_progress: bool = True,
) -> float:
    """
    Compute the global RMSE over the full (masked) cube for a given x_CP.

    This evaluates the same least–squares objective as the Kaczmarz solver
    in a single streaming pass over the cube. It is optimized for the common
    case cp_flux_ref is None (no column–flux normalization).

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing /DataCube and /HyperCube/models.
    x_CP : ndarray, shape (C, P)
        Current solution in the solver's internal basis. If cp_flux_ref is
        not None, this is the normalized basis used by Kaczmarz; otherwise
        it is in the physical basis.
    s_ranges : list of (int, int)
        Spatial tile ranges covering the full spaxel axis S. Should match
        the tiling used by the solver.
    keep_idx : ndarray or None
        Wavelength indices that are kept after applying the mask. If None,
        all wavelengths are used.
    w_lam_sqrt : ndarray or None
        sqrt(λ-weights) on the masked grid. If weighted=True and this is
        not None, residuals are multiplied by w_lam_sqrt before forming
        ||R||^2.
    cp_flux_ref, inv_cp_flux_ref : ndarray or None
        Column–flux scaling arrays of shape (C, P). If cp_flux_ref is None,
        no column-normalization is assumed and x_CP is used directly.
    dset_slots, dset_bytes, dset_w0 : int
        Chunk-cache parameters for /HyperCube/models.
    weighted : bool, optional
        If True (default) and w_lam_sqrt is provided, compute a λ-weighted
        RMSE. If False, compute an unweighted RMSE.
    show_progress : bool, optional
        If True, wrap the tile loop in a tqdm progress bar.

    Returns
    -------
    rmse : float
        Global (optionally weighted) RMSE over all spaxels and masked
        wavelengths.
    """
    # Normalize inputs
    keep_idx = None if keep_idx is None else np.asarray(
        keep_idx, dtype=np.int64
    )
    use_weight = bool(weighted and (w_lam_sqrt is not None))

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]           # (S, L)
        M  = f["/HyperCube/models"]   # (S, C, P, L)
        try:
            M.id.set_chunk_cache(dset_slots, dset_bytes, dset_w0)
        except Exception:
            pass

        S, L = map(int, DC.shape)
        _, C, P, Lm = map(int, M.shape)
        if Lm != L:
            raise RuntimeError(
                f"_compute_global_rmse: models L={Lm} vs data L={L}"
            )

        if x_CP.shape != (C, P):
            raise ValueError(
                f"_compute_global_rmse: x_CP shape {x_CP.shape} "
                f"!= (C,P)=({C},{P})"
            )

        # Push any column normalization onto x_CP once; in your current
        # runs cp_flux_ref is None, so this is just a cheap copy.
        if inv_cp_flux_ref is not None:
            x_eff = (
                np.asarray(x_CP, np.float64, order="C")
                * np.asarray(inv_cp_flux_ref, np.float64, order="C")
            )
        else:
            x_eff = np.asarray(x_CP, np.float64, order="C")

        if keep_idx is None:
            Lk = L
        else:
            Lk = int(keep_idx.size)

        if use_weight:
            wvec = np.asarray(w_lam_sqrt, np.float64).ravel()
            if wvec.size != Lk:
                raise RuntimeError(
                    f"_compute_global_rmse: w_lam_sqrt length "
                    f"{wvec.size} != Lk={Lk}"
                )
            # Harden λ-weights once
            if not np.all(np.isfinite(wvec)):
                print(
                    "[RMSE] WARNING: non-finite λ-weights detected; "
                    "cleaning with nan_to_num.",
                    flush=True,
                )
                wvec = np.nan_to_num(
                    wvec, nan=0.0, posinf=0.0, neginf=0.0
                )
        else:
            wvec = None

        num = 0.0
        den = 0  # scalar residual count

        iterator = s_ranges
        pbar = None
        if show_progress:
            pbar = tqdm(
                s_ranges,
                total=len(s_ranges),
                desc="[Kaczmarz-MP] global RMSE",
                mininterval=1.0,
                dynamic_ncols=True,
            )
            iterator = pbar

        for (s0, s1) in iterator:
            s0 = int(s0)
            s1 = int(s1)
            if s0 >= s1:
                continue
            Sblk = s1 - s0

            # Data slice
            Y = np.asarray(DC[s0:s1, :], np.float64, order="C")
            if keep_idx is not None:
                Y = Y[:, keep_idx]  # (Sblk, Lk)

            # Clean any non-finite data just in case
            if not np.all(np.isfinite(Y)):
                Y = np.nan_to_num(
                    Y, nan=0.0, posinf=0.0, neginf=0.0, copy=False
                )

            if wvec is not None:
                Yw = Y * wvec[None, :]
            else:
                Yw = Y

            # Model slice, weighted or unweighted
            yhat_w = np.zeros((Sblk, Lk), dtype=np.float64)

            for c in range(C):
                A = np.asarray(
                    M[s0:s1, c, :, :], dtype=np.float32, order="C"
                )  # (Sblk, P, L)
                if keep_idx is not None:
                    A = A[:, :, keep_idx]  # (Sblk, P, Lk)

                A = A.astype(np.float64, copy=False)
                if wvec is not None:
                    A = A * wvec[None, None, :]  # λ-weights

                # Clean design slice if needed
                if not np.all(np.isfinite(A)):
                    A = np.nan_to_num(
                        A, nan=0.0, posinf=0.0, neginf=0.0, copy=False
                    )

                xc = np.asarray(x_eff[c, :], np.float64, copy=False)

                # yhat_w += Σ_p xc[p] * A[:, p, :]
                yhat_w += np.tensordot(xc, A, axes=(0, 1))

            Rw = Yw - yhat_w

            # Final guard: ensure residuals are finite before accumulation
            if not np.all(np.isfinite(Rw)):
                Rw = np.nan_to_num(
                    Rw, nan=0.0, posinf=0.0, neginf=0.0, copy=False
                )

            num += float(np.sum(Rw * Rw))
            den += int(Rw.size)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    if (not np.isfinite(num)) or (not np.isfinite(den)) or (den <= 0):
        print(
            "[RMSE] WARNING: aggregate num/den non-finite or den<=0; "
            "returning +inf RMSE.",
            flush=True,
        )
        return float("inf")

    return float(math.sqrt(num / float(den)))

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

# ---------------------------- Worker ---------------------------------

def _worker_tile_job_with_R(args):
    r"""
    Worker for one S-tile and a contiguous c-band.

    This routine performs a robust, mask/weight-aware Kaczmarz-style update
    over a band of components for a single spatial tile. It is designed to be
    numerically stable and inclusive of common data challenges (NaNs/Infs,
    near-zero columns, masked wavelengths), while preserving physical zeros in
    the design matrix (no input normalization).

    Key behaviors
    -------------
    • Sanitize inputs in-place: non-finite values in `A` and `R` → 0.
    • Freeze columns whose (tile-local, weighted) energy is numerically tiny.
    • Diagonal preconditioning:
        dx_p = lr * g_p / sum_s,λ ( (√w_λ A_{s,p,λ})^2 ).
    • Trust region on the model change:
        || √w ⊙ (Δy) || ≤ τ · || √w ⊙ R ||  with τ≈0.7 by default.
    • Backtracking on the step scale α using an **O(1) quadratic evaluation**
      of the weighted RMSE:
         ||Rw + a·ΔRw||² = rr + 2a·cr + a²·rd,
      where rr=⟨Rw,Rw⟩, rd=⟨ΔRw,ΔRw⟩, cr=⟨Rw,ΔRw⟩. This avoids repeated
      array sums over Sblk×Lk during line search.
    • Blend local denominators with global column energy:
        denom ← max(denom, β · E_global[c, :]).
    • Return per-band global update-energy estimates for a global step cap.
    • Propose a small set of high-score columns per band for optional,
      tile-local NNLS polishing upstream (the actual polish is coordinated
      outside this worker).

    Parameters
    ----------
    args : tuple
        (h5_path, s0, s1, keep_idx, c_start, c_stop, x_band, lr,
         project_nonneg, R, w_band, dset_slots, dset_bytes, dset_w0,
         E_global_band, beta_blend, w_lam_sqrt)

        See caller for shapes and semantics. In brief:
        - R is the current (unweighted) residual for the tile, shape (Sblk, Lk).
        - w_lam_sqrt is √(λ-weights) on the (masked) λ-grid or None.
        - x_band is a view of the current weights for this band (c_start:c_stop).
        - E_global_band supplies per-(band,pop) global energies for blending.

    Returns
    -------
    R_delta : ndarray, shape (Sblk, Lk), float64
        The **unweighted** residual update accumulated over the band
        (i.e., what should be added to R by the coordinator, after an
        optional outer backtracking / global cap).
    dx_list : list of tuples
        [(c, dx_c, upd_energy_sq), ...] where dx_c has shape (P,) and
        `upd_energy_sq = Σ_p (dx_c[p]^2 · E_global[c,p])` for global
        step capping upstream.
    nnls_candidates : list of (global_cp_index, score)
        Proposed (component·P + pop) indices and their scores for a
        later tile-local NNLS polish step.

    Notes
    -----
    • All λ-weighting is applied symmetrically to `A` and `R` via `√w` for
      gradient/denominator and trust-region logic. The returned `R_delta`
      remains **unweighted** so that the coordinator can combine band deltas
      and apply any further global backtracking consistently.
    • Nonnegativity projection is respected. If projection clips some
      entries, ΔR is recomputed for the *final* dx on this band to keep the
      update self-consistent.
    """
    (h5_path, s0, s1, keep_idx,
     c_start, c_stop, x_band, x_prior_band,
     lr, project_nonneg, R,
     dset_slots, dset_bytes, dset_w0,
     E_global_band, beta_blend,
     w_lam_sqrt,
     inv_ref_band) = args   # (band_size, P)


    bt_steps   = int(os.environ.get("CUBEFIT_BT_STEPS", "3"))
    bt_factor  = float(os.environ.get("CUBEFIT_BT_FACTOR", "0.5"))
    tau_trust  = float(os.environ.get("CUBEFIT_TRUST_TAU", "0.7"))
    eps        = float(os.environ.get("CUBEFIT_EPS", "1e-12"))
    rel_zero   = float(os.environ.get("CUBEFIT_ZERO_COL_REL", "1e-12"))
    abs_zero   = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-24"))
    dbg        = os.environ.get("CUBEFIT_DEBUG_SAFE", "0") == "1"
    nnls_prop_per_band = int(os.environ.get("CUBEFIT_NNLS_PROP_PER_BAND", "6"))
    nnls_l2 = float(os.environ.get("CUBEFIT_NNLS_L2", "0.0"))
    if not np.isfinite(nnls_l2) or nnls_l2 < 0.0:
        nnls_l2 = 0.0

    bt_steps = np.max((0, bt_steps))
    beta_blend = float(beta_blend)

    def _finite_inplace(arr, name, stats_dict):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = 0.0
            if stats_dict is not None:
                stats_dict[name] = int(np.sum(bad))

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]  # (S,C,P,L)
        try:
            M.id.set_chunk_cache(dset_slots, dset_bytes, dset_w0)
        except Exception:
            pass

        Sblk = int(s1 - s0)
        P    = int(M.shape[2])
        L    = int(M.shape[3])
        Lk   = L if keep_idx is None else int(keep_idx.size)

        R = np.asarray(R, dtype=np.float64, order="C")  # (Sblk,Lk)
        stats = {}
        _finite_inplace(R, "R_nonfinite", stats)

        # Weighted residual (√w * R) for LS computations
        if w_lam_sqrt is not None:
            wvec = np.asarray(w_lam_sqrt, np.float64).ravel()
            if wvec.size != R.shape[1]:
                raise RuntimeError(
                    f"w_lam_sqrt length {wvec.size} != Lk={R.shape[1]}"
                )
            Rw = R * wvec[None, :]
        else:
            Rw = R

        rmse_before_tile = float(np.sqrt(np.mean(R * R)))  # unweighted (for info)
        R_delta = np.zeros((Sblk, Lk), dtype=np.float64)
        dx_list = []
        nnls_candidates = []

        for bi, c in enumerate(range(c_start, c_stop)):
            A = np.asarray(M[s0:s1, c, :, :], np.float32, order="C")
            if keep_idx is not None:
                A = A[:, :, keep_idx]  # (Sblk,P,Lk)

            inv_ref = (
                np.asarray(inv_ref_band[bi, :], dtype=np.float64)
                if inv_ref_band is not None
                else np.ones((P,), dtype=np.float64)
            )
            # Broadcast over (Sblk, P, Lk)
            A = A * inv_ref[None, :, None]
            _finite_inplace(A, f"A_nonfinite_c{c}", stats)

            # λ-weighted view for gradient/denominator
            if w_lam_sqrt is not None:
                A_w = A * w_lam_sqrt[None, None, :]
            else:
                A_w = A

            # Gradient g = (√w A)^T (√w R)
            g = np.zeros((P,), dtype=np.float64)
            for s in range(Sblk):
                g += A_w[s, :, :].astype(np.float64, copy=False) @ Rw[s, :]

            # Per-column denom on the same weighted view
            col_denom = np.sum(np.square(A_w, dtype=np.float64), axis=(0, 2))

            # Freeze numerically tiny columns (tile-local)
            if np.any(col_denom > 0):
                med_energy = float(np.median(col_denom[col_denom > 0]))
            else:
                med_energy = 0.0
            tiny_col = np.max((abs_zero, rel_zero * med_energy))
            freeze = col_denom <= tiny_col
            if freeze.any():
                g[freeze] = 0.0
                col_denom = np.where(freeze, np.inf, col_denom)

            # Blend with global column energy (stabilizes sparse tiles)
            if E_global_band is not None:
                Eg_row = np.asarray(E_global_band[bi, :], dtype=np.float64)
                if Eg_row.size == col_denom.size:
                    col_denom = np.maximum(col_denom, beta_blend * Eg_row)

            # Optional L2 term.
            #
            # If x_prior_band is not None:
            #   J_L2 = 0.5 * nnls_l2 * ||x_c - x_prior_c||^2   (seed as prior)
            # If x_prior_band is None:
            #   J_L2 = 0.5 * nnls_l2 * ||x_c||^2               (ridge to 0)
            #
            # Our g currently carries +A^T W R, i.e. the negative gradient
            # of the data term. The L2 term contributes
            #   -nnls_l2 * (x_c - x_prior_c)   or   -nnls_l2 * x_c
            # to this same descent direction, restricted to non-frozen cols.
            if nnls_l2 > 0.0:
                x_c = np.asarray(
                    x_band[bi, :], dtype=np.float64, copy=False
                )

                if x_prior_band is not None:
                    # pull toward prior
                    x_prior_c = np.asarray(
                        x_prior_band[bi, :],
                        dtype=np.float64,
                        copy=False,
                    )
                    delta = x_c - x_prior_c
                    if freeze.any():
                        mask = ~freeze
                        g[mask] -= nnls_l2 * delta[mask]
                    else:
                        g -= nnls_l2 * delta
                else:
                    # No prior supplied: pure ridge to zero
                    if freeze.any():
                        mask = ~freeze
                        g[mask] -= nnls_l2 * x_c[mask]
                    else:
                        g -= nnls_l2 * x_c

            invD = 1.0 / np.maximum(col_denom, eps)
            dx_c = float(lr) * (g * invD)  # (P,)

            # Propose NNLS candidates (optional polish)
            if nnls_prop_per_band > 0:
                score = np.abs(g) / (np.sqrt(np.maximum(col_denom, eps)) + eps)
                if np.any(freeze):
                    score[freeze] = 0.0
                k_keep = int(np.min((nnls_prop_per_band, int(np.count_nonzero(score > 0.0)))))
                if k_keep > 0:
                    top_p = np.argpartition(score, -k_keep)[-k_keep:]
                    ordr  = np.argsort(score[top_p])[::-1]
                    top_p = top_p[ordr]
                    top_global = (int(c) * P + top_p).astype(np.int64)
                    for j in range(top_p.size):
                        nnls_candidates.append(
                            (int(top_global[j]), float(score[top_p[j]]))
                        )

            # ΔR for α=1, computed in the **unweighted** space
            R_delta_band = np.zeros((Sblk, Lk), dtype=np.float64)
            for s in range(Sblk):
                R_delta_band[s, :] -= (
                    A[s, :, :].astype(np.float64, copy=False).T @ dx_c
                )

            # Trust region & backtracking in the **weighted** space
            if w_lam_sqrt is not None:
                Rw_delta_band = R_delta_band * wvec[None, :]
            else:
                Rw_delta_band = R_delta_band

            # Trust-region cap on α based on norms
            rd_norm = float(np.linalg.norm(Rw_delta_band))
            if rd_norm > 0.0:
                r_norm = float(np.linalg.norm(Rw))
                alpha_max = np.min((1.0, (tau_trust * r_norm) / rd_norm))
            else:
                alpha_max = 1.0

            # ---- Quadratic, O(1) evaluation for line search ----
            # ||Rw + a*Δ||^2 = rr + 2a*cr + a^2*rd
            rr  = float(np.dot(Rw.ravel(),           Rw.ravel()))
            rd2 = float(np.dot(Rw_delta_band.ravel(), Rw_delta_band.ravel()))
            cr  = float(np.dot(Rw.ravel(),           Rw_delta_band.ravel()))
            den = float(Rw.size) or 1.0

            def _rmse_w_at(a: float) -> float:
                val = rr + 2.0 * a * cr + (a * a) * rd2
                if val < 0.0:  # numerical safety
                    val = 0.0
                return math.sqrt(val / den)

            rmse_before_w = math.sqrt(rr / den)
            alpha = float(alpha_max)
            rmse_after_w = _rmse_w_at(alpha)

            if not (rmse_after_w < rmse_before_w):
                a = alpha
                for _ in range(bt_steps):
                    a *= bt_factor
                    if a <= 0.0:
                        break
                    if _rmse_w_at(a) < rmse_before_w:
                        alpha = a
                        break
                else:
                    alpha = a  # may be 0

            # finalize dx; if we clip nonnegativity, recompute ΔR / ΔRw for final dx
            dx_c = alpha * dx_c
            if project_nonneg:
                over_neg = dx_c < -x_band[bi, :]
                if np.any(over_neg):
                    dx_c[over_neg] = -x_band[bi, :][over_neg]
                    R_delta_band.fill(0.0)
                    for s in range(Sblk):
                        R_delta_band[s, :] -= (
                            A[s, :, :].astype(np.float64, copy=False).T @ dx_c
                        )
                    if w_lam_sqrt is not None:
                        Rw_delta_band = R_delta_band * wvec[None, :]
            else:
                if alpha != 1.0:
                    R_delta_band  *= alpha
                    Rw_delta_band *= alpha

            # Per-band global update-energy estimate (for a global cap)
            Eg_row = None
            if E_global_band is not None:
                Eg_row = np.asarray(E_global_band[bi, :], np.float64)
            if Eg_row is not None:
                upd_energy_sq = float(
                    np.sum((dx_c.astype(np.float64) ** 2) * Eg_row)
                )
            else:
                upd_energy_sq = 0.0

            R_delta += R_delta_band
            dx_list.append((c, dx_c, upd_energy_sq))

        if dbg and stats:
            try:
                print("[SAFE]", {k: int(v) for k, v in stats.items()})
            except Exception:
                pass

        return R_delta, dx_list, nnls_candidates

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

def solve_global_kaczmarz_cchunk_mp(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,   # (C,) or None
    x0: Optional[np.ndarray] = None,              # 1-D (C*P,) or None
    tracker: Optional[object] = None,             # FitTracker or None
    ratio_cfg: cu.RatioCfg | None = None,
) -> tuple[np.ndarray, dict]:

    pool = None
    try:

        def _set_chunk_cache(dset, cfg_obj):
            try:
                dset.id.set_chunk_cache(
                    cfg_obj.dset_slots, cfg_obj.dset_bytes, cfg_obj.dset_w0
                )
            except Exception:
                pass

        # ---- guard: ensure HyperCube normalization mode is declared
        mode = _assert_norm_mode(h5_path, expect=None)
        print(f"[Kaczmarz-MP] HyperCube norm.mode = '{mode}'", flush=True)

        # ------------------- read dims, chunks, mask -------------------
        t0 = time.perf_counter()
        with open_h5(h5_path, role="reader") as f:
            DC = f["/DataCube"]           # (S,L) float64
            M  = f["/HyperCube/models"]   # (S,C,P,L) float32
            _set_chunk_cache(M, cfg)

            S, L = map(int, DC.shape)
            _, C, P, Lm = map(int, M.shape)
            if Lm != L:
                raise RuntimeError(f"L mismatch: models L={Lm} vs data L={L}.")

            mask = cu._get_mask(f) if cfg.apply_mask else None
            keep_idx_lam = np.flatnonzero(mask) if mask is not None else None
            Lk = int(keep_idx_lam.size) if keep_idx_lam is not None else L
            # ------------------- column-flux scaling -------------------
            # cp_flux_ref[c,p] is the reference flux for column (c,p) in the *physical* basis
            cp_flux_ref = cu._ensure_cp_flux_ref(h5_path, keep_idx=keep_idx_lam)  # shape (C,P) or None
            cp_flux_ref = None

            if cp_flux_ref is not None:
                print("[Kaczmarz-MP] Using column-flux scaling.", flush=True)
                print(f"[Kaczmarz-MP] {'cp_flux_ref'}.min={np.min(cp_flux_ref):.3e}, "
                      f"max={np.max(cp_flux_ref):.3e}, "
                      f"median={np.median(cp_flux_ref[cp_flux_ref>0]):.3e}",
                      flush=True)
                cp_flux_ref = np.asarray(cp_flux_ref, np.float64).reshape(C, P)
                inv_cp_flux_ref = 1.0 / np.maximum(cp_flux_ref, 1.0e-30)
            else:
                # no scaling; everything stays physical
                cp_flux_ref = None
                inv_cp_flux_ref = None

            # ------------------- λ-weights (feature emphasis) -------------
            lamw_enable = os.environ.get(
                "CUBEFIT_LAMBDA_WEIGHTS_ENABLE", "1"
            ).lower() not in ("0", "false", "no", "off")
            lamw_dset   = os.environ.get(
                "CUBEFIT_LAMBDA_WEIGHTS_DSET", "/HyperCube/lambda_weights"
            )
            lamw_floor  = float(os.environ.get("CUBEFIT_LAMBDA_MIN_W", "1e-6"))
            lamw_auto   = os.environ.get(
                "CUBEFIT_LAMBDA_WEIGHTS_AUTO", "1"
            ).lower() not in ("0", "false", "no", "off")

            if lamw_enable:
                print(
                    f"[Kaczmarz-MP] Reading λ-weights from '{lamw_dset}' "
                    f"(floor={lamw_floor}, auto={lamw_auto})...", flush=True
                )
                try:
                    w_full = cu.read_lambda_weights(
                        h5_path, dset_name=lamw_dset, floor=lamw_floor
                    )
                except Exception:
                    w_full = cu.ensure_lambda_weights(
                        h5_path, dset_name=lamw_dset
                    ) if lamw_auto else np.ones(L, dtype=np.float64)
                print(
                    "[Kaczmarz-MP] λ-weights min={:.3e}, max={:.3e}, mean={:.3e}"
                    .format(float(np.min(w_full)),
                            float(np.max(w_full)),
                            float(np.mean(w_full))),
                    flush=True
                )
                w_lam = w_full[keep_idx_lam] if keep_idx_lam is not None else w_full
                # Use sqrt(w) on both A and R (weighted LS)
                w_lam_sqrt = np.sqrt(np.clip(w_lam, lamw_floor, None)).astype(
                    np.float64, order="C"
                )
            else:
                w_lam_sqrt = None

            # Spaxel tiling (follow /HyperCube/models chunking)
            s_tile = int(M.chunks[0]) if (M.chunks and M.chunks[0] > 0) else 128
            s_ranges = [(s0, np.min((S, s0 + s_tile))) for s0 in range(0, S, s_tile)]

            # ordering + global ||Y|| while we touch DC
            norms = []
            Y_glob_norm2 = 0.0
            for (ss0, ss1) in s_ranges:
                Yt = np.asarray(DC[ss0:ss1, :], np.float64)
                if keep_idx_lam is not None:
                    Yt = Yt[:, keep_idx_lam]
                norms.append(float(np.linalg.norm(Yt)))
                Y_glob_norm2 += float(np.sum(Yt * Yt))

        print(
            f"[Kaczmarz-MP] DataCube S={S}, L={L} (kept Lk={Lk}), "
            f"Hypercube C={C}, P={P}, s_tile={s_tile}, "
            f"epochs={cfg.epochs}, lr={cfg.lr}, "
            f"processes={cfg.processes}, blas_threads={cfg.blas_threads}",
            flush=True
        )

        # --- Persist orbit_weights and push into tracker ---
        print("[Kaczmarz-MP] Preparing orbit weights...", flush=True)
        print("[Kaczmarz-MP] _canon_orbit_weights(...)...", flush=True)
        w_prior = _canon_orbit_weights(h5_path, orbit_weights, C, P)
        print("[Kaczmarz-MP] done.", flush=True)
        if (w_prior is not None) and (tracker is not None) and hasattr(tracker, "set_orbit_weights"):
            print("[Kaczmarz-MP] Setting orbit weights in tracker...", flush=True)
            tracker.set_orbit_weights(w_prior)  # this mirrors your SP path behavior
            print("[Kaczmarz-MP] done.", flush=True)

        # sort by descending norm (hard / bright tiles first)
        paired = sorted(zip(norms, s_ranges), key=lambda t: -t[0])
        norms_sorted = [t[0] for t in paired]
        s_ranges_sorted = [t[1] for t in paired]

        # Optional tile budget for quick polish runs
        print("[Kaczmarz-MP] Applying max_tiles constraint...", flush=True)
        max_tiles = cfg.max_tiles
        if max_tiles is not None:
            max_tiles = int(max_tiles)
            if 0 < max_tiles < len(s_ranges_sorted):
                rng = np.random.default_rng(
                    int(os.environ.get("CUBEFIT_POLISH_SEED", "12345"))
                )

                k_bright = max(1, int(math.ceil(0.7 * max_tiles)))
                k_rand   = max(0, max_tiles - k_bright)

                bright_idx = np.arange(k_bright, dtype=int)
                tail_idx = np.arange(k_bright, len(s_ranges_sorted), dtype=int)
                if k_rand > 0 and tail_idx.size > 0:
                    rng.shuffle(tail_idx)
                    rand_idx = tail_idx[:k_rand]
                    keep_tiles_idx = np.concatenate([bright_idx, rand_idx])
                else:
                    keep_tiles_idx = bright_idx

                keep_tiles_idx = np.sort(keep_tiles_idx)
                s_ranges = [s_ranges_sorted[i] for i in keep_tiles_idx]
                norms_sorted = [norms_sorted[i] for i in keep_tiles_idx]
            else:
                s_ranges = s_ranges_sorted
        else:
            s_ranges = s_ranges_sorted
        print(f"[Kaczmarz-MP] Using {len(s_ranges)} tiles for fitting.", flush=True)

        Y_glob_norm = float(np.sqrt(Y_glob_norm2))


        # ------------------- global column energy & knobs -----------------
        print("[Kaczmarz-MP] Reading global column energy...", flush=True)
        E_global = read_global_column_energy(h5_path)  # (C,P) float64
        print("[Kaczmarz-MP] done.", flush=True)
        tau_global = float(os.environ.get("CUBEFIT_GLOBAL_TAU", "0.5"))
        beta_blend = float(os.environ.get("CUBEFIT_GLOBAL_ENERGY_BLEND", "1e-2"))

        # ------------------- x init --------------------------------------
        if x0 is None:
            # Start from zero *in the normalized basis*
            x_CP = np.zeros((C, P), dtype=np.float64, order="C")
        else:
            x0 = np.asarray(x0, np.float64).ravel(order="C")
            if x0.size != C * P:
                raise ValueError(f"x0 length {x0.size} != C*P={C*P}.")

            X_phys = x0.reshape(C, P)  # x0 is always in the physical basis

            if cp_flux_ref is not None:
                # Encode physical weights into the normalized basis the solver uses
                x_CP = X_phys * cp_flux_ref   # x_norm = x_phys * cp_flux_ref
            else:
                x_CP = X_phys.copy(order="C")

        # Tiny symmetry breaking (optional)
        sym_eps  = float(os.environ.get("CUBEFIT_SYMBREAK_EPS", "1e-6"))
        sym_mode = os.environ.get("CUBEFIT_SYMBREAK_MODE", "qr").lower()
        if sym_eps > 0.0 and sym_mode != "off" and (
            x0 is None or np.count_nonzero(x_CP) == 0
        ):
            print("[Kaczmarz-MP] Applying symmetry breaking...", flush=True)
            rng = np.random.default_rng(int(os.environ.get("CUBEFIT_SEED",
                                                        "12345")))
            if sym_mode == "qr":
                Rmat = rng.standard_normal((C, P))
                Q, _ = np.linalg.qr(Rmat.T, mode="reduced")
                Q = Q[:, :C].T  # (C,P)
                x_CP += sym_eps * np.abs(Q)
            else:
                x_CP += sym_eps * rng.random((C, P))
            if cfg.project_nonneg:
                np.maximum(x_CP, 0.0, out=x_CP)
            print("[Kaczmarz-MP] done.", flush=True)
        # x_CP holds the normalized weights actually updated by Kaczmarz.
        # Optionally keep a fixed prior (e.g. NNLS seed) for L2.
        use_prior = os.environ.get(
            "CUBEFIT_USE_NNLS_PRIOR", "1"
        ).lower() not in ("0", "false", "no", "off")

        if use_prior:
            print("[Kaczmarz-MP] Using NNLS seed as prior.", flush=True)
            x_CP_prior = x_CP.copy(order="C")
        else:
            print("[Kaczmarz-MP] Seed prior disabled.", flush=True)
            x_CP_prior = None

        # ----- global RMSE configuration (weighted/unweighted & guard) -----
        rmse_weighted = os.environ.get(
            "CUBEFIT_RMSE_WEIGHTED", "1"
        ).lower() not in ("0", "false", "no", "off")

        # Tolerance for "best epoch" tracking (used only for logging).
        rmse_guard_tol = float(
            os.environ.get("CUBEFIT_RMSE_GUARD_TOL", "0.0")
        )

        print(
            f"[Kaczmarz-MP] Global RMSE will be computed "
            f"{'with' if rmse_weighted else 'without'} λ-weights.",
            flush=True,
        )

        # # ------------------- global RMSE for the seed ----------------------
        # print("[Kaczmarz-MP] Computing global RMSE for the seed...", flush=True)
        # rmse_seed_raw = _compute_global_rmse(
        #     h5_path,
        #     x_CP,
        #     s_ranges=s_ranges,
        #     keep_idx=keep_idx,
        #     w_lam_sqrt=w_lam_sqrt,
        #     cp_flux_ref=cp_flux_ref,
        #     inv_cp_flux_ref=inv_cp_flux_ref,
        #     dset_slots=cfg.dset_slots,
        #     dset_bytes=cfg.dset_bytes,
        #     dset_w0=cfg.dset_w0,
        #     weighted=rmse_weighted,
        # )
        # rmse_seed = _safe_scalar_rmse(rmse_seed_raw, "seed")
        # print(
        #     f"[Kaczmarz-MP] seed global RMSE = {rmse_seed:.6e}",
        #     flush=True,
        # )

        # best_rmse = float(rmse_seed)
        # best_x_CP = x_CP.copy(order="C")

        best_rmse_proxy = np.inf
        best_x_CP       = x_CP.copy()

        # ------------------- ratio penalty (argument-driven) ------------------
        print(f"[Kaczmarz-MP] ratio_cfg pre-normalization: {type(ratio_cfg)}", flush=True)
        try:
            if ratio_cfg is not None and not isinstance(ratio_cfg, cu.RatioCfg):
                # Be conservative: only accept actual dict-like configs
                ratio_cfg = cu.RatioCfg(**dict(ratio_cfg))
            print(f"[Kaczmarz-MP] ratio_cfg post-normalization: {type(ratio_cfg)}",
                flush=True)
        except Exception as e:
            # If anything goes wrong, disable ratio and log
            print(f"[Kaczmarz-MP] ratio_cfg normalization failed: {e!r}. "
                f"Disabling ratio regularizer.", flush=True)
            ratio_cfg = None
        rc = ratio_cfg

        have_ratio = (orbit_weights is not None and rc is not None and rc.use)
        print(f"[Kaczmarz-MP] have_ratio = {have_ratio}", flush=True)
        if have_ratio:
            print("[Kaczmarz-MP] Preparing ratio regularizer...", flush=True)
            w_in = np.asarray(orbit_weights, np.float64).ravel(order="C")
            if w_in.size != C:
                raise ValueError(f"orbit_weights length {w_in.size} != C={C}.")

            # Base target mixture (normalize with a floor)
            w_target = np.maximum(w_in, rc.minw)
            w_target = w_target / np.maximum(np.sum(w_target), 1.0)

            # Optional anchor: mixture implied by the input x0 (also in physical units)
            if x0 is not None and x0.size == C * P:
                x0_mat = np.asarray(x0, np.float64).reshape(C, P, order="C")
                s0 = np.sum(np.maximum(x0_mat, 0.0), axis=1)  # physical mass per component
                S0 = np.sum(s0)
                t_x0 = (s0 / np.maximum(S0, 1.0)) if S0 > 0.0 else w_target
            else:
                t_x0 = None

            rng = np.random.default_rng()

            # Choose which target mixture to use at each update; for now just w_target.
            # You can later blend w_target and t_x0 if rc supports that.
            def _get_target_mix() -> np.ndarray:
                return w_target

            def _ratio_update_in_place(x_norm_mat: np.ndarray) -> None:
                """
                Mass-preserving multiplicative update toward the target mixture
                in *physical* space. x_norm_mat is the solver's internal normalized
                weights (x_norm = x_phys * cp_flux_ref).
                """
                # 1) Convert to physical weights
                if cp_flux_ref is not None:
                    x_phys = x_norm_mat * inv_cp_flux_ref     # x_phys = x_norm / cp_flux_ref
                else:
                    x_phys = x_norm_mat

                # 2) Compute physical mixture per component
                s = np.sum(x_phys, axis=1)                    # (C,)
                S = np.sum(s)
                if not np.isfinite(S) or S <= 0.0:
                    return

                sh = s / np.maximum(S, 1.0)                   # current physical mixture ŝ

                t_vec = _get_target_mix()                     # target physical mixture (sum=1)

                active = np.isfinite(sh) & np.isfinite(t_vec) & (t_vec > 0.0)
                idx = np.nonzero(active)[0]
                if idx.size == 0:
                    return

                # Optional stochastic subsampling
                if rc.prob < 1.0:
                    keep = rng.random(idx.size) < rc.prob
                    idx = idx[keep]
                    if idx.size == 0:
                        return
                if (rc.batch > 0) and (idx.size > rc.batch):
                    idx = rng.choice(idx, size=int(rc.batch), replace=False)

                # 3) Compute multiplicative factors (in log space) to move ŝ → t_vec
                e = np.log(np.maximum(sh[idx], rc.minw)) - np.log(
                    np.maximum(t_vec[idx], rc.minw)
                )
                f_idx = np.exp(-rc.eta * e)
                f_idx = np.clip(f_idx, 1.0 / rc.gamma, rc.gamma)

                F = np.ones((C,), dtype=np.float64)
                F[idx] = f_idx

                denom = np.sum(sh * F)
                if not np.isfinite(denom) or denom <= 0.0:
                    return

                # Renormalize to keep total mass fixed
                F = F / np.maximum(denom, 1.0e-30)

                # 4) Apply update in *physical* space, then convert back to normalized
                x_phys *= F[:, None]
                if cfg.project_nonneg:
                    np.maximum(x_phys, 0.0, out=x_phys)

                if cp_flux_ref is not None:
                    x_norm_mat[:, :] = x_phys * cp_flux_ref   # back to normalized basis
                else:
                    x_norm_mat[:, :] = x_phys

            print("[Kaczmarz-MP] Ratio regularizer ready.", flush=True)

        # ------------------- bands & pool -------------------
        print(f"[Kaczmarz-MP] Spinning up workers with {cfg.processes} processes...",
            flush=True)

        # Compute band layout from the requested process count
        print("[Kaczmarz-MP] Spinning up workers (entry)...", flush=True)
        nprocs_req = max(1, int(cfg.processes))
        print(f"[Kaczmarz-MP] nprocs_req = {nprocs_req}", flush=True)
        band_size = np.ceil(C / nprocs_req).astype(int)
        print(f"[Kaczmarz-MP] band_size = {band_size}", flush=True)
        bands: List[Tuple[int, int]] = []
        c0 = 0
        for i in range(nprocs_req):
            c1 = np.min((C, c0 + band_size))
            print(f"[Kaczmarz-MP] band loop i={i}, c0={c0}, c1={c1}", flush=True)
            if c1 > c0:
                bands.append((c0, c1))
            c0 = c1
        nprocs = len(bands)
        print(f"[Kaczmarz-MP] Using {nprocs} processes, band_size={band_size}.",
            flush=True)

        print(
            f"[Kaczmarz-MP] Using {nprocs} processes, band_size={band_size}.",
            flush=True,
        )

        # If we only have one band, we can skip mp.Pool entirely. This is both
        # a useful fallback and a very clean way to see if the hang is MP-only.
        use_pool = nprocs > 1

        if not use_pool:
            print(
                "[Kaczmarz-MP] Single-process mode "
                "(no multiprocessing pool will be used).",
                flush=True,
            )
            pool = None
        else:
            ctx_name = os.environ.get("CUBEFIT_MP_CTX", "forkserver")
            print(
                f"[Kaczmarz-MP] Using multiprocessing context '{ctx_name}'",
                flush=True,
            )
            try:
                ctx = mp.get_context(ctx_name)
            except ValueError:
                print(
                    "[Kaczmarz-MP] Context '{ctx_name}' unavailable; "
                    "falling back to 'spawn'.",
                    flush=True,
                )
                ctx = mp.get_context("spawn")

            # Anti-deadlock knobs in the parent
            os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
            os.environ.setdefault("OMP_NUM_THREADS", str(cfg.blas_threads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg.blas_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(cfg.blas_threads))

            max_tasks = int(os.environ.get("CUBEFIT_WORKER_MAXTASKS", "0"))
            ping_timeout = float(
                os.environ.get("CUBEFIT_POOL_PING_TIMEOUT", "5.0")
            )
            renew_each_epoch = os.environ.get(
                "CUBEFIT_POOL_RENEW_EVERY_EPOCH", "0"
            ).lower() not in ("0", "false", "no", "off")

            def _make_pool():
                print(
                    f"[Kaczmarz-MP] Creating pool with {nprocs} workers...",
                    flush=True,
                )
                ppool = ctx.Pool(
                    processes=nprocs,
                    initializer=_worker_init,
                    initargs=(int(cfg.blas_threads),),
                    maxtasksperchild=(None if max_tasks <= 0 else max_tasks),
                )
                print("[Kaczmarz-MP] Pool created.", flush=True)
                return ppool

            pool = _make_pool()
            print(
                "[Kaczmarz-MP] Workers ready. Starting epochs...",
                flush=True,
            )

        # Optional integer-shift diagnostic (cheap sampling)
        want_shift_diag = os.environ.get(
            "CUBEFIT_SHIFT_DIAG", "0"
        ).lower() not in ("0", "false", "no", "off")

        x = x_CP.ravel(order="C")   # view; keep and reuse everywhere

        try:
            for ep in range(cfg.epochs):
                if use_pool:
                    for ep in range(cfg.epochs):
                        if use_pool:
                            if (not _pool_ok(pool, timeout=ping_timeout)) or renew_each_epoch:
                                try:
                                    pool.close()
                                    pool.join()
                                except Exception:
                                    try:
                                        pool.terminate()
                                    except Exception:
                                        pass
                                pool = _make_pool()

                pbar = tqdm(total=len(s_ranges),
                            desc=f"[Kaczmarz-MP] epoch {ep+1}/{cfg.epochs}",
                            mininterval=2.0, dynamic_ncols=True)
                pbar.refresh()

                # Per-epoch target mixture t_norm (sum=1)
                if have_ratio:
                    if rc.anchor == "x0" and (t_x0 is not None):
                        t_epoch = t_x0.copy()
                    elif rc.anchor == "auto" and (t_x0 is not None):
                        t_epoch = t_x0.copy()
                    else:
                        t_epoch = w_target.copy()

                    if rc.epoch_renorm:
                        # keep a small floor to avoid zeros
                        t_epoch = np.maximum(t_epoch, rc.minw)
                        t_epoch = t_epoch / np.maximum(np.sum(t_epoch), 1.0)

                    t_norm = t_epoch
                else:
                    t_norm = None

                # --------- epoch-level RMSE proxy accumulator --------------
                # We sum ||R||^2 over all tiles using the *pre-update* R
                # (unweighted, masked wavelengths) and track the total count.
                rmse_sum_sq = 0.0
                rmse_count  = 0

                for tile_idx, (s0, s1) in enumerate(s_ranges):
                    Sblk = s1 - s0

                    # ---------- Build residual R = Y - yhat ----------
                    with open_h5(h5_path, role="reader") as f:
                        DC = f["/DataCube"]; M  = f["/HyperCube/models"]
                        try:
                            M.id.set_chunk_cache(cfg.dset_slots,
                                                cfg.dset_bytes, cfg.dset_w0)
                        except Exception:
                            pass

                        Y = np.asarray(DC[s0:s1, :], np.float64, order="C")
                        if keep_idx is not None:
                            Y = Y[:, keep_idx]

                        yhat = np.zeros((Sblk, Lk), np.float64)
                        for c in range(C):
                            A = np.asarray(M[s0:s1, c, :, :], np.float32, order="C")
                            if keep_idx is not None:
                                A = A[:, :, keep_idx]
                            if cp_flux_ref is not None:
                                # same cp normalization here
                                A = A * (inv_cp_flux_ref[c, :][None, :, None])
                            xc_norm = x_CP[c, :].astype(np.float64, copy=False)
                            for s in range(Sblk):
                                yhat[s, :] += xc_norm @ A[s, :, :]
                        R = Y - yhat


                    # --------- aggregate epoch RMSE proxy (pre-updates) -----
                    if not np.all(np.isfinite(R)):
                        bad = ~np.isfinite(R)
                        n_bad = int(bad.sum())
                        print(
                            f"[Kaczmarz-MP] WARNING: non-finite residuals on "
                            f"tile {tile_idx} (bad={n_bad}); zeroing.",
                            flush=True,
                        )
                        R = np.nan_to_num(
                            R, nan=0.0, posinf=0.0, neginf=0.0, copy=False
                        )
                    rmse_sum_sq += float(np.sum(R * R))
                    rmse_count  += int(R.size)

                    # Optional: light shift diagnostic on 1 spaxel in the tile
                    if want_shift_diag and Sblk > 0:
                        s_pick = s0
                        try:
                            y_obs = np.asarray(DC[s_pick, :], np.float64)
                            if keep_idx is not None:
                                y_obs = y_obs[keep_idx]
                            y_fit = yhat[s_pick - s0, :]
                            sh = _xcorr_int_shift(y_obs, y_fit)
                            if sh != 0:
                                print(f"[diag] spaxel {s_pick}: "
                                    f"data↔model integer shift = {sh} px")
                        except Exception:
                            pass

                    # ---------- RMSE BEFORE ANY UPDATES ----------
                    rmse_before = float(np.sqrt(np.mean(R * R)))

                    # Optional hard guard:
                    rmse_cap = float(os.environ.get("CUBEFIT_RMSE_ABORT",
                        "1e7"))
                    if (rmse_cap > 0.0) and (rmse_before > rmse_cap):
                        if tracker is not None:
                            tracker.on_batch_rmse(rmse_cap)
                        # Skip updates for this tile; keep x_CP and R as-is
                        continue

                    if tracker is not None:
                        tracker.on_batch_rmse(rmse_before)

                    # Cosine LR decay across tiles
                    if cfg.epochs >= 1:
                        frac = (tile_idx + 1) / len(s_ranges)
                        lr_tile = float(cfg.lr) * (0.5 + 0.5 * np.cos(
                            np.pi * frac
                        ))
                    else:
                        lr_tile = float(cfg.lr)

                    # ---------- Workers ----------
                    jobs = []
                    for (c_start, c_stop) in bands:
                        x_band = x_CP[c_start:c_stop, :].copy()
                        if x_CP_prior is not None:
                            x_prior_band = x_CP_prior[c_start:c_stop, :].copy()
                        else:
                            x_prior_band = None
                        E_band = E_global[c_start:c_stop, :]  # (band_size, P)
                        inv_ref_band = (
                            inv_cp_flux_ref[c_start:c_stop, :]
                            if cp_flux_ref is not None else None
                        )

                        jobs.append(
                            (
                                h5_path,
                                s0,
                                s1,
                                keep_idx,
                                c_start,
                                c_stop,
                                x_band,
                                x_prior_band,
                                float(lr_tile),
                                bool(cfg.project_nonneg),
                                R.copy(),
                                cfg.dset_slots,
                                cfg.dset_bytes,
                                cfg.dset_w0,
                                E_band,
                                float(beta_blend),
                                w_lam_sqrt,
                                inv_ref_band,
                            )
                        )

                    if use_pool:
                        results = pool.map(_worker_tile_job_with_R, jobs)
                    else:
                        # Single-process fallback: call the worker directly
                        results = [
                            _worker_tile_job_with_R(job) for job in jobs
                        ]

                    # --------------- aggregate & tile backtracking ---------------
                    R_delta_agg = np.zeros_like(R)
                    band_updates = []
                    cand_pairs = []
                    upd_energy_sq_total = 0.0

                    for item in results:
                        if isinstance(item, tuple) and len(item) == 3:
                            R_delta, dx_list, cands = item
                            if cands:
                                cand_pairs.extend(cands)
                        else:
                            R_delta, dx_list = item
                        R_delta_agg += R_delta
                        for tup in dx_list:
                            if len(tup) == 3:
                                c, dx_c, e2 = tup
                                upd_energy_sq_total += float(e2)
                            else:
                                c, dx_c = tup
                            band_updates.append((c, dx_c))

                    bt_steps_tile  = int(
                        np.max((0, int(os.environ.get("CUBEFIT_TILE_BT_STEPS", "6"))))
                    )
                    bt_factor_tile = float(
                        os.environ.get("CUBEFIT_TILE_BT_FACTOR", "0.5")
                    )

                    alpha = 1.0
                    rmse_after = float(
                        np.sqrt(np.mean((R + alpha * R_delta_agg) ** 2))
                    )
                    for _ in range(bt_steps_tile):
                        if rmse_after < rmse_before:
                            break
                        alpha *= bt_factor_tile
                        if alpha <= 0.0:
                            break
                        rmse_after = float(
                            np.sqrt(np.mean((R + alpha * R_delta_agg) ** 2))
                        )

                    if not np.isfinite(rmse_after):
                        alpha *= 0.0
                        rmse_after = rmse_before
                    if tracker is not None:
                        tracker.on_batch_rmse(rmse_after)

                    # -------- GLOBAL trust region (global energy & ||Y||) -------
                    if (upd_energy_sq_total > 0.0) and (Y_glob_norm > 0.0):
                        global_step_norm = float(
                            np.sqrt(upd_energy_sq_total)
                        ) * alpha
                        cap = float(tau_global * Y_glob_norm)
                        if global_step_norm > cap:
                            alpha *= float(cap / np.max((1e-12, global_step_norm)))

                    # ----------------- apply α ONCE to x and R -------------------
                    # ----------------- APPLY α (NaN-hardened) -----------------
                    # If backtracking produced nonsense, neutralize it.
                    if not np.isfinite(alpha):
                        alpha = 0.0

                    # R update: sanitize the aggregated residual step if needed.
                    if not np.all(np.isfinite(R_delta_agg)):
                        R_delta_agg = np.nan_to_num(
                            R_delta_agg, nan=0.0, posinf=0.0, neginf=0.0
                        )

                    R += alpha * R_delta_agg

                    # x updates: sanitize each band’s dx before applying.
                    for (c, dx_c) in band_updates:
                        if not np.all(np.isfinite(dx_c)):
                            # Zero-out any NaN/Inf in the step; keep shape, no copy.
                            dx_c = np.nan_to_num(dx_c, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                        x_CP[c, :] += alpha * dx_c
                        if cfg.project_nonneg:
                            np.maximum(x_CP[c, :], 0.0, out=x_CP[c, :])

                    # Safety net: if any NaNs still leaked into x_CP, zero them.
                    if not np.all(np.isfinite(x_CP)):
                        bad = ~np.isfinite(x_CP)
                        x_CP[bad] = 0.0

                    # ---------- optional ratio penalty --------------------------
                    if have_ratio and ((tile_idx % rc.tile_every) == 0):
                        _ratio_update_in_place(x_CP)

                    # ---------- Optional: tile-local NNLS polish ----------------
                    nnls_enable = os.environ.get(
                        "CUBEFIT_NNLS_ENABLE", "1"
                    ).lower() not in ("0", "false", "no", "off")
                    nnls_every = int(os.environ.get("CUBEFIT_NNLS_EVERY", "5"))
                    nnls_max_cols = int(
                        os.environ.get("CUBEFIT_NNLS_MAX_COLS", "256")
                    )
                    nnls_max_bytes = int(os.environ.get(
                        "CUBEFIT_NNLS_MAX_BYTES", str(1_000_000_000)
                    ))
                    nnls_sub_L = int(os.environ.get("CUBEFIT_NNLS_SUB_L", "0"))
                    nnls_solver = os.environ.get(
                        "CUBEFIT_NNLS_SOLVER", "nnls"
                    ).lower()  # "pg"|"fista"|"nnls"|"lsq"
                    nnls_max_iter = int(
                        os.environ.get("CUBEFIT_NNLS_MAX_ITER", "200")
                    )
                    nnls_min_improve = float(os.environ.get("CUBEFIT_NNLS_MIN_IMPROVE", "0.9995"))
                    nnls_l2 = float(os.environ.get("CUBEFIT_NNLS_L2", "0.0"))
                    if not np.isfinite(nnls_l2) or nnls_l2 < 0.0:
                        nnls_l2 = 0.0
                    # allow a small worsening of ratio misfit when polishing (>=1 means allow)
                    # e.g. 1.02 allows up to +2% worse local ratio misfit; 1.00 = never worsen.
                    nnls_ratio_worsen = float(os.environ.get("CUBEFIT_NNLS_RATIO_WORSEN", "1.02"))

                    do_nnls = (
                        nnls_enable and ((tile_idx % nnls_every) == 0)
                        and (len(cand_pairs) > 0)
                    )

                    if do_nnls:
                        # Deduplicate by best score per global index
                        idxs = np.asarray(
                            [int(t[0]) for t in cand_pairs], np.int64
                        )
                        scrs = np.asarray(
                            [float(t[1]) for t in cand_pairs], np.float64
                        )
                        best = {}
                        for i, sc in zip(idxs, scrs):
                            if (i not in best) or (sc > best[i]):
                                best[i] = sc
                        uniq_idx = np.fromiter(best.keys(), dtype=np.int64)
                        uniq_scr = np.fromiter(best.values(), dtype=np.float64)
                        order = np.argsort(uniq_scr)[::-1]
                        uniq_idx = uniq_idx[order]

                        Lk_loc = int(R.shape[1])
                        if (nnls_sub_L > 0) and (Lk_loc > nnls_sub_L):
                            rng = np.random.default_rng(12345 + tile_idx)
                            lam_sel = np.sort(
                                rng.choice(Lk_loc, size=int(nnls_sub_L),
                                        replace=False)
                            )
                        else:
                            lam_sel = np.arange(Lk_loc, dtype=np.int64)

                        rows = int(Sblk * lam_sel.size)
                        bytes_per_col = int(rows * 8)  # float64
                        cap_by_mem = int(np.max((1, nnls_max_bytes // np.max((1, bytes_per_col)))))
                        K_use = int(np.min((uniq_idx.size, nnls_max_cols, cap_by_mem)))

                        if (K_use >= 2) and (rows >= 2):
                            W_idx = uniq_idx[:K_use]

                            # λ-weights on the subsample
                            if w_lam_sqrt is not None:
                                wlam_sel = w_lam_sqrt[lam_sel].astype(
                                    np.float64, copy=False
                                )
                            else:
                                wlam_sel = np.ones(lam_sel.size, np.float64)
                            sqrt_w_rows = np.tile(wlam_sel, Sblk).astype(
                                np.float64
                            )  # (rows,)

                            # Build B and xW on the subsample
                            with open_h5(h5_path, role="reader") as f:
                                M = f["/HyperCube/models"]
                                PP = int(M.shape[2])
                                groups = {}
                                for j, g in enumerate(W_idx):
                                    cc = int(g // PP); pp = int(g % PP)
                                    groups.setdefault(cc, []).append((pp, j))

                                B = np.empty((rows, K_use), dtype=np.float64)
                                xW = np.zeros((K_use,), dtype=np.float64)
                                xW_prior = np.zeros(
                                    (K_use,), dtype=np.float64
                                )

                                for cc, plist in groups.items():
                                    A_c = np.asarray(M[s0:s1, cc, :, :],
                                                    np.float32, order="C")
                                    if keep_idx is not None:
                                        A_c = A_c[:, :, keep_idx]
                                    A_c = A_c[:, :, lam_sel]
                                    for pp, j in plist:
                                        col = np.asarray(
                                            M[s0:s1, cc, pp, :],
                                            dtype=np.float64,
                                        )
                                        if keep_idx is not None:
                                            col = col[:, keep_idx]
                                        if cp_flux_ref is not None:
                                            col *= float(
                                                inv_cp_flux_ref[int(cc),
                                                                int(pp)]
                                            )
                                        B[:, int(j)] = col
                                        xW[int(j)] = float(
                                            x_CP[int(cc), int(pp)]
                                        )
                                        xW_prior[int(j)] = float(
                                            x_CP_prior[int(cc), int(pp)]
                                        )


                            r_sub = R[:, lam_sel].reshape(rows, order="C")
                            y_rhs = B @ xW + r_sub

                            # Weighted system
                            B_w     = B * sqrt_w_rows[:, None]
                            y_rhs_w = y_rhs * sqrt_w_rows

                            # Optional L2 via Tikhonov augmentation:
                            # J(x) = 0.5||B_w x - y_rhs_w||^2
                            #      + 0.5*nnls_l2*||x||^2
                            if nnls_l2 > 0.0:
                                lam_sqrt = float(np.sqrt(nnls_l2))
                                # Augment with sqrt(λ) I and 0 target
                                B_aug = np.vstack([
                                    B_w,
                                    lam_sqrt * np.eye(K_use, dtype=np.float64)
                                ])
                                y_aug = np.concatenate([
                                    y_rhs_w,
                                    np.zeros(K_use, dtype=np.float64)
                                ])
                            else:
                                B_aug = B_w
                                y_aug = y_rhs_w

                            xW_new = None
                            if nnls_solver in ("pg", "fista"):
                                # Column-normalize local system (conditioning)
                                col_norm = np.linalg.norm(B_w, axis=0)
                                col_norm = np.where(
                                    col_norm > 0.0, col_norm, 1.0
                                )
                                Bn = B_w / col_norm
                                x = xW / col_norm

                                if nnls_l2 > 0.0:
                                    x_prior_norm = xW_prior / col_norm
                                else:
                                    x_prior_norm = np.zeros_like(x)

                                # FISTA with backtracking (fast & robust)
                                if K_use <= 2048:
                                    L_est = float(
                                        np.linalg.norm(Bn, ord=2) ** 2
                                        + nnls_l2
                                    )
                                else:
                                    L_est = float(
                                        (Bn ** 2).sum(axis=0).max()
                                        + nnls_l2
                                    )

                                t = 1.0
                                z = np.maximum(0.0, x.copy())
                                x_old = z.copy()

                                def _f_and_grad(zvec):
                                    r = Bn @ zvec - y_rhs_w
                                    if nnls_l2 > 0.0:
                                        diff = zvec - x_prior_norm
                                        f = (
                                            0.5 * float(r @ r)
                                            + 0.5 * nnls_l2
                                            * float(diff @ diff)
                                        )
                                        g = Bn.T @ r + nnls_l2 * diff
                                    else:
                                        f = 0.5 * float(r @ r)
                                        g = Bn.T @ r
                                    return f, g, r

                                fz, gz, rz = _f_and_grad(z)
                                step = 1.0 / max(L_est, 1e-6)

                                for _it in range(nnls_max_iter):
                                    # Armijo backtracking
                                    while True:
                                        x_try = z - step * gz
                                        np.maximum(x_try, 0.0, out=x_try)
                                        r_try = Bn @ x_try - y_rhs_w
                                        if nnls_l2 > 0.0:
                                            diff_try = x_try - x_prior_norm
                                            f_try = (
                                                0.5 * float(r_try @ r_try)
                                                + 0.5 * nnls_l2
                                                * float(diff_try @ diff_try)
                                            )
                                        else:
                                            f_try = 0.5 * float(r_try @ r_try)

                                        if (
                                            f_try
                                            <= fz
                                            - 1e-4 * step * float(gz @ gz)
                                            or step < 1e-12
                                        ):
                                            break
                                        step *= 0.5

                                    # FISTA momentum
                                    t_new = 0.5 * (
                                        1.0 + np.sqrt(1.0 + 4.0 * t * t)
                                    )
                                    z = x_try + ((t - 1.0) / t_new) * (
                                        x_try - x_old
                                    )
                                    x_old = x_try
                                    t = t_new
                                    fz, gz, rz = _f_and_grad(z)
                                    if float(rz @ rz) < 1e-12:
                                        break

                                xW_new = x_try * col_norm

                            elif nnls_solver == "lsq":
                                try:
                                    from scipy.optimize import lsq_linear
                                    res = lsq_linear(
                                        B_aug, y_aug,
                                        bounds=(0.0, np.inf),
                                        method="trf",
                                        max_iter=nnls_max_iter,
                                        verbose=0,
                                    )
                                    xW_new = np.maximum(0.0, res.x)
                                except Exception:
                                    xW_new = None
                            elif nnls_solver == "nnls":
                                try:
                                    from scipy.optimize import nnls as _scipy_nnls
                                    xW_new, _ = _scipy_nnls(B_aug, y_aug)
                                except Exception:
                                    xW_new = None

                            # Commit only if we actually solved something and it helped enough
                            if xW_new is not None:
                                dxW = xW_new - xW
                                if np.any(dxW != 0.0):

                                    # --- Weighted RMSE accept criterion (unchanged) ---
                                    def _wrmse(vec_1d: np.ndarray) -> float:
                                        w2 = sqrt_w_rows * sqrt_w_rows
                                        num = float(np.dot(w2, vec_1d * vec_1d))
                                        den = float(np.sum(w2)) or 1.0
                                        return math.sqrt(num / den)

                                    r_after = r_sub - (B @ dxW)   # NOTE: unweighted residual update
                                    rmse_before = _wrmse(r_sub)
                                    rmse_after  = _wrmse(r_after)
                                    ok_rmse = (rmse_after / max(1e-12, rmse_before)) < nnls_min_improve

                                    # --- ratio-drift guard vs global prior w_prior ---
                                    ok_ratio = True
                                    if have_ratio:
                                        # totals before
                                        s_before = x_CP.sum(axis=1)             # (C,)
                                        S_b = float(np.sum(s_before))
                                        if S_b > 0.0:
                                            mix_b = s_before / S_b

                                            # apply dxW only to the touched (c,p) indices
                                            comp_of_j = (W_idx // P).astype(np.int64)
                                            dS = np.zeros(C, dtype=np.float64)
                                            np.add.at(dS, comp_of_j, dxW)

                                            s_after = s_before + dS
                                            S_a = float(np.sum(s_after))
                                            if S_a > 0.0:
                                                mix_a = s_after / S_a
                                                # L1 deviation from target mixture w_prior
                                                dev_b = float(np.sum(np.abs(mix_b - w_prior)))
                                                dev_a = float(np.sum(np.abs(mix_a - w_prior)))
                                                ok_ratio = (dev_a <= dev_b * nnls_ratio_worsen + 1e-18)

                                    if ok_rmse and ok_ratio:
                                        # Commit to x_CP and to the *unweighted* residual R
                                        for j, g in enumerate(W_idx):
                                            cc = int(g // P); pp = int(g % P)
                                            x_CP[cc, pp] = float(max(0.0, x_CP[cc, pp] + dxW[j]))
                                        R[:, lam_sel] = r_after.reshape(Sblk, lam_sel.size, order="C")
                                    # else: reject this NNLS polish

                    # ---------- end NNLS polish ----------

                    if tracker is not None:
                        tracker.on_progress(
                            epoch=ep + 1,
                            spax_done=tile_idx + 1,
                            spax_total=len(s_ranges),
                            rmse_ewma=None
                        )
                        tracker.maybe_snapshot_x(x_CP, epoch=ep,
                                                rmse=rmse_after)

                    pbar.update(1)
                    pbar.refresh()

                pbar.close()

                # --------- finalize epoch-level RMSE proxy -----------------
                if rmse_count > 0:
                    mean_sq = rmse_sum_sq / float(rmse_count)
                else:
                    mean_sq = 0.0

                if (not np.isfinite(mean_sq)) or (mean_sq < 0.0):
                    print(
                        f"[Kaczmarz-MP] WARNING: epoch {ep+1} RMSE(proxy) "
                        f"mean_sq={mean_sq!r} non-finite or negative; "
                        f"setting RMSE(proxy)=+inf.",
                        flush=True,
                    )
                    rmse_epoch_proxy = float("inf")
                else:
                    rmse_epoch_proxy = float(math.sqrt(mean_sq))

                print(
                    f"[Kaczmarz-MP] epoch {ep+1} RMSE(proxy) = "
                    f"{rmse_epoch_proxy:.6e}",
                    flush=True,
                )

                if tracker is not None:
                    try:
                        tracker.on_epoch_end(
                            ep + 1,
                            {"rmse_epoch_proxy": rmse_epoch_proxy},
                            block=False,
                        )
                    except TypeError:
                        tracker.on_epoch_end(
                            ep + 1,
                            {"rmse_epoch_proxy": rmse_epoch_proxy},
                        )

                if w_prior is not None:
                    t0_sb = time.perf_counter()
                    orbBand, orbStep = softbox_params_smooth(
                        eq=ep, E=cfg.epochs)
                    cu.apply_component_softbox_energy(
                        x_CP, E_global,
                        (orbit_weights if orbit_weights is not None else np.ones(x_CP.shape[0])),
                        band=float(orbBand), step=float(orbStep),
                        min_target=1e-10
                    )

                    dt_sb = time.perf_counter() - t0_sb
                    print(f"[softbox] epoch {ep+1}: took {dt_sb:.4f}s",
                        flush=True)

                # ---------- optional orbit weights enforcement -----------
                t0_ob = time.perf_counter()
                if orbit_weights is not None and rc is not None and rc.epoch_project:
                    cu.project_to_component_weights(
                        x_CP,
                        orbit_weights,
                        E_cp=E_global,     # (C,P) from /HyperCube/col_energy
                        minw=1e-10,
                        beta=rc.epoch_beta  # e.g. 0.1–0.3
                    )

                    dt_ob = time.perf_counter() - t0_ob
                    print(f"[orbit-weights] epoch {ep+1}: took {dt_ob:.4f}s",
                        flush=True)
                    t0_ob = time.perf_counter()
                    x[:] = x_CP.ravel(order="C")  # keep your flattened view in sync
                    dt_ob = time.perf_counter() - t0_ob
                    print(f"[orbit-weights] epoch {ep+1}: sync took {dt_ob:.4f}s",
                        flush=True)

                    t0_usage = time.perf_counter()
                    # --- global energy L1-to-target diagnostic (robust) ---
                    try:
                        if E_global is not None:
                            print("[global energy] computing L1-to-target...", flush=True)
                            X64 = np.asarray(x_CP, dtype=np.float64)
                            E64 = np.asarray(E_global, dtype=np.float64)

                            # Sanitize E_global and X64 to avoid NaN/Inf poisoning
                            bad_E = ~np.isfinite(E64)
                            bad_X = ~np.isfinite(X64)
                            n_bad_E = int(bad_E.sum())
                            n_bad_X = int(bad_X.sum())
                            if n_bad_E or n_bad_X:
                                print(f"[global energy] WARNING: non-finite entries detected "
                                    f"in X/E (X bad={n_bad_X}, E bad={n_bad_E}); zeroing.",
                                    flush=True)
                                if n_bad_E:
                                    E64[bad_E] = 0.0
                                if n_bad_X:
                                    X64[bad_X] = 0.0

                            # Energy–weighted usage per component
                            s = (X64 * E64).sum(axis=1)  # (C,)
                        else:
                            # Plain usage
                            X64 = np.asarray(x_CP, dtype=np.float64)
                            bad_X = ~np.isfinite(X64)
                            if bad_X.any():
                                print(f"[global energy] WARNING: non-finite entries in X "
                                    f"(bad={int(bad_X.sum())}); zeroing.", flush=True)
                                X64[bad_X] = 0.0
                            s = X64.sum(axis=1)

                        # Now compute L1 safely
                        s = np.maximum(s, 0.0)
                        S = float(np.sum(s))
                        if not np.isfinite(S) or S <= 0.0:
                            print("[global energy] S is non-finite or <=0; skipping L1 diagnostic.",
                                flush=True)
                        else:
                            s_frac = s / S

                            # t is your target mix; make sure it's finite too
                            t_vec = (t_x0 if (rc.anchor == "x0" and t_x0 is not None) else w_target)
                            t = np.asarray(t_vec, np.float64).ravel(order="C")
                            if t.size == C * P:
                                t = t.reshape(C, P, order="C").sum(axis=1)
                            elif t.size != C:
                                raise ValueError(f"[ratio] target len {t.size} not in {{C, C*P}}")

                            t = np.maximum(np.nan_to_num(t, nan=0.0, posinf=0.0,
                                neginf=0.0),
                                rc.minw if rc is not None else 1e-10)
                            T = float(np.sum(t))
                            if not np.isfinite(T) or T <= 0.0:
                                print("[global energy] target sum non-finite or <=0; skipping L1.",
                                    flush=True)
                            else:
                                t_frac = t / T
                                diff = s_frac - t_frac
                                # clean diff just in case
                                diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                                l1 = float(np.sum(np.abs(diff)))
                                print(f"[ratio] epoch L1-to-target = {l1:.5e}", flush=True)

                    except Exception as e:
                        # Last-resort guard so the solver never 'hangs' here
                        print(f"[global energy] ERROR while computing L1-to-target: {e!r}",
                            flush=True)
                    dt_usage = time.perf_counter() - t0_usage
                    print(f"[ratio] epoch {ep+1}: usage calc took {dt_usage:.4f}s",
                        flush=True)

                print(f"[Kaczmarz-MP] epoch {ep+1}/{cfg.epochs} snapshotting...",
                    flush=True)

                try:
                    if tracker is not None:
                        tracker.maybe_snapshot_x(x_CP, epoch=ep+1,
                            rmse=rmse_after, force=True)
                except Exception:
                    pass

                # # ------------------- global RMSE for this epoch -------------
                # print(
                #     f"[Kaczmarz-MP] Computing global RMSE for "
                #     f"epoch {ep+1}...",
                #     flush=True,
                # )
                # rmse_epoch_raw = _compute_global_rmse(
                #     h5_path,
                #     x_CP,
                #     s_ranges=s_ranges,
                #     keep_idx=keep_idx,
                #     w_lam_sqrt=w_lam_sqrt,
                #     cp_flux_ref=cp_flux_ref,
                #     inv_cp_flux_ref=inv_cp_flux_ref,
                #     dset_slots=cfg.dset_slots,
                #     dset_bytes=cfg.dset_bytes,
                #     dset_w0=cfg.dset_w0,
                #     weighted=rmse_weighted,
                # )
                # rmse_epoch = _safe_scalar_rmse(rmse_epoch_raw, "epoch")
                # print(
                #     f"[Kaczmarz-MP] epoch {ep+1} global RMSE = "
                #     f"{rmse_epoch:.6e}",
                #     flush=True,
                # )

                # Track best epoch by global RMSE (proxy)
                if rmse_epoch_proxy + rmse_guard_tol < best_rmse_proxy:
                    best_rmse_proxy = rmse_epoch_proxy
                    best_x_CP = x_CP.copy()
                    print(
                        f"[Kaczmarz-MP] New best RMSE "
                        f"{best_rmse_proxy:.6e} at epoch {ep+1}.",
                        flush=True,
                    )
                else:
                    print(
                        f"[Kaczmarz-MP] RMSE did not improve over "
                        f"best={best_rmse_proxy:.6e}.",
                        flush=True,
                    )

                print(f"[Kaczmarz-MP] epoch {ep+1}/{cfg.epochs} housekeeping "
                    f"done.", flush=True)   

            elapsed = time.perf_counter() - t0

            if np.isfinite(best_rmse_proxy):
                x_CP[:, :] = best_x_CP

            # Sanitize the final solution and convert to physical basis
            # np.nan_to_num(
            #     x_CP, nan=0.0, posinf=0.0, neginf=0.0, copy=False
            # )
            np.nan_to_num(
                best_x_CP, nan=0.0, posinf=0.0, neginf=0.0, copy=False
            )

            if cp_flux_ref is not None:
                # Decode normalized solution back to physical weights
                X_norm = best_x_CP
                X_phys = X_norm * inv_cp_flux_ref   # x_phys = x_norm / cp_flux_ref
            else:
                X_phys = best_x_CP

            x_out = np.asarray(
                X_phys, np.float64, order="C"
            ).ravel(order="C")

            return x_out, dict(
                epochs=cfg.epochs,
                elapsed_sec=elapsed,
                # rmse_seed=rmse_seed,
                rmse_best=best_rmse_proxy,
            )
        finally:
            if pool is not None:
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    try:
                        pool.terminate()
                    except Exception:
                        pass

    except Exception as e:
        print(
            "[Kaczmarz-MP] FATAL exception in "
            "solve_global_kaczmarz_cchunk_mp:",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        # re-raise so the pipeline still fails loudly
        raise

# ------------------------------------------------------------------------------

def solve_global_kaczmarz_global_step_mp(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    tracker: Optional[object] = None,
    ratio_cfg: cu.RatioCfg | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Global-step Kaczmarz/gradient solver over the full HyperCube.

    This variant treats the whole cube as a single least-squares problem
    with objective

        J(x) = 0.5 * sum_{s,λ} w_λ (Y_{s,λ} - yhat_{s,λ}(x))^2
               + 0.5 * λ * ||x||^2,

    where x is the component–population weight matrix in the solver's
    normalized basis (x_CP). Each epoch does:

    1. Sweep over all spatial tiles, using the current x_CP to build the
       residual R_tile = Y_tile - yhat_tile(x_CP).
    2. For each tile and band of components, accumulate local contributions
       to the global gradient g_CP and diagonal preconditioner D_CP using
       `_worker_tile_global_grad_band`.
    3. After all tiles, take a single global projected step

           dx = lr * (g_eff / D_eff),
           x_CP <- max(0, x_CP + dx),

       with an optional global trust-region cap based on E_global and
       ||Y||.

    There are **no per-tile Kaczmarz updates** in this solver, no tile-level
    line search, and no tile-local NNLS polish. The only objective it
    optimizes is the global LS (plus optional L2), making its behaviour
    much easier to interpret relative to the NNLS seed.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 cube with /DataCube and /HyperCube/models.
    cfg : MPConfig
        Configuration for tiling, multiprocessing, epochs, LR, etc.
    orbit_weights : ndarray or None, optional
        Ignored in the pure-LS global step, but kept for API parity.
        Global orbit/ratio regularization is not applied here.
    x0 : ndarray or None, optional
        Initial weights in the **physical** basis, shape (C*P,). If None,
        start from zeros in the solver's normalized basis.
    tracker : object or None, optional
        Optional FitTracker-like object with methods:
            - on_batch_rmse(rmse: float)
            - on_progress(epoch, spax_done, spax_total, rmse_ewma=None)
            - on_epoch_end(epoch, stats_dict, block=True/False)
            - maybe_snapshot_x(x_CP, epoch, rmse, force=False)
    ratio_cfg : cu.RatioCfg or None, optional
        Ignored in this pure-LS solver. Present only for API compatibility.

    Returns
    -------
    x_out : ndarray, shape (C*P,), float64
        Final physical component–population weights, flattened in C-major
        order.
    info : dict
        Dictionary with basic run metadata, currently:
            - epochs : int
            - elapsed_sec : float

    Notes
    -----
    • This solver assumes `mode='model'` normalization on the HyperCube.
    • Column-flux normalization (`cp_flux_ref`) is allowed but disabled in
      your current runs; when present it is handled in the "normalized"
      basis internally and decoded back to physical at the end.
    • All global orbit/ratio/softbox regularization is intentionally
      disabled here to focus purely on the LS objective. You can layer it
      back later if this behaves well.
    """
    try:
        t0 = time.perf_counter()

        def _set_chunk_cache(dset, cfg_obj):
            try:
                dset.id.set_chunk_cache(
                    cfg_obj.dset_slots, cfg_obj.dset_bytes, cfg_obj.dset_w0
                )
            except Exception:
                pass

        # Normalization mode sanity check
        mode = _assert_norm_mode(h5_path, expect=None)
        print(f"[Kaczmarz-GLOBAL] HyperCube norm.mode = '{mode}'",
              flush=True)

        # ------------------- read dims, mask, cp_flux_ref -------------------
        with open_h5(h5_path, role="reader") as f:
            DC = f["/DataCube"]           # (S, L) float64
            M  = f["/HyperCube/models"]   # (S, C, P, L) float32
            _set_chunk_cache(M, cfg)

            S, L = map(int, DC.shape)
            _, C, P, Lm = map(int, M.shape)
            if Lm != L:
                raise RuntimeError(
                    f"L mismatch: models L={Lm} vs data L={L}."
                )

            mask = cu._get_mask(f) if cfg.apply_mask else None
            keep_idx = np.flatnonzero(mask) if mask is not None else None
            Lk = int(keep_idx.size) if keep_idx is not None else L

            # Column-flux scaling (model basis), explicitly gated by env
            cp_enable = os.environ.get(
                "CUBEFIT_CP_FLUX_ENABLE", "0"
            ).lower() not in ("0", "false", "no", "off")

            if cp_enable:
                cp_flux_ref = cu._ensure_cp_flux_ref(
                    h5_path, keep_idx=keep_idx
                )  # shape (C, P) or None
            else:
                cp_flux_ref = None

            if cp_flux_ref is not None:
                print(
                    "[Kaczmarz-GLOBAL] Using column-flux scaling.",
                    flush=True,
                )
                cp_flux_ref = np.asarray(
                    cp_flux_ref, np.float64
                ).reshape(C, P)
                print(
                    "[Kaczmarz-GLOBAL] cp_flux_ref: "
                    f"min={np.min(cp_flux_ref):.3e}, max={np.max(cp_flux_ref):.3e}, median={np.median(cp_flux_ref[cp_flux_ref > 0.0]):.3e}",
                    flush=True,
                )
                inv_cp_flux_ref = 1.0 / np.maximum(
                    cp_flux_ref, 1.0e-30
                )
            else:
                cp_flux_ref = None
                inv_cp_flux_ref = None

            # λ-weights (feature emphasis)
            lamw_enable = os.environ.get(
                "CUBEFIT_LAMBDA_WEIGHTS_ENABLE", "1"
            ).lower() not in ("0", "false", "no", "off")
            lamw_dset = os.environ.get(
                "CUBEFIT_LAMBDA_WEIGHTS_DSET",
                "/HyperCube/lambda_weights",
            )
            lamw_floor = float(
                os.environ.get("CUBEFIT_LAMBDA_MIN_W", "1e-6")
            )
            lamw_auto = os.environ.get(
                "CUBEFIT_LAMBDA_WEIGHTS_AUTO", "1"
            ).lower() not in ("0", "false", "no", "off")

            if lamw_enable:
                print(
                    "[Kaczmarz-GLOBAL] Reading λ-weights from "
                    f"'{lamw_dset}' (floor={lamw_floor}, auto={lamw_auto})",
                    flush=True,
                )
                try:
                    w_full = cu.read_lambda_weights(
                        h5_path, dset_name=lamw_dset, floor=lamw_floor
                    )
                except Exception:
                    w_full = (
                        cu.ensure_lambda_weights(
                            h5_path, dset_name=lamw_dset
                        )
                        if lamw_auto
                        else np.ones(L, dtype=np.float64)
                    )
                print(
                    "[Kaczmarz-GLOBAL] λ-weights "
                    f"min={np.min(w_full):.3e}, max={np.max(w_full):.3e}, mean={np.mean(w_full):.3e}",
                    flush=True,
                )
                w_lam = w_full[keep_idx] if keep_idx is not None else w_full
                w_lam_sqrt = np.sqrt(
                    np.clip(w_lam, lamw_floor, None)
                ).astype(np.float64, order="C")
            else:
                w_lam_sqrt = None

            # Spaxel tiling from HyperCube chunking
            s_tile = int(M.chunks[0]) if (M.chunks and M.chunks[0] > 0) \
                else 128
            s_ranges = [
                (s0, int(np.min((S, s0 + s_tile))))
                for s0 in range(0, S, s_tile)
            ]

            # Tile norms and global ||Y||
            norms = []
            Y_glob_norm2 = 0.0
            for (ss0, ss1) in s_ranges:
                Yt = np.asarray(DC[ss0:ss1, :], np.float64)
                if keep_idx is not None:
                    Yt = Yt[:, keep_idx]
                norms.append(float(np.linalg.norm(Yt)))
                Y_glob_norm2 += float(np.sum(Yt * Yt))

        print(
            f"[Kaczmarz-GLOBAL] DataCube S={S}, L={L} (kept Lk={Lk}), "
            f"Hypercube C={C}, P={P}, s_tile={s_tile}, epochs={cfg.epochs}, lr={cfg.lr}, "
            f"processes={cfg.processes}, blas_threads={cfg.blas_threads}",
            flush=True,
        )

        # # Sort tiles by descending norm (brightest first)
        # paired = sorted(
        #     zip(norms, s_ranges), key=lambda t: -t[0]
        # )
        # norms_sorted = [t[0] for t in paired]
        # s_ranges_sorted = [t[1] for t in paired]

        # # Optional tile budget (max_tiles)
        # max_tiles = cfg.max_tiles
        # print("[Kaczmarz-GLOBAL] Applying max_tiles constraint...",
        #       flush=True)
        # if max_tiles is not None:
        #     max_tiles = int(max_tiles)
        #     if 0 < max_tiles < len(s_ranges_sorted):
        #         import math
        #         rng_sel = np.random.default_rng(
        #             int(os.environ.get(
        #                 "CUBEFIT_POLISH_SEED", "12345"
        #             ))
        #         )

        #         k_bright = max(1, int(math.ceil(0.7 * max_tiles)))
        #         k_rand = max(0, max_tiles - k_bright)

        #         bright_idx = np.arange(k_bright, dtype=int)
        #         tail_idx = np.arange(
        #             k_bright, len(s_ranges_sorted), dtype=int
        #         )
        #         if k_rand > 0 and tail_idx.size > 0:
        #             rng_sel.shuffle(tail_idx)
        #             rand_idx = tail_idx[:k_rand]
        #             keep_idx_tiles = np.concatenate(
        #                 [bright_idx, rand_idx]
        #             )
        #         else:
        #             keep_idx_tiles = bright_idx

        #         keep_idx_tiles = np.sort(keep_idx_tiles)
        #         s_ranges = [s_ranges_sorted[i] for i in keep_idx_tiles]
        #         norms_sorted = [norms_sorted[i] for i in keep_idx_tiles]
        #     else:
        #         s_ranges = s_ranges_sorted
        # else:
        #     s_ranges = s_ranges_sorted

        print("[Kaczmarz-MP] Applying max_tiles constraint...", flush=True)
        max_tiles = cfg.max_tiles
        # ---- Build tile list ----
        s_ranges_all = list(s_ranges)  # keep the full list for fairness

        # ---------------- probe tiles for accept/backtracking ----------------
        n_probe = int(os.environ.get("CUBEFIT_PROBE_TILES", "6"))
        n_probe = max(1, min(n_probe, len(s_ranges_all)))

        probe_seed = int(os.environ.get("CUBEFIT_PROBE_SEED", "24680"))
        probe_tiles = _choose_tiles_fair_spread(
            s_ranges, n_probe, seed=probe_seed
        )

        probe_growth_max = float(
            os.environ.get("CUBEFIT_PROBE_GROWTH_MAX", "1.10")
        )
        if (not np.isfinite(probe_growth_max)) or (probe_growth_max <= 1.0):
            probe_growth_max = 1.10

        bt_max = int(os.environ.get("CUBEFIT_PROBE_BT_MAX", "8"))
        bt_max = max(1, bt_max)

        bt_shrink = float(os.environ.get("CUBEFIT_PROBE_BT_SHRINK", "0.5"))
        if (not np.isfinite(bt_shrink)) or (bt_shrink <= 0.0) or (bt_shrink >= 1.0):
            bt_shrink = 0.5

        allow_sign_flip = os.environ.get(
            "CUBEFIT_PROBE_SIGN_FLIP", "1"
        ).lower() not in ("0", "false", "no", "off")

        tile_norms: list[tuple[float, tuple[int, int]]] = []
        with open_h5(h5_path, role="reader") as f:
            DC = f["/DataCube"]              # (S, L)
            for (s0, s1) in s_ranges_all:
                Y = np.asarray(DC[s0:s1, :], np.float64, order="C")
                if keep_idx is not None:
                    Y = Y[:, keep_idx]       # (Sblk, Lk)
                tile_norms.append((float(np.linalg.norm(Y)), (s0, s1)))

        # Brightness ordering (optional, but fine as an iteration order)
        tile_norms.sort(key=lambda t: t[0], reverse=True)

        if max_tiles is not None and max_tiles < len(s_ranges_all):
            # Fair subset in space, then keep brightness ordering within that subset.
            fair_keep = _choose_tiles_fair_spread(
                s_ranges_all,
                int(max_tiles),
                seed=int(os.environ.get("CUBEFIT_GLOBAL_TILE_SEED", "12345")),
            )
            fair_keep_set = set(fair_keep)
            s_ranges = [t for _, t in tile_norms if t in fair_keep_set]
        else:
            s_ranges = [t for _, t in tile_norms]

        print(
            f"[Kaczmarz-GLOBAL] Using {len(s_ranges)} tiles for fitting.",
            flush=True,
        )

        Y_glob_norm = float(np.sqrt(Y_glob_norm2))

        # Global column energy (for trust-region scaling)
        print("[Kaczmarz-GLOBAL] Reading global column energy...",
            flush=True)
        E_global = read_global_column_energy(h5_path)  # (C, P) float64
        print("[Kaczmarz-GLOBAL] done.", flush=True)

        # ------------------- optional orbit_weights prior -------------------
        # Here we only prepare the target mix; the actual use is an optional
        # projection after each global step (see below).
        w_prior = None
        orbit_weights_arr = None
        if orbit_weights is not None:
            orbit_weights_arr = np.asarray(
                orbit_weights, np.float64
            ).ravel(order="C")
            if orbit_weights_arr.size != C:
                raise ValueError(
                    f"orbit_weights length {orbit_weights_arr.size} != C={C}."
                )
            w_prior = orbit_weights_arr.copy()

            if tracker is not None and hasattr(
                tracker, "set_orbit_weights"
            ):
                try:
                    tracker.set_orbit_weights(w_prior)
                except Exception:
                    pass

        tau_global = float(
            os.environ.get("CUBEFIT_GLOBAL_TAU", "0.5")
        )
        beta_blend = float(
            os.environ.get("CUBEFIT_GLOBAL_ENERGY_BLEND", "1e-2")
        )

        # ------------------- x initialization -------------------
        if x0 is None:
            x_CP = np.zeros((C, P), dtype=np.float64, order="C")
        else:
            x0 = np.asarray(x0, np.float64).ravel(order="C")
            if x0.size != C * P:
                raise ValueError(
                    f"x0 length {x0.size} != C*P={C*P}."
                )
            X_phys = x0.reshape(C, P)  # physical basis
            if cp_flux_ref is not None:
                x_CP = X_phys * cp_flux_ref
            else:
                x_CP = X_phys.copy(order="C")

        # Small symmetry breaking if starting from exact zero
        sym_eps = float(
            os.environ.get("CUBEFIT_SYMBREAK_EPS", "1e-6")
        )
        sym_mode = os.environ.get(
            "CUBEFIT_SYMBREAK_MODE", "qr"
        ).lower()
        if (
            sym_eps > 0.0
            and sym_mode != "off"
            and (x0 is None or np.count_nonzero(x_CP) == 0)
        ):
            print("[Kaczmarz-GLOBAL] Applying symmetry breaking...",
                  flush=True)
            rng_sb = np.random.default_rng(
                int(os.environ.get("CUBEFIT_SEED", "12345"))
            )
            if sym_mode == "qr":
                Rmat = rng_sb.standard_normal((C, P))
                Q, _ = np.linalg.qr(Rmat.T, mode="reduced")
                Q = Q[:, :C].T  # (C, P)
                x_CP += sym_eps * np.abs(Q)
            else:
                x_CP += sym_eps * rng_sb.random((C, P))
            if cfg.project_nonneg:
                np.maximum(x_CP, 0.0, out=x_CP)
            print("[Kaczmarz-GLOBAL] done.", flush=True)

        # This global-step solver ignores orbit_weights / ratio_cfg for now
        have_ratio = False
        w_prior = None

        # ------------------- multiprocessing setup -------------------
        print(
            f"[Kaczmarz-GLOBAL] Spinning up workers with "
            f"{cfg.processes} processes...",
            flush=True,
        )

        nprocs_req = max(1, int(cfg.processes))
        band_size = int(np.ceil(C / nprocs_req))
        bands: list[tuple[int, int]] = []
        c0 = 0
        for _i in range(nprocs_req):
            c1 = int(np.min((C, c0 + band_size)))
            if c1 > c0:
                bands.append((c0, c1))
            c0 = c1
        nprocs = len(bands)
        print(
            f"[Kaczmarz-GLOBAL] Using {nprocs} processes, "
            f"band_size={band_size}.",
            flush=True,
        )

        use_pool = nprocs > 1
        if not use_pool:
            print(
                "[Kaczmarz-GLOBAL] Single-process mode "
                "(no multiprocessing pool).",
                flush=True,
            )
            pool = None
        else:
            ctx_name = os.environ.get(
                "CUBEFIT_MP_CTX", "forkserver"
            )
            print(
                f"[Kaczmarz-GLOBAL] Using multiprocessing context "
                f"'{ctx_name}'",
                flush=True,
            )
            try:
                ctx = mp.get_context(ctx_name)
            except ValueError:
                print(
                    "[Kaczmarz-GLOBAL] Context '{ctx_name}' unavailable; "
                    "falling back to 'spawn'.",
                    flush=True,
                )
                ctx = mp.get_context("spawn")

            os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
            os.environ.setdefault(
                "OMP_NUM_THREADS", str(cfg.blas_threads)
            )
            os.environ.setdefault(
                "OPENBLAS_NUM_THREADS", str(cfg.blas_threads)
            )
            os.environ.setdefault(
                "MKL_NUM_THREADS", str(cfg.blas_threads)
            )

            max_tasks = int(
                os.environ.get("CUBEFIT_WORKER_MAXTASKS", "0")
            )
            ping_timeout = float(
                os.environ.get(
                    "CUBEFIT_POOL_PING_TIMEOUT", "5.0"
                )
            )
            renew_each_epoch = os.environ.get(
                "CUBEFIT_POOL_RENEW_EVERY_EPOCH", "0"
            ).lower() not in ("0", "false", "no", "off")

            def _make_pool():
                print(
                    f"[Kaczmarz-GLOBAL] Creating pool with {nprocs} "
                    f"workers...",
                    flush=True,
                )
                ppool = ctx.Pool(
                    processes=nprocs,
                    initializer=_worker_init,
                    initargs=(int(cfg.blas_threads),),
                    maxtasksperchild=(
                        None if max_tasks <= 0 else max_tasks
                    ),
                )
                print("[Kaczmarz-GLOBAL] Pool created.", flush=True)
                return ppool

            pool = _make_pool()

        want_shift_diag = os.environ.get(
            "CUBEFIT_SHIFT_DIAG", "0"
        ).lower() not in ("0", "false", "no", "off")

        # Kacz-style global step hyperparameters
        eps      = float(os.environ.get("CUBEFIT_EPS", "1e-12"))
        rel_zero = float(os.environ.get("CUBEFIT_ZERO_COL_REL", "1e-12"))
        abs_zero = float(os.environ.get("CUBEFIT_ZERO_COL_ABS", "1e-24"))

        # Allow disabling the tiny-column freeze for diagnostics.
        # If CUBEFIT_ZERO_COL_FREEZE in {0, false, no, off}, no columns are
        # frozen based on (rel_zero, abs_zero).
        zero_col_freeze = os.environ.get(
            "CUBEFIT_ZERO_COL_FREEZE", "1"
        ).lower() not in ("0", "false", "no", "off")

        kacz_l2 = float(os.environ.get("CUBEFIT_KACZ_L2", "0.0"))
        if not np.isfinite(kacz_l2) or kacz_l2 < 0.0:
            kacz_l2 = 0.0

        # Optional orbit-mix projection strength (0 = off)
        orbit_beta = float(
            os.environ.get("CUBEFIT_ORBIT_BETA", "0.0")
        )
        if not np.isfinite(orbit_beta) or orbit_beta < 0.0:
            orbit_beta = 0.0

        rmse_cap = float(
            os.environ.get("CUBEFIT_RMSE_ABORT", "0.0")
        )

        frac_bad_prev = None
        dx_base_norm_prev = None

        try:
            for ep in range(cfg.epochs):
                if use_pool:
                    if (not _pool_ok(pool, timeout=ping_timeout)) or \
                       renew_each_epoch:
                        try:
                            pool.close()
                            pool.join()
                        except Exception:
                            try:
                                pool.terminate()
                            except Exception:
                                pass
                        pool = _make_pool()

                # Global accumulators for this epoch
                g_tot = np.zeros((C, P), dtype=np.float64)
                D_tot = np.zeros((C, P), dtype=np.float64)
                rmse_sum_sq = 0.0
                rmse_count = 0

                pbar = tqdm(
                    total=len(s_ranges),
                    desc=(
                        f"[Kaczmarz-GLOBAL] epoch "
                        f"{ep+1}/{cfg.epochs}"
                    ),
                    mininterval=2.0,
                    dynamic_ncols=True,
                )
                pbar.refresh()

                for tile_idx, (s0, s1) in enumerate(s_ranges):
                    Sblk = s1 - s0

                    # ---------- Build residual R = Y - yhat ----------
                    with open_h5(h5_path, role="reader") as f:
                        DC = f["/DataCube"]
                        M  = f["/HyperCube/models"]
                        try:
                            M.id.set_chunk_cache(
                                cfg.dset_slots,
                                cfg.dset_bytes,
                                cfg.dset_w0,
                            )
                        except Exception:
                            pass

                        Y = np.asarray(
                            DC[s0:s1, :], np.float64, order="C"
                        )
                        if keep_idx is not None:
                            Y = Y[:, keep_idx]

                        # Model prediction with current global x_CP
                        yhat = np.zeros((Sblk, Lk), np.float64)
                        for c in range(C):
                            A = np.asarray(
                                M[s0:s1, c, :, :],
                                np.float32,
                                order="C",
                            )
                            if keep_idx is not None:
                                A = A[:, :, keep_idx]
                            if cp_flux_ref is not None:
                                A = (
                                    A
                                    * inv_cp_flux_ref[c, :][
                                        None, :, None
                                    ]
                                )
                            xc_norm = x_CP[c, :].astype(
                                np.float64, copy=False
                            )
                            # yhat[s, :] += xc_norm @ A[s, :, :]
                            yhat += np.tensordot(
                                xc_norm, A, axes=(0, 1)
                            )

                        R = Y - yhat

                    # Optional per-tile shift diagnostic
                    if want_shift_diag and Sblk > 0:
                        s_pick = s0
                        try:
                            y_obs = np.asarray(
                                DC[s_pick, :], np.float64
                            )
                            if keep_idx is not None:
                                y_obs = y_obs[keep_idx]
                            y_fit = yhat[s_pick - s0, :]
                            sh = _xcorr_int_shift(y_obs, y_fit)
                            if sh != 0:
                                print(
                                    f"[diag] spaxel {s_pick}: "
                                    "data↔model integer shift "
                                    f"= {int(sh)} px",
                                    flush=True,
                                )
                        except Exception:
                            pass

                    # NaN/Inf guard on R
                    if not np.all(np.isfinite(R)):
                        bad = ~np.isfinite(R)
                        n_bad = int(bad.sum())
                        print(
                            "[Kaczmarz-GLOBAL] WARNING: non-finite "
                            f"residuals on tile {tile_idx} "
                            f"(bad={n_bad}); zeroing.",
                            flush=True,
                        )
                        R = np.nan_to_num(
                            R,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                            copy=False,
                        )

                    # Per-tile RMSE (unweighted) before this global step
                    rmse_before = float(
                        np.sqrt(np.mean(R * R))
                    )

                    if (rmse_cap > 0.0) and (rmse_before > rmse_cap):
                        if tracker is not None:
                            tracker.on_batch_rmse(rmse_cap)
                        # Skip this tile in gradient accumulation
                        pbar.update(1)
                        pbar.refresh()
                        continue

                    if tracker is not None:
                        tracker.on_batch_rmse(rmse_before)

                    # Accumulate into the epoch-level global RMSE proxy
                    rmse_sum_sq += float(np.sum(R * R))
                    rmse_count += int(R.size)

                    # ---------- Gradient jobs for this tile ----------
                    jobs = []
                    inv_ref_band_full = (
                        inv_cp_flux_ref
                        if cp_flux_ref is not None
                        else None
                    )
                    for (c_start, c_stop) in bands:
                        x_band = x_CP[c_start:c_stop, :].copy(order="C")
                        inv_ref_band = (
                            inv_ref_band_full[c_start:c_stop, :]
                            if inv_ref_band_full is not None
                            else None
                        )
                        jobs.append(
                            (
                                h5_path,
                                int(s0),
                                int(s1),
                                keep_idx,
                                int(c_start),
                                int(c_stop),
                                x_band,
                                R.copy(order="C"),
                                w_lam_sqrt,
                                inv_ref_band,
                                cfg.dset_slots,
                                cfg.dset_bytes,
                                cfg.dset_w0,
                            )
                        )

                    if use_pool:
                        results = pool.map(
                            _worker_tile_global_grad_band, jobs
                        )
                    else:
                        results = [
                            _worker_tile_global_grad_band(job)
                            for job in jobs
                        ]

                    # Aggregate band contributions into global g_tot, D_tot
                    for (c_start, c_stop), (g_band, D_band) in zip(
                        bands, results
                    ):
                        c_start = int(c_start)
                        c_stop = int(c_stop)
                        g_tot[c_start:c_stop, :] += g_band
                        D_tot[c_start:c_stop, :] += D_band

                    if tracker is not None and False:
                        tracker.on_progress(
                            epoch=ep + 1,
                            spax_done=tile_idx + 1,
                            spax_total=len(s_ranges),
                            rmse_ewma=None,
                        )

                    pbar.update(1)
                    # pbar.refresh()

                pbar.close()

                # --------- finalize epoch-level RMSE proxy ----------
                if rmse_count > 0:
                    mean_sq = rmse_sum_sq / float(rmse_count)
                else:
                    mean_sq = 0.0

                if (not np.isfinite(mean_sq)) or (mean_sq < 0.0):
                    print(
                        "[Kaczmarz-GLOBAL] WARNING: epoch "
                        f"{ep+1} RMSE(proxy) mean_sq={mean_sq!r} "
                        "non-finite or negative; setting +inf.",
                        flush=True,
                    )
                    rmse_epoch_proxy = float("inf")
                else:
                    rmse_epoch_proxy = float(np.sqrt(mean_sq))

                print(
                    f"[Kaczmarz-GLOBAL] epoch {ep + 1} RMSE(proxy) = "
                    f"{rmse_epoch_proxy:.6e}",
                    flush=True,
                )

                if tracker is not None:
                    try:
                        tracker.on_epoch_end(
                            ep + 1,
                            {"rmse_epoch_proxy": rmse_epoch_proxy},
                            block=False,
                        )
                    except TypeError:
                        tracker.on_epoch_end(
                            ep + 1,
                            {"rmse_epoch_proxy": rmse_epoch_proxy},
                        )

                # --------- global Kaczmarz-style update in x_CP -----
                # Freeze numerically tiny columns based on global D_tot
                D = np.asarray(D_tot, dtype=np.float64, order="C")
                g = np.asarray(g_tot, dtype=np.float64, order="C")

                # Freeze numerically tiny columns globally (optional)
                if zero_col_freeze and np.any(D > 0):
                    med_energy = float(np.median(D[D > 0]))
                    tiny_col = np.max((abs_zero, rel_zero * med_energy))
                    freeze = D <= tiny_col
                    if freeze.any():
                        g[freeze] = 0.0
                        D = np.where(freeze, np.inf, D)
                else:
                    tiny_col = 0.0  # not used, kept for potential logging

                # Blend with global column energy to stabilize empty tiles
                if E_global is not None:
                    Eg = np.asarray(E_global, np.float64)
                    if Eg.shape == D.shape:
                        D = np.maximum(D, beta_blend * Eg)

                # Optional L2 term in the normalized basis
                if kacz_l2 > 0.0:
                    g -= kacz_l2 * x_CP

                # ---- Robust denom floor to prevent huge invD from tiny-but-nonzero D ----
                D_pos = D[np.isfinite(D) & (D > 0.0)]
                if D_pos.size:
                    D_scale = float(np.percentile(D_pos, 90.0))
                else:
                    D_scale = 0.0

                D_floor_frac = float(os.environ.get("CUBEFIT_DENOM_FLOOR_FRAC", "1e-8"))
                if (not np.isfinite(D_floor_frac)) or (D_floor_frac <= 0.0):
                    D_floor_frac = 1e-8

                D_floor = float(max(abs_zero, D_floor_frac * D_scale))
                n_clamped = 0
                if D_floor > 0.0:
                    n_clamped = int(np.count_nonzero(D < D_floor))
                    np.maximum(D, D_floor, out=D)
                print(
                    f"[Kaczmarz-GLOBAL] epoch {ep+1} denom floor = "
                    f"{D_floor:.3e} (clamped {n_clamped} cols), "
                    f" D_p90={D_scale:.3e}",
                    flush=True,
                )

                invD = 1.0 / np.maximum(D, eps)

                # ---- Adaptive effective learning rate for this epoch ----
                # Base step for lr=1.0
                dx_base = g * invD

                # ------------------------------------------------------------
                # Guard A: gradient geometry sanity check (pre-epoch)
                # ------------------------------------------------------------
                if cfg.project_nonneg:
                    frac_bad = float(np.mean(dx_base < -x_CP))
                else:
                    frac_bad = 0.0

                dx_base_norm = float(np.linalg.norm(dx_base))

                if frac_bad_prev is not None and dx_base_norm_prev is not None:
                    if (frac_bad > frac_bad_prev * 1.2) and (dx_base_norm > dx_base_norm_prev * 10.0):
                        print(
                            f"[Kaczmarz-GLOBAL] epoch {ep+1} ABORTED EARLY: "
                            f"active-set shock "
                            f"(frac_bad {frac_bad:.2f}, "
                            f"dx_norm {dx_base_norm:.2e})",
                            flush=True,
                        )
                        break

                frac_bad_prev = frac_bad
                dx_base_norm_prev = dx_base_norm

                # ------------------------------------------------------------
                # Descent-guaranteed global projected step (probe quadratic)
                # ------------------------------------------------------------
                lr = float(cfg.lr)
                x_before = x_CP.copy(order="C")

                # Start from the raw preconditioned direction.
                dx_dir = lr * dx_base

                # Enforce a *feasible projected-gradient* direction so small steps remain
                # feasible without needing "project full step once".
                if cfg.project_nonneg:
                    # Do not step negative at the boundary x=0.
                    dx_dir = np.where((x_before > 0.0) | (dx_dir > 0.0), dx_dir, 0.0)

                # Apply global trust cap to the direction (BT only scales down).
                step_scale_cap = 1.0
                if (Y_glob_norm > 0.0) and (E_global is not None):
                    Eg = np.asarray(E_global, np.float64)
                    step_energy = float(np.sum((dx_dir * dx_dir) * Eg))
                    cap = float(tau_global * Y_glob_norm)
                    if (step_energy > 0.0) and (cap > 0.0):
                        step_norm = float(np.sqrt(step_energy))
                        if step_norm > cap:
                            step_scale_cap = cap / max(step_norm, 1.0e-12)
                            dx_dir *= step_scale_cap


                def _probe_quadratic_terms(dx: np.ndarray) -> tuple[float, float, float, int]:
                    rr = 0.0
                    cr = 0.0
                    rd2 = 0.0
                    den = 0

                    with open_h5(h5_path, role="reader") as f:
                        DC = f["/DataCube"]
                        M = f["/HyperCube/models"]

                        for (ps0, ps1) in probe_tiles:
                            Y = np.asarray(DC[ps0:ps1, :], np.float64, order="C")
                            if keep_idx is not None:
                                Y = Y[:, keep_idx]

                            Sblk = int(ps1 - ps0)
                            yhat0 = np.zeros((Sblk, Lk), dtype=np.float64)
                            dyhat = np.zeros((Sblk, Lk), dtype=np.float64)

                            for c in range(C):
                                A = np.asarray(M[ps0:ps1, c, :, :], np.float32, order="C")
                                if keep_idx is not None:
                                    A = A[:, :, keep_idx]
                                if cp_flux_ref is not None:
                                    A = A * inv_cp_flux_ref[c, :][None, :, None]

                                A = A.astype(np.float64, copy=False)

                                yhat0 += np.tensordot(x_before[c, :], A, axes=(0, 1))
                                dyhat += np.tensordot(dx[c, :], A, axes=(0, 1))

                            if w_lam_sqrt is not None:
                                wv = w_lam_sqrt[None, :]
                                Rw = (Y - yhat0) * wv
                                dRw = (-dyhat) * wv
                            else:
                                Rw = (Y - yhat0)
                                dRw = (-dyhat)

                            r = Rw.ravel(order="C")
                            d = dRw.ravel(order="C")

                            rr += float(np.dot(r, r))
                            cr += float(np.dot(r, d))
                            rd2 += float(np.dot(d, d))
                            den += int(r.size)

                    return rr, cr, rd2, den


                def _rmse_from_quad(rr: float, cr: float, rd2: float, den: int,
                                    a: float) -> float:
                    val = rr + 2.0 * a * cr + (a * a) * rd2
                    if val < 0.0:
                        val = 0.0
                    return float(np.sqrt(val / max(den, 1)))


                # Compute probe quadratic once.
                rr, cr, rd2, den = _probe_quadratic_terms(dx_dir)
                rmse_probe_before = float(np.sqrt(rr / max(den, 1)))

                # Directional derivative sanity: for a descent direction we need cr < 0.
                # If it is not, flip the direction once and re-evaluate.
                step_sign = 1.0
                if allow_sign_flip and np.isfinite(cr) and (cr >= 0.0):
                    step_sign = -1.0
                    dx_dir = -dx_dir
                    if cfg.project_nonneg:
                        dx_dir = np.where((x_before > 0.0) | (dx_dir > 0.0), dx_dir, 0.0)
                    rr, cr, rd2, den = _probe_quadratic_terms(dx_dir)
                    rmse_probe_before = float(np.sqrt(rr / max(den, 1)))

                # Closed-form best step along this direction (quadratic is exact here).
                a_star = 0.0
                if np.isfinite(rd2) and (rd2 > 0.0) and np.isfinite(cr):
                    a_star = float(np.clip(-cr / rd2, 0.0, 1.0))

                # Backtrack only to enforce improvement.
                accepted = False
                bt_used = 0
                step_scale = 0.0
                rmse_probe_after = rmse_probe_before

                improve_tol = float(os.environ.get("CUBEFIT_PROBE_IMPROVE_TOL", "1e-12"))
                target = rmse_probe_before * (1.0 - improve_tol)

                for bt_used in range(bt_max):
                    a = a_star * float(bt_shrink ** bt_used)
                    if a <= 0.0:
                        break
                    rmse_a = _rmse_from_quad(rr, cr, rd2, den, a)
                    if np.isfinite(rmse_a) and (rmse_a < target):
                        accepted = True
                        step_scale = a
                        rmse_probe_after = rmse_a
                        break

                if accepted:
                    dx = step_scale * dx_dir
                    x_trial = x_before + dx
                    if cfg.project_nonneg:
                        np.maximum(x_trial, 0.0, out=x_trial)
                else:
                    dx = np.zeros_like(dx_dir)
                    x_trial = x_before

                x_CP[:, :] = x_trial
                dx_applied = dx

                # Diagnostics: fraction of entries that would violate NNLS without projection
                if cfg.project_nonneg:
                    frac_clipped = float(np.mean(dx_dir < -x_before))
                else:
                    frac_clipped = 0.0

                dx_norm = float(np.linalg.norm(dx))
                dx_app_norm = float(np.linalg.norm(dx_applied))
                x_norm = float(np.linalg.norm(x_CP))
                x_min = float(np.min(x_CP))
                x_max = float(np.max(x_CP))

                if E_global is not None:
                    Eg = np.asarray(E_global, np.float64)
                    dx_E = float(np.sqrt(np.sum((dx.astype(np.float64) ** 2) * Eg)))
                    dx_app_E = float(
                        np.sqrt(np.sum((dx_applied.astype(np.float64) ** 2) * Eg))
                    )
                else:
                    dx_E = float("nan")
                    dx_app_E = float("nan")

                print(
                    f"[Kaczmarz-GLOBAL] epoch {ep + 1} "
                    f"rmse_proxy={rmse_epoch_proxy:.3e} "
                    f"probe_before={rmse_probe_before:.3e} "
                    f"probe_after={rmse_probe_after:.3e} "
                    f"accepted={int(accepted)} bt={bt_used} sign={step_sign:+.0f} "
                    f"frac_clipped={frac_clipped:.3f} "
                    f"| dx_norm={dx_norm:.3e} dx_app_norm={dx_app_norm:.3e} "
                    f"step_scale={step_scale:.3e} "
                    f"| dx_E_norm={dx_E:.3e} dx_app_E_norm={dx_app_E:.3e} "
                    f"| x_norm={x_norm:.3e} x_min={x_min:.3e} x_max={x_max:.3e}",
                    flush=True,
                )

                if not np.all(np.isfinite(x_CP)):
                    bad = ~np.isfinite(x_CP)
                    print(
                        "[Kaczmarz-GLOBAL] WARNING: non-finite entries "
                        f"in x_CP after update (bad={int(bad.sum())}); "
                        "zeroing.",
                        flush=True,
                    )
                    x_CP[bad] = 0.0

                # Optional global projection toward orbit_weights
                if (
                    orbit_beta > 0.0
                    and orbit_weights_arr is not None
                    and E_global is not None
                ):
                    try:
                        cu.project_to_component_weights(
                            x_CP,
                            orbit_weights_arr,
                            E_cp=E_global,
                            minw=1.0e-10,
                            beta=float(orbit_beta),
                        )
                    except Exception as e:
                        print(
                            "[Kaczmarz-GLOBAL] WARNING: "
                            "orbit_weights projection failed: "
                            f"{e!r}",
                            flush=True,
                        )

                # Snapshot after the global step
                if tracker is not None:
                    try:
                        tracker.maybe_snapshot_x(
                            x_CP,
                            epoch=ep + 1,
                            rmse=rmse_epoch_proxy,
                            force=True,
                        )
                    except Exception:
                        pass

                print(
                    f"[Kaczmarz-GLOBAL] epoch {ep + 1}/{cfg.epochs} done.",
                    flush=True,
                )

            elapsed = time.perf_counter() - t0

            # Convert back to physical basis
            np.nan_to_num(
                x_CP,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                copy=False,
            )

            if cp_flux_ref is not None:
                X_norm = x_CP
                X_phys = X_norm * inv_cp_flux_ref
            else:
                X_phys = x_CP

            x_out = np.asarray(
                X_phys, np.float64, order="C"
            ).ravel(order="C")

            return x_out, dict(
                epochs=cfg.epochs,
                elapsed_sec=elapsed,
            )
        finally:
            if use_pool and (pool is not None):
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    try:
                        pool.terminate()
                    except Exception:
                        pass
    except Exception:
        print(
            "[Kaczmarz-GLOBAL] FATAL exception in "
            "solve_global_kaczmarz_global_step_mp:",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        raise

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
