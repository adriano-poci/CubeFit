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
"""
# kaczmarz_solver_cchunk_mp.py


from __future__ import annotations, print_function

import os, sys, traceback
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from contextlib import contextmanager

import multiprocessing as mp
import numpy as np
import h5py
from tqdm import tqdm

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
     c_start, c_stop, x_band, lr, project_nonneg, R, w_band,
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
            inv_ref = np.asarray(inv_ref_band[bi, :], dtype=np.float64) if inv_ref_band is not None else np.ones((P,), dtype=np.float64)
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

            invD = 1.0 / np.maximum(col_denom, eps)
            dx_c = float(lr) * (g * invD)  # (P,)

            if w_band is not None:
                dx_c *= float(w_band[bi])

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
            keep_idx = np.flatnonzero(mask) if mask is not None else None
            Lk = int(keep_idx.size) if keep_idx is not None else L
            # ------------------- column-flux scaling -------------------
            # cp_flux_ref[c,p] is the reference flux for column (c,p) in the *physical* basis
            cp_flux_ref = cu._ensure_cp_flux_ref(h5_path, keep_idx=keep_idx)  # shape (C,P) or None
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
                w_lam = w_full[keep_idx] if keep_idx is not None else w_full
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
                if keep_idx is not None:
                    Yt = Yt[:, keep_idx]
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
                import math
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
                    keep_idx = np.concatenate([bright_idx, rand_idx])
                else:
                    keep_idx = bright_idx

                keep_idx = np.sort(keep_idx)
                s_ranges = [s_ranges_sorted[i] for i in keep_idx]
                norms_sorted = [norms_sorted[i] for i in keep_idx]
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
        print("[Kaczmarz-MP] Applying symmetry breaking (if needed)...", flush=True)
        sym_eps  = float(os.environ.get("CUBEFIT_SYMBREAK_EPS", "1e-6"))
        sym_mode = os.environ.get("CUBEFIT_SYMBREAK_MODE", "qr").lower()
        if sym_eps > 0.0 and sym_mode != "off" and (
            x0 is None or np.count_nonzero(x_CP) == 0
        ):
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
        # ------------------- ratio penalty (argument-driven) -------------------
        t_norm = None  # per-epoch normalized target, sum=1

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

        have_ratio = (
            (orbit_weights is not None)
            and (ratio_cfg is not None)
            and bool(getattr(ratio_cfg, "use", False))
        )
        print(f"[Kaczmarz-MP] have_ratio = {have_ratio}", flush=True)
        if have_ratio:
            print("[Kaczmarz-MP] Preparing ratio regularizer...", flush=True)
            w_in = np.asarray(orbit_weights, np.float64).ravel(order="C")
            if w_in.size != C:
                raise ValueError(f"orbit_weights length {w_in.size} != C={C}.")

            rc = ratio_cfg

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
                    if not _pool_ok(pool, timeout=ping_timeout) or renew_each_epoch:
                        try:
                            pool.close(); pool.join()
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

                for tile_idx, (s0, s1) in enumerate(s_ranges):
                    Sblk = s1 - s0

                    # ---------- Build residual R = Y - yhat ----------
                    with open_h5(h5_path, role="reader") as f:
                        DC = f["/DataCube"]; M  = f["/HyperCube/models"]
                        try: M.id.set_chunk_cache(cfg.dset_slots,
                            cfg.dset_bytes, cfg.dset_w0)
                        except Exception: pass

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

                    if tracker is not None:
                        tracker.on_batch_rmse(float(np.sqrt(np.mean(R * R))))

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
                        w_band = None if w_full is None else w_full[
                            c_start:c_stop
                        ].copy()
                        E_band = E_global[c_start:c_stop, :]  # (band_size, P)
                        inv_ref_band = inv_cp_flux_ref[c_start:c_stop, :] if cp_flux_ref is not None else None
                        # (band_size,P)
                        jobs.append(
                            (
                                h5_path,
                                s0,
                                s1,
                                keep_idx,
                                c_start,
                                c_stop,
                                x_band,
                                float(lr_tile),
                                bool(cfg.project_nonneg),
                                R.copy(),
                                w_band,
                                cfg.dset_slots,
                                cfg.dset_bytes,
                                cfg.dset_w0,
                                E_band,
                                float(beta_blend),
                                w_lam_sqrt,
                                inv_ref_band
                            ))

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

                    rmse_before = float(np.sqrt(np.mean(R * R)))
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
                    nnls_l2         = float(os.environ.get("CUBEFIT_NNLS_L2", "0.0"))
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

                                for cc, plist in groups.items():
                                    A_c = np.asarray(M[s0:s1, cc, :, :],
                                                    np.float32, order="C")
                                    if keep_idx is not None:
                                        A_c = A_c[:, :, keep_idx]
                                    A_c = A_c[:, :, lam_sel]
                                    for pp, j in plist:
                                        col = A_c[:, int(pp), :].astype(
                                            np.float64, copy=False
                                        ).reshape(rows, order="C")
                                        # normalize this (c,p) column
                                        if cp_flux_ref is not None:
                                            col *= float(inv_cp_flux_ref[int(cc), int(pp)])
                                        B[:, int(j)] = col
                                        xW[int(j)]   = float(x_CP[int(cc), int(pp)])


                            r_sub = R[:, lam_sel].reshape(rows, order="C")
                            y_rhs = B @ xW + r_sub

                            # Weighted system
                            B_w     = B * sqrt_w_rows[:, None]
                            y_rhs_w = y_rhs * sqrt_w_rows

                            xW_new = None
                            if nnls_solver in ("pg", "fista"):
                                # Column-normalize local system (conditioning)
                                col_norm = np.linalg.norm(B_w, axis=0)
                                col_norm = np.where(col_norm > 0.0, col_norm, 1.0)
                                Bn = B_w / col_norm
                                x  = xW / col_norm

                                # FISTA with backtracking (fast & robust)
                                if K_use <= 2048:
                                    L_est = float(np.linalg.norm(Bn, ord=2)**2
                                                + nnls_l2)
                                else:
                                    L_est = float(
                                        (Bn**2).sum(axis=0).max() + nnls_l2
                                    )
                                t = 1.0
                                z = np.maximum(0.0, x.copy())
                                x_old = z.copy()

                                def _f_and_grad(zvec):
                                    r = Bn @ zvec - y_rhs_w
                                    f = 0.5 * float(r @ r) + 0.5 * nnls_l2 * \
                                        float(zvec @ zvec)
                                    g = Bn.T @ r + nnls_l2 * zvec
                                    return f, g, r

                                fz, gz, rz = _f_and_grad(z)
                                step = 1.0 / max(L_est, 1e-6)

                                for _it in range(nnls_max_iter):
                                    # Armijo backtracking
                                    while True:
                                        x_try = z - step * gz
                                        np.maximum(x_try, 0.0, out=x_try)
                                        r_try = Bn @ x_try - y_rhs_w
                                        f_try = 0.5 * float(r_try @ r_try) \
                                                + 0.5 * nnls_l2 * \
                                                float(x_try @ x_try)
                                        if f_try <= fz - 1e-4 * step * \
                                            float(gz @ gz) or step < 1e-12:
                                            break
                                        step *= 0.5
                                    # FISTA momentum
                                    t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t*t))
                                    z = x_try + ((t - 1.0) / t_new) * (x_try - x_old)
                                    x_old = x_try
                                    t = t_new
                                    fz, gz, rz = _f_and_grad(z)
                                    if float(rz @ rz) < 1e-12:
                                        break

                                xW_new = x_try * col_norm

                            elif nnls_solver == "lsq":
                                try:
                                    from scipy.optimize import lsq_linear
                                    res = lsq_linear(B_w, y_rhs_w,
                                                    bounds=(0.0, np.inf),
                                                    method="trf",
                                                    max_iter=nnls_max_iter,
                                                    verbose=0)
                                    xW_new = np.maximum(0.0, res.x)
                                except Exception:
                                    xW_new = None
                            elif nnls_solver == "nnls":
                                try:
                                    from scipy.optimize import nnls as _scipy_nnls
                                    xW_new, _ = _scipy_nnls(B_w, y_rhs_w)
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

                if w_prior is not None:
                    t0_sb = time.perf_counter()
                    orbBand, orbStep = softbox_params_smooth(eq=ep, E=cfg.epochs)
                    cu.apply_component_softbox_energy(
                        x_CP, E_global,
                        (orbit_weights if orbit_weights is not None else np.ones(x_CP.shape[0])),
                        band=float(orbBand), step=float(orbStep), min_target=1e-10
                    )

                    dt_sb = time.perf_counter() - t0_sb
                    print(f"[softbox] epoch {ep+1}: took {dt_sb:.4f}s", flush=True)

                # ---------- optional orbit weights enforcement -----------
                t0_ob = time.perf_counter()
                if orbit_weights is not None and rc.epoch_project:
                    cu.project_to_component_weights(
                        x_CP,
                        orbit_weights,
                        E_cp=E_global,     # (C,P) from /HyperCube/col_energy
                        minw=1e-10,
                        beta=rc.epoch_beta  # e.g. 0.1–0.3
                    )

                    dt_ob = time.perf_counter() - t0_ob
                    print(f"[orbit-weights] epoch {ep+1}: took {dt_ob:.4f}s", flush=True)
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

                            t = np.maximum(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), rc.minw)
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
                        tracker.on_epoch_end(ep + 1, dict(), block=False)
                except Exception:
                    pass
                try:
                    if tracker is not None:
                        tracker.maybe_snapshot_x(x_CP, epoch=ep+1,
                            rmse=rmse_after, force=True)
                except Exception:
                    pass
                print(f"[Kaczmarz-MP] epoch {ep+1}/{cfg.epochs} housekeeping "
                    f"done.", flush=True)

            elapsed = time.perf_counter() - t0

            np.nan_to_num(x_CP, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

            # ------------------- finalize / convert back to physical basis -------------------
            if cp_flux_ref is not None:
                # Decode normalized solution back to physical weights
                X_norm = x_CP
                X_phys = X_norm * inv_cp_flux_ref   # x_phys = x_norm / cp_flux_ref
            else:
                X_phys = x_CP

            x_out = np.asarray(X_phys, np.float64, order="C").ravel(order="C")

            return x_out, dict(epochs=cfg.epochs, elapsed_sec=elapsed)
        finally:
            pool.close()
            pool.join()
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
