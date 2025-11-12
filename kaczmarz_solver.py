# -*- coding: utf-8 -*-
r"""
    kaczmarz_solver.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Randomised Kaczmarz NNLS solvers for CubeFit: supports per-aperture, global
    (full-cube), block-constrained, and penalized fits. Designed for extremely
    large, memory-efficient IFU data decomposition.

    Notes
    -----
    * No direct HDF5 I/O here — everything flows through HyperCubeReader.
    * Progress bar shows per-tile ETA and total tile count for steady progress.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Basic per-aperture solver. 2025
v1.1:   Parallel global fit, block-sum constraints, soft penalties. 2025
v1.2:   Initialisation from stacked NNLS, custom momentum/projection logic. 2025
v1.3:   Complete re-write to use HDF5. 7 September 2025
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np
import time
from tqdm.auto import tqdm

import CubeFit.cube_utils as cu

@dataclass
class SolverCfg:
    # Core loop
    epochs: int = 1
    pixels_per_aperture: int = 256
    lr: float = 0.25
    project_nonneg: bool = True
    row_order: str = "random"            # "random" | "sequential"

    # BLAS threading inside solver: None => respect environment
    blas_threads: Optional[int] = None

    # Block Kaczmarz controls (enable when >1)
    block_rows: int = 1                  # e.g., 64–256
    block_norm: str = "sumsq"            # reserved for future variants

    # Ratio penalty (component-level)
    ratio_use: Optional[bool] = None     # None => auto (use if priors given)
    ratio_anchor: str | int = "auto"     # "auto" or int in [0,C)
    ratio_eta: Optional[float] = None    # default: 0.05 * lr
    ratio_prob: float = 0.02
    ratio_batch: int = 1
    ratio_min_weight: float = 1e-3

    # Misc
    seed: Optional[int] = None
    verbose: bool = True

def solve_global_kaczmarz(
    reader,
    cfg,
    *,
    orbit_weights: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
    on_progress: Callable[[int, dict], None] | None = None,
    progress_interval_sec: float = 300.0,
    on_batch_rmse: Callable[[float], None] | None = None,
):
    """
    Streaming, memory-lean Kaczmarz (global NNLS).

    The solver iterates over spaxels and a subset of pixel rows per spaxel.
    For each selected pixel `l`, it promotes the column `A[:, l]` to float64
    and performs a standard Kaczmarz update on the global coefficient vector
    `x` (kept in float64). No per-column scaling by priors is applied.

    Optionally, a *scale-invariant* ratio penalty can be injected to nudge the
    per-component sums toward prior *fractions* without fixing the absolute
    flux scale. Let `s_c = sum_p x_{c,p}`. For an anchor component `i`
    (typically the largest prior weight), we stochastically apply rows of the
    form:
        s_c - (w_c / w_i) * s_i ≈ 0
    with a small step size and low frequency. This preserves mixture ratios
    while allowing the overall normalization to be set by the data.

    Data updates:
      For L_eff pixels in a spaxel, pick K rows, promote a single column
      A[:, l] from f32 → f64 (into a pre-allocated buffer) and do a Kaczmarz
      step on the global x (f64).

    Ratio penalty (scale-invariant):
      Let s_c = sum_p x_{c,p}. Pick an anchor i (largest prior by default).
      Stochastically apply s_c - (w_c/w_i)*s_i ≈ 0 with small step (eta) and
      low freq. This preserves mixture ratios but not absolute scale.

    Callbacks:
      - on_progress(epoch, stats)     ~ every progress_interval_sec
      - on_batch_rmse(rmse)              once per spaxel (K-row RMSE)
      - on_epoch_end(epoch, stats)    end of epoch
    """

    rng = np.random.default_rng(getattr(cfg, "seed", None))
    C = int(reader.nComp)
    P = int(reader.nPop)
    N = C * P

    # ---------- priors -> per-component fractions (scale-free) ----------
    w_c = None
    if orbit_weights is not None:
        w_c = np.asarray(orbit_weights, dtype=np.float64).ravel()
        if w_c.size not in (C, N):
            raise ValueError(
                f"orbit_weights has size {w_c.size}, expected {C} or {N}"
            )
        if w_c.size == N:
            w_c = w_c.reshape(C, P).sum(axis=1)
        tot = float(w_c.sum())
        w_c = (w_c / tot) if tot > 0 else None

    # ---------- ratio penalty config ----------
    use_ratio = (
        cu._cfg_bool(cfg, "ratio_use", None)
        if getattr(cfg, "ratio_use", None) is not None
        else (w_c is not None)
    )
    lr = cu._cfg_float(cfg, "lr", 0.25)
    eta_pen = cu._cfg_float(cfg, "ratio_eta", 0.05 * lr)
    p_ratio = cu._cfg_float(cfg, "ratio_prob", 0.02)
    ratio_batch = cu._cfg_int(cfg, "ratio_batch", 1)
    min_w_frac = cu._cfg_float(cfg, "ratio_min_weight", 1e-3)
    if use_ratio and w_c is None:
        use_ratio = False

    if use_ratio:
        ra = getattr(cfg, "ratio_anchor", "auto")
        # Treat None the same as "auto" for robustness.
        if (ra is None) or (ra == "auto"):
            thr = float(min_w_frac) * float(w_c.max())
            good = (w_c >= thr)
            if not np.any(good):
                good = np.ones_like(w_c, dtype=bool)
            i_anchor = int(np.argmax(w_c * good))
        else:
            try:
                i_anchor = int(ra)
            except Exception as e:
                raise ValueError(
                    "ratio_anchor must be 'auto' or an int in [0, C)"
                ) from e
            if not (0 <= i_anchor < C):
                raise ValueError("ratio_anchor out of range")
        cand = np.arange(C, dtype=int)
        cand = cand[(cand != i_anchor) & (w_c > (min_w_frac * w_c.max()))]
        r_ci = np.empty(cand.size, dtype=np.float64)
        r_ci[:] = w_c[cand] / max(w_c[i_anchor], 1e-18)
    else:
        i_anchor, cand, r_ci = None, np.array([], int), np.array([], float)

    # ---------- solution vector ----------
    if x0 is None:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64, order="C").copy()

    # per-component sums cached lazily
    s_comp: Optional[np.ndarray] = None

    # ---------- core config ----------
    epochs = cu._cfg_int(cfg, "epochs", 1)
    K_req = cu._cfg_int(cfg, "pixels_per_aperture", 256)
    order = str(getattr(cfg, "row_order", "random"))
    proj_nn = cu._cfg_bool(cfg, "project_nonneg", True)
    blas_threads = getattr(cfg, "blas_threads", None)

    # block controls
    B = cu._cfg_int(cfg, "block_rows", 1)
    block_norm = str(getattr(cfg, "block_norm", "sumsq"))

    # ---------- small helpers ----------
    def _ratio_update(x_vec: np.ndarray,
                      s_vec: np.ndarray,
                      c: int, i: int, r: float) -> None:
        # residual for s_c - r * s_i ≈ 0
        e = s_vec[c] - r * s_vec[i]
        den = P * (1.0 + r * r)
        if den <= 0.0:
            return
        dc = -eta_pen * e / den
        di = +r * eta_pen * e / den
        c0 = c * P
        i0 = i * P
        x_vec[c0:c0 + P] += dc
        x_vec[i0:i0 + P] += di
        if proj_nn:
            bc = x_vec[c0:c0 + P].copy()
            bi = x_vec[i0:i0 + P].copy()
            np.maximum(x_vec[c0:c0 + P], 0.0, out=x_vec[c0:c0 + P])
            np.maximum(x_vec[i0:i0 + P], 0.0, out=x_vec[i0:i0 + P])
            s_vec[c] += float(x_vec[c0:c0 + P].sum() - bc.sum())
            s_vec[i] += float(x_vec[i0:i0 + P].sum() - bi.sum())
        else:
            s_vec[c] += P * dc
            s_vec[i] += P * di

    # ---------- main loop ----------
    t0 = time.perf_counter()
    with cu.blas_threads_ctx(blas_threads):
        for ep in range(epochs):
            verbose = cu._cfg_bool(cfg, "verbose", True)
            pbar = tqdm(
                total=reader.nSpat,
                desc=f"[Kaczmarz] epoch {ep + 1}/{epochs}",
                unit="spax",
                dynamic_ncols=True,
                leave=(ep == epochs - 1),
                disable=not verbose,
                mininterval=1.0,
            )
            last_tick = [time.perf_counter()]

            def _maybe_progress():
                if on_progress is None or progress_interval_sec is None:
                    return
                t_now = time.perf_counter()
                if t_now - last_tick[0] >= float(progress_interval_sec):
                    stats_epoch = {
                        "spaxels_done": int(pbar.n),
                        "nSpax": int(reader.nSpat),
                        "pixels_per_aperture": cu._cfg_int(cfg,
                            "pixels_per_aperture", 256),
                        "N": int(reader.nComp * reader.nPop),
                        "block_rows": cu._cfg_int(cfg, "block_rows", 1),
                    }
                    on_progress(ep + 1, stats_epoch)
                    last_tick[0] = t_now

            for s0, s1 in reader.spaxel_tiles():
                for s in range(s0, s1):
                    A_f32, y = reader.read_spaxel_plane(s)
                    L_eff = int(A_f32.shape[1])
                    if L_eff != y.size:
                        raise RuntimeError(
                            f"L mismatch at spaxel {s}: A has {L_eff}, "
                            f"y has {y.size}"
                        )

                    K = min(K_req, L_eff)
                    if order == "sequential":
                        idxK = np.arange(K, dtype=np.int64)
                    else:
                        idxK = rng.choice(L_eff, size=K, replace=False)

                    if B <= 1:
                        # ----- per-pixel classic Kaczmarz -----
                        a64 = np.empty(N, dtype=np.float64)
                        for l in idxK:
                            a64[:] = A_f32[:, l]
                            r = y[l] - np.dot(a64, x)
                            den = np.dot(a64, a64) + 1e-18
                            x += lr * (r / den) * a64
                    else:
                        # ----- block Kaczmarz (BLAS-2 GEMV) -----
                        Bmax = B
                        A_blk = np.empty((N, Bmax), dtype=np.float64, order="F")
                        y_blk = np.empty(Bmax, dtype=np.float64)
                        for b0 in range(0, K, B):
                            b1 = min(K, b0 + B)
                            m = b1 - b0
                            cols = idxK[b0:b1]
                            # Gather columns into A_blk[:, :m] as f64
                            np.copyto(
                                A_blk[:, :m],
                                A_f32[:, cols].astype(np.float64, copy=False),
                            )
                            y_blk[:m] = y[cols]
                            rB = y_blk[:m] - A_blk[:, :m].T @ x
                            if block_norm == "column":
                                # per-column normalization
                                denom = np.maximum(
                                    np.einsum("in,in->n",
                                              A_blk[:, :m], A_blk[:, :m],
                                              optimize=True),
                                    1e-18,
                                )
                                g = (A_blk[:, :m] * (rB / denom)).sum(axis=1)
                            else:
                                # sumsq of the whole block
                                denom = float(
                                    np.einsum("in,in->",
                                              A_blk[:, :m], A_blk[:, :m],
                                              optimize=True)
                                ) + 1e-18
                                g = (A_blk[:, :m] @ rB) / denom
                            x += lr * g

                    # occasional ratio penalty
                    if use_ratio and cand.size and (rng.random() < p_ratio):
                        if s_comp is None:
                            s_comp = x.reshape(C, P).sum(axis=1)
                        for _ in range(ratio_batch):
                            j = int(rng.integers(cand.size))
                            c_idx = int(cand[j])
                            _ratio_update(
                                x, s_comp, c_idx, int(i_anchor),
                                float(r_ci[j])
                            )
                        if rng.random() < 0.1:
                            s_comp = x.reshape(C, P).sum(axis=1)

                    if proj_nn:
                        np.maximum(x, 0, out=x)

                    # optional per-spaxel RMSE sampling
                    if on_batch_rmse is not None:
                        yhat = A_f32[:, idxK].T @ x
                        r = y[idxK] - yhat
                        rmse = float(np.sqrt(np.mean(r * r)))
                        on_batch_rmse(rmse)  # NEW API: single float

                    pbar.update(1)
                    _maybe_progress()

            pbar.close()
            if on_epoch_end is not None:
                stats_epoch = dict(
                    epoch=ep + 1,
                    elapsed_sec=None,  # fill by caller if desired
                    pixels_per_aperture=K_req,
                    blas_threads=blas_threads,
                    N=N,
                    block_rows=B,
                )
                on_epoch_end(ep + 1, stats_epoch)

    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed_sec": elapsed,
        "epochs": epochs,
        "pixels_per_aperture": K_req,
        "blas_threads": blas_threads,
        "N": N,
        "block_rows": B,
        "block_norm": block_norm,
        "ratio_used": bool(use_ratio),
        "ratio_anchor": (None if not use_ratio else int(i_anchor)),
        "ratio_prob": (p_ratio if use_ratio else 0.0),
        "ratio_eta": (eta_pen if use_ratio else 0.0),
        "ratio_batch": (ratio_batch if use_ratio else 0),
        "ratio_min_weight": (min_w_frac if use_ratio else 0.0),
    }
    return x, stats
