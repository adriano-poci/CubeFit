# -*- coding: utf-8 -*-
r"""
    kaczmarz_solver_batched.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Experimental: batched/tiled Kaczmarz with super-block GEMV/GEMM

    This module sits alongside kaczmarz_solver.py and exposes a solver
    that aggregates *columns from many spaxels* into large super-blocks
    to maximize BLAS throughput (OpenBLAS/MKL threads), while keeping
    HDF5 I/O sequential and chunk-aligned.

    Design highlights
    -----------------
    * Reads per-spaxel planes via your existing HyperCubeReader API.
    * Builds a large column super-block A[:, J] by concatenating K rows
      from many spaxels (in memory), computes a single block update:
          g  = A @ r / denom    with  r = y[J] - A.T @ x
    * Repeats across the S-tile; ratio penalty (if enabled) is injected
      sparsely, exactly like your current solver.

    Notes
    -----
    * Single-process, multi-threaded BLAS by default (set blas_threads).
    * Memory cap controls the super-block width (columns) — see cfg.
    * Callbacks (on_progress/on_epoch_end/on_batch_rmse) match your
      existing solver signatures, so PipelineRunner can reuse them.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   2025/09/28
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
import math
import time
import numpy as np
from tqdm.auto import tqdm

# Reuse your config helpers / context manager for BLAS threads
import CubeFit.cube_utils as cu
# Reuse your existing config dataclass to keep knobs consistent
from CubeFit.kaczmarz_solver import SolverCfg  # existing module, same repo


# --------------------------- Batched config -----------------------------

@dataclass
class BatchedCfg(SolverCfg):
    """
    Extends SolverCfg with batching knobs. Defaults are conservative and
    work within ~8–12 GiB for a 74.5k-by-4096 super-block (float32 A).

    Parameters
    ----------
    s_tile : int
        Number of spaxels to aggregate before flushing the super-block.
        Align with your reader's tile size (e.g., 128).
    super_cols_max : int
        Max columns in the super-block (controls memory). Each column is
        one pixel row from some spaxel. Memory ~ (N * super_cols_max * 4B).
    super_cols_min : int
        Minimum columns before triggering an update. Lets the solver
        start early if tiles are small.
    """
    s_tile: int = 128
    super_cols_max: int = 4096
    super_cols_min: int = 1024


# --------------------------- Solver entrypoint --------------------------

def solve_global_kaczmarz_batched(
    reader,
    cfg: BatchedCfg,
    *,
    orbit_weights: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    on_epoch_end: Callable[[np.ndarray, int, dict], None] | None = None,
    on_progress: Callable[[np.ndarray, int, dict], None] | None = None,
    progress_interval_sec: float = 300.0,
    on_batch_rmse: Callable[[float], None] | None = None,
) -> Tuple[np.ndarray, dict]:
    """
    Batched/tiled Kaczmarz that builds *large column super-blocks*
    across many spaxels to leverage BLAS throughput.

    The public API matches your existing solve_global_kaczmarz enough
    for drop-in use in PipelineRunner. The key difference is that
    updates are computed on a big in-memory block spanning many
    (spaxel, pixel) columns at once, which improves CPU utilization.

    Returns
    -------
    x : (C*P,) float64
        Final solution vector.
    stats : dict
        Run metadata (epochs, elapsed_sec, block sizes, ratio flags).
    """
    # -------------------- shapes & constants --------------------
    rng = np.random.default_rng(getattr(cfg, "seed", None))
    C = int(reader.nComp)
    P = int(reader.nPop)
    N = C * P

    # -------------------- priors → fractions --------------------
    w_c = None
    if orbit_weights is not None:
        w_c = np.asarray(orbit_weights, dtype=np.float64).ravel()
        if w_c.size not in (C, N):
            raise ValueError(f"orbit_weights has size {w_c.size}, "
                             f"expected {C} or {N}")
        if w_c.size == N:
            w_c = w_c.reshape(C, P).sum(axis=1)
        tot = float(w_c.sum())
        w_c = (w_c / tot) if tot > 0 else None

    # ratio penalty flags/params (same semantics as baseline solver)
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
    proj_nn = cu._cfg_bool(cfg, "project_nonneg", True)
    verbose = cu._cfg_bool(cfg, "verbose", True)
    order = str(getattr(cfg, "row_order", "random"))

    i_anchor, cand, r_ci = None, np.array([], int), np.array([], float)
    if use_ratio and w_c is not None:
        ra = getattr(cfg, "ratio_anchor", "auto")
        if (ra is None) or (ra == "auto"):
            thr = float(min_w_frac) * float(w_c.max())
            good = (w_c >= thr)
            if not np.any(good):
                good = np.ones_like(w_c, dtype=bool)
            i_anchor = int(np.argmax(w_c * good))
        else:
            i_anchor = int(ra)
            if not (0 <= i_anchor < C):
                raise ValueError("ratio_anchor out of range")
        cand = np.arange(C, dtype=int)
        cand = cand[(cand != i_anchor) & (w_c > (min_w_frac * w_c.max()))]
        r_ci = np.empty(cand.size, dtype=np.float64)
        r_ci[:] = w_c[cand] / max(w_c[i_anchor], 1e-18)
    else:
        use_ratio = False

    # -------------------- x init --------------------
    if x0 is None:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64, order="C").copy()

    # cached per-component sums for ratio updates
    s_comp: Optional[np.ndarray] = None

    # -------------------- batching knobs --------------------
    K_req = cu._cfg_int(cfg, "pixels_per_aperture", 256)
    S_tile = int(getattr(cfg, "s_tile", 128))
    J_max = int(getattr(cfg, "super_cols_max", 4096))
    J_min = int(getattr(cfg, "super_cols_min", 1024))
    blas_threads = getattr(cfg, "blas_threads", None)
    epochs = cu._cfg_int(cfg, "epochs", 1)

    # helpers for ratio penalty
    def _ratio_update(x_vec: np.ndarray,
                      s_vec: np.ndarray,
                      c: int, i: int, r: float) -> None:
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

    # -------------------- main loop --------------------
    t0 = time.perf_counter()
    with cu.blas_threads_ctx(blas_threads):
        for ep in range(epochs):
            pbar = tqdm(
                total=reader.nSpat,
                desc=f"[Kaczmarz-B] epoch {ep + 1}/{epochs}",
                unit="spax",
                dynamic_ncols=True,
                leave=(ep == epochs - 1),
                disable=not verbose,
                mininterval=1.0,
            )
            last_tick = [time.perf_counter()]

            # progress hook (solver-native: 3 args)
            def _maybe_progress():
                if on_progress is None or progress_interval_sec is None:
                    return
                t_now = time.perf_counter()
                if t_now - last_tick[0] >= float(progress_interval_sec):
                    stats_epoch = {
                        "spaxels_done": int(pbar.n),
                        "nSpax": int(reader.nSpat),
                        "pixels_per_aperture": K_req,
                        "N": int(reader.nComp * reader.nPop),
                        "super_cols_max": J_max,
                        "s_tile": S_tile,
                    }
                    on_progress(x, ep + 1, stats_epoch)
                    last_tick[0] = t_now

            # --- super-block buffers (allocated once, grown as needed)
            A_sb = None     # (N, J) float64 or float32 for memory
            y_sb = None     # (J,) float64
            j_fill = 0      # current fill of super-block
            s_in_tile = 0   # spaxels accumulated in current tile

            def _flush_superblock():
                """Apply one block update with current A_sb[:, :j_fill]."""
                nonlocal x, A_sb, y_sb, j_fill
                if j_fill == 0:
                    return
                # r = y - A.T @ x
                # Use float64 math; A_sb may be float32 to save RAM.
                A_view = A_sb[:, :j_fill].astype(np.float64, copy=False)
                y_view = y_sb[:j_fill]
                r = y_view - A_view.T @ x
                # denom: whole-block sumsq (stable), avoid division by 0
                denom = float(np.einsum("ij,ij->", A_view, A_view,
                                        optimize=True)) + 1e-18
                g = (A_view @ r) / denom
                x += lr * g
                if proj_nn:
                    np.maximum(x, 0.0, out=x)
                j_fill = 0  # reset; keep buffers for reuse

            # iterate over spaxel tiles from the reader
            for s0, s1 in reader.spaxel_tiles(tile_size=S_tile):
                s_in_tile = 0
                for s in range(s0, s1):
                    A_f32, y = reader.read_spaxel_plane(s)  # (N,Ls), (Ls,)
                    Ls = int(A_f32.shape[1])
                    if Ls != y.size:
                        raise RuntimeError(
                            f"L mismatch at spaxel {s}: A has {Ls}, y has "
                            f"{y.size}"
                        )

                    # choose K rows for this spaxel
                    K = min(K_req, Ls)
                    if order == "sequential":
                        idxK = np.arange(K, dtype=np.int64)
                    else:
                        idxK = rng.choice(Ls, size=K, replace=False)

                    # allocate super-block lazily (float32 for RAM)
                    if A_sb is None or A_sb.shape[1] < J_max:
                        # first alloc or grow to J_max
                        J_cap = J_max
                        A_sb = np.empty((N, J_cap), dtype=np.float32,
                                        order="F")
                        y_sb = np.empty(J_cap, dtype=np.float64)

                    # append this spaxel's K columns; if overflow, flush first
                    need = K
                    if j_fill + need > A_sb.shape[1]:
                        _flush_superblock()

                    # gather columns into the super-block
                    cols = idxK
                    # Copy A[:, cols] (float32) and y[cols] (float64)
                    np.copyto(A_sb[:, j_fill:j_fill + K],
                              A_f32[:, cols].astype(np.float32, copy=False))
                    y_sb[j_fill:j_fill + K] = y[cols].astype(np.float64,
                                                             copy=False)
                    j_fill += K
                    s_in_tile += 1
                    pbar.update(1)

                    # optional per-spaxel RMSE sample (using the K rows)
                    if on_batch_rmse is not None:
                        yhat = A_f32[:, cols].T @ x
                        r_s = y[cols] - yhat
                        rmse = float(np.sqrt(np.mean(r_s * r_s)))
                        on_batch_rmse(rmse)

                    # opportunistic ratio penalty (sparse)
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

                    # If we have enough columns, do a block update now
                    if j_fill >= J_min:
                        _flush_superblock()

                    _maybe_progress()

                # end of this S_tile — force a flush
                _flush_superblock()

            # end epoch
            if on_epoch_end is not None:
                stats_epoch = dict(
                    epoch=ep + 1,
                    elapsed_sec=None,
                    pixels_per_aperture=K_req,
                    blas_threads=blas_threads,
                    N=N,
                    s_tile=S_tile,
                    super_cols_max=J_max,
                )
                on_epoch_end(x, ep + 1, stats_epoch)

    # stats
    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed_sec": elapsed,
        "epochs": epochs,
        "pixels_per_aperture": K_req,
        "blas_threads": blas_threads,
        "N": N,
        "s_tile": S_tile,
        "super_cols_max": J_max,
        "super_cols_min": J_min,
        "ratio_used": bool(use_ratio),
        "ratio_anchor": (None if not use_ratio else int(i_anchor)),
        "ratio_prob": (p_ratio if use_ratio else 0.0),
        "ratio_eta": (eta_pen if use_ratio else 0.0),
        "ratio_batch": (ratio_batch if use_ratio else 0),
        "ratio_min_weight": (min_w_frac if use_ratio else 0.0),
    }
    return x, stats
