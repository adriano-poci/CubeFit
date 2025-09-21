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
from tqdm import tqdm

@dataclass
class SolverCfg:
    epochs: int = 1
    pixels_per_aperture: int = 256
    lr: float = 0.25
    project_nonneg: bool = True
    row_order: str = "random"        # "random" or "sequential"
    blas_threads: Optional[int] = None
    verbose: bool = True
    seed: Optional[int] = None
    # ratio-penalty (scale-invariant guidance from component priors)
    ratio_use: Optional[bool] = None   # None -> auto (True if orbit_weights provided)
    ratio_prob: float = 0.02           # chance to apply after a spaxel
    ratio_eta: Optional[float] = None  # defaults to 0.1 * lr
    ratio_anchor: str | int = "auto"   # "auto" (argmax weight) or integer index
    ratio_min_weight: float = 1e-5     # ignore tiny prior weights
    ratio_batch: int = 2               # how many constraints per trigger

# Optional control of BLAS/OpenMP threads (safe no-op if unavailable)
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None  # gracefully degrade if not installed

def _blas_ctx(nthreads: Optional[int]):
    """
    Context manager that limits BLAS/OpenMP threads to `nthreads`.
    If threadpoolctl is unavailable or nthreads is falsy, it's a no-op.
    """
    if _threadpool_limits and nthreads and int(nthreads) > 0:
        return _threadpool_limits(limits=int(nthreads))
    class _Nop:
        def __enter__(self): return None
        def __exit__(self, *exc): return False
    return _Nop()

def solve_global_kaczmarz(
    reader,
    cfg,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    on_epoch_end: Optional[Callable[[np.ndarray, int, dict], None]] = None,
    on_progress: Optional[Callable[[np.ndarray, int, dict], None]] = None,
    progress_interval_sec: Optional[float] = 300.0,
    on_batch_rmse: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, dict]:
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
      - on_progress(x, epoch, stats)     ~ every progress_interval_sec
      - on_batch_rmse(rmse)              once per spaxel (K-row RMSE)
      - on_epoch_end(x, epoch, stats)    end of epoch

    Parameters
    ----------
    reader
        A HyperCube reader exposing:
          - nComp (C), nPop (P), nSpat (S)
          - spaxel_tiles() -> iterable of (s0, s1)
          - read_spaxel_plane(s) -> (A_f32 (N,L_eff), y_f64 (L_eff,))
        where N = C * P.
    cfg
        Solver configuration object with fields (defaults shown):
          - epochs: int = 1
          - pixels_per_aperture: int = 256
          - lr: float = 0.25
          - project_nonneg: bool = True
          - row_order: str = "random"  # or "sequential"
          - blas_threads: Optional[int] = None
          - verbose: bool = True
          - seed: Optional[int] = None
        Ratio-penalty knobs (optional):
          - ratio_use: Optional[bool] = None
          - ratio_prob: float = 0.02
          - ratio_eta: Optional[float] = None  # default 0.1 * lr
          - ratio_anchor: "auto" or int = "auto"
          - ratio_min_weight: float = 1e-5
          - ratio_batch: int = 2
    orbit_weights
        Optional prior weights; length C or C*P. If provided, they are
        collapsed to per-component and normalized to fractions for ratio
        targets. If absent or degenerate, ratio penalty is disabled unless
        `cfg.ratio_use` is explicitly True (not recommended).
    x0
        Optional warm start vector of length N (= C*P). If None, zeros are
        used.
    on_epoch_end
        Optional callback `(x, epoch_index, stats_dict) -> None` invoked after
        each epoch. `x` is passed by reference (no copy). The callback should
        perform any I/O needed (checkpointing, metrics, etc.).

    Returns
    -------
    x : np.ndarray
        Final coefficient vector of shape (N,) in float64.
    stats : dict
        Summary statistics including elapsed seconds and ratio settings.

    Notes
    -----
    * All math uses float64 in-core; inputs from the reader may be float32.
    * Ratio rows are O(P) updates (uniform shifts within a component block).
    * Row scaling: the Kaczmarz update uses `denom = <a,a> + 1e-18`.
    """

    rng = np.random.default_rng(getattr(cfg, "seed", None))
    C = int(reader.nComp)
    P = int(reader.nPop)
    N = C * P

    # ----- priors → component fractions -----
    w_c = None
    if orbit_weights is not None:
        ow = np.asarray(orbit_weights, np.float64).ravel()
        if ow.size == C:
            w_c = ow.copy()
        elif ow.size == N:
            w_c = ow.reshape(C, P).sum(axis=1)
        else:
            raise ValueError(f"orbit_weights size {ow.size} not in {{C,N}}")
        tot = float(w_c.sum())
        if tot > 0:
            w_c = w_c / tot
        else:
            w_c = None

    use_ratio = bool(cfg.ratio_use) if cfg.ratio_use is not None else (w_c is not None)
    eta_pen = float(cfg.ratio_eta if cfg.ratio_eta is not None else 0.1 * float(getattr(cfg, "lr", 0.25)))
    p_ratio = float(getattr(cfg, "ratio_prob", 0.02))
    ratio_batch = int(getattr(cfg, "ratio_batch", 2))
    min_w = float(getattr(cfg, "ratio_min_weight", 1e-5))

    if use_ratio and w_c is not None:
        if cfg.ratio_anchor == "auto":
            good = (w_c >= min_w)
            if not np.any(good): good = np.ones_like(w_c, bool)
            i_anchor = int(np.argmax(w_c * good))
        else:
            i_anchor = int(cfg.ratio_anchor)
            if not (0 <= i_anchor < C):
                raise ValueError("ratio_anchor out of range")
        cand = np.arange(C, dtype=int)
        cand = cand[(cand != i_anchor) & (w_c > min_w)]
        r_ci = np.empty(cand.size, dtype=np.float64)
        r_ci[:] = w_c[cand] / max(w_c[i_anchor], 1e-18)
    else:
        i_anchor, cand, r_ci = None, np.array([], int), np.array([], float)

    # ----- solution and buffers -----
    x = np.zeros(N, np.float64) if x0 is None else np.asarray(x0, np.float64, order="C").copy()
    a64 = np.empty(N, np.float64)

    # ----- cfg -----
    epochs = int(getattr(cfg, "epochs", 1))
    K_req  = int(getattr(cfg, "pixels_per_aperture", 256))
    lr     = float(getattr(cfg, "lr", 0.25))
    order  = str(getattr(cfg, "row_order", "random"))
    proj_nn = bool(getattr(cfg, "project_nonneg", True))
    blas_threads = getattr(cfg, "blas_threads", None)
    verbose = bool(getattr(cfg, "verbose", True))

    def _ratio_update(x_vec: np.ndarray, sums: np.ndarray, c: int, i: int, r: float):
        e = sums[c] - r * sums[i]
        den = P * (1.0 + r*r)
        if den <= 0.0: return
        dc = - eta_pen * e / den
        di = + r       * eta_pen * e / den
        c0 = c * P; i0 = i * P
        x_vec[c0:c0+P] += dc
        x_vec[i0:i0+P] += di
        if proj_nn:
            xb = x_vec[c0:c0+P]; xi = x_vec[i0:i0+P]
            bc = xb.sum(); bi = xi.sum()
            np.maximum(xb, 0.0, out=xb)
            np.maximum(xi, 0.0, out=xi)
            sums[c] += float(xb.sum() - bc)
            sums[i] += float(xi.sum() - bi)
        else:
            sums[c] += P * dc
            sums[i] += P * di

    t0 = time.perf_counter()
    next_progress = t0 + (float(progress_interval_sec) if progress_interval_sec else float("inf"))

    with _blas_ctx(blas_threads):
        for ep in range(epochs):
            if verbose:
                print(f"[Kaczmarz] epoch {ep+1}/{epochs}")
            pbar = tqdm(total=reader.nSpat,
                        desc=f"[Kaczmarz] epoch {ep+1}/{epochs}",
                        unit="spax",
                        dynamic_ncols=True,
                        leave=(ep == epochs - 1),
                        disable=not verbose,
                        mininterval=1.0)

            s_comp = None  # lazily refreshed component sums

            for s0, s1 in reader.spaxel_tiles():
                for s in range(s0, s1):
                    A32, y = reader.read_spaxel_plane(s)   # (N,L_eff) f32, (L_eff,) f64
                    L_eff = int(A32.shape[1])
                    if L_eff == 0:
                        pbar.update(1); continue
                    K = min(K_req, L_eff)
                    if order == "sequential":
                        idx = np.arange(K, dtype=np.int64)
                    else:
                        idx = rng.choice(L_eff, size=K, replace=False)

                    rsq_sum = 0.0
                    for l in idx:
                        a64[:] = A32[:, l]          # promote single column
                        r = y[l] - np.dot(a64, x)
                        den = np.dot(a64, a64) + 1e-18
                        step = lr * (r / den)
                        x += step * a64
                        rsq_sum += float(r*r)

                    if on_batch_rmse is not None:
                        on_batch_rmse((rsq_sum / K) ** 0.5)

                    if use_ratio and cand.size and (rng.random() < p_ratio):
                        if s_comp is None:
                            s_comp = x.reshape(C, P).sum(axis=1)
                        for _ in range(ratio_batch):
                            j = int(rng.integers(cand.size))
                            _ratio_update(x, s_comp, int(cand[j]), int(i_anchor), float(r_ci[j]))
                        if rng.random() < 0.1:
                            s_comp = x.reshape(C, P).sum(axis=1)

                    # time-gated progress callback
                    if on_progress is not None and time.perf_counter() >= next_progress:
                        stats_ep = {"epoch": ep + 1, "N": N}
                        try:
                            on_progress(x, ep + 1, stats_ep)
                        except Exception:
                            pass
                        next_progress = time.perf_counter() + (float(progress_interval_sec) if progress_interval_sec else float("inf"))

                    pbar.update(1)

            if proj_nn:
                np.maximum(x, 0.0, out=x)

            pbar.close()
            if on_epoch_end is not None:
                stats_epoch = {"epoch": ep + 1,
                               "elapsed_sec": time.perf_counter() - t0,
                               "pixels_per_aperture": K_req,
                               "blas_threads": blas_threads,
                               "N": N}
                try:
                    on_epoch_end(x, ep + 1, stats_epoch)
                except Exception:
                    pass

    stats = {"elapsed_sec": time.perf_counter() - t0,
             "epochs": epochs,
             "N": N,
             "ratio_used": bool(use_ratio),
             "ratio_anchor": (None if not use_ratio else int(i_anchor)),
             "ratio_prob": (p_ratio if use_ratio else 0.0),
             "ratio_eta": (eta_pen if use_ratio else 0.0),
             "ratio_batch": (ratio_batch if use_ratio else 0),
             "ratio_min_weight": (min_w if use_ratio else 0.0),
             "pixels_per_aperture": K_req}
    return x, stats