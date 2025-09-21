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

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

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

    # --- NEW (ratio-penalty) ---
    ratio_use: Optional[bool] = None   # None -> auto (True if orbit_weights given)
    ratio_prob: float = 0.02           # probability to apply ratio penalty after a spaxel
    ratio_eta: Optional[float] = None  # default: 0.1 * lr
    ratio_anchor: str | int = "auto"   # "auto" (argmax w) or explicit component index
    ratio_min_weight: float = 1e-5     # ignore tiny-weight components
    ratio_batch: int = 2               # how many ratio rows per trigger

def _blas_ctx(nthreads: Optional[int]):
    if threadpool_limits and nthreads and nthreads > 0:
        return threadpool_limits(limits=int(nthreads))
    class _Nop:
        def __enter__(self): return None
        def __exit__(self, *a): return False
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

    Core data updates:
      • For each selected pixel l, promote column A[:, l] (float32) to a
        float64 buffer and perform a standard Kaczmarz step on x (float64).
      • No per-column scaling by priors is applied.

    Optional scale-invariant ratio penalty:
      • Let s_c = sum_p x_{c,p}. For an anchor component i (usually the
        largest prior weight), stochastically enforce
            s_c - (w_c / w_i) * s_i ≈ 0
        with small step-size and low frequency. This nudges mixture ratios
        without fixing absolute normalization.

    Callbacks:
      • on_epoch_end(x, epoch, stats): called once per epoch (x is live).
      • on_progress(x, epoch, stats): called every ~progress_interval_sec
        within an epoch (x is live). Set interval to None to disable.
      • on_batch_rmse(rmse): called once per spaxel with RMSE over that
        spaxel’s selected rows; useful for a tracker EWMA.

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

    # None-aware config reader
    def _cfg_val(obj, name, default):
        v = getattr(obj, name, None)
        return default if v is None else v

    # Base learning rate used in several defaults
    lr = float(_cfg_val(cfg, "lr", 0.25))

    # ---- Priors: per-component targets (length C) for ratios ----
    w_c = None
    if orbit_weights is not None:
        w_c = np.asarray(orbit_weights, dtype=np.float64).ravel()
        if w_c.size != C:
            raise ValueError(
                f"orbit_weights must have length C={C}; got {w_c.size}"
            )
        # Do not normalize: ratios w_c[c]/w_c[i] are scale-invariant.

        # ---- Ratio configuration (mass + shape preserving) ----
        use_ratio = (_cfg_val(cfg, "ratio_use", None)
                    if hasattr(cfg, "ratio_use") else None)
        use_ratio = (w_c is not None) if (use_ratio is None) else bool(use_ratio)

        # Gentle defaults so data rows dominate scale
        ratio_prob   = float(_cfg_val(cfg, "ratio_prob", 0.005))
        ratio_batch  = int(_cfg_val(cfg, "ratio_batch", 1))
        eta_pen      = float(_cfg_val(cfg, "ratio_eta", 0.05 * lr))
        min_w_frac   = float(_cfg_val(cfg, "ratio_min_weight", 1e-3))

    if use_ratio and (w_c is not None):
        wmax = float(np.max(w_c)) if w_c.size else 0.0
        thr = max(min_w_frac * max(wmax, 1.0), 0.0)

        anch = getattr(cfg, "ratio_anchor", "auto")
        if anch == "auto":
            good = (w_c >= thr)
            if not np.any(good):
                good = np.ones_like(w_c, dtype=bool)
            i_anchor = int(np.argmax(w_c * good))
        else:
            i_anchor = int(anch)
            if not (0 <= i_anchor < C):
                raise ValueError("ratio_anchor out of range")

        cand = np.where(
            (np.arange(C) != i_anchor) & (w_c >= thr)
        )[0].astype(int)
        # Precompute ratios r_ci = w_c / w_anchor (avoid div-by-zero).
        denom = max(w_c[i_anchor], np.finfo(float).tiny)
        r_ci = (w_c[cand] / denom).astype(np.float64, copy=False)
    else:
        i_anchor = None
        cand = np.array([], dtype=int)
        r_ci = np.array([], dtype=np.float64)

    # ---- Solution vector and buffers ----
    if x0 is None:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64, order="C").copy()
    a64 = np.empty(N, dtype=np.float64)  # one-column buffer

    # ---- Core Kaczmarz config ----
    epochs   = int(_cfg_val(cfg, "epochs", 1))
    K_req    = int(_cfg_val(cfg, "pixels_per_aperture", 256))
    # lr already computed above
    order    = str(_cfg_val(cfg, "row_order", "random"))
    proj_nn  = bool(_cfg_val(cfg, "project_nonneg", True))
    blas_threads = _cfg_val(cfg, "blas_threads", None)
    verbose  = bool(_cfg_val(cfg, "verbose", True))

    # ---- Mass- and shape-preserving ratio update ----
    def ratio_update(
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        c: int,
        i: int,
        r: float,
        eps: float = 1e-18,
    ) -> None:
        """
        Enforce s_c ≈ r * s_i by rescaling blocks c and i only.
        - Preserves population *shape* within each block.
        - Preserves total mass: Δs_c + Δs_i = 0.
        - Effective step is eta_pen/(1+r) to keep dynamics stable.
        """
        sc = float(s_vec[c])
        si = float(s_vec[i])
        g = sc - r * si
        if abs(g) < 1e-30:
            return

        # Mass-conserving pair step (effective step includes 1/(1+r)).
        eff = eta_pen / (1.0 + r)
        ds_c = -eff * g
        ds_i = +eff * g

        c0, i0 = c * P, i * P
        xc = x_vec[c0:c0 + P]
        xi = x_vec[i0:i0 + P]

        # Directions follow current shapes (fallbacks if empty).
        if sc > eps:
            dir_c = xc / (sc + eps)
        elif si > eps:
            dir_c = xi / (si + eps)              # borrow anchor shape
        else:
            dir_c = np.full(P, 1.0 / P, x_vec.dtype)

        if si > eps:
            dir_i = xi / (si + eps)
        elif sc > eps:
            dir_i = xc / (sc + eps)
        else:
            dir_i = np.full(P, 1.0 / P, x_vec.dtype)

        xc_new = xc + ds_c * dir_c
        xi_new = xi + ds_i * dir_i

        if proj_nn:
            sc_old, si_old = float(xc.sum()), float(xi.sum())
            np.maximum(xc_new, 0.0, out=xc_new)
            np.maximum(xi_new, 0.0, out=xi_new)
            s_vec[c] += float(xc_new.sum() - sc_old)
            s_vec[i] += float(xi_new.sum() - si_old)
        else:
            s_vec[c] += ds_c
            s_vec[i] += ds_i

        x_vec[c0:c0 + P] = xc_new
        x_vec[i0:i0 + P] = xi_new

    # ---- Main loop ----
    t0 = time.perf_counter()
    with _blas_ctx(blas_threads):
        for ep in range(epochs):
            if verbose:
                print(f"[Kaczmarz] epoch {ep + 1}/{epochs}")
            t_ep0 = time.perf_counter()
            pbar = tqdm(
                total=reader.nSpat,
                desc=f"[Kaczmarz] epoch {ep + 1}/{epochs}",
                unit="spax",
                dynamic_ncols=True,
                leave=(ep == epochs - 1),
                disable=not verbose,
                mininterval=1.0,
            )
            s_comp = None
            last_tick = time.time()

            for s0, s1 in reader.spaxel_tiles():
                for s in range(s0, s1):
                    A_f32, y = reader.read_spaxel_plane(s)
                    L_eff = int(A_f32.shape[1])
                    if L_eff != y.size:
                        raise RuntimeError(
                            f"L mismatch for spaxel {s}: "
                            f"A has {L_eff}, y has {y.size}"
                        )

                    # Choose K pixel rows
                    K = min(K_req, L_eff)
                    if order == "sequential":
                        idx = np.arange(K, dtype=np.int64)
                    else:
                        idx = rng.choice(L_eff, size=K, replace=False)

                    # Optional RMSE accumulator (per spaxel)
                    if on_batch_rmse is not None:
                        se_sum = 0.0

                    # Data Kaczmarz updates
                    for l in idx:
                        a64[:] = A_f32[:, l]
                        rres = y[l] - np.dot(a64, x)
                        denom = np.dot(a64, a64) + 1e-18
                        x += lr * (rres / denom) * a64
                        if on_batch_rmse is not None:
                            se_sum += float(rres * rres)

                    if on_batch_rmse is not None:
                        rmse = float((se_sum / max(K, 1)) ** 0.5)
                        on_batch_rmse(rmse)

                    # Occasional ratio adjustment (cheap, O(P))
                    if use_ratio and cand.size and (rng.random() < ratio_prob):
                        if s_comp is None:
                            s_comp = x.reshape(C, P).sum(axis=1)
                        for _ in range(ratio_batch):
                            j = int(rng.integers(cand.size))
                            c_idx = int(cand[j])
                            ratio_update(
                                x, s_comp, c_idx, int(i_anchor), float(r_ci[j])
                            )
                        # Occasionally refresh sums to limit drift after proj
                        if rng.random() < 0.1:
                            s_comp = x.reshape(C, P).sum(axis=1)

                    # Heartbeat for dashboards
                    if (on_progress is not None) and (progress_interval_sec):
                        now = time.time()
                        if (now - last_tick) >= float(progress_interval_sec):
                            on_progress(
                                x,
                                ep + 1,
                                {
                                    "epoch": ep + 1,
                                    "elapsed_sec": now - t_ep0,
                                    "pixels_per_aperture": K_req,
                                    "blas_threads": blas_threads,
                                    "N": N,
                                },
                            )
                            last_tick = now

                    pbar.update(1)

            if proj_nn:
                np.maximum(x, 0, out=x)

            pbar.close()

            if on_epoch_end is not None:
                stats_epoch = {
                    "epoch": ep + 1,
                    "elapsed_sec": time.perf_counter() - t_ep0,
                    "pixels_per_aperture": K_req,
                    "blas_threads": blas_threads,
                    "N": N,
                }
                on_epoch_end(x, ep + 1, stats_epoch)

    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed_sec": elapsed,
        "epochs": epochs,
        "pixels_per_aperture": K_req,
        "blas_threads": blas_threads,
        "N": N,
        "ratio_used": bool(use_ratio),
        "ratio_anchor": (None if not use_ratio else int(i_anchor)),
        "ratio_prob": (ratio_prob if use_ratio else 0.0),
        "ratio_eta_base": (eta_pen if use_ratio else 0.0),
        "ratio_batch": (ratio_batch if use_ratio else 0),
        "ratio_min_weight_frac": (min_w_frac if use_ratio else 0.0),
        "ratio_mass_preserving": bool(use_ratio),
        "ratio_shape_preserving": bool(use_ratio),
    }
    return x, stats