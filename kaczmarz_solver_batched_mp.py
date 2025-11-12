# -*- coding: utf-8 -*-
r"""
    kaczmarz_solver_batched_mp.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Multi-process batched/tiled Kaczmarz with super-block GEMV/GEMM.

    Save this file alongside your existing kaczmarz_solver.py.
    Use from PipelineRunner by importing and calling
    `solve_global_kaczmarz_batched_mp(...)` instead of the single-process one.

    Design
    ------
    - Coordinator process:
    * Owns the master solution vector x in shared memory (float64).
    * Feeds (s0, s1) spaxel tiles into a task queue.
    * Receives tile-gradients g from workers and applies: x += lr * g.
    * (Optional) applies ratio penalty updates sparsely.
    * Emits progress & batch-RMSE to your callbacks.

    - Worker processes (N_procs of them):
    * Each opens its own HyperCubeReader (new file handle ⇒ SWMR-safe).
    * Iterates assigned spaxel tiles; for each tile, builds a large
        column super-block A[:, J] (float32) and y[J] (float64).
    * Computes r = y - A.T @ x_snapshot and the block gradient
        g = (A @ r) / ||A||_F^2, then sends g to the coordinator.
    * Sends batch RMSE samples (per spaxel) to the coordinator.

    I/O & BLAS
    ----------
    - Reads are sequential and chunk-aligned (spaxel-tiles).
    - Use a few processes (e.g., 4) with many BLAS threads each (e.g., 12).
    - Set BLAS threads inside workers to avoid oversubscription.

    Requirements
    ------------
    - A "reader factory" we can use *in each process*. We try to derive one
    from the `reader` you pass in:
        reader.__class__(reader.h5_path, cfg=reader.cfg)
    If that’s not available, pass `reader_ctor` explicitly.

    Public API
    ----------
    `solve_global_kaczmarz_batched_mp(reader, cfg, *, reader_ctor=None, processes=4, blas_threads=12, ...)`
    matches your current solver’s callbacks:

        on_epoch_end(epoch, stats_dict)
        on_progress(epoch, stats_dict)
        on_batch_rmse(rmse_float)

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   2025/09/28
"""

from __future__ import annotations

import os
import time
import math
import importlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any, Dict
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue  # put near other imports
from multiprocessing.synchronize import Event as MPEvent
import queue

# Reuse your config helpers (same as in your repo)
import CubeFit.cube_utils as cu
# Reuse the baseline SolverCfg so the knobs line up with your pipeline
from CubeFit.kaczmarz_solver import SolverCfg


# --------------------------- MP config -----------------------------

@dataclass
class MPBatchedCfg(SolverCfg):
    """
    Extends SolverCfg with multi-process batching knobs.

    s_tile          : spaxels aggregated per worker "tile" (match chunking)
    super_cols_max  : max columns in the super-block (controls RAM)
    super_cols_min  : threshold to flush and apply a block update
    processes       : number of worker processes
    blas_threads    : OpenBLAS/MKL threads per worker
    """
    s_tile: int = 128
    super_cols_max: int = 4096
    super_cols_min: int = 1024
    processes: int = 4
    blas_threads: Optional[int] = 12


# --------------------------- reader ctor helpers -------------------

def _np_from_shared(shared_arr, n: int) -> np.ndarray:
    """Create a NumPy float64 view over a multiprocessing.Array/RawArray."""
    try:
        buf = shared_arr.get_obj()
    except Exception:
        buf = shared_arr
    return np.frombuffer(buf, dtype=np.float64, count=n)

def _set_worker_blas_threads(n: Optional[int]) -> None:
    # Set before heavy BLAS is first used in the spawned process
    if n is None:
        return
    for k in ("OPENBLAS_NUM_THREADS",
              "OMP_NUM_THREADS",
              "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(int(n))

# --------------------------- worker -------------------------------

def _worker_loop(
    wid: int,
    ctor_pickle: Tuple[str, str, str, Dict[str, Any]] | None,
    reader_ctor_callable: Optional[Callable[[], Any]],
    cfg_dict: Dict[str, Any],
    x_shared,  # multiprocessing.Array('d', N)
    N: int,
    task_q: MPQueue,
    grad_q: MPQueue,
    rmse_q: MPQueue,
    prog_q: MPQueue,
    stop_ev: MPEvent,
) -> None:
    """
    Worker entrypoint. Each worker creates its own reader and loops over
    (s0,s1) tiles pulled from task_q, computing block-gradients and
    sending them to grad_q. RMSE samples go to rmse_q; progress ticks
    go to prog_q.
    """
    # Configure BLAS threads *inside* the worker
    _set_worker_blas_threads(cfg_dict.get("blas_threads"))

    # Reconstruct reader
    reader = None
    if reader_ctor_callable is not None:
        reader = reader_ctor_callable()
    elif ctor_pickle is not None:
        mod_name, cls_name, h5_path, kwargs = ctor_pickle
        ReaderCls = getattr(importlib.import_module(mod_name), cls_name)
        reader = ReaderCls(h5_path, **kwargs)
    else:
        raise RuntimeError("[worker] No reader constructor available.")

    # Unpack cfg knobs we use frequently
    rng = np.random.default_rng(cfg_dict.get("seed", None))
    order = str(cfg_dict.get("row_order", "random"))
    K_req = int(cfg_dict.get("pixels_per_aperture", 256))
    S_tile = int(cfg_dict.get("s_tile", 128))
    J_max = int(cfg_dict.get("super_cols_max", 4096))
    J_min = int(cfg_dict.get("super_cols_min", 1024))
    lr = float(cfg_dict.get("lr", 0.25))
    proj_nn = bool(cfg_dict.get("project_nonneg", True))

    # Shared x view (read-only snapshot semantics)
    x_sh = _np_from_shared(x_shared, N)

    # Super-block buffers (float32 for A, float64 for y)
    A_sb = np.empty((N, J_max), dtype=np.float32, order="F")
    y_sb = np.empty(J_max, dtype=np.float64)
    j_fill = 0

    # Convenience flush: compute g for current SB and send to coordinator
    def _flush_superblock():
        nonlocal j_fill
        if j_fill == 0:
            return
        A_view = A_sb[:, :j_fill].astype(np.float64, copy=False)
        y_view = y_sb[:j_fill]
        # Snapshot x once per flush (cheap, ~600 KiB)
        x_snap = x_sh.copy()
        r = y_view - A_view.T @ x_snap
        denom = float(np.einsum("ij,ij->", A_view, A_view, optimize=True)) + 1e-18
        g = (A_view @ r) / denom
        grad_q.put(g.astype(np.float64, copy=False))
        j_fill = 0

    # Tile loop: tasks are (s0, s1) ranges
    while not stop_ev.is_set():
        try:
            job = task_q.get(timeout=0.5)
        except Exception:
            continue
        if job is None:  # sentinel
            break
        s0, s1 = map(int, job)

        # Iterate spaxels in this tile
        for s in range(s0, s1):
            if stop_ev.is_set():
                break
            A_f32, y = reader.read_spaxel_plane(s)  # (N, Ls), (Ls,)
            Ls = int(A_f32.shape[1])
            if Ls != y.size:
                raise RuntimeError(f"[worker{wid}] L mismatch: A has {Ls}, y has {y.size}")

            # Choose K rows for this spaxel
            K = min(K_req, Ls)
            if order == "sequential":
                idxK = np.arange(K, dtype=np.int64)
            else:
                idxK = rng.choice(Ls, size=K, replace=False)

            # If not enough capacity, flush then continue filling
            if j_fill + K > J_max:
                _flush_superblock()

            # Append columns into the super-block
            # Copy A[:, idxK] (float32) and y[idxK] (float64)
            np.copyto(A_sb[:, j_fill:j_fill + K], A_f32[:, idxK].astype(np.float32, copy=False))
            y_sb[j_fill:j_fill + K] = y[idxK].astype(np.float64, copy=False)
            j_fill += K

            # Per-spaxel RMSE sample at the chosen K rows
            try:
                yhat = (A_f32[:, idxK].T @ x_sh)
                r_s = y[idxK] - yhat
                rmse = float(np.sqrt(np.mean(r_s * r_s)))
                rmse_q.put(rmse)
            except Exception:
                pass

            # Progress tick (one per spaxel)
            try:
                prog_q.put(1)
            except Exception:
                pass

            # Opportunistic early update to keep latency low
            if j_fill >= J_min:
                _flush_superblock()

        # End of tile: force a flush
        _flush_superblock()

    # End worker
    try:
        _flush_superblock()
    except Exception:
        pass
    # Let the coordinator know we’re done by sending an empty progress tick
    try:
        prog_q.put(0)
    except Exception:
        pass

# --------------------------- coordinator (public API) -----------------

def solve_global_kaczmarz_batched_mp(
    reader,
    cfg: MPBatchedCfg,
    *,
    orbit_weights: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
    on_progress: Callable[[int, dict], None] | None = None,
    progress_interval_sec: float = 300.0,
    on_batch_rmse: Callable[[float], None] | None = None,
) -> Tuple[np.ndarray, dict]:
    """
    Multi-process batched/tiled Kaczmarz with shared-x gradient reduction.

    Parameters
    ----------
    reader : a HyperCubeReader-like object (already opened in the parent)
      Must expose: nSpat, nComp, nPop, spaxel_tiles(tile_size), read_spaxel_plane(s).
      Used here only to (a) discover sizes, and (b) enumerate tiles.
    reader_ctor : Optional[() -> reader]
      If provided, workers call this to get a fresh reader in their process.
      If None, we attempt to derive a ctor from `reader` as:
        reader.__class__(reader.h5_path, cfg=reader.cfg)

    Returns
    -------
    x : (C*P,) float64
    stats : dict
    """
    # Shapes
    C = int(reader.nComp)
    P = int(reader.nPop)
    S = int(reader.nSpat)
    N = C * P

    # Ratio penalty preparation (same semantics as your baseline solver)
    w_c = None
    use_ratio = False
    if orbit_weights is not None:
        w_c = np.asarray(orbit_weights, dtype=np.float64).ravel()
        if w_c.size == N:
            w_c = w_c.reshape(C, P).sum(axis=1)
        elif w_c.size != C:
            raise ValueError(f"orbit_weights has size {w_c.size}, expected {C} or {N}")
        tot = float(w_c.sum())
        w_c = (w_c / tot) if tot > 0 else None
        use_ratio = True

    # Pull cfg knobs
    lr = cu._cfg_float(cfg, "lr", 0.25)
    epochs = cu._cfg_int(cfg, "epochs", 1)
    proj_nn = cu._cfg_bool(cfg, "project_nonneg", True)
    pixels_per_aperture = cu._cfg_int(cfg, "pixels_per_aperture", 256)
    order = str(getattr(cfg, "row_order", "random"))
    s_tile = int(getattr(cfg, "s_tile", 128))
    j_max = int(getattr(cfg, "super_cols_max", 4096))
    j_min = int(getattr(cfg, "super_cols_min", 1024))
    n_procs = int(getattr(cfg, "processes", 4))
    blas_threads = getattr(cfg, "blas_threads", 12)

    # Ratio params (applied by coordinator, sparsely)
    eta_pen = cu._cfg_float(cfg, "ratio_eta", 0.05 * lr)
    p_ratio = cu._cfg_float(cfg, "ratio_prob", 0.02)
    ratio_batch = cu._cfg_int(cfg, "ratio_batch", 1)
    min_w_frac = cu._cfg_float(cfg, "ratio_min_weight", 1e-3)

    i_anchor = None
    cand = np.array([], dtype=int)
    r_ci = np.array([], dtype=float)
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
        use_ratio = cu._cfg_bool(cfg, "ratio_use", True)
    else:
        use_ratio = cu._cfg_bool(cfg, "ratio_use", False)

    # Init x in shared memory
    if x0 is None:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy(order="C")
    x_shared = mp.Array('d', int(N), lock=False)
    x_view = _np_from_shared(x_shared, N)
    x_view[:] = x  # seed

    # --- Build a PICKLEABLE ctor spec for workers (no callables) ---
    # We rely on the parent reader having attributes .h5_path and .cfg.
    # In PipelineRunner we already did:
    #   setattr(reader, "h5_path", self.h5_path)
    #   setattr(reader, "cfg", reader_cfg)
    # So we can serialize the class path + minimal kwargs here.
    try:
        cls = reader.__class__
        mod = cls.__module__
        name = cls.__name__
        h5_path = getattr(reader, "h5_path", None)
        cfg_obj = getattr(reader, "cfg", None)
        if h5_path is None:
            raise RuntimeError("reader.h5_path is None; attach it before calling the MP solver.")
        kwargs = {"cfg": cfg_obj} if (cfg_obj is not None) else {}
        ctor_pickle = (mod, name, h5_path, kwargs)
        ctor_callable = None # critical: never pass callables to spawned workers
    except Exception as e:
        raise RuntimeError(
            "Cannot derive a pickleable reader constructor for workers."
        ) from e

    # Pack cfg dict passed to workers
    cfg_dict = dict(
        seed=getattr(cfg, "seed", None),
        row_order=order,
        pixels_per_aperture=pixels_per_aperture,
        s_tile=s_tile,
        super_cols_max=j_max,
        super_cols_min=j_min,
        lr=lr,
        project_nonneg=proj_nn,
        blas_threads=blas_threads,
    )

    # Build tile list (randomize order if requested)
    # Reader decides the tile size from its own cfg (ReaderCfg.s_tile).
    tiles = list(reader.spaxel_tiles())
    s_tile_eff = int(getattr(reader, "s_tile", s_tile))
    if order == "random":
        rng = np.random.default_rng(cfg_dict["seed"])
        rng.shuffle(tiles)

    # Queues and processes
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=max(16, 2 * n_procs))
    grad_q = ctx.Queue(maxsize=max(16, 4 * n_procs))
    rmse_q = ctx.Queue(maxsize=max(16, 8 * n_procs))
    prog_q = ctx.Queue(maxsize=max(16, 8 * n_procs))
    stop_ev = ctx.Event()

    workers = []
    for wid in range(n_procs):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                wid,
                ctor_pickle,
                ctor_callable,
                cfg_dict,
                x_shared,
                N,
                task_q,
                grad_q,
                rmse_q,
                prog_q,
                stop_ev,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Coordinator loop
    t0 = time.perf_counter()
    try:
        for ep in range(epochs):
            # Progress bar first
            spax_done = 0
            last_tick = time.perf_counter()
            pbar = tqdm(
                total=S,
                desc=f"[Kaczmarz-MP] epoch {ep + 1}/{epochs}",
                unit="spax",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=(ep == epochs - 1),
                disable=not cu._cfg_bool(cfg, "verbose", True),
            )

            # Stream tiles non-blockingly to avoid queue back-pressure
            tile_iter = iter(tiles)

            # Prime the queue without blocking
            while True:
                try:
                    s0, s1 = next(tile_iter)
                except StopIteration:
                    break
                try:
                    task_q.put((int(s0), int(s1)), block=False)
                except queue.Full:
                    break

            # Per-component sums for ratio updates
            s_comp = x_view.reshape(C, P).sum(axis=1)

            # Drain queues until all spaxels reported
            while spax_done < S:
                # Opportunistically top-up the task queue (a few tries per loop)
                for _ in range(2 * n_procs):
                    try:
                        s0, s1 = next(tile_iter)
                    except StopIteration:
                        break
                    try:
                        task_q.put((int(s0), int(s1)), block=False)
                    except queue.Full:
                        break

                # Apply any gradients waiting
                applied = 0
                while True:
                    try:
                        g = grad_q.get_nowait()
                    except Exception:
                        break
                    x_view[:] = x_view + lr * g
                    if proj_nn:
                        np.maximum(x_view, 0.0, out=x_view)
                    applied += 1

                # Optional ratio updates (unchanged)
                if applied and use_ratio and r_ci.size:
                    if np.random.random() < float(p_ratio):
                        s_comp = x_view.reshape(C, P).sum(axis=1)
                        for _ in range(ratio_batch):
                            j = int(np.random.randint(0, r_ci.size))
                            c_idx = int(cand[j]); i = int(i_anchor)
                            e = s_comp[c_idx] - float(r_ci[j]) * s_comp[i]
                            den = P * (1.0 + float(r_ci[j]) ** 2)
                            if den > 0:
                                dc = -eta_pen * e / den
                                di = +float(r_ci[j]) * eta_pen * e / den
                                c0 = c_idx * P; i0 = i * P
                                x_view[c0:c0+P] += dc
                                x_view[i0:i0+P] += di
                                if proj_nn:
                                    bc = x_view[c0:c0+P].copy()
                                    bi = x_view[i0:i0+P].copy()
                                    np.maximum(x_view[c0:c0+P], 0.0, out=x_view[c0:c0+P])
                                    np.maximum(x_view[i0:i0+P], 0.0, out=x_view[i0:i0+P])
                                    s_comp[c_idx] += float(x_view[c0:c0+P].sum() - bc.sum())
                                    s_comp[i]     += float(x_view[i0:i0+P].sum() - bi.sum())
                                else:
                                    s_comp[c_idx] += P * dc
                                    s_comp[i]     += P * di

                # Emit batch RMSE upstream if any (unchanged)
                if on_batch_rmse is not None:
                    while True:
                        try:
                            rmse = rmse_q.get_nowait()
                        except Exception:
                            break
                        on_batch_rmse(float(rmse))

                # Update progress (unchanged)
                prog_drain = 0
                while True:
                    try:
                        tick = prog_q.get_nowait()
                    except Exception:
                        break
                    if tick > 0:
                        spax_done += int(tick)
                        pbar.update(int(tick))
                    prog_drain += 1

                # Periodic on_progress (unchanged)
                if (on_progress is not None) and (time.perf_counter() - last_tick >= float(progress_interval_sec)):
                    stats_epoch = {
                        "spaxels_done": int(spax_done),
                        "nSpax": int(S),
                        "pixels_per_aperture": int(pixels_per_aperture),
                        "N": int(N),
                        "s_tile": int(s_tile_eff),
                        "super_cols_max": int(j_max),
                        "processes": int(n_procs),
                        "blas_threads": blas_threads,
                        "ratio_used": bool(use_ratio),
                    }
                    on_progress(ep + 1, stats_epoch)
                    last_tick = time.perf_counter()

                # Small sleep to avoid busy-wait if nothing flowed
                if (applied == 0) and (prog_drain == 0):
                    time.sleep(0.01)

            pbar.close()

            # Epoch end callback
            if on_epoch_end is not None:
                stats_epoch = dict(
                    epoch=ep + 1,
                    elapsed_sec=None,
                    pixels_per_aperture=pixels_per_aperture,
                    N=N,
                    s_tile=s_tile_eff,
                    super_cols_max=j_max,
                    processes=n_procs,
                    blas_threads=blas_threads,
                    ratio_used=bool(use_ratio),
                )
                on_epoch_end(ep + 1, stats_epoch)

    finally:
        # Stop workers
        try:
            stop_ev.set()
        except Exception:
            pass
        # Drain task queue and push sentinels
        try:
            while True:
                task_q.get_nowait()
        except Exception:
            pass
        for _ in workers:
            task_q.put(None)
        for p in workers:
            try:
                p.join(timeout=5.0)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass

    elapsed = time.perf_counter() - t0
    return x_view.copy(), dict(
        elapsed_sec=elapsed,
        epochs=epochs,
        pixels_per_aperture=pixels_per_aperture,
        N=N,
        s_tile=s_tile_eff,
        super_cols_max=j_max,
        processes=n_procs,
        blas_threads=blas_threads,
        ratio_used=bool(use_ratio),
        ratio_anchor=(None if not use_ratio else int(i_anchor)),
        ratio_prob=(p_ratio if use_ratio else 0.0),
        ratio_eta=(eta_pen if use_ratio else 0.0),
        ratio_batch=(ratio_batch if use_ratio else 0),
        ratio_min_weight=(min_w_frac if use_ratio else 0.0),
    )
