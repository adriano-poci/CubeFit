r"""
    fit_tracker.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Sidecar-based, non-blocking fit tracker for live Kaczmarz runs.

    - Writes ONLY to <main>.fit.<pid>.<ts>.h5 (a sidecar), never the main HDF5.
    - Uses a bounded mp.Queue so the solver never blocks on I/O.
    - No SWMR; no file locking on the main file.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Added `maybe_snapshot_x` to `NullTracker` for consistency. 1 January 2026
"""

from __future__ import annotations
import queue as _queue
import os, time, math, json, multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue  # put near other imports
from dataclasses import dataclass
from typing import Optional
import numpy as np
import h5py

# for lock-clear + detection
from CubeFit.hdf5_manager import _h5clear, _looks_like_lock_error
import CubeFit.cube_utils as cu
from CubeFit.logger import get_logger

logger = get_logger()

# ---------------------------- configuration ----------------------------------

@dataclass
class TrackerConfig:
    ring_size: int = 96
    metrics_interval_sec: float = 300.0
    diag_seed: int = 12345

# ------------------------------ writer proc -----------------------------------

def _writer_main(h5_path: str, cfg: TrackerConfig, rx: MPQueue) -> None:
    # Resolve/construct a sidecar path...
    sidecar = cu._find_latest_sidecar(h5_path)
    if not sidecar:
        sidecar = cu._default_sidecar_path(h5_path)
    if not sidecar:
        base = str(h5_path) if h5_path else "cube"
        sidecar = f"{base}.fit.{os.getpid()}.{int(time.time())}.h5"

    if os.environ.get("HDF5_USE_FILE_LOCKING") is None:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # lock-aware open
    retries, backoff, last_exc = 3, 0.4, None
    f = None
    for attempt in range(retries + 1):
        try:
            f = h5py.File(sidecar, "a", libver="latest")
            break
        except OSError as e:
            last_exc = e
            try:
                looks_lock = _looks_like_lock_error(e)
            except Exception:
                looks_lock = ("Unable to synchronously open file" in str(e)
                              or "open for write" in str(e)
                              or "open for read-only" in str(e))
            if (attempt == retries) or (not looks_lock):
                raise
            try:
                _h5clear(sidecar)
            except Exception:
                pass
            time.sleep(backoff * (attempt + 1))

    with f:
        gfit = f.require_group("/Fit")
        gfit.attrs["source_main_h5"] = str(h5_path)

        try:
            f.flush()
            f.swmr_mode = True
            logger.log(
                f"[FitTracker] SWMR writer enabled on sidecar: {sidecar}")
        except Exception as e:
            logger.log("[FitTracker] SWMR enable failed:")
            logger.log_exc(e)

        # history datasets (small)
        if "rmse_hist" not in gfit:
            gfit.create_dataset("rmse_hist", shape=(0,), maxshape=(None,), chunks=(8192,), dtype="f4")
        if "rmse_ewma" not in gfit:
            gfit.create_dataset("rmse_ewma", shape=(0,), maxshape=(None,), chunks=(8192,), dtype="f4")
        if "progress" not in gfit:
            gfit.create_dataset("progress", shape=(0, 4), maxshape=(None, 4), chunks=(2048, 4), dtype="f4")


        # lazy x datasets are created only on first save_x
        def _save_x(vec: np.ndarray, epoch: float, rmse: float) -> None:
            x = np.asarray(vec, dtype=np.float64).ravel(order="C")
            if "x_last" not in gfit:
                gfit.create_dataset("x_last", data=x, dtype="f8", maxshape=(x.size,), chunks=(x.size,))
            else:
                ds = gfit["x_last"]
                if ds.shape != (x.size,):
                    del gfit["x_last"]
                    gfit.create_dataset("x_last", data=x, dtype="f8", maxshape=(x.size,), chunks=(x.size,))
                else:
                    ds[...] = x
            gfit["x_last"].attrs["epoch"] = float(epoch)
            gfit["x_last"].attrs["rmse"]  = float(rmse)

            # optional x_best (keep best-by-RMSE)
            try:
                keep = False
                if "x_best" not in gfit:
                    keep = True
                else:
                    cur = float(gfit["x_best"].attrs.get("rmse", np.inf))
                    keep = (rmse < cur)
                if keep:
                    if "x_best" not in gfit:
                        gfit.create_dataset("x_best", data=x, dtype="f8", maxshape=(x.size,), chunks=(x.size,))
                    else:
                        dsb = gfit["x_best"]
                        if dsb.shape != (x.size,):
                            del gfit["x_best"]
                            gfit.create_dataset("x_best", data=x, dtype="f8", maxshape=(x.size,), chunks=(x.size,))
                        else:
                            dsb[...] = x
                    gfit["x_best"].attrs["epoch"] = float(epoch)
                    gfit["x_best"].attrs["rmse"]  = float(rmse)
            except Exception as e:
                logger.log("[FitTracker] x_best update failed:")
                logger.log_exc(e)
        # ---------- helpers for x snapshots ----------
        def _ensure_x_ds(N: int) -> None:
            """Create ring + last datasets if absent, sized by N."""
            if N <= 0:
                return
            if "x_last" not in gfit:
                gfit.create_dataset("x_last", shape=(N,), dtype="f4", chunks=(N,))
            if "x_ring" not in gfit:
                ring = gfit.create_dataset(
                    "x_ring",
                    shape=(cfg.ring_size, N),
                    maxshape=(cfg.ring_size, N),
                    chunks=(1, N),
                    dtype="f4",
                )
                gfit.create_dataset("x_epoch", shape=(cfg.ring_size,), dtype="i4")
                gfit.create_dataset("x_ts",    shape=(cfg.ring_size,), dtype="f8")
                gfit.create_dataset("x_rmse",  shape=(cfg.ring_size,), dtype="f4")
                gfit.attrs["x_head"] = np.int64(0)

        def _append_x(x32: np.ndarray, epoch: int, rmse: float | None) -> None:
            """Append into the ring and update x_last."""
            N = int(x32.size)
            _ensure_x_ds(N)
            head = int(gfit.attrs.get("x_head", 0))
            idx  = head % int(cfg.ring_size)
            gfit["x_ring"][idx, :] = x32
            gfit["x_epoch"][idx]   = int(epoch) if epoch is not None else -1
            gfit["x_ts"][idx]      = float(time.time())
            gfit["x_rmse"][idx]    = float(rmse) if (rmse is not None and np.isfinite(rmse)) else np.nan
            gfit["x_last"][:]      = x32
            gfit.attrs["x_head"]   = np.int64(head + 1)

        # batching knobs
        FLUSH_EVERY = int(os.environ.get("CUBEFIT_TRACKER_FLUSH_EVERY", "128"))
        FLUSH_INTERVAL = float(os.environ.get("CUBEFIT_TRACKER_FLUSH_SEC", "5.0"))
        pending = 0
        t_last = time.monotonic()

        # main loop with timeout, so we can flush even with no messages
        while True:
            try:
                msg = rx.get(timeout=1.0)
            except _queue.Empty:
                # idle tick: flush if needed
                now = time.monotonic()
                if (pending > 0) and (now - t_last >= FLUSH_INTERVAL):
                    try:
                        f.flush()
                    except Exception as e:
                        logger.log("[FitTracker] periodic flush failed:")
                        logger.log_exc(e)
                    pending = 0
                    t_last = now
                continue

            if msg is None or msg.get("op") == "stop":
                break

            try:
                op = msg.get("op")
                if op == "rmse_batch":
                    r = float(msg["value"]); e = float(msg.get("ewma", r))
                    d1 = gfit["rmse_hist"]; n1 = d1.shape[0]; d1.resize((n1+1,)); d1[n1] = r
                    d2 = gfit["rmse_ewma"]; n2 = d2.shape[0]; d2.resize((n2+1,)); d2[n2] = e
                    pending += 1

                elif op == "progress":
                    epoch = float(msg.get("epoch", 0))
                    done  = float(msg.get("done", 0))
                    total = float(msg.get("total", 0))
                    ewma  = float(msg.get("ewma") or np.nan)
                    dp = gfit["progress"]; n = dp.shape[0]
                    dp.resize((n+1, 4)); dp[n, :] = (epoch, done, total, ewma)
                    pending += 1

                elif op == "epoch_end":
                    epoch = float(msg.get("epoch", 0))
                    dp = gfit["progress"]; n = dp.shape[0]
                    dp.resize((n+1, 4)); dp[n, :] = (epoch, np.nan, np.nan, np.nan)
                    pending += 1

                elif op == "save_x":
                    _save_x(
                        np.asarray(msg["x"], np.float64),
                        float(msg.get("epoch", -1)),
                        float(msg.get("rmse", np.nan)),
                    )
                    pending += 1

                elif op == "x_snapshot":
                    _append_x(
                        np.asarray(msg["x"], np.float32),
                        msg.get("epoch"),
                        msg.get("rmse"),
                    )
                    pending += 1

                # batch/interval flush
                now = time.monotonic()
                if pending >= FLUSH_EVERY or (now - t_last) >= FLUSH_INTERVAL:
                    try:
                        f.flush()
                        pending = 0
                        t_last = now
                    except Exception as e:
                        logger.log("[FitTracker] flush failed:")
                        logger.log_exc(e)

            except Exception as e:
                logger.log("[FitTracker] message handling error:")
                logger.log_exc(e)

        # final flush
        try:
            f.flush()
        except Exception:
            pass

# --------------------------------- public API --------------------------------

class FitTracker:
    """
    Non-blocking tracker façade. Sends tiny messages to a sidecar writer proc.
    """
    def __init__(self, h5_path: str, cfg: Optional[TrackerConfig] = None):
        self.h5_path = str(h5_path)
        self.cfg = cfg or TrackerConfig()
        self._ewma = None
        self._alpha = 0.02

        # Sampling & queue knobs (env-overridable)
        self._rmse_stride = int(os.environ.get("CUBEFIT_RMSE_STRIDE", "16"))
        self._rmse_ctr = 0

        prefer = (os.environ.get("FITTRACKER_START", "spawn")).lower()
        avail = mp.get_all_start_methods()
        order = [m for m in (prefer, "spawn", "forkserver", "fork") if m in avail]

        last_err = None
        self._q = None
        self._proc = None
        self._start_method = None

        self._snap_last_t = 0.0
        self._snap_min_sec = float(os.environ.get("CUBEFIT_X_SNAPSHOT_SEC", "3600"))  # seconds between snapshots


        for m in order:
            try:
                ctx = mp.get_context(m)
                self._q = ctx.Queue(maxsize=int(os.environ.get("CUBEFIT_TRACKER_QSIZE", "8192")))
                self._proc = ctx.Process(target=_writer_main, args=(self.h5_path, self.cfg, self._q))
                self._proc.daemon = False
                self._proc.start()
                self._start_method = m
                break
            except Exception as e:
                last_err = e
                self._q = None
                self._proc = None
                continue

        if self._proc is None or self._q is None:
            raise RuntimeError("FitTracker: could not start writer") from last_err

    # ------------ public methods used by PipelineRunner / solver ---------------

    def set_meta(self, N: int) -> None:
        self._try_put({"op": "set_meta", "N": int(N)})

    def set_orbit_weights(self, w: np.ndarray) -> None:
        self._try_put({"op": "set_orbit_weights",
                       "w": np.asarray(w, np.float64).ravel()})

    def on_batch_rmse(self, rmse: float, *, block: bool = False) -> None:
        """
        Non-blocking RMSE push with optional subsampling.
        - Updates local EWMA always.
        - Enqueues at most every `self._rmse_stride` batches.
        - Queue put is non-blocking by default; drops if full.
        """
        try:
            r = float(rmse)
        except Exception:
            return
        if not np.isfinite(r):
            return

        
        # ---- drop pathological outliers to avoid huge spikes ----
        if self._ewma is not None:
            # Use the current EWMA as a scale; protect against 0.
            scale = max(self._ewma, 1.0)
            # Factor 1e3 is deliberately generous; tune if needed.
            if r > 1e3 * scale:
                # Ignore this sample completely; don't update EWMA or write to disk.
                return
        # ----------------------------------------------------------------

        # standard EWMA update
        if self._ewma is None:
            self._ewma = r
        else:
            self._ewma = (1.0 - self._alpha) * self._ewma + self._alpha * r

        self._rmse_ctr += 1
        if (self._rmse_ctr % max(1, self._rmse_stride)) != 0:
            return

        self._try_put({"op": "rmse_batch", "value": r, "ewma": float(self._ewma)}, block=block)

    def on_progress(self, epoch: int, spax_done: int, spax_total: int, *, rmse_ewma: Optional[float] = None) -> None:
        self._try_put({"op": "progress", "epoch": int(epoch),
                    "done": int(spax_done), "total": int(spax_total),
                    "ewma": float(rmse_ewma) if rmse_ewma is not None else None}, block=False)

    def on_epoch_end(self, epoch: int, stats: dict) -> None:
        self._try_put({"op": "epoch_end", "epoch": int(epoch), "stats": dict(stats)}, block=False)

    def maybe_save(self, x_final: np.ndarray, stats: dict) -> None:
        self._try_put({"op": "save_x",
            "x": np.asarray(x_final, np.float32).ravel(order="C"),
            "epoch": int(stats.get("epochs", -1)),
            "rmse": float(stats.get("rmse_epoch_last",
                stats.get("rmse_final", math.nan)))},
            block=False)

    def maybe_snapshot_x(self, x: np.ndarray, *, epoch: int | None = None,
                        rmse: float | None = None, force: bool = False) -> bool:
        """
        Non-blocking, time-gated snapshot of the current global solution vector.
        - Downcasts to float32 before enqueue to cut payload in half.
        - Throttled to at most one snapshot every _snap_min_sec unless force=True.
        Returns True if enqueued, False if skipped/dropped.
        """
        now = time.monotonic()
        if (not force) and ((now - self._snap_last_t) < max(1.0, self._snap_min_sec)):
            return False
        self._snap_last_t = now

        try:
            x32 = np.asarray(x, np.float32).ravel(order="C")
        except Exception:
            return False

        return self._try_put({"op": "x_snapshot", "x": x32,
            "epoch": int(epoch) if epoch is not None else None,
            "rmse": float(rmse) if (rmse is not None and np.isfinite(rmse)) else None},
            block=False)


    def close(self, timeout: float = 2.0) -> None:
        # send a real sentinel the writer understands
        q = getattr(self, "_q", None)
        if q is not None:
            try:
                q.put_nowait(None)  # sentinel
            except Exception:
                pass

        if self._proc is not None:
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    # ----------------------------- helpers ------------------------------------

    def _try_put(self, msg, block: bool = False) -> bool:
        """
        Put `msg` into the tracker queue; non-blocking by default.
        Returns True if enqueued, False if dropped or no queue.
        """
        q = getattr(self, "_q", None)
        if q is None:
            return False
        try:
            if block:
                q.put(msg)
            else:
                q.put_nowait(msg)
            return True
        except _queue.Full:
            return False

class NullTracker:
    def set_meta(self, *a, **k): pass
    def set_orbit_weights(self, *a, **k): pass
    def on_progress(self, *a, **k): pass
    def on_batch_rmse(self, *a, **k): pass
    def on_epoch_end(self, *a, **k): pass
    def maybe_save(self, *a, **k): pass
    def maybe_snapshot_x(self, *a, **k): pass
    def close(self, *a, **k): pass
