#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, math, subprocess, multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import h5py

from CubeFit.hdf5_manager import open_h5
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# ---------------------------- public config ---------------------------------

@dataclass
class TrackerConfig:
    ring_size: int = 96
    metrics_interval_sec: float = 300.0
    val_interval_sec: float = 3600.0
    ckpt_interval_sec: float = 1800.0
    # Validation disabled in this minimal tracker (always NaN)
    diag_seed: int = 12345

# -------------------------- writer-side helpers ------------------------------

def _try_h5clear(path: str) -> None:
    try:
        subprocess.run(["h5clear", "-s", path],
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass

def _open_append_lockfree(path: str, tries: int = 6, backoff: float = 0.5):
    # Locking OFF only in the writer process
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    last = None
    for k in range(tries):
        try:
            return h5py.File(path, "a", libver="latest", locking=False)
        except TypeError:
            # Older h5py without 'locking' kwarg:
            return h5py.File(path, "a", libver="latest")
        except (BlockingIOError, OSError) as e:
            last = e
            _try_h5clear(path)
            time.sleep(backoff * (k + 1))
    raise last

def _ensure_fit_root(f: h5py.File) -> h5py.Group:
    if "/Fit" not in f:
        return f.create_group("/Fit")
    return f["/Fit"]

def _ensure_schema_lazy(f: h5py.File, CP: int, ring: int) -> None:
    """Create datasets on first save; cheap if they exist."""
    fitg = _ensure_fit_root(f)
    if "x_latest" not in fitg:
        fitg.create_dataset("x_latest", shape=(CP,), dtype="f8")
    if "x_best" not in fitg:
        fitg.create_dataset("x_best", shape=(CP,), dtype="f8")
    if "x_ring" not in fitg:
        xr = fitg.create_dataset("x_ring", shape=(ring, CP), dtype="f8",
                                 chunks=(1, CP))
        xr.attrs["head"] = np.int64(0)
    if "metrics" not in fitg:
        m = fitg.create_group("metrics")
        # Simple columnar, extensible datasets
        m.create_dataset("epoch", (0,), maxshape=(None,), dtype="i8", chunks=True)
        m.create_dataset("time_sec", (0,), maxshape=(None,), dtype="f8", chunks=True)
        m.create_dataset("train_rmse_ewma", (0,), maxshape=(None,), dtype="f8", chunks=True)

def _append_metric_row(f: h5py.File, epoch: int, tsec: float, rmse: float) -> None:
    m = f["/Fit/metrics"]
    for name, val in (("epoch", epoch), ("time_sec", tsec), ("train_rmse_ewma", rmse)):
        ds = m[name]
        n = ds.shape[0]
        ds.resize((n + 1,))
        ds[n] = val

# ------------------------------ writer main ---------------------------------

def _writer_main(h5_path: str, cfg: TrackerConfig, q: MPQueue) -> None:
    """
    Child process. Owns the only append handle.
    IMPORTANT: Always writes to the sidecar '<file>.fit.h5' to avoid any
    concurrency with the main HDF5 that the solver reads from.

    - Disables HDF5 file locking in this process.
    - Opens with locking=False and a small raw-data chunk cache to keep RSS flat.
    - Defers dataset creation until first 'save' (instant startup).
    """

    # 1) No file locks in this writer process
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # 2) Sidecar path (always)
    target = h5_path + ".fit.h5"

    # 3) Open the sidecar with small rdcc (raw data chunk cache)
    #    This keeps per-dataset writer cache small and avoids RSS creep.
    def _open_sidecar(path: str):
        last = None
        for k in range(6):
            try:
                return h5py.File(
                    path, "a", libver="latest", locking=False,
                    rdcc_nbytes=8 * 1024 * 1024,   # 8 MiB
                    rdcc_nslots=1 << 15,           # ~32k slots
                    rdcc_w0=0.75,
                )
            except TypeError:
                # Older h5py: no locking/rdcc kwargs; rely on env var
                return h5py.File(path, "a", libver="latest")
            except (BlockingIOError, OSError) as e:
                last = e
                try:
                    subprocess.run(["h5clear", "-s", path], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
                time.sleep(0.5 * (k + 1))
        raise last

    with _open_sidecar(target) as f:
        # Minimal, lazy schema
        if "/Fit" not in f:
            fitg = f.create_group("/Fit")
        else:
            fitg = f["/Fit"]
        try:
            fitg.attrs.modify("using_sidecar", True)
        except Exception:
            pass

        have_schema = False
        ring = int(getattr(cfg, "ring_size", 96))

        def _ensure_schema(CP: int):
            nonlocal have_schema
            if have_schema:
                return
            if "x_latest" not in fitg:
                fitg.create_dataset("x_latest", shape=(CP,), dtype="f8")
            if "x_best" not in fitg:
                fitg.create_dataset("x_best", shape=(CP,), dtype="f8")
            if "x_ring" not in fitg:
                xr = fitg.create_dataset("x_ring", shape=(ring, CP), dtype="f8",
                                         chunks=(1, CP))
                xr.attrs["head"] = np.int64(0)
            if "metrics" not in fitg:
                m = fitg.create_group("metrics")
                for name, dt in (("epoch","i8"),("time_sec","f8"),("train_rmse_ewma","f8")):
                    m.create_dataset(name, (0,), maxshape=(None,), dtype=dt, chunks=True)
            have_schema = True

        def _append_metric(epoch: int, tsec: float, rmse: float):
            m = fitg["metrics"]
            for nm, val in (("epoch", epoch), ("time_sec", tsec), ("train_rmse_ewma", rmse)):
                ds = m[nm]; n = ds.shape[0]; ds.resize((n + 1,)); ds[n] = val

        # Main loop
        while True:
            msg = q.get()  # small bounded queue on parent side
            if not isinstance(msg, dict):
                continue
            op = msg.get("op", "")

            if op == "stop":
                try: f.flush()
                except Exception: pass
                return

            elif op == "set_orbit":
                ow = np.asarray(msg.get("ow", []), np.float64).ravel()
                if ow.size:
                    if "orbit_weights" in fitg:
                        del fitg["orbit_weights"]
                    fitg.create_dataset("orbit_weights", data=ow, dtype="f8")
                    try: f.flush()
                    except Exception: pass

            elif op == "save":
                x = msg.get("x", None)
                epoch = int(msg.get("epoch", -1))
                tsec  = float(msg.get("time_sec", time.time()))
                rmse  = float(msg.get("train_rmse_ewma", math.nan))

                if isinstance(x, np.ndarray) and x.ndim == 1 and x.size:
                    CP = int(x.size); _ensure_schema(CP)
                    x64 = np.asarray(x, np.float64)
                    fitg["x_latest"][...] = x64
                    xr = fitg["x_ring"]; head = int(xr.attrs.get("head", 0))
                    xr[head % xr.shape[0], :] = x64
                    xr.attrs.modify("head", (head + 1) % xr.shape[0])

                try:
                    _append_metric(epoch, tsec, rmse)
                except Exception:
                    pass

                try: f.flush()
                except Exception:
                    pass

# ------------------------------ public class --------------------------------

class FitTracker:
    """
    Non-blocking tracker. If the writer falls behind, updates are dropped
    rather than blocking the solver.
    """
    def __init__(self, h5_path: str, cfg: Optional[TrackerConfig] = None) -> None:
        self.h5_path = str(h5_path)
        self.cfg = cfg or TrackerConfig()
        self._ewma: Optional[float] = None
        self._alpha = 0.02  # EWMA decay
        # Start method preference: env → fork → forkserver → spawn
        prefer = os.environ.get("FITTRACKER_START", "").lower()
        methods = mp.get_all_start_methods()
        order = ([prefer] if prefer else []) + \
                [m for m in ("fork", "forkserver", "spawn") if m and m in methods and m != prefer]
        last_err = None
        self._q: MPQueue | None = None
        self._proc = None
        for m in order:
            try:
                ctx = mp.get_context(m)
                self._q = ctx.Queue(maxsize=8)
                self._proc = ctx.Process(target=_writer_main,
                                         args=(self.h5_path, self.cfg, self._q))
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

    # ------------- API the solver / runner calls -----------------------------

    def set_orbit_weights(self, ow: np.ndarray) -> None:
        if self._q is None:
            return
        try:
            self._q.put_nowait({"op": "set_orbit", "ow": np.asarray(ow, np.float64)})
        except Exception:
            pass

    def on_batch(self, rmse: float) -> None:
        if not np.isfinite(rmse):
            return
        if self._ewma is None:
            self._ewma = float(rmse)
        else:
            self._ewma = (1.0 - self._alpha) * self._ewma + self._alpha * float(rmse)

    def maybe_save(self, x: np.ndarray, epoch: int) -> None:
        if self._q is None:
            return
        msg = {
            "op": "save",
            "x": np.asarray(x, np.float64).ravel(),
            "epoch": int(epoch),
            "time_sec": time.time(),
            "train_rmse_ewma": float(self._ewma) if self._ewma is not None else math.nan,
        }
        try:
            self._q.put_nowait(msg)
        except Exception:
            pass

    def close(self, timeout: float = 2.0) -> None:
        if self._q is not None:
            try:
                self._q.put_nowait({"op": "stop"})
            except Exception:
                pass
        if self._proc is not None:
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception:
                    pass

# ------------------------------ Null tracker --------------------------------

class NullTracker:
    """No-op fallback with the same methods."""
    def __init__(self, *a, **k): pass
    def set_orbit_weights(self, *a, **k): pass
    def on_batch(self, *a, **k): pass
    def maybe_save(self, *a, **k): pass
    def close(self, *a, **k): pass

def copy_fit_sidecar_back(main_path: str):
    sidecar = main_path + ".fit.h5"
    if not os.path.exists(sidecar): return
    with h5py.File(main_path, "a", libver="latest", locking=False) as F, \
         h5py.File(sidecar,  "r", libver="latest") as G:
        if "/Fit/x_best" in G:
            if "/Fit" not in F: F.create_group("/Fit")
            if "/Fit/x_best" in F: del F["/Fit/x_best"]
            F.create_dataset("/Fit/x_best", data=np.asarray(G["/Fit/x_best"][...], np.float64), dtype="f8")
        # copy small metrics if you want:
        # (avoid huge copies; typically a few thousand rows is fine)
        if "/Fit/metrics" in G and "/Fit/metrics" not in F:
            G.copy("/Fit/metrics", F["/Fit"])
