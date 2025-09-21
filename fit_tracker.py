#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, time, math, subprocess, multiprocessing as mp
from dataclasses import dataclass
from typing import Optional
import numpy as np, h5py

# ---------------------------- configuration ----------------------------------

@dataclass
class TrackerConfig:
    ring_size: int = 96
    metrics_interval_sec: float = 300.0
    # (we keep only what's needed right now)
    diag_seed: int = 12345

# ------------------------------- utilities -----------------------------------

def _try_h5clear(path: str) -> None:
    try:
        subprocess.run(["h5clear", "-s", path],
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass

def _open_sidecar(path: str):
    """
    Open <path>.fit.h5 with a *small* raw chunk cache and locking disabled.
    Never touches the main file; avoids SWMR/locking entirely.
    """
    side = path + ".fit.h5"
    # tiny cache to keep RSS flat
    try:
        return h5py.File(side, "a", libver="latest", locking=False,
                         rdcc_nbytes=8 * 1024 * 1024,   # 8 MiB
                         rdcc_nslots=1 << 15,           # ~32k
                         rdcc_w0=0.75)
    except TypeError:
        # Older h5py without 'locking' kwarg
        return h5py.File(side, "a", libver="latest")

# ------------------------------- writer proc ---------------------------------

def _writer_main(h5_path: str, cfg: TrackerConfig, q: mp.Queue) -> None:
    """
    Child process: owns the only append handle (to the *sidecar*).
    Messages:
      {"op": "save", "x": np.ndarray1d, "epoch": int, "time_sec": float, "train_rmse_ewma": float}
      {"op": "set_orbit", "ow": np.ndarray1d}
      {"op": "stop"}
    """
    # Completely disable file locking in this process.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    f = _open_sidecar(h5_path)
    fitg = f.require_group("/Fit")

    def _ensure_schema(CP: int):
        if "x_latest" not in fitg:
            fitg.create_dataset("x_latest", shape=(CP,), dtype="f8")
        if "x_best" not in fitg:
            fitg.create_dataset("x_best", shape=(CP,), dtype="f8")
        if "x_ring" not in fitg:
            xr = fitg.create_dataset("x_ring", shape=(cfg.ring_size, CP),
                                     dtype="f8", chunks=(1, CP))
            xr.attrs["head"] = np.int64(0)
        if "metrics" not in fitg:
            m = fitg.create_group("metrics")
            m.create_dataset("epoch", (0,), maxshape=(None,), dtype="i8", chunks=True)
            m.create_dataset("time_sec", (0,), maxshape=(None,), dtype="f8", chunks=True)
            m.create_dataset("train_rmse_ewma", (0,), maxshape=(None,), dtype="f8", chunks=True)

    def _append_metric(epoch: int, tsec: float, rmse: float):
        m = fitg["metrics"]
        for nm, val in (("epoch", epoch), ("time_sec", tsec), ("train_rmse_ewma", rmse)):
            ds = m[nm]
            n = ds.shape[0]
            ds.resize((n + 1,))
            ds[n] = val

    try:
        while True:
            msg = q.get()
            if not isinstance(msg, dict):
                continue
            op = msg.get("op", "")

            if op == "stop":
                try:
                    f.flush()
                except Exception:
                    pass
                break

            elif op == "set_orbit":
                ow = np.asarray(msg.get("ow", []), np.float64).ravel()
                if ow.size:
                    if "orbit_weights" in fitg:
                        del fitg["orbit_weights"]
                    fitg.create_dataset("orbit_weights", data=ow, dtype="f8")
                    try:
                        f.flush()
                    except Exception:
                        pass

            elif op == "save":
                x = msg.get("x", None)
                epoch = int(msg.get("epoch", -1))
                tsec  = float(msg.get("time_sec", time.time()))
                rmse  = float(msg.get("train_rmse_ewma", math.nan))
                if isinstance(x, np.ndarray) and x.ndim == 1 and x.size:
                    CP = int(x.size)
                    _ensure_schema(CP)
                    x64 = np.asarray(x, np.float64)
                    fitg["x_latest"][...] = x64
                    xr = fitg["x_ring"]
                    head = int(xr.attrs.get("head", 0))
                    xr[head % xr.shape[0], :] = x64
                    xr.attrs.modify("head", (head + 1) % xr.shape[0])
                try:
                    _append_metric(epoch, tsec, rmse)
                except Exception:
                    pass
                try:
                    f.flush()
                except Exception:
                    pass
    finally:
        try:
            f.close()
        except Exception:
            pass

# --------------------------------- public API --------------------------------

class FitTracker:
    """
    Non-blocking tracker. Uses a bounded queue; drops updates if full.
    Writes ONLY to '<file>.fit.h5'.
    """
    def __init__(self, h5_path: str, cfg: Optional[TrackerConfig] = None):
        self.h5_path = str(h5_path)
        self.cfg = cfg or TrackerConfig()
        self._ewma = None
        self._alpha = 0.02

        prefer = os.environ.get("FITTRACKER_START", "").lower()
        methods = mp.get_all_start_methods()
        order = ([prefer] if prefer else []) + \
                [m for m in ("fork", "forkserver", "spawn")
                 if m and m in methods and m != prefer]

        last_err = None
        self._q = None
        self._proc = None
        for m in order:
            try:
                ctx = mp.get_context(m)
                self._q = ctx.Queue(maxsize=8)  # bounded → never blocks producer
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

    def set_orbit_weights(self, ow: np.ndarray) -> None:
        if self._q is None: return
        try:
            self._q.put_nowait({"op": "set_orbit",
                                "ow": np.asarray(ow, np.float64)})
        except Exception:
            pass

    def on_batch(self, rmse: float) -> None:
        if not np.isfinite(rmse): return
        self._ewma = float(rmse) if self._ewma is None \
            else (1.0 - self._alpha) * self._ewma + self._alpha * float(rmse)

    def maybe_save(self, x: np.ndarray, epoch: int, now: float | None = None) -> None:
        if self._q is None: return
        msg = {
            "op": "save",
            "x": np.asarray(x, np.float64).ravel(),
            "epoch": int(epoch),
            "time_sec": float(now if now is not None else time.time()),
            "train_rmse_ewma": float(self._ewma) if self._ewma is not None else math.nan,
        }
        try:
            self._q.put_nowait(msg)
        except Exception:
            pass

    def close(self, timeout: float = 2.0) -> None:
        if self._q is not None:
            try: self._q.put_nowait({"op": "stop"})
            except Exception: pass
        if self._proc is not None:
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                try: self._proc.terminate()
                except Exception: pass

class NullTracker:
    def __init__(self, *a, **k): pass
    def set_orbit_weights(self, *a, **k): pass
    def on_batch(self, *a, **k): pass
    def maybe_save(self, *a, **k): pass
    def close(self, *a, **k): pass

def copy_fit_sidecar_back(main_path: str):
    """Optional: copy /Fit/x_best from sidecar → main file after the run."""
    side = main_path + ".fit.h5"
    if not os.path.exists(side): return
    with h5py.File(main_path, "a", libver="latest", locking=False) as F, \
         h5py.File(side, "r", libver="latest") as G:
        if "/Fit/x_best" in G:
            F.require_group("/Fit")
            if "/Fit/x_best" in F: del F["/Fit/x_best"]
            F.create_dataset("/Fit/x_best",
                             data=np.asarray(G["/Fit/x_best"][...], np.float64),
                             dtype="f8")
