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
v1.1:   Added `save_state` to `*Tracker` classes. 1 January 2026
v1.2:   Added atomic sidecar checkpoint support and read-side loading for matched
            `x` + solver state, enabling full resumable restarts from one
            checkpoint path. 7 August 2026
v1.3:   Removed all legacy checkpointing and tracking;
        Always generate new sidecar filename. 26 August 2026
"""

from __future__ import annotations
import queue as _queue
import os, time, json, multiprocessing as mp
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
    queue_size: int = 128

# ------------------------------ writer proc -----------------------------------

def _writer_main(h5_path: str, cfg: TrackerConfig, rx: MPQueue) -> None:
    # Resolve/construct a sidecar path...
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


        # lazy x datasets are created only on first save_x
        def _save_x(vec: np.ndarray, iteration: int) -> None:
            """Persist the physical solution vector for a solver iteration."""
            x = np.asarray(vec, dtype=np.float64).ravel(order="C")

            if "x_last" not in gfit:
                gfit.create_dataset("x_last", data=x, dtype="f8")
            else:
                ds = gfit["x_last"]
                if ds.shape != x.shape or ds.dtype != np.dtype("f8"):
                    del gfit["x_last"]
                    gfit.create_dataset("x_last", data=x, dtype="f8")
                else:
                    ds[...] = x

            gfit["x_last"].attrs["iter"] = int(iteration)
        def _save_state(state: dict) -> None:
            """Persist the solver restart state."""
            try:
                gfit.attrs["solver_state_json"] = json.dumps(
                    state, sort_keys=True)
                gfit.attrs["solver_state_ts"] = float(time.time())
            except Exception as exc:
                logger.log("[FitTracker] solver state update failed:")
                logger.log_exc(exc)

        # main loop with timeout, so we can flush even with no messages
        while True:
            try:
                msg = rx.get()
            except Exception as exc:
                logger.log("[FitTracker] queue read failed:")
                logger.log_exc(exc)
                break

            if msg is None or msg.get("op") == "stop":
                break

            try:
                op = msg.get("op")

                if op == "save_checkpoint":
                    x = np.asarray(
                        msg["x"], dtype=np.float64).ravel(order="C")
                    state = dict(msg.get("state", {}))
                    iteration = int(state.get("iter", -1))

                    if iteration < 0:
                        raise ValueError(
                            "Checkpoint state has no valid iteration.")

                    _save_x(x, iteration)
                    _save_state(state)
                    f.flush()

                    logger.log(
                        "[FitTracker] checkpoint persisted: "
                        f"iter={iteration}, n={x.size}.")

            except Exception as exc:
                logger.log("[FitTracker] message handling error:")
                logger.log_exc(exc)

        try:
            f.flush()
        except Exception:
            pass

# ------------------------------------------------------------------------------

def load_checkpoint(
    sidecar_path: str | None,
    expected_size: int | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    """
    Load the latest coefficient vector and matching solver state.

    Parameters
    ----------
    sidecar_path : str or None
        FitTracker sidecar path.
    expected_size : int or None, optional
        Expected number of coefficients.

    Returns
    -------
    x : ndarray or None
        Latest coefficient vector.
    state : dict or None
        Matching solver restart state.

    Raises
    ------
    None

    Examples
    --------
    >>> x, state = load_checkpoint("cube.h5.fit.123.h5", 6900)
    """
    if not sidecar_path:
        logger.log("[FitTracker] Resume rejected: no sidecar path.")
        return None, None

    if not os.path.exists(sidecar_path):
        logger.log(
            f"[FitTracker] Resume rejected: sidecar does not exist: "
            f"{sidecar_path}")
        return None, None

    try:
        with h5py.File(sidecar_path, "r", libver="latest", swmr=True) as f:
            fit = f.get("/Fit")

            if fit is None:
                logger.log(
                    f"[FitTracker] Resume rejected: no /Fit group in "
                    f"{sidecar_path}.")
                return None, None

            if "x_last" not in fit:
                logger.log(
                    f"[FitTracker] Resume rejected: no /Fit/x_last in "
                    f"{sidecar_path}.")
                return None, None

            x = np.asarray(
                fit["x_last"][...], dtype=np.float64).ravel(order="C")

            if expected_size is not None and x.size != int(expected_size):
                logger.log(
                    f"[FitTracker] Resume rejected: /Fit/x_last has "
                    f"{x.size} coefficients; expected "
                    f"{int(expected_size)}.")
                return None, None

            if not np.all(np.isfinite(x)):
                nbad = int(np.count_nonzero(~np.isfinite(x)))
                logger.log(
                    f"[FitTracker] Resume rejected: /Fit/x_last contains "
                    f"{nbad} non-finite coefficients.")
                return None, None

            raw = fit.attrs.get("solver_state_json", None)

            if raw is None:
                logger.log(
                    "[FitTracker] Full resume unavailable: "
                    "solver_state_json is missing.")
                return x, None

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")

            try:
                state = json.loads(str(raw))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.log(
                    "[FitTracker] Full resume unavailable: could not decode "
                    f"solver_state_json: {exc}")
                return x, None

            if not isinstance(state, dict):
                logger.log(
                    "[FitTracker] Full resume unavailable: "
                    "solver_state_json is not a dictionary.")
                return x, None
            
            x_iter = int(fit["x_last"].attrs.get("iter", -1))
            state_iter = int(state.get("iter", -1))

            if x_iter < 0 or state_iter < 0:
                logger.log(
                    "[FitTracker] Full resume unavailable: checkpoint iteration "
                    f"is missing, x={x_iter}, state={state_iter}.")
                return x, None

            if x_iter != state_iter:
                logger.log(
                    "[FitTracker] Full resume unavailable: checkpoint iteration "
                    f"mismatch, x={x_iter}, state={state_iter}.")
                return x, None

            logger.log(
                "[FitTracker] Loaded full checkpoint: "
                f"iter={state_iter}, phase={state.get('phase', None)}, "
                f"final={state.get('final', None)}, n={x.size}.")

            return x, state

    except Exception as exc:
        logger.log(
            f"[FitTracker] Resume rejected: could not load "
            f"{sidecar_path}: {exc}")
        return None, None

# --------------------------------- public API --------------------------------

class FitTracker:
    """
    Non-blocking tracker façade. Sends tiny messages to a sidecar writer proc.
    """
    def __init__(self, h5_path: str, cfg: Optional[TrackerConfig] = None):
        self.h5_path = str(h5_path)
        self.cfg = cfg or TrackerConfig()

        prefer = (os.environ.get("FITTRACKER_START", "spawn")).lower()
        avail = mp.get_all_start_methods()
        order = [m for m in (prefer, "spawn", "forkserver", "fork") if m in avail]

        last_err = None
        self._q = None
        self._proc = None
        self._start_method = None


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

    def save_checkpoint(self, x: np.ndarray, state: dict, *,
        block: bool = False) -> bool:
        """
        Persist a solution vector and its matching solver state.

        Parameters
        ----------
        x : ndarray
            Physical solution vector.
        state : dict
            Full resumable solver state.
        block : bool, optional
            If True, block while queueing the checkpoint.

        Returns
        -------
        bool
            True if the checkpoint was queued successfully.

        Raises
        ------
        None

        Examples
        --------
        >>> tracker.save_checkpoint(x, state, block=True)
        """
        try:
            x = np.asarray(x, dtype=np.float64).ravel(order="C")
            state = dict(state)
        except Exception:
            return False

        return self._try_put({
            "op": "save_checkpoint",
            "x": x,
            "state": state,
        }, block=block)

    def close(self, timeout: float = 30.0) -> None:
        """Flush pending checkpoints and stop the writer process."""
        q = getattr(self, "_q", None)

        if q is not None:
            try:
                q.put(None)
            except Exception:
                pass

        if self._proc is not None:
            self._proc.join(timeout=timeout)

            if self._proc.is_alive():
                logger.log(
                    "[FitTracker] Writer did not stop cleanly; terminating.")
                try:
                    self._proc.terminate()
                    self._proc.join(timeout=5.0)
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
    """No-op tracker implementing the active checkpoint interface."""

    def save_checkpoint(self, *args, **kwargs) -> bool:
        """Discard a checkpoint request."""
        return False

    def close(self, *args, **kwargs) -> None:
        """Close the no-op tracker."""
        return None
