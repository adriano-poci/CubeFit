# -*- coding: utf-8 -*-
r"""
    hdf5_manager.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Manages HDF5 data storage for CubeFit pipeline, including creation, loading,
    and validation of large, chunked arrays (templates, data cube, LOSVD, weights).
    Supports buffered template grids for safe convolution.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Initial HDF5 design and validation. 7 September 2025


HDF5 manager for CubeFit.

This module provides a single, store-centric API around the HDF5 base file:
    - Persist core datasets (/LOSVD, /DataCube, /Templates, /Mask, /X_global)
    - Compute and persist spectral rebin operator (/RebinMatrix, /R_T)
      using an exact, flux-conserving bin-overlap scheme equivalent to
      the original `linRebin(lamRange, spectrum, M)`.
    - Compute and persist template FFTs (/TemplatesFFT) and a re-sized
      version consistent with the rebin rows (/TemplatesFFT_R).
    - Keep a small dimension manifest in root attributes.

Everything downstream (builder, solver, reader) reads from the file; nothing
needs optional overrides in memory.
"""

from __future__ import annotations
import numpy as np
import json
from dataclasses import dataclass
from pathlib import Path
from types import NoneType
from typing import Optional, Tuple, Sequence
import os, time, errno, subprocess, tempfile, shutil, fcntl
from contextlib import contextmanager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for BL legend proxies
import h5py

from dynamics.IFU.Constants import Constants

CTS = Constants()
C_KMS = CTS.c

# --------------------------------------------------------------------------- #
# Logger (safe fallback)
# --------------------------------------------------------------------------- #

try:
    # If your project already exposes a global `logger` module attribute,
    # import it so we can call `logger.log(...)`. Otherwise, use a fallback.
    from CubeFit.logger import get_logger

    logger = get_logger()
except Exception:  # pragma: no cover
    class _Logger:
        def log(self, *parts) -> None:
            try:
                msg = " ".join(str(p) for p in parts)
            except Exception:
                msg = "<log message rendering failed>"
            print(msg)
    logger = _Logger()  # type: ignore

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

# -------- Robust HDF5 open helpers (single entry-point) ---------------------
# Ensure HDF5 locking is disabled on shared filesystems unless the user overrides.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# ------------------------------------------------------------------------------

def invalidate_done(h5_path: str):
    with open_h5(h5_path, role="writer") as f:
        g = f.require_group("/HyperCube")
        g.attrs["complete"] = False
        if "_done" in g:
            del g["_done"]

# ------------------------------------------------------------------------------

def print_hypercube_done_status(h5_path: str) -> None:
    """
    Print a compact resume of /HyperCube/_done and basic dataset geometry.
    """

    with open_h5(h5_path, "reader") as f:
        assert "/HyperCube/models" in f, "No /HyperCube/models in file."
        M = f["/HyperCube/models"]
        S, C, P, L = map(int, M.shape)
        chunks = M.chunks or (S, 1, P, L)
        print(f"models shape={M.shape} chunks={chunks}")

        assert "/HyperCube/_done" in f, "No /HyperCube/_done bitmap."
        D = np.asarray(f["/HyperCube/_done"][...])
        total = int(D.size)
        done  = int(D.sum())
        print(f"_done: {done}/{total} tiles complete "
              f"({100.0*done/max(1,total):.1f}%)")

        # Show a few finished tile indices (grid space)
        idx = np.argwhere(D != 0)
        if idx.size:
            print("first finished tiles (is,ic,ip):", idx[:5].tolist())
        else:
            print("no finished tiles yet.")

# ------------------------------------------------------------------------------

def _vel_to_pixel_shift(v_kms: np.ndarray | float, dlog: float) -> np.ndarray | float:
    """Convert velocity (km/s) → pixel shift on a log-λ grid with step dlog."""
    return np.log1p(np.asarray(v_kms, dtype=np.float64) / C_KMS) / float(dlog)

def choose_minimal_template_bounds(
    *,
    TemPix_log: np.ndarray,      # (T,)   template log-λ grid
    ObsPix_log: np.ndarray,      # (L,)   observed log-λ grid
    VelPix_kms: np.ndarray,      # (V,)   LOSVD velocity grid (km/s)
    LOSVD: np.ndarray | None = None,  # optional (S,V,C) to auto-trim tails
    mode: str = "robust",        # "robust" or "conservative"
    eps: float = 1e-3,           # central (1-eps) mass if mode="robust"
    safety_pixels: int = 8       # extra guard on each side (template pixels)
) -> dict:
    """
    Compute minimal safe template bounds (on the template grid) so that:
      - linear convolution by the LOSVD kernel never needs signal outside the band
      - rebinning to ObsPix is clean (no edge artefacts)

    Returns:
      {
        'i_min': int, 'i_max': int,     # slice TemPix_log[i_min : i_max+1]
        'tmin': float, 'tmax': float,   # log-λ bounds actually chosen
        'k_left': int, 'k_right': int,  # required guard in template pixels
        'dlog': float
      }
    """
    TemPix_log = np.asarray(TemPix_log, dtype=np.float64).ravel()
    ObsPix_log = np.asarray(ObsPix_log, dtype=np.float64).ravel()
    VelPix_kms = np.asarray(VelPix_kms, dtype=np.float64).ravel()

    if TemPix_log.size < 3 or ObsPix_log.size < 3 or VelPix_kms.size < 3:
        raise ValueError("TemPix_log, ObsPix_log, VelPix_kms must be 1-D arrays of length ≥ 3.")

    dlog = float(np.median(np.diff(TemPix_log)))

    # 1) effective velocity span to cover
    if mode.lower() == "robust" and LOSVD is not None:
        H = np.asarray(LOSVD, dtype=np.float64)
        if H.ndim != 3 or H.shape[1] != VelPix_kms.size:
            raise ValueError("LOSVD must have shape (S,V,C) matching len(VelPix_kms).")
        H_sum = H.sum(axis=(0, 2))  # (V,)
        total = H_sum.sum()
        if not np.isfinite(total) or total <= 0:
            v_left, v_right = float(VelPix_kms.min()), float(VelPix_kms.max())
        else:
            pdf = H_sum / total
            cdf = np.cumsum(pdf)
            v_left  = float(np.interp(eps/2,       cdf, VelPix_kms))
            v_right = float(np.interp(1.0 - eps/2, cdf, VelPix_kms))
    else:
        v_left, v_right = float(VelPix_kms.min()), float(VelPix_kms.max())

    # 2) guard in template pixels (allow asymmetric)
    k_left  = int(np.ceil(abs(_vel_to_pixel_shift(min(0.0, v_left),  dlog))))
    k_right = int(np.ceil(abs(_vel_to_pixel_shift(max(0.0, v_right), dlog))))
    k_left  += int(safety_pixels)
    k_right += int(safety_pixels)

    # 3) target template bounds around observed range
    t_target_min = ObsPix_log[0]  - k_left  * dlog
    t_target_max = ObsPix_log[-1] + k_right * dlog

    i_min = int(np.searchsorted(TemPix_log, t_target_min, side="right") - 1)
    i_max = int(np.searchsorted(TemPix_log, t_target_max, side="left"))
    i_min = max(0, i_min)
    i_max = min(TemPix_log.size - 1, i_max)

    tmin = float(TemPix_log[i_min])
    tmax = float(TemPix_log[i_max])

    # sanity: ensure margins are sufficient in pixel units
    left_margin_px  = int(np.floor((ObsPix_log[0]  - tmin) / dlog))
    right_margin_px = int(np.floor((tmax - ObsPix_log[-1]) / dlog))
    if left_margin_px < k_left or right_margin_px < k_right:
        i_min = max(0, i_min - (k_left  - left_margin_px  + 1))
        i_max = min(TemPix_log.size - 1, i_max + (k_right - right_margin_px + 1))
        tmin, tmax = float(TemPix_log[i_min]), float(TemPix_log[i_max])

    return dict(i_min=i_min, i_max=i_max, tmin=tmin, tmax=tmax,
                k_left=k_left, k_right=k_right, dlog=dlog)
            
# ------------------------------------------------------------------------------

def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, out.stdout.strip()
    except Exception as e:
        return False, str(e)

def _h5repack_inplace(path: Path) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp-repack")
    ok, out = _run(["h5repack", str(path), str(tmp)])
    if ok:
        try:
            os.replace(tmp, path)
            logger.log(f"[HDF5] h5repack succeeded; replaced {path}")
        except Exception as e:
            try:
                os.remove(tmp)
            except Exception:
                pass
            return False
        return True
    else:
        try:
            logger.log(f"[HDF5] h5repack failed: {out}")
        except Exception:
            pass
        return False

def _open_once(path: str, mode: str):
    """One attempt to open an HDF5 file with libver='latest'."""
    # NB: SWMR read requires explicit swmr=True; we don’t enable it by default
    # because most pipelines don’t publish SWMR metadata consistently.
    return h5py.File(path, mode, libver="latest")

# ---------- GLOBAL FILE LOCK (one writer at a time across processes) ----------
@contextmanager
def _writer_lock(base_path: str | Path, timeout: float = 300.0, poll: float = 0.2):
    """
    Cross-process OS lock using fcntl on a sidecar .lock file.
    Ensures exactly one writer for a given .h5 at a time.
    """
    lock_path = Path(str(base_path) + ".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if (time.time() - start) > timeout:
                    raise TimeoutError(f"Timeout acquiring writer lock: {lock_path}")
                time.sleep(poll)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

# Treat these as "fatal" only if truly unavoidable.
_LOCK_STRS = ("Unable to synchronously open file (unable to lock file",
              "file is already open for write",
              "file is already open for read-only")

def _looks_like_lock_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(s in msg for s in _LOCK_STRS)

def _h5clear(path: str | Path) -> bool:
    """Try to clear SWMR/consistency flags in-place. Returns True on success."""
    p = str(path)
    try:
        # Avoid getting stuck on filesystems with broken locking.
        env = dict(os.environ)
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        r = subprocess.run(["h5clear", "-s", p], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True)
        if r.returncode == 0:
            logger.log(f"[HDF5] h5clear -s succeeded on {p}")
            return True
        else:
            logger.log(f"[HDF5] h5clear failed rc={r.returncode}: {r.stderr.strip()}")
            return False
    except FileNotFoundError:
        logger.log("[HDF5] h5clear not found on PATH.")
        return False
    except Exception as e:
        logger.log(f"[HDF5] h5clear unexpected error: {e}")
        return False

@contextmanager
def open_h5(path: str | Path,
            role: str = "reader",
            retries: int = 3,
            backoff: float = 0.4,
            *,
            swmr: bool | None = None,
            locking: bool | None = None):
    """
    Robust HDF5 open with lock handling and modest raw-chunk cache.
    role='reader' -> read-only; role='writer' -> append/update.
    If swmr is True and role='reader', open in SWMR-read mode.
    """
    p = Path(path)
    if role not in ("reader", "writer"):
        raise ValueError("role must be 'reader' or 'writer'")

    # Allow caller to force HDF5 file locking on/off for this process
    if locking is not None:
        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE" if locking else "FALSE"
    elif os.environ.get("HDF5_USE_FILE_LOCKING") is None:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    mode = "r" if role == "reader" else "a"
    last_exc = None
    for attempt in range(retries + 1):
        try:
            rdcc_nbytes = int(os.environ.get("CUBEFIT_RDCC_NBYTES", str(4 * 1024**3)))
            rdcc_nslots = int(os.environ.get("CUBEFIT_RDCC_NSLOTS", "400_003"))
            rdcc_w0     = float(os.environ.get("CUBEFIT_RDCC_W0", "0.9"))

            kwargs = dict(libver="latest",
                          rdcc_nbytes=rdcc_nbytes,
                          rdcc_nslots=rdcc_nslots,
                          rdcc_w0=rdcc_w0)

            # h5py allows swmr=... only for read-only opens
            if role == "reader" and (swmr is True):
                kwargs["swmr"] = True

            f = h5py.File(p, mode, **kwargs)
            try:
                yield f
            finally:
                try: f.flush()
                except Exception: pass
                f.close()
            return
        except OSError as e:
            last_exc = e
            # Only retry on likely lock/consistency errors
            if not _looks_like_lock_error(e) or attempt == retries:
                raise
            logger.log(f"[HDF5] open({mode}) failed with lock: {e} (attempt {attempt+1}/{retries})")
            try: _h5clear(p)
            except Exception: pass
            time.sleep(backoff * (attempt + 1))
    raise last_exc

# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H5Dims:
    """Dimensions recorded in the base HDF5 file."""
    nSpat: int   # spatial apertures
    nLSpec: int  # observed spectrum length
    nTSpec: int  # template spectrum length (time-domain)
    nVel: int    # LOSVD length
    nComp: int   # number of LOSVD components
    nPop: int    # number of stellar population templates

# --------------------------------------------------------------------------- #
# Flux-conserving rebin helpers (exact linRebin equivalent)
# --------------------------------------------------------------------------- #

def _bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """
    Compute bin edges from approximately uniform centers. Interior edges are
    midpoints; the first/last edges are extrapolated by half a step.
    """
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.ndim != 1 or c.size < 2:
        raise ValueError("centers must be 1-D with length >= 2.")
    step0 = c[1] - c[0]
    stepN = c[-1] - c[-2]
    mid = 0.5 * (c[:-1] + c[1:])
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[1:-1] = mid
    edges[0] = c[0] - 0.5 * step0
    edges[-1] = c[-1] + 0.5 * stepN
    return edges

def _build_linrebin_matrix_from_edges(old_edges: np.ndarray,
                                      new_edges: np.ndarray) -> np.ndarray:
    """
    Build a dense, flux-conserving rebin matrix R such that

        new[j] = sum_i old[i] * overlap(old_i, new_j) / width(new_j),

    where old_i = [old_edges[i], old_edges[i+1]] and new_j is defined
    similarly. This matches the semantics of the reference linRebin.
    """
    old_edges = np.asarray(old_edges, dtype=np.float64).ravel()
    new_edges = np.asarray(new_edges, dtype=np.float64).ravel()
    if old_edges.ndim != 1 or new_edges.ndim != 1:
        raise ValueError("edges must be 1-D.")
    if np.any(np.diff(old_edges) <= 0) or np.any(np.diff(new_edges) <= 0):
        raise ValueError("edges must be strictly increasing.")

    n_src = old_edges.size - 1
    n_obs = new_edges.size - 1
    R = np.zeros((n_obs, n_src), dtype=np.float64)

    i = 0
    for j in range(n_obs):
        a, b = new_edges[j], new_edges[j + 1]
        width_new = b - a
        # advance i to first old bin that may overlap
        while i < n_src and old_edges[i + 1] <= a:
            i += 1
        ii = i
        while ii < n_src and old_edges[ii] < b:
            left = max(a, old_edges[ii])
            right = min(b, old_edges[ii + 1])
            overlap = right - left
            if overlap > 0:
                R[j, ii] = overlap / width_new
            ii += 1
    return R

def _build_linrebin_matrix_from_range(lam_range: Tuple[float, float],
                                      n_src: int, n_obs: int) -> np.ndarray:
    """
    Convenience wrapper to build R from a common wavelength range and counts.
    """
    lam_lo, lam_hi = float(lam_range[0]), float(lam_range[1])
    old_edges = np.linspace(lam_lo, lam_hi, num=n_src + 1, dtype=np.float64)
    new_edges = np.linspace(lam_lo, lam_hi, num=n_obs + 1, dtype=np.float64)
    return _build_linrebin_matrix_from_edges(old_edges, new_edges)

# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #

class H5Manager:
    """
    HDF5 manager that owns the base file layout and derived artifacts.

    Typical usage:
        mgr = H5Manager(base_h5, tem_pix=tem, obs_pix=obs)
        mgr.populate_from_arrays(losvd=..., datacube=..., templates=...)
        mgr.ensure_rebin_and_resample()  # /RebinMatrix, /R_T, /TemplatesFFT_R

    After this, the builder and solver read everything they need from disk.
    """

    def __init__(self,
                 base_path: str | Path,
                 *,
                 compression: str = "gzip",
                 clevel: int = 4,
                 shuffle: bool = True,
                 tem_pix: Optional[np.ndarray] = None,
                 obs_pix: Optional[np.ndarray] = None) -> None:
        self.base_path = Path(base_path)
        self.compression = compression
        self.clevel = int(clevel)
        self.shuffle = bool(shuffle)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

        # If grids are provided, persist them and build rebin artifacts.
        if tem_pix is not None and obs_pix is not None:
            self.set_spectral_grids(tem_pix, obs_pix)

    # ------------------------------- Base IO -------------------------------- #

    def _open_ro(self):
        """Robust read-only open via global wrapper."""
        return open_h5(self.base_path, role="reader")

    def _open_rw(self):
        """Robust read/write open via global wrapper."""
        return open_h5(self.base_path, role="writer")

    def init_base(self, dims: H5Dims) -> None:
        """
        Ensure core datasets exist with correct shape/dtype. Recreate any that
        mismatch. Record dims in file attributes.
        """
        nS, nL, nT = int(dims.nSpat), int(dims.nLSpec), int(dims.nTSpec)
        nV, nC, nP = int(dims.nVel), int(dims.nComp), int(dims.nPop)

        def req(name: str, shape, dtype, chunks):
            with self._open_rw() as f:
                if name in f:
                    ds = f[name]
                    if tuple(ds.shape) == tuple(shape) and str(ds.dtype) == dtype:
                        return
                    del f[name]
                f.create_dataset(
                    name, shape=shape, dtype=dtype, chunks=chunks,
                    compression=self.compression, compression_opts=self.clevel,
                    shuffle=self.shuffle, fletcher32=False
                )

        # LOSVD kernels: (nSpat, nVel, nComp)
        req("/LOSVD", (nS, nV, nC), "f8",
            (min(32, nS), nV, min(16, nC)))

        # Observed spectra: (nSpat, nLSpec)
        req("/DataCube", (nS, nL), "f8", (min(32, nS), nL))

        # Templates: time-domain (nPop, nTSpec)
        req("/Templates", (nP, nT), "f8", (min(32, nP), nT))

        # Templates FFT (same width as time-domain length)
        req("/TemplatesFFT", (nP, nT), "complex128",
            (min(32, nP), nT))

        with self._open_rw() as f:
            # Store scalar attrs (portable in HDF5)
            f.attrs["nSpat"] = int(nS)
            f.attrs["nLSpec"] = int(nL)
            f.attrs["nTSpec"] = int(nT)
            f.attrs["nVel"]   = int(nV)
            f.attrs["nComp"]  = int(nC)
            f.attrs["nPop"]   = int(nP)
            f.attrs["dims_schema"] = "scalar_v1"  # optional hint
        logger.log("[H5Manager] Base initialized.")

    @property
    def models_path(self) -> str:
        """The HDF5 file that contains /HyperCube/models (same as base_path)."""
        return str(self.base_path)

    def ensure_models(self, *, shape: tuple, chunks: tuple, dtype: str = "f4") -> None:
        """
        Ensure /HyperCube/models exists with the given shape/chunks/dtype.
        Recreate if incompatible. Stores basic attrs on the /HyperCube group.
        """
        S, C, P, L = map(int, shape)
        cS, cC, cP, cL = map(int, chunks)
        with self._open_rw() as f:
            hg = f.require_group("/HyperCube")
            if "models" in hg:
                ds = hg["models"]
                ok_shape = tuple(ds.shape) == (S, C, P, L)
                ok_dtype = str(ds.dtype) == dtype or (dtype == "f4" and str(ds.dtype) == "float32")
                ok_chunks = (getattr(ds, "chunks", None) == (cS, cC, cP, cL))
                if not (ok_shape and ok_dtype and ok_chunks):
                    del hg["models"]
                    ds = None
            else:
                ds = None

            if ds is None:
                ds = hg.create_dataset(
                    "models",
                    shape=(S, C, P, L),
                    chunks=(cS, cC, cP, cL),
                    dtype=dtype,
                    compression=self.compression,
                    compression_opts=self.clevel,
                    shuffle=self.shuffle,
                )
                logger.log(f"[HDF5] Created /HyperCube/models: shape={(S,C,P,L)}, "
                        f"chunks={(cS,cC,cP,cL)}, dtype={dtype}")
            else:
                logger.log(f"[HDF5] Reusing /HyperCube/models: shape={ds.shape}, "
                        f"chunks={ds.chunks}, dtype={ds.dtype}")

            # stash basic info on the group for quick inspection
            hg.attrs["chunks_S"] = cS
            hg.attrs["chunks_C"] = cC
            hg.attrs["chunks_P"] = cP
            hg.attrs["chunks_L"] = cL
            # clear completion flag (builder will set it true on success)
            hg.attrs["complete"] = False

    def get_manifest(self) -> dict | None:
        """Return manifest dict from /HyperCube attrs, or None if absent."""
        with h5py.File(self.base_path, "r", libver="latest") as f:
            if "/HyperCube" not in f:
                return None
            g = f["/HyperCube"]
            # preferred: JSON string
            if "manifest_json" in g.attrs:
                try:
                    return json.loads(g.attrs["manifest_json"])
                except Exception:
                    return None
            # fallback: older scalar attrs (if you had them)
            keys = ("shape_S", "shape_C", "shape_P", "shape_L", "dtype_models")
            if all(k in g.attrs for k in keys):
                return {
                    "shape": (int(g.attrs["shape_S"]), int(g.attrs["shape_C"]),
                            int(g.attrs["shape_P"]), int(g.attrs["shape_L"])),
                    "chunks": (int(g.attrs.get("chunks_S", 0)),
                            int(g.attrs.get("chunks_C", 0)),
                            int(g.attrs.get("chunks_P", 0)),
                            int(g.attrs.get("chunks_L", 0))),
                    "dtype_models": str(g.attrs["dtype_models"]),
                }
        return None

    def set_manifest(self, *, shape: tuple, chunks: tuple, dtype_models: str = "f4",
                    extra: dict | None = None) -> None:
        """
        Write manifest to /HyperCube attrs (as JSON). Safe to call repeatedly.
        """
        man = {
            "shape": tuple(map(int, shape)),
            "chunks": tuple(map(int, chunks)),
            "dtype_models": dtype_models,
            "models_path": str(self.base_path),   # << required by HyperCubeReader
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            try:
                json.dumps(extra)
                man["extra"] = extra
            except Exception:
                man["extra"] = str(extra)

        with self._open_rw() as f:
            g = f.require_group("/HyperCube")
            g.attrs["manifest_json"] = json.dumps(man)
            # also store a few scalars for quick grep/debug
            S, C, P, L = man["shape"]
            g.attrs["shape_S"] = S; g.attrs["shape_C"] = C
            g.attrs["shape_P"] = P; g.attrs["shape_L"] = L
            cS, cC, cP, cL = man["chunks"]
            g.attrs["chunks_S"] = cS; g.attrs["chunks_C"] = cC
            g.attrs["chunks_P"] = cP; g.attrs["chunks_L"] = cL
            g.attrs["dtype_models"] = dtype_models

    def models_complete(self) -> bool:
        """Return True if /HyperCube.attrs['complete'] is True."""
        with self._open_ro() as f:
            if "/HyperCube" not in f:
                return False
            return bool(f["/HyperCube"].attrs.get("complete", False))

    def mark_models_complete(self, flag: bool = True) -> None:
        """Set /HyperCube.attrs['complete'] flag."""
        with self._open_rw() as f:
            g = f.require_group("/HyperCube")
            g.attrs["complete"] = bool(flag)
        logger.log(f"[HDF5] /HyperCube complete={bool(flag)}")

    def _compute_template_guard_pixels(self, tem_pix: np.ndarray, vel_pix: np.ndarray, *, safety_pad_px: int = 64) -> int:
        tem_pix = np.asarray(tem_pix, dtype=float).ravel()
        vel_pix = np.asarray(vel_pix, dtype=float).ravel()
        if tem_pix.size < 3: raise ValueError("tem_pix must have ≥3 points.")
        if vel_pix.size < 2: raise ValueError("vel_pix must have ≥2 points.")
        dlog_tem = float(np.median(np.diff(tem_pix)))
        vmax = float(np.max(np.abs(vel_pix)))
        kmax = int(np.ceil(np.log1p(vmax / C_KMS) / dlog_tem))
        return int(kmax + int(safety_pad_px))

    def _build_R_T_dense(self, tem_pix: np.ndarray, obs_pix: np.ndarray) -> np.ndarray:
        tem_pix = np.asarray(tem_pix, dtype=float).ravel()
        obs_pix = np.asarray(obs_pix, dtype=float).ravel()
        if not (np.all(np.diff(tem_pix) > 0) and np.all(np.diff(obs_pix) > 0)):
            raise ValueError("tem_pix and obs_pix must be strictly increasing (log-λ).")
        T, L = tem_pix.size, obs_pix.size
        R_T = np.zeros((T, L), dtype=np.float32)
        j  = np.searchsorted(tem_pix, obs_pix, side="right") - 1
        j  = np.clip(j, 0, T - 2)
        j1 = j + 1
        t0, t1 = tem_pix[j], tem_pix[j1]
        denom = (t1 - t0); denom[denom == 0.0] = 1.0
        w1 = (obs_pix - t0) / denom; w0 = 1.0 - w1
        cols = np.arange(L)
        R_T[j,  cols] += w0.astype(np.float32, copy=False)
        R_T[j1, cols] += w1.astype(np.float32, copy=False)
        return R_T

    def get_spectral_grids(self) -> tuple[np.ndarray, np.ndarray]:
        with self._open_ro() as f:
            tem = np.asarray(f["/TemPix"][...], dtype=np.float64)
            obs = np.asarray(f["/ObsPix"][...], dtype=np.float64)
        return tem, obs

    def get_velocity_grid(self) -> np.ndarray | None:
        with self._open_ro() as f:
            if "/VelPix" not in f: return None
            return np.asarray(f["/VelPix"][...], dtype=np.float64)

    def get_template_pop_shape(self) -> tuple[int, ...]:
        with self._open_ro() as f:
            Tds = f["/Templates"]
            a = Tds.attrs.get("pop_shape", None)
            if a is None:
                P = int(Tds.shape[0])
                return (P,)
            return tuple(int(x) for x in a)

    def read_templates_unflattened(self) -> np.ndarray:
        with self._open_ro() as f:
            T2 = np.asarray(f["/Templates"][...], dtype=np.float64)  # (P,T)
            pop_shape = self.get_template_pop_shape()
            P_expected = int(np.prod(pop_shape, dtype=np.int64))
            if T2.shape[0] != P_expected:
                return T2
            return T2.reshape(*pop_shape, T2.shape[1])

    def _write_dims_attrs(self, f, dims: dict) -> None:
        """
        Persist dims in multiple robust forms:
          - / attrs: dims_json (JSON string)
          - / attrs: dims.<key>=value (individual ints for quick access)
          - file attrs: dims_json (duplicate for convenience)
        """
        # normalize to ints
        d = {k: int(v) for k, v in dims.items()}
        s = json.dumps(d)

        # Root group scalars for quick lookups
        for k, v in d.items():
            f["/"].attrs[f"dims.{k}"] = int(v)

        # JSON copies (root and file)
        f["/"].attrs["dims_json"] = s
        f.attrs["dims_json"] = s

    def _read_dims_attrs(self, f) -> dict:
        """
        Read dims from file/group attrs, handling both JSON and scalar forms.
        Falls back to dataset shapes if needed.
        """
        # Prefer JSON on root
        if "dims_json" in f["/"].attrs:
            val = f["/"].attrs["dims_json"]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            try:
                return {k: int(v) for k, v in json.loads(val).items()}
            except Exception:
                pass

        # Reconstruct from scalars
        out = {}
        for k in ("nSpat", "nLSpec", "nTSpec", "nVel", "nComp", "nPop"):
            key = f"dims.{k}"
            if key in f["/"].attrs:
                out[k] = int(f["/"].attrs[key])

        # Fallback to dataset shapes if needed
        if not out:
            if "/DataCube" in f:
                S, L = map(int, f["/DataCube"].shape)
                out["nSpat"], out["nLSpec"] = S, L
            if "/LOSVD" in f:
                S2, V, C = map(int, f["/LOSVD"].shape)
                out.setdefault("nVel", V)
                out.setdefault("nComp", C)
            if "/Templates" in f:
                P, T = map(int, f["/Templates"].shape)
                out.setdefault("nPop", P)
                out.setdefault("nTSpec", T)
        return out

    def _write_guard_attrs(self, f, guard_info: dict) -> None:
        """
        Persist guard_info in robust forms:
          - / attrs: guard.Kguard_px, guard.dlog_tem, ...
          - / attrs: guard_info_json (JSON string)
        """
        # coerce to plain Python ints/floats
        d = {
            "Kguard_px": int(guard_info["Kguard_px"]),
            "dlog_tem": float(guard_info["dlog_tem"]),
            "obs_lo": float(guard_info["obs_lo"]),
            "obs_hi": float(guard_info["obs_hi"]),
            "tem_lo": float(guard_info["tem_lo"]),
            "tem_hi": float(guard_info["tem_hi"]),
            "safety_pad_px": int(guard_info["safety_pad_px"]),
        }
        # scalar attrs on root for quick lookups
        for k, v in d.items():
            f["/"].attrs[f"guard.{k}"] = v
        # JSON copy
        f["/"].attrs["guard_info_json"] = json.dumps(d)

    def _read_guard_attrs(self) -> dict:
        """
        Read guard_info from the file; returns a dict with keys matching _write_guard_attrs.
        """
        with self._open_ro() as f:
            root = f["/"].attrs
            if "guard_info_json" in root:
                val = root["guard_info_json"]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                try:
                    return {k: (int(v) if k.endswith("_px") or k == "Kguard_px" else float(v))
                            for k, v in json.loads(val).items()}
                except Exception:
                    pass
            # fall back to scalar attrs if JSON missing
            out = {}
            for k in ("Kguard_px", "dlog_tem", "obs_lo", "obs_hi", "tem_lo", "tem_hi", "safety_pad_px"):
                key = f"guard.{k}"
                if key in root:
                    out[k] = root[key]
            return out

    def _write_1d(self, f, name: str, arr, dtype, *,
                units: str | None = None,
                semantic: str | None = None,
                chunk_elems: int = 4096):
        """Robust writer for 1-D arrays with overwrite semantics & attrs."""
        a = np.asarray(arr, dtype=dtype).ravel()
        if name in f:
            del f[name]                      # safe here; populate_* runs pre-SWMR
        ds = f.create_dataset(
            name,
            data=a,
            dtype=dtype,
            chunks=(min(int(a.size), int(chunk_elems)),),
            compression=self.compression,
            compression_opts=self.clevel,
            shuffle=self.shuffle,
        )
        if units is not None:
            ds.attrs["units"] = units
        if semantic is not None:
            ds.attrs["semantic"] = semantic
        return ds
    
    def populate_from_arrays(
        self,
        *,
        losvd: np.ndarray,
        datacube: np.ndarray,
        templates: np.ndarray,
        mask: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        tem_pix: np.ndarray | None = None,
        obs_pix: np.ndarray | None = None,
        vel_pix: np.ndarray | None = None,
        safety_pad_px: int = 64,
        xpix: np.ndarray | None = None,     # length nPix, float
        ypix: np.ndarray | None = None,     # length nPix, float
        binnum: np.ndarray | None = None,   # length nPix, int in [0, S)
        orbit_weights: np.ndarray | None = None, 
    ) -> dict:
        """
        Expected input shapes (strict):
        datacube : (L_obs, S)   # wavelength-major
        losvd    : (S, V, C)
        templates: (T_tem, nMetals, nAges, nAlphas) # spectral axis is first
        mask     : (L_obs,) optional

        Stores:
        /DataCube   -> (S, L_obs) float64
        /LOSVD      -> (S, V, C)  float64
        /Templates  -> (P, T_c)   float64 (flattened populations, cropped spectral)
        /TemPix     -> (T_c,)     float64
        /ObsPix     -> (L_obs,)   float64
        /VelPix     -> (V,)       float64
        /R_T        -> (T_c, L_obs) float32
        attrs on /Templates to reconstruct N-D population axes and crop indices.
        /XPix       -> (nPix,) float64
        /YPix       -> (nPix,) float64
        /BinNum     -> (nPix,) int32   (pixel → spatial-bin index in [0..S-1])
        /HyperCube/data_flux -> (S,) float64
            Per-spaxel mean of observed flux over *unmasked* wavelengths (λ),
            computed only over unmasked wavelengths. Masked wavelengths
            are written as 0.0. This vector is the solver/builder’s
            canonical "L_vec".
        """

        # ---------- normalize arrays ----------
        losvd_in     = np.asarray(losvd)
        datacube_in  = np.asarray(datacube)
        templates_in = np.asarray(templates)

        # ---------- source grids from file if not passed ----------
        base_exists = Path(self.base_path).exists()

        if base_exists:
            # Only read if the file already exists
            with self._open_ro() as f_ro:
                if tem_pix is None:
                    if "/TemPix" not in f_ro:
                        raise RuntimeError("Missing /TemPix; pass tem_pix or call "
                                        "set_spectral_grids(...).")
                    tem_pix = np.asarray(f_ro["/TemPix"][...], dtype=np.float64)
                else:
                    tem_pix = np.asarray(tem_pix, dtype=np.float64)

                if obs_pix is None:
                    if "/ObsPix" not in f_ro:
                        raise RuntimeError("Missing /ObsPix; pass obs_pix or call "
                                        "set_spectral_grids(...).")
                    obs_pix = np.asarray(f_ro["/ObsPix"][...], dtype=np.float64)
                else:
                    obs_pix = np.asarray(obs_pix, dtype=np.float64)

                if vel_pix is not None:
                    vel_pix = np.asarray(vel_pix, dtype=np.float64)
        else:
            # Fresh run: no base file yet. If you supplied the grids, persist them.
            if tem_pix is None or obs_pix is None:
                raise RuntimeError(
                    "Base HDF5 does not exist yet and spectral grids were not "
                    "provided. Pass tem_pix and obs_pix (and optionally vel_pix) "
                    "on the first run."
                )
            tem_pix = np.asarray(tem_pix, dtype=np.float64)
            obs_pix = np.asarray(obs_pix, dtype=np.float64)
            if vel_pix is not None:
                vel_pix = np.asarray(vel_pix, dtype=np.float64)

            # Create the file and write the grids + rebin artifacts so downstream
            # code can read them
            self.set_spectral_grids(tem_pix, obs_pix)
            # (No read here; we just wrote them.)

        # ---------- strict shape checks & normalization ----------
        # datacube comes as (L, S) → store as (S, L)
        if datacube_in.ndim != 2:
            raise ValueError(f"datacube must be 2-D (L,S). Got "
                            f"{datacube_in.ndim}D with shape "
                            f"{datacube_in.shape}.")
        L_obs = int(obs_pix.size)
        if datacube_in.shape[0] != L_obs:
            raise ValueError(f"datacube first dim must equal len(ObsPix)={L_obs}, "
                            f"got {datacube_in.shape}.")
        datacube_SL = datacube_in.T  # (S, L)

        # losvd is (S, V, C)
        if losvd_in.ndim != 3:
            raise ValueError(f"losvd must be 3-D (S,V,C). Got "
                            f"{losvd_in.ndim}D with shape {losvd_in.shape}.")
        S, L = map(int, datacube_SL.shape)
        S2, V, C = map(int, losvd_in.shape)
        if S2 != S:
            raise ValueError(f"LOSVD S mismatch: datacube S={S}, losvd S={S2}.")
        if int(vel_pix.size) != V:
            raise ValueError(f"VelPix length {vel_pix.size} != LOSVD V {V}.")

        # templates have spectral axis FIRST with length len(tem_pix)
        T_len = int(tem_pix.size)
        if templates_in.ndim < 1:
            raise ValueError("templates must be at least 1-D.")
        if templates_in.shape[0] != T_len:
            raise ValueError(
                f"templates spectral axis (axis 0) length "
                f"{templates_in.shape[0]} != len(tem_pix) {T_len}."
            )

        templates_in_shape = tuple(templates_in.shape)          # (T, *pop_axes)
        orig_t_axis = 0
        pop_shape = tuple(int(s) for s in templates_in_shape[1:]) or (1,)

        # move spectral axis to last, then flatten populations -> (P, T_len)
        tmpl = np.moveaxis(templates_in, 0, -1)                 # (*pop_axes, T)
        templates_PT = tmpl.reshape(-1, T_len)                  # (P, T)
        P, T = map(int, templates_PT.shape)

        # ---------- compute guard (template pixels) & crop spectral axis ----------
        Kguard = self._compute_template_guard_pixels(
            tem_pix, vel_pix, safety_pad_px=safety_pad_px
        )
        dlog_tem = float(np.median(np.diff(tem_pix)))
        lam_lo   = float(obs_pix.min()) - Kguard * dlog_tem
        lam_hi   = float(obs_pix.max()) + Kguard * dlog_tem

        i_lo = int(np.searchsorted(tem_pix, lam_lo, side="left"))
        i_hi = int(np.searchsorted(tem_pix, lam_hi, side="right"))
        i_lo = max(0, i_lo)
        i_hi = min(T, i_hi)
        if i_hi - i_lo < 2:
            raise ValueError(
                "Templates do not span the required guard. "
                f"Need [{lam_lo:.6f}, {lam_hi:.6f}] in log-λ, but tem_pix runs "
                f"{tem_pix[0]:.6f}..{tem_pix[-1]:.6f}. Provide a wider template "
                "grid."
            )

        clipped_left  = (i_lo == 0) and (tem_pix[0] > lam_lo)
        clipped_right = (i_hi == T) and (tem_pix[-1] < lam_hi)
        if clipped_left or clipped_right:
            eff_lo_px = int(round(max(0.0,
                            (obs_pix.min() - tem_pix[0]) / dlog_tem)))
            eff_hi_px = int(round(max(0.0,
                            (tem_pix[-1] - obs_pix.max()) / dlog_tem)))
            raise ValueError(
                "Template guard insufficient after cropping: "
                f"effective guard (px) left={eff_lo_px}, right={eff_hi_px}, "
                f"required={Kguard}."
            )

        tem_pix_c   = tem_pix[i_lo:i_hi].copy()
        templates_c = templates_PT[:, i_lo:i_hi].copy()
        T_c = int(tem_pix_c.size)
        if T_c < 2:
            raise ValueError("Cropped template grid too small.")

        # ---------- build rebin operator for cropped grid ----------
        R_T = self._build_R_T_dense(tem_pix_c, obs_pix)  # (T_c, L), float32

        # ---------- per-pixel metadata validation ----------
        nPix = None
        if xpix is not None:
            xpix = np.asarray(xpix).ravel()
            nPix = int(xpix.size)
        if ypix is not None:
            ypix = np.asarray(ypix).ravel()
            nPix_y = int(ypix.size)
            if nPix is None:
                nPix = nPix_y
            elif nPix_y != nPix:
                raise RuntimeError(f"/XPix length ({nPix}) and /YPix length "
                                f"({nPix_y}) mismatch.")
        if binnum is not None:
            bn = np.asarray(binnum).ravel()
            nPix_b = int(bn.size)
            if nPix is None:
                nPix = nPix_b
            elif nPix_b != nPix:
                raise RuntimeError(f"/BinNum length ({nPix_b}) and /XPix length "
                                f"({nPix}) mismatch.")
            if not np.issubdtype(bn.dtype, np.integer):
                raise ValueError("binnum must be integer indices mapping pixels "
                                "to spatial bins [0..S-1].")
            if bn.size == 0:
                raise ValueError("binnum must be non-empty if provided.")
            if (bn.min() < 0) or (bn.max() >= S):
                raise ValueError("binnum contains out-of-range indices; expected "
                                f"in [0, {S-1}].")

        # ---------- write everything ----------
        with self._open_rw() as f:
            def _write(name, data, **create_kw):
                if name in f:
                    del f[name]
                return f.create_dataset(name, data=data, **create_kw)

            # grids
            _write("/TemPix", tem_pix_c, dtype=np.float64)
            _write("/ObsPix", obs_pix,   dtype=np.float64)
            _write("/VelPix", vel_pix,   dtype=np.float64)

            # core arrays
            _write("/DataCube",
                datacube_SL.astype(np.float64, copy=False),
                dtype=np.float64)  # (S,L)
            _write("/LOSVD",
                losvd_in.astype(np.float64, copy=False),
                dtype=np.float64)  # (S,V,C)

            Tds = _write("/Templates",
                        templates_c.astype(np.float64, copy=False),
                        dtype=np.float64)  # (P, T_c)

            if mask is not None:
                mask = np.asarray(mask, dtype=bool).ravel()
                if mask.size != L_obs:
                    raise ValueError(f"mask length {mask.size} != L {L_obs}.")
                frac_true = float(np.mean(mask))
                logger.log(
                    f"[Mask] True==keep; fraction kept = {frac_true:.3f}")
                _write("/Mask", mask, dtype=np.bool_)
            if x_init is not None:
                x_init = np.asarray(x_init, dtype=np.float64).ravel()
                if x_init.size != C * P:
                    raise ValueError(f"x_init length {x_init.size} != C*P={C*P}.")
                _write("/X_global", x_init, dtype=np.float64)

            # operators
            _write("/R_T", R_T, dtype=np.float32)

            # dims (post-crop)
            dims = dict(nSpat=S, nLSpec=L_obs, nTSpec=T_c, nVel=V, nComp=C,
                        nPop=P)
            self._write_dims_attrs(f, dims)

            # template metadata (reconstruction + crop)
            Tds.attrs["orig_shape"]  = np.asarray(templates_in_shape,
                                                dtype=np.int64)
            Tds.attrs["orig_t_axis"] = np.int64(orig_t_axis)
            Tds.attrs["pop_shape"]   = np.asarray(pop_shape, dtype=np.int64)
            Tds.attrs["crop_i_lo"]   = np.int64(i_lo)
            Tds.attrs["crop_i_hi"]   = np.int64(i_hi)
            Tds.attrs["T_len_in"]    = np.int64(T)
            Tds.attrs["T_len_out"]   = np.int64(T_c)

            # guard metadata
            guard_info = dict(
                Kguard_px=int(Kguard),
                dlog_tem=float(dlog_tem),
                obs_lo=float(obs_pix.min()),
                obs_hi=float(obs_pix.max()),
                tem_lo=float(tem_pix_c.min()),
                tem_hi=float(tem_pix_c.max()),
                safety_pad_px=int(safety_pad_px),
            )
            self._write_guard_attrs(f, guard_info)

            # ---------- NEW: write per-pixel metadata (only if provided) ----------
            if xpix is not None:
                self._write_1d(
                    f, "/XPix", xpix, np.float64,
                    units="pixel or arcsec (user-supplied)",
                    semantic="image-plane X coordinate per detector pixel",
                )
            if ypix is not None:
                self._write_1d(
                    f, "/YPix", ypix, np.float64,
                    units="pixel or arcsec (user-supplied)",
                    semantic="image-plane Y coordinate per detector pixel",
                )
            if binnum is not None:
                self._write_1d(
                    f, "/BinNum", binnum, np.int32,
                    units="index",
                    semantic="pixel→spatial-bin mapping [0..S-1]",
                )

            # ---------- compute & store /HyperCube/data_flux (S,) -----------
            # Per-spaxel mean over unmasked wavelengths (λ). Masked λ are ignored.
            if "/HyperCube" not in f:
                f.create_group("/HyperCube")

            if mask is None:
                with np.errstate(invalid="ignore"):
                    L_vec = np.nanmean(datacube_SL, axis=1).astype(np.float64)  # (S,)
            else:
                use = np.asarray(mask, dtype=bool).ravel()
                if use.size != L_obs:
                    raise ValueError(f"mask length {use.size} != L {L_obs}.")
                if np.any(use):
                    with np.errstate(invalid="ignore"):
                        L_vec = np.nanmean(datacube_SL[:, use], axis=1).astype(np.float64)
                else:
                    L_vec = np.zeros(S, dtype=np.float64)

            L_vec[~np.isfinite(L_vec)] = 0.0

            if "/HyperCube/data_flux" in f:
                del f["/HyperCube/data_flux"]
            ds = f.create_dataset("/HyperCube/data_flux", data=L_vec, dtype=np.float64)
            ds.attrs["semantic"] = "per-spaxel mean flux over unmasked λ"
            ds.attrs["source"] = "/DataCube + /Mask"
            ds.attrs["stat"] = "mean"

            # ---------- write /CompWeights if provided ----------
            if orbit_weights is not None:
                w = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")
                if w.size not in (C, C * P):
                    raise ValueError(
                        f"orbit_weights length must be C ({C}) or C*P ({C*P}); got {w.size}."
                    )
                if w.size == C * P:
                    w = w.reshape(C, P, order="C").sum(axis=1)  # reduce to component level

                # unit-sum normalize for canonical storage
                s = float(np.sum(w))
                w = (w / np.maximum(s, 1.0e-30)) if s > 0.0 else np.zeros_like(w)

                if "/CompWeights" in f:
                    del f["/CompWeights"]
                ds_w = f.create_dataset("/CompWeights", data=w, dtype=np.float64)
                ds_w.attrs["semantic"] = "prior component weights w_c (unit-sum)"
                ds_w.attrs["normalized"] = bool(np.isclose(np.sum(w), 1.0))

        return dict(nSpat=S, nLSpec=L_obs, nTSpec=T_c, nVel=V, nComp=C, nPop=P)

    # -------------------------- Spectral grids & rebin ------------------------

    def set_spectral_grids(self, tem_pix: np.ndarray,
                           obs_pix: np.ndarray) -> Tuple[int, int]:
        """
        Persist spectral grids (/TemPix, /ObsPix) and a canonical /LamRange
        inferred from tem_pix. Then compute and persist /RebinMatrix and /R_T.

        Returns (N_src, nL) where:
          N_src = len(tem_pix)  (template time-domain length)
          nL    = len(obs_pix)  (observed spectrum length)
        """
        self._store_pix(tem_pix, obs_pix)
        N_src, nL = self.ensure_rebin()
        return N_src, nL

    def _store_pix(self, tem_pix: np.ndarray, obs_pix: np.ndarray) -> None:
        """
        Store spectral grids and canonical wavelength range derived from
        template centers. Idempotent.
        """
        tem_pix = np.asarray(tem_pix, dtype=np.float64).ravel()
        obs_pix = np.asarray(obs_pix, dtype=np.float64).ravel()
        if tem_pix.ndim != 1 or tem_pix.size < 2:
            raise ValueError("tem_pix must be 1-D with length >= 2.")
        if obs_pix.ndim != 1 or obs_pix.size < 2:
            raise ValueError("obs_pix must be 1-D with length >= 2.")
        if not (np.all(np.diff(tem_pix) > 0) and np.all(np.diff(obs_pix) > 0)):
            raise ValueError("tem_pix and obs_pix must be strictly increasing.")

        tem_edges = _bin_edges_from_centers(tem_pix)
        lam_lo, lam_hi = float(tem_edges[0]), float(tem_edges[-1])

        with self._open_rw() as f:
            def put1d(name, data):
                data = np.asarray(data, dtype=np.float64).ravel()
                if name in f:
                    ds = f[name]
                    if tuple(ds.shape) == (data.size,) and str(ds.dtype) == "float64":
                        ds[...] = data
                        return
                    del f[name]
                f.create_dataset(
                    name, data=data, dtype="f8", chunks=(data.size,),
                    compression=self.compression, compression_opts=self.clevel,
                    shuffle=self.shuffle
                )

            put1d("/TemPix", tem_pix)
            put1d("/ObsPix", obs_pix)

            lam_range = np.array([lam_lo, lam_hi], dtype=np.float64)
            if "/LamRange" in f:
                ds = f["/LamRange"]
                if tuple(ds.shape) == (2,) and str(ds.dtype) == "float64":
                    # Only rewrite if noticeably different
                    if not np.allclose(ds[...], lam_range, rtol=0, atol=1e-12):
                        ds[...] = lam_range
                else:
                    del f["/LamRange"]
                    f.create_dataset("/LamRange", data=lam_range, dtype="f8",
                                     chunks=(2,))
            else:
                f.create_dataset("/LamRange", data=lam_range, dtype="f8",
                                 chunks=(2,))

        logger.log(f"[H5Manager] Stored /TemPix(N={tem_pix.size}), "
                   f"/ObsPix(N={obs_pix.size}), "
                   f"/LamRange=({lam_lo:.6f},{lam_hi:.6f}).")

    def set_velocity_grid(self, vel_pix: np.ndarray) -> int:
        """
        Persist /VelPix (km/s), validate monotonicity, and return nVel.

        vel_pix : 1-D array of velocity-bin centers (km/s) corresponding to
                  the second axis of /LOSVD (shape: nVel).
        """
        v = np.asarray(vel_pix, dtype=np.float64).ravel()
        if v.size < 2:
            raise ValueError("vel_pix must be 1-D with length >= 2")
        if not (np.all(np.isfinite(v)) and np.all(np.diff(v) > 0)):
            raise ValueError("vel_pix must be strictly increasing and finite")

        with self._open_rw() as f:
            if "/VelPix" in f:
                ds = f["/VelPix"]
                if ds.shape == (v.size,) and str(ds.dtype) == "float64":
                    ds[...] = v
                else:
                    del f["/VelPix"]
                    f.create_dataset("/VelPix", data=v, dtype="f8",
                                     chunks=(min(4096, v.size),))
            else:
                f.create_dataset("/VelPix", data=v, dtype="f8",
                                 chunks=(min(4096, v.size),))
            f["/VelPix"].attrs["units"] = "km/s"
            f["/VelPix"].attrs["center_definition"] = "bin_centers"

        return int(v.size)

    def ensure_rebin(self) -> Tuple[int, int]:
        """
        Ensure /RebinMatrix (nL, N_src) and /R_T (N_src, nL) exist, using the
        exact flux-conserving (bin-overlap) rebin that matches linRebin.

        Uses /LamRange (from template grid), and the lengths of /TemPix and
        /ObsPix to determine sizes. Idempotent.

        Returns (N_src, nL).
        """
        with self._open_rw() as f:
            for req in ("/LamRange", "/TemPix", "/ObsPix"):
                if req not in f:
                    raise RuntimeError(
                        f"Missing {req}. Call set_spectral_grids(...) first."
                    )

            lam_lo, lam_hi = map(float, f["/LamRange"][...])
            N_src = int(f["/TemPix"].shape[0])
            nL = int(f["/ObsPix"].shape[0])

            need_R = ("/RebinMatrix" not in f or
                      tuple(f["/RebinMatrix"].shape) != (nL, N_src) or
                      str(f["/RebinMatrix"].dtype) != "float64")
            need_RT = ("/R_T" not in f or
                       tuple(f["/R_T"].shape) != (N_src, nL) or
                       str(f["/R_T"].dtype) != "float64")

            if need_R or need_RT:
                R = _build_linrebin_matrix_from_range((lam_lo, lam_hi),
                                                      N_src, nL)
                if need_R:
                    if "/RebinMatrix" in f:
                        del f["/RebinMatrix"]
                    f.create_dataset(
                        "/RebinMatrix", data=R, dtype="f8",
                        chunks=(min(1024, nL), N_src),
                        compression=self.compression, compression_opts=self.clevel,
                        shuffle=self.shuffle
                    )
                if need_RT:
                    if "/R_T" in f:
                        del f["/R_T"]
                    f.create_dataset(
                        "/R_T", data=R.T, dtype="f8",
                        chunks=(min(4096, N_src), nL),
                        compression=self.compression, compression_opts=self.clevel,
                        shuffle=self.shuffle
                    )
                logger.log(f"[H5Manager] Computed /RebinMatrix({nL},{N_src}) "
                           "and /R_T.")
            else:
                logger.log("[H5Manager] Rebin already present; reusing.")

            return N_src, nL

    def ensure_rebin_and_resample(self) -> tuple[int, int, int]:
        """
        Ensure /R_T, /RebinMatrix and the template FFT caches exist and match current grids.

        Returns
        -------
        (P, T, L): population count, template length (cropped), observed length
        """

        with self._open_rw() as f:
            # ---- required grids ----
            if "/TemPix" not in f or "/ObsPix" not in f:
                raise RuntimeError("Missing /TemPix or /ObsPix; call set_spectral_grids(...) first.")
            tem_pix = np.asarray(f["/TemPix"][...], dtype=np.float64)   # (T,)
            obs_pix = np.asarray(f["/ObsPix"][...], dtype=np.float64)   # (L,)
            T = int(tem_pix.size)
            L = int(obs_pix.size)

            # ---- rebin operator(s): /R_T (T,L), /RebinMatrix (L,T) ----
            need_rt = ("/R_T" not in f) or (f["/R_T"].shape != (T, L))
            need_rm = ("/RebinMatrix" not in f) or (f["/RebinMatrix"].shape != (L, T))
            if need_rt or need_rm:
                R_T = self._build_R_T_dense(tem_pix, obs_pix)  # (T,L) float32
                if need_rt:
                    if "/R_T" in f:
                        del f["/R_T"]
                    f.create_dataset("/R_T", data=R_T, dtype=np.float32)
                if need_rm:
                    if "/RebinMatrix" in f:
                        del f["/RebinMatrix"]
                    f.create_dataset("/RebinMatrix", data=R_T.T, dtype=np.float32)
                # optional: keep them in sync if only one existed but was mismatched
            # sanity (after potential writes)
            R_T = f["/R_T"]
            assert R_T.shape == (T, L)

            # ---- templates present? ----
            if "/Templates" not in f:
                raise RuntimeError("Missing /Templates; call populate_from_arrays(...) first.")
            Tds = f["/Templates"]
            P, T_templates = map(int, Tds.shape)
            if T_templates != T:
                # If someone changed /TemPix after writing /Templates, fix is to re-run populate_from_arrays.
                # We continue using the templates' spectral length for FFTs, but warn via shape mismatch.
                T = T_templates  # keep internal consistency for FFT build below

            # ---- template FFT caches ----
            nfreq = T // 2 + 1  # rfft length
            need_fft  = ("/TemplatesFFT"    not in f) or (f["/TemplatesFFT"].shape    != (P, nfreq))
            need_fftR = ("/TemplatesFFT_R"  not in f) or (f["/TemplatesFFT_R"].shape  != (P, nfreq))

            if need_fft or need_fftR:
                # load templates (P,T) as float64 for numerical stability
                T_mat = np.asarray(Tds[...], dtype=np.float64, order="C")
                F  = np.fft.rfft(T_mat, n=T, axis=1)             # (P, nfreq) complex128
                FR = np.fft.rfft(T_mat[:, ::-1], n=T, axis=1)    # reversed for true convolution

                if need_fft:
                    if "/TemplatesFFT" in f:
                        del f["/TemplatesFFT"]
                    f.create_dataset("/TemplatesFFT", data=F.astype(np.complex64, copy=False))

                if need_fftR:
                    if "/TemplatesFFT_R" in f:
                        del f["/TemplatesFFT_R"]
                    f.create_dataset("/TemplatesFFT_R", data=FR.astype(np.complex64, copy=False))

        # Return basic dims
        return (P, T, L)

# --------------------------------------------------------------------------- #
# Convenience wrappers
# ------------------------------------------------------------------------------

def populate_base_from_arrays(base_h5: str | Path,
                              *,
                              losvd: np.ndarray,
                              datacube: np.ndarray,
                              templates: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              x_init: Optional[np.ndarray] = None,
                              tem_pix: Optional[np.ndarray] = None,
                              obs_pix: Optional[np.ndarray] = None,
                              vel_pix: Optional[np.ndarray] = None) -> H5Dims:
    """
    Module-level convenience wrapper for H5Manager.populate_from_arrays.
    """
    mgr = H5Manager(base_h5)
    return mgr.populate_from_arrays(
        losvd=losvd, datacube=datacube, templates=templates,
        mask=mask, x_init=x_init,
        tem_pix=tem_pix, obs_pix=obs_pix, vel_pix=vel_pix
    )

def set_spectral_grids(base_h5: str | Path,
                       tem_pix: np.ndarray,
                       obs_pix: np.ndarray) -> Tuple[int, int]:
    """
    Module-level wrapper to persist spectral grids and build rebin artifacts.
    """
    mgr = H5Manager(base_h5)
    return mgr.set_spectral_grids(tem_pix, obs_pix)

# ------------------------------------------------------------------------------

def recompress_hypercube_models(
    base_h5: str | Path,
    *,
    dataset: str = "/HyperCube/models",
    compression: str = "gzip", # or "lzf"
    compression_opts: Optional[int] = 4, # gzip level (1–9), None = lib default
    shuffle: bool = True,
    chunk_overrides: Optional[Tuple[int,int,int,int]] = None,  # (S,C,P,L)
    temp_name: str = "models_tmp",
    keep_backup: bool = False,
) -> dict:
    """
    Re-encode `/HyperCube/models` with new filters by copying chunk-by-chunk to a
    sibling dataset, then swapping names. Keeps a **single writer** open, in line
    with open_h5(...) usage elsewhere. File size on disk may not shrink until you
    run `h5repack` (see compact_file_via_h5repack below).

    Returns a dict with before/after chunking and (if available) on-disk sizes.
    """
    start = time.time()
    base_h5 = Path(base_h5)

    with open_h5(base_h5, "writer") as f:
        if dataset not in f:
            raise RuntimeError(f"{dataset} not found in {base_h5}")
        old = f[dataset]
        parent = old.parent  # /HyperCube

        shape = tuple(int(x) for x in old.shape)           # (S,C,P,L)
        dtype = old.dtype
        old_chunks = old.chunks or (shape[0], 1, min(64, shape[2]), shape[3])
        chunks = tuple(chunk_overrides) if chunk_overrides else old_chunks

        # fresh temp
        if temp_name in parent:
            del parent[temp_name]

        kwargs = dict(shape=shape, dtype=dtype, chunks=chunks)
        if compression not in (None, "none", 0, False):
            kwargs.update(compression=compression,
                          compression_opts=compression_opts,
                          shuffle=bool(shuffle))
        new = parent.create_dataset(temp_name, **kwargs)

        # copy attributes
        for k, v in old.attrs.items():
            new.attrs[k] = v

        # chunked copy using old chunk grid (minimize read amplification)
        S_ch, C_ch, P_ch, L_ch = old_chunks
        for s0 in range(0, shape[0], S_ch):
            s1 = min(s0 + S_ch, shape[0])
            for c0 in range(0, shape[1], C_ch):
                c1 = min(c0 + C_ch, shape[1])
                for p0 in range(0, shape[2], P_ch):
                    p1 = min(p0 + P_ch, shape[2])
                    new[s0:s1, c0:c1, p0:p1, :] = old[s0:s1, c0:c1, p0:p1, :]

        # atomically swap; optionally keep backup
        backup_name = None
        if keep_backup:
            backup_name = f"{Path(dataset).name}_backup_{int(time.time())}"
            parent.move(Path(dataset).name, backup_name)
        else:
            del parent[Path(dataset).name]
        parent.move(temp_name, Path(dataset).name)

        # try to report storage sizes
        old_size = None
        new_size = None
        try:
            if backup_name:
                old_size = parent[backup_name].id.get_storage_size()
            new_size = parent[Path(dataset).name].id.get_storage_size()
        except Exception:
            pass

        return {
            "dataset": dataset,
            "shape": shape,
            "dtype": str(dtype),
            "chunks_old": tuple(old_chunks),
            "chunks_new": tuple(chunks),
            "compression": compression,
            "compression_opts": compression_opts,
            "shuffle": bool(shuffle),
            "old_size_bytes": old_size,
            "new_size_bytes": new_size,
            "elapsed_sec": time.time() - start,
            "backup_dataset": backup_name,
        }

def compact_file_via_h5repack(path: str | Path) -> Path:
    """
    Physically compact an HDF5 file by rewriting it with `h5repack`.
    All handles to the file must be closed before calling this function.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".repacked")
    try:
        subprocess.run(["h5repack", "-i", str(path), "-o", str(tmp), "-v"], check=True)
    except FileNotFoundError:
        raise RuntimeError("h5repack not found in PATH; cannot compact file.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"h5repack failed: {e}")
    tmp.replace(path)
    return path

# ------------------------------------------------------------------------------

def _get_obs_grid(f) -> np.ndarray:
    L = int(f["/DataCube"].shape[1])
    if "/ObsPix" in f:
        lam = np.asarray(f["/ObsPix"][...], dtype=np.float64).reshape(L)
    else:
        lam = np.arange(L, dtype=np.float64)
    return lam  # (L,)

def _get_tem_grid(f) -> np.ndarray:
    if "/Templates" not in f:
        raise RuntimeError("Missing /Templates")
    T = int(f["/Templates"].shape[1])
    if "/TemPix" in f:
        lamT = np.asarray(f["/TemPix"][...], dtype=np.float64).reshape(T)
    else:
        lamT = np.arange(T, dtype=np.float64)
    return lamT  # (T,)

def _get_vel_grid(f, V: int) -> np.ndarray:
    # Try common names; otherwise index grid
    for k in ("/VelGrid", "/VelPix", "/VelocityGrid"):
        if k in f:
            vg = np.asarray(f[k][...], dtype=np.float64).reshape(V)
            return vg
    return np.arange(V, dtype=np.float64)

def _load_mask(f, L: int) -> Optional[np.ndarray]:
    if "/Mask" in f:
        m = np.asarray(f["/Mask"][...], dtype=bool).reshape(L)
        return m  # True = keep (reader semantics)
    return None

def read_observed_spectrum(base_h5: str | Path, s: int, *, apply_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (λ_obs, y_s)."""
    with open_h5(base_h5, "reader") as f:
        Y = f["/DataCube"]
        L = int(Y.shape[1])
        lam = _get_obs_grid(f)
        y = np.asarray(Y[s, :], dtype=np.float64, order="C")
        if apply_mask:
            m = _load_mask(f, L)
            if m is not None:
                lam, y = lam[m], y[m]
        return lam, y

def read_template_spectrum(
    base_h5: str | Path, p: int, *,
    rebin_to_obs: bool = True,
    apply_mask: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a single template spectrum p.
      If rebin_to_obs=False: (λ_tem, T_native) on native grid (T).
      If rebin_to_obs=True:  (λ_obs,  T_obs)   on observed grid (L) using /RebinMatrix or /R_T.
    """
    with open_h5(base_h5, "reader") as f:
        Tds = f["/Templates"]                # (P, T)
        T_native = np.asarray(Tds[p, :], dtype=np.float64, order="C")  # (T,)
        if not rebin_to_obs:
            return _get_tem_grid(f), T_native

        # Rebin to observed grid (L)
        L = int(f["/DataCube"].shape[1])
        lam_obs = _get_obs_grid(f)
        # Try /RebinMatrix first, else /R_T; handle orientation automatically
        T = T_native.shape[0]
        T_obs = None
        if "/RebinMatrix" in f:
            R = np.asarray(f["/RebinMatrix"][...])
            if R.shape == (L, T):
                T_obs = (R @ T_native)
            elif R.shape == (T, L):
                T_obs = (T_native @ R)
        if T_obs is None and "/R_T" in f:
            RT = np.asarray(f["/R_T"][...])
            if RT.shape == (T, L):
                T_obs = (T_native @ RT)
            elif RT.shape == (L, T):
                T_obs = (RT @ T_native)
        if T_obs is None:
            raise RuntimeError("No compatible /RebinMatrix or /R_T found for rebinning.")
        T_obs = np.asarray(T_obs, dtype=np.float64, order="C")

        if apply_mask:
            m = _load_mask(f, L)
            if m is not None:
                lam_obs, T_obs = lam_obs[m], T_obs[m]
        return lam_obs, T_obs

def read_losvd(base_h5: str | Path, s: int, c: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (v_grid, LOSVD_{s,c}(v))."""
    with open_h5(base_h5, "reader") as f:
        LSV = f["/LOSVD"]                    # (S, V, C)
        V = int(LSV.shape[1])
        vgrid = _get_vel_grid(f, V)
        los = np.asarray(LSV[s, :, c], dtype=np.float64, order="C")
        return vgrid, los

def read_model_basis(base_h5: str | Path, s: int, c: int, p: int, *, apply_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (λ_obs, basis) where basis = models[s, c, p, :] — post‑convolution basis for that (s,c,p)."""
    with open_h5(base_h5, "reader") as f:
        M = f["/HyperCube/models"] # (S, C, P, L)
        if M.ndim != 4:
            raise RuntimeError("/HyperCube/models must be rank-4 (S,C,P,L)")
        L = int(M.shape[3])
        lam = _get_obs_grid(f)
        basis = np.asarray(M[s, c, p, :], dtype=np.float64, order="C")
        if apply_mask:
            m = _load_mask(f, L)
            if m is not None:
                lam, basis = lam[m], basis[m]
        return lam, basis

def reconstruct_model_spectrum(base_h5: str | Path, s: int, x_global: np.ndarray, *, apply_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (λ_obs, model_spectrum) for a given spaxel s and global weights x_global (length C*P).
    Uses the same contraction over (C,P) as in your reconstruction helper.  :contentReference[oaicite:12]{index=12}
    """
    with open_h5(base_h5, "reader") as f:
        M = f["/HyperCube/models"]           # (S, C, P, L)
        if M.ndim != 4:
            raise RuntimeError("/HyperCube/models must be rank-4 (S,C,P,L)")
        S, C, P, L = map(int, M.shape)
        lam = _get_obs_grid(f)
        x = np.asarray(x_global, dtype=np.float64).reshape(C, P)  # (C,P)
        slab = np.asarray(M[s, :, :, :], dtype=np.float64, order="C")  # (C,P,L)
        y_model = np.tensordot(slab, x, axes=([0, 1], [0, 1]))         # (L,)
        if apply_mask:
            m = _load_mask(f, L)
            if m is not None:
                lam, y_model = lam[m], y_model[m]
        return lam, y_model

# ------------------------------------------------------------------------------

def plot_prefit_panel(
    base_h5: str | Path,
    s: int, c: int, p: int,
    *,
    include_mix: bool = False,
    x_global: Optional[np.ndarray] = None,
):
    """
    2x2 panel for a given (s,c,p):
      (1) Observed y_s(λ)
      (2) Template T_p rebinned to observed grid (pre‑convolution)
      (3) LOSVD_{s,c}(v)
      (4) Post‑convolution basis models[s,c,p,:]
    If include_mix=True and x_global is provided, overlays reconstructed model A_s^T x.
    """
    galaxy = Path(base_h5).stem.split("_")[0]
    pDir = Path(base_h5).parent/galaxy
    pDir.mkdir(parents=True, exist_ok=True)

    lam_obs, y = read_observed_spectrum(base_h5, s, apply_mask=True)
    lam_t,  T  = read_template_spectrum(base_h5, p, rebin_to_obs=True, apply_mask=True)
    v, los     = read_losvd(base_h5, s, c)
    lam_b, b   = read_model_basis(base_h5, s, c, p, apply_mask=True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    axs[0,0].plot(lam_obs, y)
    axs[0,0].set_title(f"Observed spectrum (s={s})")
    axs[0,0].set_xlabel("λ")
    axs[0,0].set_ylabel("Flux")

    axs[0,1].plot(lam_t, T)
    axs[0,1].set_title(f"Template p={p} (rebinned, pre‑conv)")
    axs[0,1].set_xlabel("λ")

    axs[1,0].plot(v, los)
    axs[1,0].set_title(f"LOSVD (s={s}, c={c})")
    axs[1,0].set_xlabel("velocity bin")

    axs[1,1].plot(lam_b, b, label=f"basis (s={s},c={c},p={p})")
    if include_mix and x_global is not None:
        lam_m, y_m = reconstruct_model_spectrum(base_h5, s, x_global, apply_mask=True)
        axs[1,1].plot(lam_m, y_m, linestyle="--", label="A_s^T x (mixture)")
        axs[1,1].legend()
    axs[1,1].set_title("Post‑convolution basis / mixture")
    axs[1,1].set_xlabel("λ")

    fig.savefig(pDir/'prefit')
    plt.close('all')

# ------------------------------------------------------------------------------

def _component_scale(f, s: int, c: int, eps: float = 1e-30) -> tuple[float, str, float]:
    """
    Compute the scale applied to (s,c) columns in /HyperCube/models.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 handle.
    s : int
        Spaxel index.
    c : int
        Component index.
    eps : float, optional
        Numerical floor to avoid divide-by-zero.

    Returns
    -------
    scale_sc : float
        The multiplicative factor actually applied to every (p,λ) for this
        (s,c) in the models array.
    mode : str
        'data' or 'model' (the normalization mode).
    frac : float
        For data-mode only: A[s,c] / sum_c A[s,c]. For model-mode this is
        0.0 (unused).

    Examples
    --------
    >>> with open_h5(h5, 'reader') as f:
    ...     scale, mode, frac = _component_scale(f, 1033, 124)
    ...     print(scale, mode, frac)
    """
    mode = str(f["/HyperCube"].attrs.get("norm.mode", "model")).lower()
    A_sc = float(f["/HyperCube/norm/losvd_amp"][s, c])  # A[s,c]
    if mode == "data":
        a_sum = float(f["/HyperCube/norm/losvd_amp_sum"][s])  # Σ_c A[s,c]
        Ls = float(f["/HyperCube/data_flux"][s])              # masked mean data
        if a_sum <= 0.0 or Ls <= 0.0:
            return 0.0, mode, 0.0
        frac = A_sc / max(a_sum, eps)
        return Ls * frac, mode, frac
    else:
        return A_sc, mode, 0.0

# ------------------------------------------------------------------------------

def live_prefit_snapshot_from_models(
    h5_path: str,
    *,
    max_spaxels: int = 6,
    max_components: int = 2,
    max_templates: int = 4,         # used in TR panel
    max_sc_pairs: int = 3,          # NEW: number of (s,c) pairs in BR
    templates_per_pair: int = 2,    # NEW: curves per (s,c) in BR
    out_png: str = "prefit_live.png",
    shade_alpha: float = 0.28,
    q_lo: float = 0.02,
    q_hi: float = 0.98,
    pad_frac: float = 0.20,
    seed: int | None = None,
) -> str:
    """
    SWMR-safe diagnostic snapshot while the HyperCube build is running.

    Panels:
      TL  Observed spectra (prefer distinct S-tiles)
      TR  Templates rebinned (pre-convolution)
      BL  LOSVD (native) + resampled kernel
      BR  ACTUAL /HyperCube/models for several diverse (s,c) pairs

    - Masked pixels are shaded but do NOT influence autoscaling.
    - BR now overlays multiple (s,c) pairs from distinct tiles for better coverage.
    """
    rng = np.random.default_rng(seed)

    # ---------- helpers ----------
    def _true_runs(mask_bool_1d: np.ndarray):
        idx = np.flatnonzero(mask_bool_1d)
        if idx.size == 0:
            return []
        brk = np.where(np.diff(idx) != 1)[0] + 1
        starts = np.concatenate([idx[[0]], idx[brk]])
        ends   = np.concatenate([idx[brk - 1], idx[[-1]]])
        return list(zip(starts, ends))

    def _shade_mask(ax, lam_obs, mask_raw):
        if mask_raw is None:
            return
        lam = np.asarray(lam_obs, dtype=float).ravel()
        if lam.size < 2:
            return
        keep_mask = np.asarray(mask_raw, dtype=bool).ravel()   # True == keep
        bad_mask  = ~keep_mask                                  # shaded regions
        dlam = float(np.median(np.diff(lam)))
        half = abs(dlam) * 0.5
        x_min, x_max = float(lam[0]), float(lam[-1])
        for i0, i1 in _true_runs(bad_mask):
            left  = max(float(lam[max(i0, 0)])  - half, x_min)
            right = min(float(lam[min(i1, lam.size - 1)]) + half, x_max)
            if right <= left:
                right = min(left + abs(dlam), x_max)
            ax.axvspan(left, right, color="k", alpha=shade_alpha, lw=0, zorder=5)

    def _set_ylim_from_unmasked(ax, ys, mask_raw, qlo=q_lo, qhi=q_hi, pad=pad_frac):
        vals = []
        if mask_raw is None:
            for y in ys:
                z = np.asarray(y, dtype=float).ravel()
                if z.size:
                    vals.append(z[np.isfinite(z)])
        else:
            keep = np.asarray(mask_raw, dtype=bool).ravel()     # True == keep
            for y in ys:
                z = np.asarray(y, dtype=float).ravel()
                if z.size:
                    if keep.size == z.size:
                        vals.append(z[keep & np.isfinite(z)])
                    else:
                        vals.append(z[np.isfinite(z)])
        if not vals:
            return
        cat = np.concatenate([v for v in vals if v.size], dtype=float)
        if cat.size == 0:
            return
        y0 = float(np.quantile(cat, qlo)); y1 = float(np.quantile(cat, qhi))
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
            y0 = float(np.nanmin(cat)); y1 = float(np.nanmax(cat))
            if not np.isfinite(y0) or not np.isfinite(y1):
                return
            if y1 == y0:
                y1 = y0 + 1.0
        pad_abs = pad * (y1 - y0 if y1 > y0 else 1.0)
        ax.set_ylim(y0 - pad_abs, y1 + pad_abs)

    # ---------- SWMR read ----------
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        if "/HyperCube/models" not in f or "/HyperCube/_done" not in f:
            raise RuntimeError("Hypercube not initialized yet (missing /HyperCube/models or /HyperCube/_done).")

        lam_obs = np.asarray(f["/ObsPix"][...], dtype=float)
        tem_log = np.asarray(f["/TemPix"][...], dtype=float)
        vel_pix = np.asarray(f["/VelPix"][...], dtype=float)

        # Rebin orientation (T,L)
        R = np.asarray(f["/R_T"][...])
        if R.shape == (tem_log.size, lam_obs.size):
            R_T = R
        elif R.shape == (lam_obs.size, tem_log.size):
            R_T = R.T
        else:
            raise RuntimeError(f"Incompatible /R_T shape {R.shape} vs T={tem_log.size}, L={lam_obs.size}")

        mask = None
        if "/Mask" in f:
            m = np.asarray(f["/Mask"][...]).ravel()
            mask = np.asarray(m != 0, dtype=bool)

        M    = f["/HyperCube/models"]   # (S,C,P,L)
        DONE = f["/HyperCube/_done"]    # (Sgrid,Cgrid,Pgrid)

        for obj in (M, DONE):
            try:
                obj.refresh()
            except Exception:
                try:
                    obj.id.refresh()
                except Exception:
                    pass

        S, C, P, L = map(int, M.shape)
        s_chunk, c_chunk, p_chunk, l_chunk = M.chunks
        done_arr = np.asarray(DONE[...], dtype=bool)

        sgrid = int(np.ceil(S / s_chunk))
        cgrid = int(np.ceil(C / c_chunk))
        pgrid = int(np.ceil(P / p_chunk))
        if done_arr.shape[:3] != (sgrid, cgrid, pgrid):
            sgrid, cgrid, pgrid = done_arr.shape[:3]

        # --- Prefer diverse S-tiles for TL/BR selections
        good_si = np.where(done_arr.any(axis=(1, 2)))[0]
        if good_si.size == 0:
            raise RuntimeError("No completed tiles yet; try again later.")

        def srange(si):
            s0 = int(si * s_chunk); s1 = int(min(S, s0 + s_chunk))
            return s0, s1

        # TL spaxels: try one from each of as many distinct S-tiles as possible
        rng.shuffle(good_si)
        spaxel_list = []
        for si in good_si:
            s0, s1 = srange(si)
            if s1 > s0:
                spaxel_list.append(int(rng.integers(s0, s1)))
            if len(spaxel_list) >= max_spaxels:
                break
        # fallback: fill from first tile if we still need more
        if len(spaxel_list) < max_spaxels:
            s0, s1 = srange(good_si[0])
            need = max_spaxels - len(spaxel_list)
            extras = rng.choice(np.arange(s0, s1), size=min(need, s1 - s0), replace=False)
            spaxel_list.extend(map(int, extras))
        sp_sel = np.array(spaxel_list[:max_spaxels], dtype=int)

        # TR: choose some templates (global)
        TDS = f["/Templates"]
        p_pool = np.arange(P)
        rng.shuffle(p_pool)
        p_sel_tr = np.array(sorted(p_pool[:min(max_templates, P)]), dtype=int)
        T_sel = np.asarray(TDS[p_sel_tr, :], dtype=float)
        # SAFE (BLAS-free) path: (B,T) x (T,L) -> (B,L) via broadcast & sum
        TR_mat = np.sum(T_sel[:, :, None] * R_T[None, :, :], axis=1, dtype=np.float64)

        # BL: components for LOSVD variety
        co_pool = np.arange(C)
        rng.shuffle(co_pool)
        co_sel = np.array(sorted(co_pool[:min(max_components, C)]), dtype=int)

        LOS = f["/LOSVD"]  # (S,V,C)
        def _has_signal(s_idx: int, c_idx: int) -> bool:
            row = np.asarray(LOS[s_idx, :, c_idx], dtype=float)
            return np.isfinite(row).any() and float(row.sum()) > 0.0

        # --- BR: build diverse (s,c) PAIRS from distinct tiles with any P done
        # boolean grid of finished (si,ci) pairs
        done_si_ci = done_arr.any(axis=2)            # (sgrid, cgrid)
        pairs = np.argwhere(done_si_ci)              # list of [si, ci]
        rng.shuffle(pairs)

        sc_pairs = []
        used_si, used_ci = set(), set()

        # pass 1: prefer new si and new ci
        for si, ci in pairs:
            if si not in used_si and ci not in used_ci:
                # pick a concrete s, c index inside those tiles
                s0, s1 = srange(int(si))
                if s1 <= s0:
                    continue
                s_idx = int(rng.integers(s0, s1))
                c_idx = int(min(C - 1, int(ci) * c_chunk + rng.integers(0, max(1, c_chunk))))
                if not _has_signal(s_idx, c_idx):
                    continue
                sc_pairs.append((s_idx, c_idx, int(si), int(ci)))
                used_si.add(int(si)); used_ci.add(int(ci))
                if len(sc_pairs) >= max_sc_pairs:
                    break

        # pass 2: fill remaining slots, relaxing the uniqueness
        if len(sc_pairs) < max_sc_pairs:
            for si, ci in pairs:
                s0, s1 = srange(int(si))
                if s1 <= s0:
                    continue
                s_idx = int(rng.integers(s0, s1))
                c_idx = int(min(C - 1, int(ci) * c_chunk + rng.integers(0, max(1, c_chunk))))
                if not _has_signal(s_idx, c_idx):
                    continue
                sc_pairs.append((s_idx, c_idx, int(si), int(ci)))
                if len(sc_pairs) >= max_sc_pairs:
                    break

        if len(sc_pairs) == 0:
            # Fallback: scan for any non-zero (s,c)
            for s_idx in range(S):
                for c_idx in range(C):
                    if _has_signal(s_idx, c_idx):
                        sc_pairs.append(
                            (s_idx, c_idx, int(s_idx // s_chunk), int(c_idx // c_chunk))
                        )
                        break
                if sc_pairs:
                    break
            if len(sc_pairs) == 0:
                raise RuntimeError("All candidate (s,c) have zero LOSVD mass.")

        # For each (si,ci), pick templates from its finished P-tiles
        pj_done = done_arr  # alias
        per_pair_curves = []
        for s_idx, c_idx, si, ci in sc_pairs:
            pj_tiles = np.where(pj_done[si, ci, :])[0]
            p_cands = []
            for pj in pj_tiles:
                p0 = int(pj * p_chunk); p1 = int(min(P, p0 + p_chunk))
                p_cands.extend(range(p0, p1))
            if not p_cands:
                p_cands = list(range(P))
            rng.shuffle(p_cands)
            p_sel_pair = list(sorted(p_cands[:min(templates_per_pair, len(p_cands))]))
            per_pair_curves.append((s_idx, c_idx, p_sel_pair))
        bl_pairs = [(s_idx, c_idx) for (s_idx, c_idx, _) in per_pair_curves]

        # ---------------- figure ----------------
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # TL: observed spectra (diverse S-tiles)
        DC = f["/DataCube"]  # (S,L)
        Y_rows = [np.asarray(DC[s, :], dtype=float) for s in sp_sel]
        _set_ylim_from_unmasked(axs[0, 0], Y_rows, mask)
        for s, y in zip(sp_sel, Y_rows):
            axs[0, 0].plot(lam_obs, y, lw=1, label=f"s={int(s)}")
        axs[0, 0].set_title("Observed spectra (completed S-tiles)")
        axs[0, 0].set_xlabel("λ (observed grid)")
        axs[0, 0].set_ylabel("Flux")
        _shade_mask(axs[0, 0], lam_obs, mask)
        axs[0, 0].legend(fontsize=9)

        # TR: templates (rebinned, pre-convolution)
        _set_ylim_from_unmasked(axs[0, 1], [row for row in TR_mat], mask)
        for p, y in zip(p_sel_tr, TR_mat):
            axs[0, 1].plot(lam_obs, y, lw=1, label=f"p={int(p)}")
        axs[0, 1].set_title("Templates (rebinned, pre-convolution)")
        axs[0, 1].set_xlabel("λ (observed grid)")
        _shade_mask(axs[0, 1], lam_obs, mask)
        axs[0, 1].legend(fontsize=9)

        # BL: LOSVD (native) + resampled kernel (+ amplitude-scaled kernel on twin y)
        def _kernel_from_losvd(los_row, V):
            dlog = float(np.median(np.diff(tem_log)))
            k_min = int(np.floor(np.log1p(V.min() / C_KMS) / dlog))
            k_max = int(np.ceil (np.log1p(V.max() / C_KMS) / dlog))
            k_offsets = np.arange(k_min, k_max + 1, dtype=int)
            v_for_k  = C_KMS * np.expm1(k_offsets * dlog)
            il = np.searchsorted(V, v_for_k, side="right") - 1
            ir = il + 1
            oob = (il < 0) | (ir >= V.size)
            il = np.clip(il, 0, V.size - 1)
            ir = np.clip(ir, 0, V.size - 1)
            denom = (V[ir] - V[il]); denom[denom == 0.0] = 1.0
            t = (v_for_k - V[il]) / denom
            Hk = (1.0 - t) * los_row[il] + t * los_row[ir]
            Hk[oob] = 0.0
            s = Hk.sum()
            if s > 0:
                Hk /= s  # unit-area kernel for shape-only view
            return v_for_k, Hk

        los_handles = []
        LOS = f["/LOSVD"]  # (S,V,C)

        # twin y-axis for amplitude-scaled kernels (flux units)
        ax_los = axs[1, 0]
        ax_kflux = ax_los.twinx()
        ax_kflux.set_ylabel("kernel × scale(s,c)  (flux units)")

        # global norm mode (same for all components)
        norm_mode = str(f["/HyperCube"].attrs.get("norm.mode", "model")).lower()

        for (s_idx, c_idx) in bl_pairs:
            los = np.asarray(LOS[s_idx, :, c_idx], dtype=float)
            tot = los.sum()
            if tot > 0:
                los = los / tot  # unit-sum for native LOSVD display

            v_for_k, Hk = _kernel_from_losvd(los, vel_pix)
            # actual per-(s,c) scale used in /HyperCube/models
            scale_sc, mode_used, frac = _component_scale(f, int(s_idx), int(c_idx))

            # native LOSVD (solid)
            if norm_mode == "data":
                lbl = f"(s={int(s_idx)}, c={int(c_idx)})  frac={frac:.3e}"
            else:
                lbl = f"(s={int(s_idx)}, c={int(c_idx)})"
            line_los, = ax_los.plot(vel_pix, los, lw=1.5, label=lbl)
            los_handles.append(line_los)

            # unit-area resampled kernel (dashed) — same color as LOSVD
            ax_los.plot(v_for_k, Hk, lw=1.0, ls="--", color=line_los.get_color())

            # amplitude-scaled kernel on twin axis (dotted) — same color
            if scale_sc > 0.0:
                ax_kflux.plot(v_for_k, scale_sc * Hk, lw=1.0, ls=":",
                            alpha=0.9, color=line_los.get_color())

        ax_los.set_title(f"LOSVD (native) and resampled kernel  [mode={norm_mode}]")
        ax_los.set_xlabel("velocity (km/s)")
        ax_los.grid(alpha=0.25)

        # Style legend (explains line styles)
        lg_styles = ax_los.legend(
            handles=[
                Line2D([], [], lw=1.5, label="LOSVD (native)"),
                Line2D([], [], lw=1.0, ls="--", label="kernel (unit-area)"),
                Line2D([], [], lw=1.0, ls=":", label="kernel × scale(s,c)"),
            ],
            loc="lower right", fontsize=8, frameon=False
        )
        ax_los.add_artist(lg_styles)

        # Per-curve legend showing which (spaxel, component) each LOSVD is
        ax_los.legend(
            handles=los_handles, loc="upper left",
            fontsize=7, frameon=False, ncol=2, title="(s, c)"
        )

        # BR: ACTUAL /HyperCube/models for diverse (s,c) pairs
        br_all_rows = []
        for pair_id, (s_idx, c_idx, p_list) in enumerate(per_pair_curves, start=1):
            for p in p_list:
                y = np.asarray(M[s_idx, c_idx, int(p), :], dtype=float)
                br_all_rows.append(y)
        _set_ylim_from_unmasked(axs[1, 1], br_all_rows, mask)
        for pair_id, (s_idx, c_idx, p_list) in enumerate(per_pair_curves, start=1):
            for p in p_list:
                y = np.asarray(M[s_idx, c_idx, int(p), :], dtype=float)
                axs[1, 1].plot(
                    lam_obs, y, lw=1,
                    label=f"p={int(p)} @(s={s_idx}, c={c_idx})"
                )
        axs[1, 1].set_title("Post-convolution (ACTUAL /HyperCube/models)")
        axs[1, 1].set_xlabel("λ (observed grid)")
        _shade_mask(axs[1, 1], lam_obs, mask)
        axs[1, 1].legend(fontsize=8)

        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png
