# -*- coding: utf-8 -*-
r"""
    pipeline_runner.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    High-level CubeFit pipeline orchestration: runs per-aperture or global
    Kaczmarz NNLS fits, manages Zarr storage, provides reference and diagnostic
    NNLS fits, and supports continuum/velocity expansion and plotting.

    Notes
    -----
    * Uses the robust HDF5 open helper `open_h5(...)` everywhere to avoid
      SWMR/locking issues and double-opens.
    * No API changes: calls from kz_fitSpec remain the same.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Initial pipeline design for CubeFit. 2025
v1.1:   Added global (full-cube) Kaczmarz and block constraint support. 2025
v1.2:   Supports continuum, velocity-shift, and reference fits. 2025
v1.3:   Full workflow Zarr integration and flexible test sub-selection. 2025
v1.4:   Complete re-write to use HDF5. 7 September 2025
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import json, time, math
import numpy as np

from CubeFit.hdf5_manager import H5Manager, H5Dims, open_h5
from CubeFit.hypercube_builder import build_hypercube
from CubeFit.hypercube_reader import HyperCubeReader, ReaderCfg
from CubeFit.kaczmarz_solver import solve_global_kaczmarz, SolverCfg
from CubeFit.fit_tracker import FitTracker, NullTracker, TrackerConfig
from CubeFit.logger import get_logger

logger = get_logger()

def _load_resume_vector(h5_path: str) -> np.ndarray | None:
    """Resume priority: /Fit/x_best > /X_global > None."""
    try:
        with open_h5(h5_path, role="reader") as f:
            if "/Fit/x_best" in f:
                x = f["/Fit/x_best"][...]
                if x.size:
                    print(f"[Pipeline] Warm-start from /Fit/x_best (n={x.size}).")
                    return np.asarray(x, np.float64, order="C")
            if "/X_global" in f:
                x = f["/X_global"][...]
                if x.size:
                    print(f"[Pipeline] Warm-start from /X_global (n={x.size}).")
                    return np.asarray(x, np.float64, order="C")
    except Exception as e:
        print(f"[Pipeline] Resume probe failed: {e}")
    return None

def _build_streaming_jacobi_seed(h5_path: str,
                                 Ns: int = 16,
                                 K: int = 256,
                                 rng_seed: int = 12345,
                                 max_bytes: int = 2_000_000_000
                                 ) -> tuple[np.ndarray, dict]:
    """
    Chunk-friendly Jacobi initializer.

    Reads a contiguous spaxel tile that lies within one S-chunk of the
    /HyperCube/models dataset, loads full-L once per tile (L_chunk == L),
    then samples K pixel columns *in RAM* per spaxel to accumulate:

        d = diag(A^T A)   in float64 (no full f64 copy of A)
        b = A^T y         with y in float64

    Returns:
        x0  : (C*P,) float64  non-negative seed clip(b / (d + eps), 0)
        stats: dict with {'Ns_used', 'K_used', 's0', 's1', 'S_chunk'}
    """
    rng = np.random.default_rng(rng_seed)
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]   # (S,C,P,L) float32
        Y = f["/DataCube"]           # (S,L)     float64
        Mask = f["/Mask"] if "/Mask" in f else None

        S, C, P, L = map(int, M.shape)
        CP = C * P
        chunks = M.chunks or (S, 1, P, L)
        S_chunk = int(chunks[0])     # tile along S to reuse cached chunks

        # memory budget: one slab (S_eff,C,P,L) float32
        bytes_per_spax = C * P * L * 4
        S_cap = max(1, int(max_bytes // max(1, bytes_per_spax)))

        # choose a tile fully inside one S_chunk
        s0 = int(rng.integers(0, max(1, S // S_chunk)) * S_chunk)
        s1 = min(S, s0 + S_chunk)
        S_eff = min(Ns, S_chunk, S_cap, s1 - s0)
        s1 = s0 + S_eff

        # read once per tile
        slab32 = M[s0:s1, :, :, :][...]          # (S_eff,C,P,L) f32
        Y_tile = np.asarray(Y[s0:s1, :], np.float64)  # (S_eff,L)   f64

        # pixel pool (mask-aware)
        if Mask is not None:
            pool = np.flatnonzero(np.asarray(Mask[...], bool).ravel())
            if pool.size == 0:
                pool = np.arange(L, dtype=int)
        else:
            pool = np.arange(L, dtype=int)

        d = np.zeros(CP, dtype=np.float64)
        b = np.zeros(CP, dtype=np.float64)

        for i in range(S_eff):
            mK = int(min(K, pool.size))
            # h5py fancy indexing requires sorted indices; we also use them
            # consistently for both A and y.
            ell = rng.choice(pool, size=mK, replace=False)
            ell_sorted = np.sort(ell)

            # select columns in RAM; keep A in float32 for products,
            # promote only the *result* to float64 via dtype=...
            Aik = slab32[i, :, :, ell_sorted]         # (C,P,mK) f32
            Ak = Aik.reshape(CP, mK)                  # (CP,mK) f32
            yk = Y_tile[i, ell_sorted]                # (mK,)  f64

            # diag(AtA): float64 accumulation without upcasting Ak itself
            d += np.einsum("ik,ik->i", Ak, Ak, dtype=np.float64, optimize=True)
            # At y: float32@float64 -> float64 result
            b += Ak @ yk

        eps = 1e-12
        x0 = np.maximum(0.0, b / (d + eps))
        stats = {
            "Ns_used": int(S_eff),
            "K_used": int(min(K, pool.size)),
            "s0": int(s0),
            "s1": int(s1),
            "S_chunk": int(S_chunk),
        }
        return x0, stats

class PipelineRunner:
    """
    Orchestrates HyperCube build & global Kaczmarz fitting from an HDF5 file.

    You can construct this class **before or after** building the HyperCube:
      - If /HyperCube/models is missing, we still read dimensions from LOSVD,
        DataCube, and Templates and you can call `build_hypercube(...)`.
      - If models exist, you can go straight to `solve_all(...)`.

    Parameters
    ----------
    h5_path : str | pathlib.Path
        Path to the HDF5 file that holds:
          * /LOSVD            (nSpat, nVel, nComp)            [required]
          * /DataCube         (nSpat, nLSpec)                 [required]
          * /Templates or /TemplatesFFT(_R) (nPop, N/TSpec)   [required to build]
          * /HyperCube/models (nSpat, nComp, nPop, nLSpec)    [required to solve]
    *_, **__ :
        Extra positional/keyword args are accepted and ignored for
        backward-compatibility with older call sites.
    """
    def __init__(self, h5_path: str | Path):
        self.h5_path = str(h5_path)
        with open_h5(h5_path, "reader") as f:
            # Read dims in a robust way (JSON or scalars); fallback to dataset shapes
            dims = {}
            # Prefer root JSON
            if "/".encode() == b"/":  # no-op, just to avoid lint; real work below
                pass
            if "dims_json" in f["/"].attrs:
                val = f["/"].attrs["dims_json"]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                try:
                    dims = {k: int(v) for k, v in json.loads(val).items()}
                except Exception:
                    dims = {}
            # Try scalar attrs if JSON missing/bad
            if not dims:
                for k in ("nSpat", "nLSpec", "nTSpec", "nVel", "nComp", "nPop"):
                    key = f"dims.{k}"
                    if key in f["/"].attrs:
                        dims[k] = int(f["/"].attrs[key])

            # Fallbacks from datasets
            self.nSpat = int(dims.get("nSpat", f["/DataCube"].shape[0]))
            self.nLSpec = int(dims.get("nLSpec", f["/DataCube"].shape[1]))
            self.nComp  = int(dims.get("nComp",  f["/LOSVD"].shape[2]))
            self.nPop   = int(dims.get("nPop",   f["/Templates"].shape[0]))
            self.nVel   = int(dims.get("nVel",   f["/LOSVD"].shape[1]))
            self.nTSpec = int(dims.get("nTSpec", f["/Templates"].shape[1])) if "/Templates" in f else None
            self.has_mask = ("/Mask" in f)
            self.has_models = ("/HyperCube/models" in f)
            self.complete = bool(f["/HyperCube"].attrs.get("complete", False)) if "/HyperCube" in f else False

        logger.log(
            "[Pipeline] Initialized from HDF5: "
            f"S={self.nSpat}, C={self.nComp}, P={self.nPop}, L={self.nLSpec}, "
            f"V={self.nVel}, T={self.nTSpec if self.nTSpec is not None else 'NA'}; "
            f"mask={'yes' if self.has_mask else 'no'}; models={'yes' if self.has_models else 'no'}; "
            f"complete={self.complete}"
        )

        # Legacy fields some code may look for
        self.zarr_path = None
        self.zarr_store = None
        self.models_path = self.h5_path

    def build_inputs(self) -> None:
        H5Manager(self.h5_path).init_base(H5Dims(
            nSpat=self.nSpat, nLSpec=self.nLSpec, nTSpec=int(self.nTSpec or 0),
            nVel=self.nVel, nComp=self.nComp, nPop=self.nPop
        ))

    def build_hypercube(self, *, S=16, C=1, P=256, galaxy=None, check="auto", extra_manifest=None) -> None:
        nS, nC, nP = 128, 1, 360
        build_hypercube(
            self.h5_path, S_chunk=nS, C_chunk=nC, P_chunk=nP,
        )

        with open_h5(self.h5_path, "reader") as f:
            self.has_models = ("/HyperCube/models" in f)
            self.complete = bool(f["/HyperCube"].attrs.get("complete", False))

    def solve_all(self,
                epochs=1, pixels_per_aperture=256, lr=0.25,
                project_nonneg=True, row_order="random",
                reader_s_tile=None, reader_c_tile=None, reader_p_tile=None,
                blas_threads=None, orbit_weights=None,
                x0=None, verbose=True, seed=None,
                warm_start: str = "auto",      # 'auto'|'resume'|'jacobi'|'zeros'|'x0'
                seed_cfg: dict | None = None,
                tracker_mode: str = "async"):  # 'async'|'off'

        # ---------------- tracker (optional) ----------------
        use_tracker = (str(tracker_mode).lower() != "off")
        tracker = NullTracker()
        if use_tracker:
            try:
                logger.log('[Pipeline] Starting tracker...', flush=True)
                tracker_cfg = TrackerConfig()
                tracker = FitTracker(self.h5_path, cfg=tracker_cfg)
                logger.log('[Pipeline] Tracker started.', flush=True)
            except Exception as e:
                print(f"[Pipeline] Tracker disabled (start failed: {e})",
                    flush=True)
                tracker = NullTracker()

        # ---------------- reader ----------------
        rcfg = ReaderCfg(s_tile=reader_s_tile, c_tile=reader_c_tile, p_tile=reader_p_tile)
        reader = HyperCubeReader(self.h5_path, cfg=rcfg)

        # ---------------- solver cfg ----------------
        cfg = SolverCfg(epochs=epochs,
                        pixels_per_aperture=pixels_per_aperture,
                        lr=lr,
                        project_nonneg=project_nonneg,
                        row_order=row_order,
                        blas_threads=blas_threads,
                        verbose=verbose,
                        seed=seed)

        # ---------------- warm-start ----------------
        x_best = x_last = None
        if warm_start in ("auto", "resume"):
            try:
                with open_h5(self.h5_path, role="reader") as f:
                    if "/Fit/x_best" in f:
                        tmp = f["/Fit/x_best"][...]
                        if tmp.size: x_best = np.asarray(tmp, np.float64, order="C")
                    if "/X_global" in f:
                        tmp = f["/X_global"][...]
                        if tmp.size: x_last = np.asarray(tmp, np.float64, order="C")
            except Exception as e:
                if verbose: print(f"[Pipeline] Resume probe failed: {e}")

        wmode = warm_start
        x0_effective = None
        if warm_start == "x0" and x0 is not None:
            x0_effective = np.asarray(x0, np.float64, order="C")
        elif warm_start == "resume" and (x_best is not None or x_last is not None):
            x0_effective = x_best if x_best is not None else x_last
        elif warm_start == "jacobi":
            x0_effective, _ = _build_streaming_jacobi_seed(self.h5_path,
                                                        **(seed_cfg or {}))
        elif warm_start == "zeros":
            x0_effective = np.zeros(reader.nComp * reader.nPop, dtype=np.float64)
        else:  # auto
            if x_best is not None or x_last is not None:
                wmode = "resume"
                x0_effective = x_best if x_best is not None else x_last
            else:
                wmode = "jacobi"
                x0_effective, _ = _build_streaming_jacobi_seed(self.h5_path,
                                                            **(seed_cfg or {}))

        if verbose:
            msg = ("fresh Jacobi" if wmode=="jacobi" else
                "zeros" if wmode=="zeros" else
                "explicit x0" if wmode=="x0" else
                "resume" if wmode=="resume" else "auto")
            logger.log(f"[Pipeline] Warm-start mode: {msg}", flush=True)
            try:
                tracker.maybe_save(x0_effective, epoch=0, now=time.time())
            except Exception:
                pass

        # ---------------- callbacks ----------------
        def _on_progress(x_vec: np.ndarray, epoch_no: int, stats_epoch: dict):
            try:
                tracker.maybe_save(x_vec, epoch=epoch_no, now=time.time())
            except Exception:
                pass

        def _on_batch_rmse(rmse: float):
            try:
                tracker.on_batch(float(rmse))
            except Exception:
                pass

        progress_interval = 300.0

        # ---------------- run solver ----------------
        try:
            logger.log('[Pipeline] Starting Kaczmarz solver.', flush=True)
            x, stats = solve_global_kaczmarz(
                reader, cfg=cfg,
                orbit_weights=orbit_weights,
                x0=x0_effective,
                on_progress=_on_progress,
                progress_interval_sec=progress_interval,
                on_batch_rmse=_on_batch_rmse)
        finally:
            reader.close()

        # ---------------- persist final vector ----------------
        try:
            with open_h5(self.h5_path, role="writer") as f:
                if "/X_global" in f: del f["/X_global"]
                dset = f.create_dataset("/X_global",
                    data=np.asarray(x, np.float64),
                    dtype="f8", chunks=(min(x.size, 1<<14),))
                dset.attrs["n"] = int(x.size)
                dset.attrs["init"] = np.bytes_(wmode)
        except Exception as e:
            if verbose:
                print(f"[Pipeline] Warning: could not write /X_global: {e}")

        try:
            tracker.close()
        except Exception:
            pass

        return x, stats
