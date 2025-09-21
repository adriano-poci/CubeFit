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
                                 rng_seed: int = 12345) -> tuple[np.ndarray, dict]:
    """
    Very fast diagonal/Jacobi initializer on a small sample:
      Accumulate d = diag(A^T A), b = A^T y over Ns spaxels × K pixels.
      x0 = clip(b / (d + eps), 0).
    """
    rng = np.random.default_rng(rng_seed)
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]     # (S,C,P,L)
        Data = f["/DataCube"]          # (S,L)
        Mask = f["/Mask"] if "/Mask" in f else None

        S, C, P, L = map(int, M.shape)
        CP = C * P
        d = np.zeros(CP, dtype=np.float64)
        b = np.zeros(CP, dtype=np.float64)

        # stratified spaxel sample
        if Ns >= S:
            spaxels = np.arange(S, dtype=int)
        else:
            base = np.linspace(0, S - 1, Ns, dtype=int)
            jitter = rng.integers(0, max(1, S // max(4, Ns)), size=Ns)
            spaxels = np.unique(np.clip(base + jitter, 0, S - 1))

        # choose pixel pool once (mask-aware)
        if Mask is not None:
            mask = np.asarray(Mask[...], bool).ravel()
            pool = np.flatnonzero(mask)
            if pool.size == 0:
                pool = np.arange(L, dtype=int)
        else:
            pool = np.arange(L, dtype=int)

        for s in spaxels:
            if K < pool.size:
                ell = rng.choice(pool, size=K, replace=False)
            else:
                ell = pool
            y = np.asarray(Data[s, ell], np.float64)                  # (K,)
            # batch read K columns -> (C,P,K)
            A = np.empty((C, P, ell.size), dtype=np.float32)
            for j, k in enumerate(ell):
                A[:, :, j] = M[s, :, :, int(k)]
            A = np.asarray(A, dtype=np.float64)                       # promote once
            Ak = A.reshape(CP, ell.size)                              # (CP, K)
            d += np.einsum("ik,ik->i", Ak, Ak, optimize=True)         # diag(AtA)
            b += Ak @ y                                               # At y

        eps = 1e-12
        x0 = np.maximum(0.0, b / (d + eps))
        stats = {"Ns": int(spaxels.size), "K": int(min(K, pool.size))}
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

        # 1) Tracker: use your FitTracker as-is
        
        use_tracker = (tracker_mode != "off")
        tracker = NullTracker()
        if use_tracker:
            try:
                tracker_cfg = TrackerConfig()
                # or your existing cfg construction
                tracker = FitTracker(self.h5_path, cfg=tracker_cfg)
            # uses and writes /Fit/*  :contentReference[oaicite:2]{index=2}
            except Exception as e:
                print(f"[Pipeline] Tracker disabled (start failed: {e})")
                tracker = NullTracker()
        
        # 2) Build the reader as you currently do
        rcfg = ReaderCfg(s_tile=reader_s_tile, c_tile=reader_c_tile, p_tile=reader_p_tile)
        reader = HyperCubeReader(self.h5_path, cfg=rcfg)

        # 3) Solver config
        cfg = SolverCfg(epochs=epochs,
                        pixels_per_aperture=pixels_per_aperture,
                        lr=lr,
                        project_nonneg=project_nonneg,
                        row_order=row_order,
                        blas_threads=blas_threads,
                        verbose=verbose,
                        seed=seed)

        if orbit_weights is not None:
            try:
                print('[Pipeline] tracker.set_orbit_weights...')
                tracker.set_orbit_weights(orbit_weights)
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] tracker.set_orbit_weights warning: {e}")

        # 4) Warm-start: user x0 overrides; else resume; else smart seed
        def _build_jacobi_seed(N, *, Ns=16, K=256, seed=12345):
            # Minimal streaming Jacobi using the reader (masked, f64-safe)
            rng = np.random.default_rng(seed)
            C, P = int(self.nComp), int(self.nPop)
            N = C * P
            d = np.zeros(N, dtype=np.float64)  # diag A^T A approx
            b = np.zeros(N, dtype=np.float64)  # A^T y approx
            ns = min(Ns, self.nSpat)
            # stratified spaxel sample
            spax = np.linspace(0, self.nSpat - 1, ns, dtype=int)
            jitter = rng.integers(0, max(1, self.nSpat // max(4, ns)), size=ns)
            spax = np.unique(np.clip(spax + jitter, 0, self.nSpat - 1))
            for s in spax:
                A_f32, y = reader.read_spaxel_plane(s)
                L_eff = int(A_f32.shape[1])
                if L_eff == 0: continue
                kk = min(K, L_eff)
                cols = rng.choice(L_eff, size=kk, replace=False)
                for l in cols:
                    a = np.asarray(A_f32[:, l], dtype=np.float64)
                    d += a * a
                    b += a * float(y[l])
            x_seed = np.divide(b, d + 1e-18, out=np.zeros_like(b), where=(d > 0))
            np.maximum(x_seed, 0.0, out=x_seed)
            return x_seed

        # Gather resume candidates
        x_best = None; x_last = None
        with open_h5(self.h5_path, role="reader") as f_ro:
            if "/Fit/x_best" in f_ro:
                x_best = np.asarray(f_ro["/Fit/x_best"][...], dtype=np.float64)
            if "/X_global" in f_ro:
                x_last = np.asarray(f_ro["/X_global"][...], dtype=np.float64)

        N = int(self.nComp) * int(self.nPop)
        if x0 is not None:
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            if x0.size != N:
                raise ValueError(f"x0.size {x0.size} != N {N}")

        # Decide
        wmode = (warm_start or "auto").lower()
        if wmode == "jacobi":
            x0_effective = _build_jacobi_seed(N, **(seed_cfg or {}))
        elif wmode == "zeros":
            x0_effective = np.zeros(N, dtype=np.float64)
        elif wmode == "x0":
            if x0 is None:
                raise ValueError("warm_start='x0' but no x0 provided")
            x0_effective = x0
        elif wmode == "resume":
            x0_effective = x_best if x_best is not None else (x_last if x_last is not None else np.zeros(N))
        else:  # 'auto'
            x0_effective = (
                x0 if x0 is not None else
                (x_best if x_best is not None else
                (x_last if x_last is not None else _build_jacobi_seed(N, **(seed_cfg or {}))))
            )

        if verbose:
            msg = ("fresh Jacobi" if wmode=="jacobi" else
                "zeros" if wmode=="zeros" else
                "explicit x0" if wmode=="x0" else
                "resume" if wmode=="resume" else
                "auto")
            print(f"[Pipeline] Warm-start mode: {msg} "
                f"(seed used: {'yes' if (wmode in ('jacobi','auto') and x0_effective is not x0 and x0_effective is not x_best and x0_effective is not x_last) else 'no'})")
            # Let dashboard see the initial seed immediately
            try:
                print('[Pipeline] tracker initial maybe_save...')
                tracker.maybe_save(x0_effective, epoch=0, now=time.time())
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] tracker initial maybe_save warning: {e}")

        # 5) Wire callbacks for the solver
        def _on_progress(x_vec: np.ndarray, epoch_no: int, stats_epoch: dict):
            # Single call that lets tracker decide when/what to write; it's time-gated internally
            try:
                tracker.maybe_save(x_vec, epoch=epoch_no, now=time.time())
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] tracker.maybe_save warning: {e}")
        def _on_batch_rmse(rmse: float):
            # EWMA of training RMSE; cheap scalar update
            try:
                tracker.on_batch(float(rmse))
            except Exception:
                pass

        # Choose progress cadence ~ tracker metrics interval so we don't over-call
        progress_interval = float(getattr(tracker_cfg, "metrics_interval_sec", 300))


        # 6) Run the solver with hooks
        try:
            x, stats = solve_global_kaczmarz(reader, cfg=cfg,
                orbit_weights=orbit_weights, x0=x0_effective,
                on_progress=_on_progress,
                progress_interval_sec=progress_interval,
                on_batch_rmse=_on_batch_rmse)
        finally:
            reader.close()
        
        # 7) Write final vector to canonical place for downstream code
        try:
            with open_h5(self.h5_path, role="writer") as f:
                if "/X_global" in f: del f["/X_global"]
                dset = f.create_dataset("/X_global", data=np.asarray(x, np.float64), dtype="f8",
                                        chunks=(min(x.size, 1<<14),))
                dset.attrs["n"] = int(x.size)
                # optional: tag provenance
                dset.attrs["init"] = np.string_("resume" if x0_effective is not None else "jacobi_seed")
                if seed_cfg:
                    dset.attrs["seed_cfg"] = np.string_(str(seed_cfg))
        except Exception as e:
            if verbose:
                print(f"[Pipeline] Warning: could not write /X_global: {e}")

        return x, stats