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
v1.5:   Wrap `solve_global_kaczmarz_cchunk_mp` in `logger.capture_all_output`
            in `PipelineRunner.solve_all_mp_batched`. 4 December 2025
v1.6:   Updated `solve_all_mp_batched` `warm_start` options to match `solve_all`.
            12 December 2025
v1.7:   Read in NNLS `L2` ridge from environment variable in
            `solve_all_mp_batched`. 13 December 2025
v1.8:   Implemented mtime-based decision for sidecar vs main file loading in
            `solve_all*` resume logic. 18 December 2025
v1.9:   Added final elapsed time logging in `PipelineRunner.solve_all_mp_batched`. 1 January
            2026
v1.10:  Implemented SPG to Kaczmarz workflow. 7 January 2026
v1.11:  Store `/X_global` as 2D array always. 10 January 2026
v1.12:  Corrected Kaczmarz polish logic in `PipelineRunner.solve_all_mp_batched`.
            11 January 2026
v1.13:  Universally removed all ad-hoc scalings;
        Scale the NNLS ridge to the orbit prior strength in
            `PipelineRunner.solve_all_mp_batched`. 25 January 2026
v1.14:  Use renamed module. 27 January 2026
v1.15:  Persist `known_zero_mask` after SPG in `solve_all_mp_batched`. 3 February
            2026
v1.16:  Fixed bug in `PipelineRunner._read_latest_from_main` which incorrectly
            sliced read-in solutions assuming they were history-like. 17
            February 2026
v1.17:  Removed lingering `orbit_beta`. 23 March 2026
v1.18:  Re-implemented `orbit_beta` support in `solve_all_mp_batched` and passed
            it to the solver. 30 March 2026
v1.19:  Resolve `orbit_prior_delta`. 27 July 2026
v1.20:  Use `cube_utils.vprint` for diagnostic prints. 31 July 2026
v1.21:  Removed legacy `jacobi` warm-start option;
        Removed redundant single-core solve pathway from `PipelineRunner`;
        Removed legacy keywords in `solve_all_mp_batched`. 4 August 2026
"""

from __future__ import annotations
import pathlib as plp
from typing import Optional, Tuple
import json, time, math, os
import numpy as np
from dataclasses import dataclass

from CubeFit.hdf5_manager import H5Manager, H5Dims, open_h5
from CubeFit.hypercube_builder import build_hypercube
from CubeFit.hypercube_reader import HyperCubeReader, ReaderCfg
# from CubeFit.streaming_nnls import (
    # MPConfig, solve_streaming_nnls)
from CubeFit.streaming_nnls_augmented_rows import (
    MPConfig, solve_streaming_nnls)
from CubeFit.live_fit_dashboard import (
    render_aperture_fits_with_x, render_sfh_from_x, alpha_star_stats
)
from CubeFit.fit_tracker import FitTracker, NullTracker, TrackerConfig
import CubeFit.cube_utils as cu
from CubeFit.cube_utils import RatioCfg
from CubeFit.logger import get_logger

logger = get_logger()
vprint = cu.vprint

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

class PipelineRunner:
    """
    Orchestrates HyperCube build & global Kaczmarz fitting from an HDF5
    file.

    You can construct this class **before or after** building the
    HyperCube:
      - If /HyperCube/models is missing, we still read dimensions from
        LOSVD, DataCube, and Templates and you can call
        `build_hypercube(...)`.
      - If models exist, you can go straight to `solve_all(...)`.

    Parameters
    ----------
    h5_path : str | pathlib.Path
        Path to the HDF5 file that holds:
          * /LOSVD            (nSpat, nVel, nComp)            [required]
          * /DataCube         (nSpat, nLSpec)                 [required]
          * /Templates or /TemplatesFFT(_R) (nPop, N/TSpec)   [required]
            to build
          * /HyperCube/models (nSpat, nComp, nPop, nLSpec)    [required]
            to solve
    *_, **__ :
        Extra positional/keyword args are accepted and ignored for
        backward-compatibility with older call sites.
    """
    def __init__(self, h5_path: str | plp.Path):
        self.h5_path = str(h5_path)
        with open_h5(h5_path, "reader") as f:
            dims = {}
            if "/".encode() == b"/":
                pass
            if "dims_json" in f["/"].attrs:
                val = f["/"].attrs["dims_json"]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                try:
                    dims = {k: int(v) for k, v in json.loads(val).items()}
                except Exception:
                    dims = {}
            if not dims:
                for k in ("nSpat", "nLSpec", "nTSpec", "nVel", "nComp",
                          "nPop"):
                    key = f"dims.{k}"
                    if key in f["/"].attrs:
                        dims[k] = int(f["/"].attrs[key])

            self.nSpat = int(dims.get("nSpat", f["/DataCube"].shape[0]))
            self.nLSpec = int(dims.get("nLSpec", f["/DataCube"].shape[1]))
            self.nComp  = int(dims.get("nComp",  f["/LOSVD"].shape[2]))
            self.nPop   = int(dims.get("nPop",   f["/Templates"].shape[0]))
            self.nVel   = int(dims.get("nVel",   f["/LOSVD"].shape[1]))
            self.nTSpec = int(dims.get("nTSpec",
                f["/Templates"].shape[1])) if "/Templates" in f else None
            self.has_mask = ("/Mask" in f)
            self.has_models = ("/HyperCube/models" in f)
            self.complete = bool(f["/HyperCube"].attrs.get(
                "complete", False)) if "/HyperCube" in f else False

        logger.log(
            "[Pipeline] Initialized from HDF5: "
            f"S={self.nSpat}, C={self.nComp}, P={self.nPop}, L={self.nLSpec}, "
            f"V={self.nVel}, T={self.nTSpec if self.nTSpec is not None else 'NA'}; "
            f"mask={'yes' if self.has_mask else 'no'}; "
            f"models={'yes' if self.has_models else 'no'}; "
            f"complete={self.complete}"
        )

        self.zarr_path = None
        self.zarr_store = None
        self.models_path = self.h5_path

    def build_inputs(self) -> None:
        H5Manager(self.h5_path).init_base(H5Dims(
            nSpat=self.nSpat, nLSpec=self.nLSpec, nTSpec=int(self.nTSpec or 0),
            nVel=self.nVel, nComp=self.nComp, nPop=self.nPop
        ))

    def build_hypercube(self, *, S=16, C=1, P=256, galaxy=None, check="auto",
                        extra_manifest=None) -> None:
        nS, nC, nP = 128, 1, 360
        build_hypercube(
            self.h5_path, S_chunk=nS, C_chunk=nC, P_chunk=nP,
        )

        with open_h5(self.h5_path, "reader") as f:
            self.has_models = ("/HyperCube/models" in f)
            self.complete = bool(f["/HyperCube"].attrs.get("complete", False))

    @staticmethod
    def _read_latest_from_sidecar(sidecar_path: str, N_expected: int):
        """
        Read the most recent solution vector from a FitTracker sidecar file.

        For *resume semantics* we prefer the latest checkpoint, not the
        best-so-far checkpoint. So we try, in order:

        1) /Fit/x_last
        2) /Fit/x_epoch_last
        3) /Fit/x_snapshots[-1]
        4) /Fit/x_best
        5) /Fit/x_hist[-1] (legacy)

        Parameters
        ----------
        sidecar_path : str
            Path to the sidecar HDF5 file: <main>.fit.<pid>.<ts>.h5
        N_expected : int
            Expected flattened size (C*P).

        Returns
        -------
        x : ndarray[float64] or None
            Flattened solution vector of length N_expected, if found.
        src : str or None
            Dataset label used to load x (for logging).
        """
        if (sidecar_path is None) or (not os.path.exists(sidecar_path)):
            return None, None

        def _read_flat(g, name: str):
            if name not in g:
                return None
            ds = g[name]
            try:
                if ds.ndim == 2 and ds.shape[0] > 0:
                    v = np.asarray(ds[-1, :], np.float64, order="C")
                else:
                    v = np.asarray(ds[...], np.float64, order="C")
            except Exception:
                return None

            v = np.asarray(v, np.float64).ravel(order="C")
            if v.size != int(N_expected):
                return None
            return v

        with open_h5(sidecar_path, role="reader", swmr=True) as g:
            # Prefer latest progress for resume.
            v = _read_flat(g, "/Fit/x_last")
            if v is not None:
                return v, "/Fit/x_last"

            v = _read_flat(g, "/Fit/x_epoch_last")
            if v is not None:
                return v, "/Fit/x_epoch_last"

            if "/Fit/x_snapshots" in g and g["/Fit/x_snapshots"].shape[0] > 0:
                ds = g["/Fit/x_snapshots"]
                try:
                    v = np.asarray(ds[-1, :], np.float64, order="C")
                    v = v.ravel(order="C")
                    if v.size == int(N_expected):
                        return v, "/Fit/x_snapshots[-1]"
                except Exception:
                    pass

            # Fallback to best-so-far.
            v = _read_flat(g, "/Fit/x_best")
            if v is not None:
                return v, "/Fit/x_best"

            # Legacy fallback.
            if "/Fit/x_hist" in g and g["/Fit/x_hist"].shape[0] > 0:
                ds = g["/Fit/x_hist"]
                try:
                    v = np.asarray(ds[-1, :], np.float64, order="C")
                    v = v.ravel(order="C")
                    if v.size == int(N_expected):
                        return v, "/Fit/x_hist[-1]"
                except Exception:
                    pass

        return None, None

    @staticmethod
    def _read_latest_from_main(h5_path: str, N_expected: int):
        """
        Read the best available solution vector from the *main* HDF5.

        Priority is "most resume-correct" first:
        1) /Fit/x_last
        2) /Fit/x_epoch_last
        3) /X_global
        4) /Fit/x_best
        5) legacy fallbacks

        Returns
        -------
        x : ndarray[float64] or None
            Flattened solution vector (length N_expected), if found.
        src : str or None
            Dataset label used to load x (for logging).
        """
        if (h5_path is None) or (not os.path.exists(h5_path)):
            return None, None

        def _read_flat(f, name: str):
            if name not in f:
                return None
            ds = f[name]
            try:
                if name.startswith("/Fit/") and ds.ndim == 2 and \
                    ds.shape[0] > 0:
                    # history-like, per epoch
                    v = np.asarray(ds[-1, :], np.float64, order="C")
                else:
                    # canonical solution vector
                    v = np.asarray(ds[...], np.float64, order="C")
            except Exception:
                return None

            v = np.asarray(v, np.float64).ravel(order="C")
            if v.size != int(N_expected):
                return None
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            return v

        with open_h5(h5_path, role="reader", swmr=True) as f:
            # Resume-correct (if present)
            v = _read_flat(f, "/Fit/x_last")
            if v is not None:
                return v, "/Fit/x_last"

            v = _read_flat(f, "/Fit/x_epoch_last")
            if v is not None:
                return v, "/Fit/x_epoch_last"

            # Canonical committed solution
            v = _read_flat(f, "/X_global")
            if v is not None:
                return v, "/X_global"

            # Best-so-far fallback (if you keep it in main)
            v = _read_flat(f, "/Fit/x_best")
            if v is not None:
                return v, "/Fit/x_best"

            # Legacy candidates (only if they exist in your file)
            for name in ("/X_best", "/X_last", "/Fit/x_hist"):
                v = _read_flat(f, name)
                if v is not None:
                    return v, name

        return None, None

    @staticmethod
    def _read_solver_state(sidecar_path: str | None) -> dict:
        """
        Read the JSON-serialized streaming NNLS resume state from a sidecar.
        """
        if (sidecar_path is None) or (not os.path.exists(sidecar_path)):
            return {}

        try:
            with open_h5(sidecar_path, role="reader", swmr=True) as f:
                fit = f.get("/Fit", None)
                if fit is None:
                    return {}

                raw = fit.attrs.get("solver_state_json", None)
                if raw is None:
                    return {}

                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")

                state = json.loads(raw)
                return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def _read_seed_from_h5(self,
                        h5_path: str,
                        N_expected: int,
                        dset: str = "/Seeds/x0_nnls_patch",
                        project_nonneg: bool = True) \
                        -> tuple[np.ndarray | None, str | None]:
        """
        Read a seed solution from the main HDF5 file.
        Accepts either flat (N_expected,) or 2-D (C,P) and flattens
        C-order. If the size mismatches, trims or zero-pads with a
        warning.

        Returns
        -------
        (x0, src_label) or (None, None)
        """
        try:
            with open_h5(h5_path, role="reader") as f:
                if dset not in f:
                    return None, None
                arr = np.asarray(f[dset][...], dtype=np.float64, order="C")
                if arr.ndim == 2:
                    arr = arr.reshape(-1, order="C")
                x0 = arr.ravel(order="C")

                if x0.size != N_expected:
                    import warnings
                    warnings.warn(
                        f"[Pipeline] Seed at {dset} has length {x0.size} "
                        f"!= expected {N_expected}; "
                        f"{'trimming' if x0.size > N_expected else 'zero-padding'} "
                        f"to match.",
                        RuntimeWarning
                    )
                    if x0.size > N_expected:
                        x0 = x0[:N_expected].copy()
                    else:
                        tmp = np.zeros(N_expected, dtype=np.float64)
                        tmp[:x0.size] = x0
                        x0 = tmp

                if project_nonneg:
                    np.maximum(x0, 0.0, out=x0)

                return x0, f"{dset} (main)"
        except Exception:
            return None, None

    # ------------------------- Solve (multi-process) -------------------

    def solve_all_mp_batched(
        self,
        reader_s_tile=128,
        reader_c_tile=1,
        reader_p_tile=360,
        reader_dtype_models="float32",
        reader_apply_mask=True,
        processes=2,
        blas_threads=12,
        orbit_weights=None,
        x0=None,
        warm_start="zeros",  # default to the new seed
        tracker_mode="on",
    ):

        # --------------- Warm-start (same policy as SP path) -----------
        N_expected = int(self.nComp * self.nPop)

        if x0 is not None:
            x0_effective = np.asarray(x0, dtype=np.float64, order="C")

        elif warm_start == "seed":
            path = os.environ.get("CUBEFIT_SEED_PATH", "/Seeds/x0_nnls_patch")
            x_seed, src_seed = self._read_seed_from_h5(self.h5_path,
                N_expected, dset=path)
            if x_seed is not None:
                x0_effective = x_seed
                vprint(f"[Pipeline] Warm-start from seed {src_seed} "
                               f"(n={x0_effective.size}).")
            else:
                x0_effective = None
                vprint(f"[Pipeline] No seed found at {path}; "
                    f"continuing without warm-start.")

        elif warm_start == "resume":
            sidecar = cu._find_latest_sidecar(self.h5_path)

            vprint(f"[Pipeline] Warm-start mode: resume; "
                f"sidecar found: {sidecar if sidecar else 'none'}")
            # Always define these (avoids UnboundLocalError patterns).
            x_side, src_side = (None, None)
            x_main, src_main = (None, None)
            x_seed, src_seed = (None, None)

            # Candidate 1: newest sidecar (by filename/mtime), but may not exist.
            if sidecar is not None and os.path.exists(sidecar):
                x_side, src_side = self._read_latest_from_sidecar(
                    sidecar, N_expected
                )

            # Candidate 2: main file (committed solution).
            x_main, src_main = self._read_latest_from_main(
                self.h5_path, N_expected
            )

            def _safe_mtime(path: str | None) -> float:
                if not path:
                    return -np.inf
                try:
                    return float(os.path.getmtime(path))
                except Exception:
                    return -np.inf

            def _try_epoch(path: str | None, dset: str | None) -> float | None:
                if (path is None) or (dset is None):
                    return None
                try:
                    with open_h5(path, role="reader", swmr=True) as f:
                        if dset in f:
                            e = f[dset].attrs.get("epoch", None)
                            if e is None:
                                return None
                            e = float(e)
                            return e if np.isfinite(e) else None
                except Exception:
                    return None
                return None

            # Decide: newest progress (prefer higher epoch if both have it;
            # otherwise prefer newer file mtime).
            choose_side = False
            if x_side is not None and x_main is None:
                choose_side = True
            elif x_side is None and x_main is not None:
                choose_side = False
            elif x_side is not None and x_main is not None:
                e_side = _try_epoch(sidecar, src_side)
                e_main = _try_epoch(self.h5_path, src_main)

                if (e_side is not None) and (e_main is not None) and (e_side != e_main):
                    choose_side = (e_side > e_main)
                else:
                    choose_side = (_safe_mtime(sidecar) > _safe_mtime(self.h5_path))

            vprint(f"[Pipeline] Warm-start resume candidates: "
                f"sidecar: {src_side if src_side else 'none'}, "
                f"main: {src_main if src_main else 'none'}; "
                f"choosing {'sidecar' if choose_side else 'main' if (x_main is not None) else 'none'}.")
            x0_effective, src_label, src_file = (
                (x_side, src_side, sidecar) if choose_side
                else (x_main, src_main, self.h5_path)
            )

            # Fallback: seed (optional but robust).
            seed_used = False
            if x0_effective is None:
                seed_path = os.environ.get("CUBEFIT_SEED_PATH",
                    "/Seeds/x0_nnls_patch")
                x_seed, src_seed = self._read_seed_from_h5(
                    self.h5_path, N_expected, dset=seed_path
                )
                if x_seed is not None:
                    x0_effective = x_seed
                    src_label = src_seed
                    src_file = self.h5_path
                    choose_side = False
                    seed_used = True
                    vprint("[Pipeline] Warm-start fallback from seed "
                        f"{src_seed} (n={x0_effective.size}).")

            if x0_effective is not None and (not seed_used):
                t_side = _safe_mtime(sidecar) if sidecar else -np.inf
                t_main = _safe_mtime(self.h5_path)
                vprint(
                    f"[Pipeline] Warm-start from {src_label} "
                    f"({'sidecar' if choose_side else 'main'}: {src_file}) "
                    f"(n={x0_effective.size}); "
                    f"mtime(sidecar)={t_side:.0f}, mtime(main)={t_main:.0f}."
                )

        elif warm_start == "zeros":
            x0_effective = np.zeros(N_expected, np.float64)

        else:
            x0_effective = None

        # ---------------- Reader ----------------
        reader_cfg = ReaderCfg(
            s_tile=reader_s_tile,
            c_tile=reader_c_tile,
            p_tile=reader_p_tile,
            dtype_models=(reader_dtype_models or "float32"),
            apply_mask=bool(reader_apply_mask),
        )
        reader = HyperCubeReader(self.h5_path, cfg=reader_cfg)
        vprint("[Pipeline] Initialized from HDF5:"
            f" S={reader.nSpat}, C={reader.nComp}, P={reader.nPop}, "
            f"L={reader.nLSpec}; "
            f"mask={'yes' if reader.has_mask else 'no'}; "
            f"models={'yes' if reader.has_models else 'no'}; "
            f"complete={reader.models_complete}")

        # ---------------- Tracker wiring ----------------
        tracker = NullTracker()
        if tracker_mode != "off":
            tracker = FitTracker(self.h5_path)
            vprint("[Pipeline] Using tracker with mode:", tracker_mode)
            vprint('[Pipeline] Need to infer shapes...')
            with open_h5(self.h5_path, role="reader") as f:
                g = f.get("/HyperCube", None)
                if g is not None:
                    shp = g.attrs.get("shape")
                    if shp is not None and len(shp) == 4:
                        _, C, P, _ = map(int, shp)
                if "/HyperCube/models" in f:
                    _, C, P, _ = map(int, f["/HyperCube/models"].shape)
                if "/LOSVD" in f and "/Templates" in f:
                    _, _, C = map(int, f["/LOSVD"].shape)
                    P = int(f["/Templates"].shape[0])
            vprint(f"[Pipeline] Inferred C={C}, P={P} from HDF5")
            tracker.set_meta(N=int(C)*int(P))

        cfg = MPConfig(
            processes=int(processes),
            blas_threads=int(blas_threads),
            apply_mask=bool(reader_apply_mask),
            # orbit_beta=float(
            #     os.environ.get("CUBEFIT_ORBIT_BETA", "1e-2")
            # ),
            orbit_prior_weight=float(
                os.environ.get("CUBEFIT_ORBIT_PRIOR_WEIGHT", "1e-2")
            )
        )

        try:
            with logger.capture_all_output():
                resume_state = {}
                if warm_start == "resume" and choose_side and \
                    (src_file is not None):
                    resume_state = self._read_solver_state(src_file)

                x_solver, stats = solve_streaming_nnls(
                    self.h5_path,
                    cfg,
                    orbit_weights=orbit_weights,
                    x0=x0_effective,
                    resume_state=resume_state,
                    tracker=tracker,
                    monolithic_max_active=1000,
                )
                # x_solver, stats = solve_monolithic_nnls(self.h5_path,
                    # orbit_weights=orbit_weights, 
                    # hard_project=True)
                # cfg = MPConfig(epochs=1, processes=1, blas_threads=1, apply_mask=True)
                # x_solver, stats = monolithic_nnls_scipy(self.h5_path, cfg,
                #     orbit_weights=orbit_weights,
                #     enforce_orbit_projection=True)

        finally:
            try:
                reader.close()
            except Exception:
                pass
            if tracker is not None:
                try:
                    tracker.close()
                except Exception:
                    pass

        vprint("[Pipeline] Writing final /X_global to main HDF5...")
        with open_h5(self.h5_path, role="writer") as f_wr:

            assert x_solver.ndim == 2, "Xcp must be (C, P) before writing /X_global"

            if "/X_global" in f_wr:
                del f_wr["/X_global"]

            f_wr.create_dataset(
                "/X_global",
                data=x_solver.astype(np.float64),
                compression="gzip",
                compression_opts=4,
            )

            f_wr["/X_global"].attrs["layout"] = "C_P"
            f_wr["/X_global"].attrs["P"] = x_solver.shape[1]

            if "known_zero_mask" in stats:
                print("[pipeline] writing KNOWN_ZERO mask to /HyperCube/known_zero_mask",
                    flush=True)
                grp = f_wr.require_group("/HyperCube")
                if "known_zero_mask" in grp:
                    del grp["known_zero_mask"]
                grp.create_dataset(
                    "known_zero_mask",
                    data=stats["known_zero_mask"].astype(bool),
                    dtype="bool",
                )
        
        logger.log(
            "[Pipeline] ===================================================")
        logger.log(
            f"[Pipeline] Multi-process solve complete: epochs={epochs}, "
            f"processes={processes}, blas_threads={blas_threads}."
        )
        logger.log(
            '[Pipeline] ---------------------------------------------------')
        logger.log("[Pipeline] Final elapsed time: "
            f"{stats.get('elapsed_sec', np.nan):.2f} sec.")
        logger.log(
            "[Pipeline] ===================================================")

        return x_solver, stats
