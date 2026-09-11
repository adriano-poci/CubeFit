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
v1.22:  Reworked resume warm-starting to load a matched sidecar checkpoint with
            `load_checkpoint` and pass only the consistent `resume_state` into
            the constrained solver. 7 August 2026
v1.23:  Removed all legacy checkpointing and tracking. 26 August 2026
v1.24:  Replaced legacy `seed` warm-starting with explicit `saved_x` in
            `PipelineRunner.solve_all_mp_batched`;
        `saved_x` restores only the latest physical solution and always starts
            with fresh solver state in `PipelineRunner.solve_all_mp_batched`;
        `resume` restores a complete matched sidecar checkpoint when available,
            otherwise falling back to an x-only saved solution in
            `PipelineRunner.solve_all_mp_batched`;
        Explicit `x0` now always starts with fresh solver state in 
            `PipelineRunner.solve_all_mp_batched`. 7 September 2026
v1.25:  Added adjustable `regularisation_scale` throughout the solver pathway.
            10 September 2026
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
from CubeFit.streaming_nnls_constrained import (
    MPConfig, solve_streaming_nnls, monolithicNNLS, monolithic_nnls_scipy)
# from CubeFit.streaming_nnls_augmented_rows import (
#     MPConfig, solve_streaming_nnls)
from CubeFit.live_fit_dashboard import (
    render_aperture_fits_with_x, render_sfh_from_x, alpha_star_stats
)
from CubeFit.fit_tracker import FitTracker, NullTracker, load_checkpoint
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
        Read the latest physical solution from a tracker sidecar.

        Parameters
        ----------
        sidecar_path : str
            Tracker sidecar path.
        N_expected : int
            Expected flattened solution size.

        Returns
        -------
        x : ndarray or None
            Flattened physical solution vector.
        src : str or None
            Source dataset label.

        Raises
        ------
        None

        Examples
        --------
        Used internally for saved-x warm-start selection.
        """
        if not sidecar_path or not os.path.exists(sidecar_path):
            return None, None

        try:
            with open_h5(sidecar_path, role="reader", swmr=True) as f:
                if "/Fit/x_last" not in f:
                    return None, None

                x = np.asarray(
                    f["/Fit/x_last"][...], dtype=np.float64).ravel(order="C")

                if x.size != int(N_expected):
                    return None, None
                if not np.all(np.isfinite(x)):
                    return None, None

                return x, "/Fit/x_last"
        except Exception:
            return None, None

    @staticmethod
    def _read_latest_from_main(h5_path: str, N_expected: int):
        """
        Read the canonical solution vector from the main HDF5 file.

        Parameters
        ----------
        h5_path : str
            Main HDF5 path.
        N_expected : int
            Expected flattened solution size.

        Returns
        -------
        x : ndarray or None
            Flattened physical solution vector.
        src : str or None
            Source dataset label.

        Raises
        ------
        None

        Examples
        --------
        Used internally for saved-x warm-start selection.
        """
        if not h5_path or not os.path.exists(h5_path):
            return None, None

        try:
            with open_h5(h5_path, role="reader", swmr=True) as f:
                if "/X_global" not in f:
                    return None, None

                x = np.asarray(
                    f["/X_global"][...], dtype=np.float64).ravel(order="C")

                if x.size != int(N_expected):
                    return None, None
                if not np.all(np.isfinite(x)):
                    return None, None

                return x, "/X_global"
        except Exception:
            return None, None

    def _read_saved_x(
        self,
        N_expected: int,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        """
        Read the newest saved physical solution without solver state.

        Parameters
        ----------
        N_expected : int
            Expected flattened solution-vector length.

        Returns
        -------
        x : ndarray or None
            Latest valid saved physical solution.
        src : str or None
            Dataset from which the solution was read.
        src_file : str or None
            HDF5 file containing the selected solution.

        Raises
        ------
        None

        Examples
        --------
        >>> x, src, path = runner._read_saved_x(
        ...     runner.nComp * runner.nPop)
        """
        sidecar = cu._find_latest_sidecar(self.h5_path)

        x_side, src_side = None, None
        if sidecar is not None:
            x_side, src_side = self._read_latest_from_sidecar(
                sidecar, N_expected)

        x_main, src_main = self._read_latest_from_main(
            self.h5_path, N_expected)

        def _safe_mtime(path: str | None) -> float:
            if not path:
                return -np.inf
            try:
                return float(os.path.getmtime(path))
            except Exception:
                return -np.inf

        choose_side = False

        if x_side is not None and x_main is None:
            choose_side = True
        elif x_side is not None and x_main is not None:
            choose_side = (
                _safe_mtime(sidecar) > _safe_mtime(self.h5_path)
            )

        if choose_side:
            return x_side, src_side, sidecar
        if x_main is not None:
            return x_main, src_main, self.h5_path

        return None, None, None

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
        warm_start="zeros",  # zeros, saved_x, or resume
        regularisation_scale=1.0,
        tracker_mode="on",
    ):

        # ---------------- Warm-start ----------------
        N_expected = int(self.nComp * self.nPop)

        x0_effective = None
        resume_state_effective = None

        if x0 is not None:
            x0_effective = np.asarray(
                x0, dtype=np.float64, order="C").ravel(order="C")

            if x0_effective.size != N_expected:
                raise ValueError(
                    f"x0 has size {x0_effective.size}, "
                    f"expected {N_expected}.")

            if not np.all(np.isfinite(x0_effective)):
                raise ValueError("x0 contains non-finite values.")

            vprint(
                "[Pipeline] Warm-start from explicit x0 "
                f"(n={x0_effective.size}); using fresh solver state.")

        elif warm_start == "saved_x":
            x0_effective, src_label, src_file = self._read_saved_x(
                N_expected)

            if x0_effective is not None:
                vprint(
                    f"[Pipeline] Warm-start from saved x {src_label} "
                    f"({src_file}; n={x0_effective.size}).")
            else:
                vprint(
                    "[Pipeline] No saved x found; continuing without "
                    "warm-start.")

        elif warm_start == "resume":
            sidecar = cu._find_latest_sidecar(self.h5_path)

            vprint(
                "[Pipeline] Warm-start mode: resume; sidecar found: "
                f"{sidecar if sidecar else 'none'}")

            x_resume = None

            if sidecar is not None:
                x_resume, resume_state_effective = load_checkpoint(
                    sidecar, expected_size=N_expected)

            if (x_resume is not None) and (resume_state_effective is not None):
                x0_effective = x_resume

                vprint(
                    "[Pipeline] Full solver-state resume from "
                    f"{sidecar}: "
                    f"iter={resume_state_effective.get('iter', None)}, "
                    f"phase={resume_state_effective.get('phase', None)}, "
                    f"final={resume_state_effective.get('final', None)}.")

            else:
                resume_state_effective = None
                x0_effective, src_label, src_file = self._read_saved_x(
                    N_expected)

                if x0_effective is not None:
                    vprint(
                        "[Pipeline] Full solver state unavailable; "
                        f"falling back to saved x {src_label} "
                        f"({src_file}; n={x0_effective.size}).")
                else:
                    vprint(
                        "[Pipeline] No full checkpoint or saved x found; "
                        "continuing without warm-start.")

        elif warm_start == "zeros":
            x0_effective = np.zeros(N_expected, dtype=np.float64)

        else:
            raise ValueError(
                "warm_start must be one of "
                "{'zeros', 'saved_x', 'resume'}.")

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

        cfg = MPConfig(processes=int(processes), blas_threads=int(blas_threads),
            apply_mask=bool(reader_apply_mask))

        try:
            with logger.capture_all_output():

                # x_solver, stats = solve_streaming_nnls(self.h5_path, cfg,
                #     orbit_weights=orbit_weights, x0=x0_effective,
                #     resume_state=resume_state_effective, tracker=tracker,
                #     monolithic_max_active=2000,
                #     regularisation_scale=regularisation_scale)
                # x_solver, stats = solve_monolithic_nnls(self.h5_path,
                    # orbit_weights=orbit_weights, 
                    # hard_project=True)
                # cfg = MPConfig(epochs=1, processes=1, blas_threads=1, apply_mask=True)
                x_solver, stats = monolithicNNLS(self.h5_path, cfg,
                    orbit_weights=orbit_weights, x0=x0_effective,
                    resume_state=resume_state_effective, tracker=tracker,
                    monolithic_max_active=2000,
                    regularisation_scale=regularisation_scale)

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

            f_wr.create_dataset("/X_global", data=x_solver.astype(np.float64),
                compression="gzip", compression_opts=4)

            f_wr["/X_global"].attrs["layout"] = "C_P"
            f_wr["/X_global"].attrs["P"] = x_solver.shape[1]

            if "known_zero_mask" in stats:
                print("[pipeline] writing KNOWN_ZERO mask to /HyperCube/known_zero_mask",
                    flush=True)
                grp = f_wr.require_group("/HyperCube")
                if "known_zero_mask" in grp:
                    del grp["known_zero_mask"]
                grp.create_dataset("known_zero_mask",
                    data=stats["known_zero_mask"].astype(bool), dtype="bool")
        
        logger.log(
            "[Pipeline] ===================================================")
        logger.log(f"[Pipeline] Multi-process solve complete:, "
            f"processes={processes}, blas_threads={blas_threads}.")
        logger.log(f"[Pipeline] Regularisation scale: {regularisation_scale}.")
        logger.log(
            '[Pipeline] ---------------------------------------------------')
        logger.log("[Pipeline] Final elapsed time: "
            f"{stats.get('elapsed_sec', np.nan):.2f} sec.")
        logger.log(
            "[Pipeline] ===================================================")

        return x_solver, stats
