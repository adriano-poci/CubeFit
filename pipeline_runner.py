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
import pathlib as plp
from typing import Optional, Tuple
import json, time, math, os
import numpy as np
from dataclasses import dataclass

from CubeFit.hdf5_manager import H5Manager, H5Dims, open_h5
from CubeFit.hypercube_builder import build_hypercube
from CubeFit.hypercube_reader import HyperCubeReader, ReaderCfg
from CubeFit.kaczmarz_solver import solve_global_kaczmarz, SolverCfg
from CubeFit.kaczmarz_solver_cchunk_mp_nnls import (
    MPConfig, solve_global_kaczmarz_cchunk_mp)
from CubeFit.live_fit_dashboard import (
    render_aperture_fits_with_x, render_sfh_from_x, alpha_star_stats
)
from CubeFit.nnls_patch import run_patch as _nnls_patch_run,\
    apply_orbit_prior_to_seed
from CubeFit.fit_tracker import FitTracker, NullTracker, TrackerConfig
import CubeFit.cube_utils as cu
from CubeFit.cube_utils import RatioCfg
from CubeFit.logger import get_logger

logger = get_logger()

# ----------------------------------------------------------------------
# Jacobi seeder
# ----------------------------------------------------------------------

def _build_streaming_jacobi_seed(
    h5_path: str,
    Ns: int = 16,
    K: int = 256,
    rng_seed: int = 12345,
    max_bytes: int = 2_000_000_000,
) -> tuple[np.ndarray, dict]:
    """
    Chunk-friendly Jacobi initializer (robust + mask-safe) with:
      • feature-aware (mean-centred in λ) matching for the seed
      • tiny NNLS polish on a working set of columns

    Env knobs (optional):
      CUBEFIT_SEED_REL_FLOOR     : float, default 1e-6
      CUBEFIT_SEED_ABS_FLOOR     : float, default 1e-24
      CUBEFIT_SEED_SIMPLEX       : 0/1,   default 1     (only if norm.mode='data')
      CUBEFIT_RDCC_SLOTS/bytes/W0: HDF5 dataset cache
      CUBEFIT_JACOBI_NNLS_K      : int,   default 128   (working-set cap)
      CUBEFIT_JACOBI_NNLS_PER_C  : int,   default 8     (per-component cap)
      CUBEFIT_JACOBI_MU_ITERS    : int,   default 10    (fallback MU iters)
    """
    rng = np.random.default_rng(rng_seed)

    rel_floor  = float(os.environ.get("CUBEFIT_SEED_REL_FLOOR", "1e-6"))
    abs_floor  = float(os.environ.get("CUBEFIT_SEED_ABS_FLOOR", "1e-24"))
    use_simplex = os.environ.get("CUBEFIT_SEED_SIMPLEX", "1").lower() \
        not in ("0", "false", "no", "off")

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"] # (S,C,P,L) float32
        Y = f["/DataCube"]         # (S,L)     float64
        Mask = f["/Mask"] if "/Mask" in f else None
        nm_attr = f["/HyperCube"].attrs.get("norm.mode", "")
        norm_mode = nm_attr.decode() if isinstance(
            nm_attr, (bytes, bytearray)) else str(nm_attr)

        try:
            M.id.set_chunk_cache(
                int(os.environ.get("CUBEFIT_RDCC_SLOTS", "100000")),
                int(os.environ.get("CUBEFIT_RDCC_BYTES",
                                   str(512 * 1024 * 1024))),
                float(os.environ.get("CUBEFIT_RDCC_W0", "0.75")),
            )
        except Exception:
            pass

        S, C, P, L = map(int, M.shape)
        CP = int(C * P)
        chunks = M.chunks or (S, 1, P, L)
        S_chunk = int(chunks[0])

        bytes_per_spax = int(C * P * L * 4)
        S_cap = int(np.max((1, int(max_bytes // np.max((1, bytes_per_spax))))))

        if S <= S_chunk:
            s0 = 0
        else:
            n_tiles = int(np.max((1, (S + S_chunk - 1) // S_chunk)))
            tile_id = int(rng.integers(0, n_tiles))
            s0 = int(np.min((tile_id * S_chunk, S - 1)))
            s0 = int((s0 // S_chunk) * S_chunk)
        S_eff = int(np.min((Ns, S_chunk, S_cap, S - s0)))
        s1 = int(s0 + S_eff)

        slab32 = np.asarray(M[s0:s1, :, :, :], np.float32, order="C")
        Y_tile = np.asarray(Y[s0:s1, :], np.float64, order="C")

        if Mask is not None:
            keep = np.asarray(Mask[...], dtype=bool).ravel()
            pool = np.flatnonzero(keep) if (keep.size == L and np.any(keep)) \
                   else np.arange(L, dtype=np.int64)
        else:
            pool = np.arange(L, dtype=np.int64)

        mK = int(np.min((K, pool.size)))
        if mK <= 0:
            raise RuntimeError("No wavelengths available for Jacobi seed.")
        ell = rng.choice(pool, size=mK, replace=False)
        ell_sorted = np.sort(ell)

        slab32 = slab32.reshape(S_eff, CP, L)
        A_sub  = slab32[:, :, ell_sorted]
        Y_sub  = Y_tile[:,  ell_sorted]

    Yc = Y_sub - np.mean(Y_sub, axis=1, keepdims=True)
    Ac = A_sub - np.mean(A_sub, axis=2, keepdims=True)

    d = np.einsum("spk,spk->p", Ac.astype(np.float64, copy=False),
                  Ac.astype(np.float64, copy=False),
                  dtype=np.float64, optimize=True)
    b = np.einsum("spk,sk->p",  Ac.astype(np.float64, copy=False),
                  Yc, dtype=np.float64, optimize=True)

    d2d = d.reshape(C, P)
    med_pos = np.zeros((C,), dtype=np.float64)
    for c in range(C):
        dc = d2d[c, :]
        pos = dc[dc > 0.0]
        med_pos[c] = float(np.median(pos)) if pos.size else 0.0
    floor_c = abs_floor + rel_floor * med_pos

    tiny_mask = d2d < floor_c[:, None]
    if np.any(tiny_mask):
        d2d[tiny_mask] = 1.0
        b.reshape(C, P)[tiny_mask] = 0.0
        d = d2d.ravel(order="C")

    eps = 1e-12
    x0 = np.maximum(0.0, b / (d + eps))

    if use_simplex and norm_mode == "data":
        X = x0.reshape(C, P)
        for c in range(C):
            s = float(np.sum(X[c, :]))
            if s > 0.0:
                X[c, :] /= s
        x0 = X.ravel(order="C")

    score = np.abs(b) / (np.sqrt(np.maximum(d, eps)) + eps)

    rows = int(S_eff * mK)
    bytes_per_col = int(rows * 8)
    K_mem = int(np.max((1, int(max_bytes // np.max((1, bytes_per_col))))))

    K_cap_env = int(os.environ.get("CUBEFIT_JACOBI_NNLS_K", "128"))
    per_c = int(np.max((1, int(os.environ.get("CUBEFIT_JACOBI_NNLS_PER_C", "8")))))
    K_cap = int(np.min((K_mem, K_cap_env, CP)))

    if K_cap > 0:
        score_cp = score.reshape(C, P)
        cand = []
        for c in range(C):
            sc = score_cp[c, :]
            k = int(np.min((per_c, np.count_nonzero(sc > 0.0))))
            if k > 0:
                top = np.argpartition(sc, -k)[-k:]
                top = top[np.argsort(sc[top])[::-1]]
                for p in top:
                    cand.append(int(c * P + int(p)))

        if len(cand) < K_cap:
            rem = K_cap - len(cand)
            order = np.argsort(score)[::-1]
            for g in order:
                if g not in cand:
                    cand.append(int(g))
                    if len(cand) >= K_cap:
                        break

        W_idx = np.array(cand[:K_cap], dtype=np.int64)

        B  = np.empty((rows, W_idx.size), dtype=np.float64)
        xW = x0[W_idx].astype(np.float64, copy=True)
        for j, g in enumerate(W_idx):
            B[:, j] = Ac[:, g, :].reshape(rows, order="C")
        y_flat = Yc.reshape(rows, order="C")

        try:
            from scipy.optimize import nnls as _scipy_nnls
            xW_new, _ = _scipy_nnls(B, y_flat)
        except Exception:
            xW_new = xW.copy()
            iters = int(np.max((0, int(os.environ.get("CUBEFIT_JACOBI_MU_ITERS",
                                                  "10")))))
            for _ in range(iters):
                By   = B @ xW_new
                BTy  = B.T @ y_flat
                BTBy = B.T @ By + 1e-12
                xW_new = np.maximum(0.0, xW_new * (BTy / BTBy))

        x0[W_idx] = xW_new

        if use_simplex and norm_mode == "data":
            X = x0.reshape(C, P)
            for c in range(C):
                s = float(np.sum(X[c, :]))
                if s > 0.0:
                    X[c, :] /= s
            x0 = X.ravel(order="C")

    stats = {
        "Ns_used": int(S_eff),
        "K_used": int(np.min((K, A_sub.shape[2]))),
        "s0": int(s0),
        "s1": int(s1),
        "S_chunk": int(S_chunk),
        "frozen_cols": int(np.sum(tiny_mask)) if 'tiny_mask' in locals() else 0,
    }
    return x0.astype(np.float64, copy=False), stats

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
                                 f["/Templates"].shape[1])) if "/Templates" \
                                 in f else None
            self.has_mask = ("/Mask" in f)
            self.has_models = ("/HyperCube/models" in f)
            self.complete = bool(f["/HyperCube"].attrs.get("complete",
                                                           False)) \
                            if "/HyperCube" in f else False

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
        with open_h5(sidecar_path, role="reader") as g:
            def _try(name):
                if name in g:
                    v = np.asarray(g[name][...], np.float64, order="C")
                    return v if v.size == N_expected else None
                return None
            x = _try("/Fit/x_best")
            src = "/Fit/x_best" if x is not None else None
            if x is None:
                x = _try("/Fit/x_last")
                src = "/Fit/x_last" if x is not None else None
            if x is None and "/Fit/x_hist" in g and g["/Fit/x_hist"].shape[0] > 0:
                v = np.asarray(g["/Fit/x_hist"][-1, ...], np.float64,
                               order="C")
                if v.size == N_expected:
                    x, src = v, "/Fit/x_hist[-1]"
            return x, src

    @staticmethod
    def _read_latest_from_main(main_path: str, N_expected: int):
        xb = xl = None
        with open_h5(main_path, role="reader") as f_ro:
            if "/Fit/x_best" in f_ro:
                v = np.asarray(f_ro["/Fit/x_best"][...], np.float64,
                               order="C")
                if v.size == N_expected: xb = v
            if "/X_global" in f_ro and xb is None:
                v = np.asarray(f_ro["/X_global"][...], np.float64, order="C")
                if v.size == N_expected: xl = v
        if xb is not None: return xb, "/Fit/x_best"
        if xl is not None: return xl, "/X_global"
        return None, None

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

    # ---------------------------- Solve (single-process) ----------------

    def solve_all(
        self,
        *,
        epochs: int = 1,
        pixels_per_aperture: int = 256,
        lr: float = 0.25,
        project_nonneg: bool = True,
        row_order: str = "random",
        # Reader knobs
        reader_s_tile: int | None = None,
        reader_c_tile: int | None = None,
        reader_p_tile: int | None = None,
        reader_dtype_models: str = "float32",
        reader_apply_mask: bool = True,
        # BLAS / compute
        blas_threads: int | None = None,
        block_rows: int | None = None,
        block_norm: str | None = None,
        # Priors / ratio
        orbit_weights: np.ndarray | None = None,
        ratio_use: bool | None = None,
        ratio_anchor: str | int | None = None,
        ratio_eta: float | None = None,
        ratio_prob: float | None = None,
        ratio_batch: int | None = None,
        ratio_min_weight: float | None = None,
        # Warm start
        x0: np.ndarray | None = None,
        warm_start: str = "nnls",  # "resume"|"jacobi"|"nnls"|
                                     # "zeros"|"seed"|"none"
        seed_cfg: dict | None = None,     # for Jacobi / nnls_patch seeding
        # Tracking
        tracker_mode: str = "sidecar",    # "sidecar" | "off"
        progress_interval_sec: float = 300.0,
        verbose: bool = True,
        seed: int | None = None,
    ):
        """
        Run the global Kaczmarz fit with a unified tracker interface.

        Requires FitTracker to expose:
        - set_orbit_weights(w_c)
        - maybe_save(x, epoch, now)
        - on_batch(rmse)
        - finalize(x)

        Returns
        -------
        x_global : (C*P,) float64
        stats    : dict
        """
        t0 = time.perf_counter()

        # ---------------- Warm-start ----------------
        N_expected = self.nComp * self.nPop

        if x0 is not None:
            x0_effective = np.asarray(x0, dtype=np.float64, order="C")

        elif warm_start == "seed":
            path = os.environ.get("CUBEFIT_SEED_PATH", "/Seeds/x0_nnls_patch")
            x_seed, src_seed = self._read_seed_from_h5(self.h5_path,
                                                       N_expected,
                                                       dset=path)
            if x_seed is not None:
                x0_effective = x_seed
                if verbose:
                    logger.log(f"[Pipeline] Warm-start from seed {src_seed} "
                               f"(n={x0_effective.size}).")
            else:
                x0_effective = None
                if verbose:
                    logger.log(f"[Pipeline] No seed found at {path}; "
                               f"continuing without warm-start.")

        elif warm_start == "resume":
            sidecar = cu._find_latest_sidecar(self.h5_path)
            x_side, src_side = (None, None)
            if sidecar is not None:
                x_side, src_side = self._read_latest_from_sidecar(
                    sidecar, N_expected
                )
            x_main, src_main = self._read_latest_from_main(self.h5_path,
                                                           N_expected)

            choose_side = False
            if x_side is not None and x_main is None:
                choose_side = True  # prefer main unless missing

            x0_effective, src_label, src_file = (
                (x_side, src_side, sidecar) if choose_side
                else (x_main, src_main, self.h5_path)
            )

            # Fallback: /Seeds/x0_nnls_patch
            if x0_effective is None:
                seed_path = os.environ.get("CUBEFIT_SEED_PATH",
                                           "/Seeds/x0_nnls_patch")
                x_seed, src_seed = self._read_seed_from_h5(self.h5_path,
                                                           N_expected,
                                                           dset=seed_path)
                if x_seed is not None:
                    x0_effective = x_seed
                    if verbose:
                        logger.log("[Pipeline] Warm-start fallback from seed "
                                   f"{src_seed} (n={x0_effective.size}).")

            if x0_effective is not None and verbose:
                if x0_effective is x_seed:
                    pass
                else:
                    logger.log(
                        f"[Pipeline] Warm-start from {src_label} "
                        f"({'sidecar' if choose_side else 'main'}: "
                        f"{src_file}) (n={x0_effective.size})."
                    )

        elif warm_start == "nnls":
            if verbose:
                logger.log("[Pipeline] Warm-start mode: nnls_patch seed (exact semantics)")

            # Mirror nnls_patch defaults: mask+lambda on, nnls solver, normalized columns, zero ridge
            res = _nnls_patch_run(
                h5_path=self.h5_path,
                s_sel=None,                 # like nnls_patch default: first min(32, S) spaxels
                k_per_comp=12,              # same default as CLI
                pick_mode="energy",
                solver="nnls",
                ridge=0.0,
                use_mask=True,
                use_lambda=True,
                lam_dset="/HyperCube/lambda_weights",
                out_dir=plp.Path(self.h5_path).parent,
                write_seed=False,
                seed_path="/Seeds/x0_nnls_patch",
                normalize_columns=True,
            )
            Xcp = np.asarray(res["x_CP"], np.float64, order="C")
            # If you have a prior, enforce it on the seed:
            if orbit_weights is not None:
                Xcp = apply_orbit_prior_to_seed(Xcp, orbit_weights,
                                                preserve_total=True, min_w_frac=1e-4)

            x0_effective = Xcp.ravel(order="C")
            if verbose:
                meta = res.get("meta", {})
                logger.log(f"[Pipeline] nnls_patch seed: rows={meta.get('rows')}, cols={meta.get('cols')}, "
                        f"mask={meta.get('mask_used')}, lambda={meta.get('lambda_used')}, "
                        f"solver={meta.get('solver')}, normcols={True}")

        elif warm_start == "jacobi":
            if verbose:
                logger.log("[Pipeline] Warm-start mode: fresh Jacobi")
            x0_effective, jacobi_stats = _build_streaming_jacobi_seed(
                self.h5_path, **(seed_cfg or {})
            )

        elif warm_start == "zeros":
            x0_effective = np.zeros(N_expected, np.float64)

        else:
            x0_effective = None

        diag_on = os.environ.get("CUBEFIT_JACOBI_DIAG", "1").lower() \
            not in ("0", "false", "no", "off")
        if warm_start == "jacobi" and x0_effective is not None and diag_on:
            try:
                s0 = int(jacobi_stats.get("s0", 0))
                s1 = int(jacobi_stats.get(
                    "s1", s0 + int(np.max([1, jacobi_stats.get("Ns_used", 4)]))
                ))
                Ns_used = int(jacobi_stats.get("Ns_used", 4))

                astats = alpha_star_stats(self.h5_path, x0_effective,
                    n_spax=int(np.min([6, Ns_used])), tile=(s0, s1))
                with open_h5(self.h5_path, role="writer") as fwr:
                    gj = fwr.require_group("/Fit/JacobiDiag")
                    for k in ("median", "min", "max", "n"):
                        gj.attrs[f"alpha_star.{k}"] = float(astats[k])

                with open_h5(self.h5_path, role="reader",
                             swmr=True) as fro:
                    S = int(fro["/DataCube"].shape[0])
                n_plot = int(np.min([6, S]))
                if n_plot > 0:
                    sample = np.linspace(0, S - 1, n_plot,
                                         dtype=int).tolist()

                    base = plp.Path(self.h5_path)
                    out_fits = str(base.with_name(base.stem
                                           + "_jacobi_fits.png"))
                    render_aperture_fits_with_x(self.h5_path, x0_effective,
                        out_fits, apertures=sample, show_residual=True,
                        title=f"Jacobi seed overlays (spaxels={sample})")

                    out_sfh = str(base.with_name(base.stem
                                          + "_jacobi_sfh.png"))
                    render_sfh_from_x(self.h5_path, x0_effective, out_sfh)

            except Exception as e:
                logger.log("[JacobiDiag] WARNING: could not run Jacobi "
                           "diagnostics:", e)


        # ---------------- Reader ----------------
        reader_cfg = ReaderCfg(
            s_tile=reader_s_tile,
            c_tile=reader_c_tile,
            p_tile=reader_p_tile,
            dtype_models=reader_dtype_models,
            apply_mask=reader_apply_mask,
        )
        reader = HyperCubeReader(self.h5_path, cfg=reader_cfg)

        # ---------------- Tracker ----------------
        tracker = NullTracker()
        if tracker_mode != "off":
            tracker_cfg = TrackerConfig()
            tracker = FitTracker(self.h5_path, cfg=tracker_cfg)

        # Push priors into tracker if provided
        if orbit_weights is not None:
            w_c = np.asarray(orbit_weights, dtype=np.float64).ravel()
            if w_c.size not in (reader.nComp, reader.nComp * reader.nPop):
                raise ValueError(
                    "orbit_weights len must be C or C*P; got "
                    f"{w_c.size} for C={reader.nComp}, P={reader.nPop}"
                )
            if w_c.size == reader.nComp * reader.nPop:
                w_c = w_c.reshape(reader.nComp, reader.nPop).sum(axis=1)
            tracker.set_orbit_weights(w_c)

        if ratio_anchor is None:
            ratio_anchor = "auto"

        cfg = SolverCfg(
            epochs=epochs,
            pixels_per_aperture=pixels_per_aperture,
            lr=lr,
            project_nonneg=project_nonneg,
            row_order=row_order,
            blas_threads=blas_threads,
            block_rows=block_rows,
            block_norm=block_norm,
            ratio_use=ratio_use,
            ratio_anchor=ratio_anchor,
            ratio_eta=ratio_eta,
            ratio_prob=ratio_prob,
            ratio_batch=ratio_batch,
            ratio_min_weight=ratio_min_weight,
            verbose=verbose,
            seed=seed,
        )

        def _on_batch_rmse(rmse: float):
            try:
                tracker.on_batch_rmse(rmse, block=False)
            except Exception:
                pass

        def _on_progress(epoch, stats_epoch):
            try:
                if tracker is not None and hasattr(tracker, "on_progress"):
                    tracker.on_progress(epoch=epoch, stats=stats_epoch)
            except Exception:
                pass

        def _on_epoch_end(epoch: int, stats: dict):
            try:
                tracker.on_epoch_end(epoch, stats)
            except Exception:
                pass

        try:
            x, stats = solve_global_kaczmarz(
                reader,
                cfg=cfg,
                orbit_weights=orbit_weights,
                x0=x0_effective,
                on_epoch_end=_on_epoch_end,
                on_progress=_on_progress,
                progress_interval_sec=progress_interval_sec,
                on_batch_rmse=_on_batch_rmse,
            )
        finally:
            reader.close()

        with open_h5(self.h5_path, role="writer") as f_wr:
            if "/X_global" in f_wr:
                del f_wr["/X_global"]
            f_wr.create_dataset("/X_global", data=x, dtype=np.float64)

        tracker.maybe_save(x, stats)
        try:
            tracker.close()
        except Exception:
            pass

        stats = dict(stats or {})
        stats["elapsed_total_sec"] = time.perf_counter() - t0
        return x, stats

    # ------------------------- Solve (multi-process) -------------------

    def solve_all_mp_batched(
        self,
        epochs=1,
        lr=0.25,
        project_nonneg=True,
        reader_s_tile=128,
        reader_c_tile=1,
        reader_p_tile=360,
        reader_dtype_models="float32",
        reader_apply_mask=True,
        processes=2,
        blas_threads=12,
        orbit_weights=None,
        x0=None,
        warm_start="nnls",  # default to the new seed
        seed_cfg=None,
        tracker_mode="on",
        verbose=True,
        ratio_cfg: RatioCfg | None = None,  
    ):

        # --------------- Warm-start (same policy as SP path) -----------
        N_expected = int(self.nComp * self.nPop)

        if x0 is not None:
            x0_effective = np.asarray(x0, dtype=np.float64, order="C")

        elif warm_start == "resume":
            sidecar = cu._find_latest_sidecar(self.h5_path)
            x_side, src_side = (None, None)
            if sidecar is not None:
                x_side, src_side = self._read_latest_from_sidecar(
                    sidecar, N_expected
                )

            x_main, src_main = self._read_latest_from_main(self.h5_path,
                                                           N_expected)

            choose_side = False
            if x_side is not None and x_main is None:
                choose_side = True

            x0_effective, src_label, src_file = (
                (x_side, src_side, sidecar) if choose_side
                else (x_main, src_main, self.h5_path)
            )

            if x0_effective is not None and verbose:
                logger.log(
                    f"[Pipeline] Warm-start from {src_label} "
                    f"({'sidecar' if choose_side else 'main'}: {src_file}) "
                    f"(n={x0_effective.size})."
                )

        elif warm_start == "nnls":
            if verbose:
                logger.log("[Pipeline] Warm-start mode: nnls_patch seed (exact semantics)")

            # Mirror nnls_patch defaults: mask+lambda on, nnls solver, normalized columns, zero ridge
            res = _nnls_patch_run(
                h5_path=self.h5_path,
                s_sel=None,                 # like nnls_patch default: first min(32, S) spaxels
                k_per_comp=12,              # same default as CLI
                pick_mode="energy",
                solver="nnls",
                ridge=0.0,
                use_mask=True,
                use_lambda=True,
                lam_dset="/HyperCube/lambda_weights",
                out_dir=plp.Path(self.h5_path).parent,
                write_seed=bool((seed_cfg or {}).get("write_seed", True)),
                seed_path="/Seeds/x0_nnls_patch",
                normalize_columns=True,
            )
            Xcp = np.asarray(res["x_CP"], np.float64, order="C")
            # If you have a prior, enforce it on the seed:
            if orbit_weights is not None:
                Xcp = apply_orbit_prior_to_seed(Xcp, orbit_weights,
                                                preserve_total=True, min_w_frac=1e-4)

            x0_effective = Xcp.ravel(order="C")
            if verbose:
                meta = res.get("meta", {})
                logger.log(f"[Pipeline] nnls_patch seed: rows={meta.get('rows')}, cols={meta.get('cols')}, "
                        f"mask={meta.get('mask_used')}, lambda={meta.get('lambda_used')}, "
                        f"solver={meta.get('solver')}, normcols={True}")

        elif warm_start == "jacobi":
            if verbose:
                logger.log("[Pipeline] Warm-start mode: fresh Jacobi")
            x0_effective, jacobi_stats = _build_streaming_jacobi_seed(
                self.h5_path, **(seed_cfg or {})
            )

        elif warm_start == "zeros":
            x0_effective = np.zeros(N_expected, np.float64)

        else:
            x0_effective = None

        diag_on = os.environ.get("CUBEFIT_JACOBI_DIAG", "1").lower() \
            not in ("0", "false", "no", "off")
        if warm_start == "jacobi" and x0_effective is not None and diag_on:
            try:
                s0 = int(jacobi_stats.get("s0", 0))
                s1 = int(jacobi_stats.get(
                    "s1", s0 + int(np.max([1, jacobi_stats.get("Ns_used", 4)]))
                ))
                Ns_used = int(jacobi_stats.get("Ns_used", 4))

                astats = alpha_star_stats(self.h5_path, x0_effective,
                    n_spax=int(np.min([6, Ns_used])), tile=(s0, s1))
                with open_h5(self.h5_path, role="writer") as fwr:
                    gj = fwr.require_group("/Fit/JacobiDiag")
                    for k in ("median", "min", "max", "n"):
                        gj.attrs[f"alpha_star.{k}"] = float(astats[k])

                with open_h5(self.h5_path, role="reader",
                             swmr=True) as fro:
                    S = int(fro["/DataCube"].shape[0])
                n_plot = int(np.min([6, S]))
                if n_plot > 0:
                    sample = np.linspace(0, S - 1, n_plot,
                                         dtype=int).tolist()

                    base = plp.Path(self.h5_path)
                    out_fits = str(base.with_name(base.stem
                                           + "_jacobi_fits.png"))
                    render_aperture_fits_with_x(self.h5_path, x0_effective,
                        out_fits, apertures=sample, show_residual=True,
                        title=f"Jacobi seed overlays (spaxels={sample})")

                    out_sfh = str(base.with_name(base.stem
                                          + "_jacobi_sfh.png"))
                    render_sfh_from_x(self.h5_path, x0_effective, out_sfh)
            except Exception as e:
                logger.log('[JacobiDiag] WARNING: could not run Jacobi '
                           'diagnostics:')
                logger.log_exc(e)

        # ---------------- Reader ----------------
        reader_cfg = ReaderCfg(
            s_tile=reader_s_tile,
            c_tile=reader_c_tile,
            p_tile=reader_p_tile,
            dtype_models=(reader_dtype_models or "float32"),
            apply_mask=bool(reader_apply_mask),
        )
        reader = HyperCubeReader(self.h5_path, cfg=reader_cfg)
        if verbose:
            logger.log(
                "[Pipeline] Initialized from HDF5:"
                f" S={reader.nSpat}, C={reader.nComp}, P={reader.nPop}, "
                f"L={reader.nLSpec}; "
                f"mask={'yes' if reader.has_mask else 'no'}; "
                f"models={'yes' if reader.has_models else 'no'}; "
                f"complete={reader.models_complete}"
            )

        # ---------------- Tracker wiring ----------------
        tracker = NullTracker()
        if tracker_mode != "off":
            tracker = FitTracker(self.h5_path)
            logger.log("[Pipeline] Using tracker with mode:", tracker_mode)
            logger.log('[Pipeline] Need to infer shapes...')
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
            logger.log(f"[Pipeline] Inferred C={C}, P={P} from HDF5")
            tracker.set_meta(N=int(C)*int(P))

        cfg = MPConfig(
            epochs=int(epochs),
            lr=float(lr),
            project_nonneg=bool(project_nonneg),
            processes=int(processes),
            blas_threads=int(blas_threads),
            apply_mask=bool(reader_apply_mask),
        )

        try:
            x_global, stats = solve_global_kaczmarz_cchunk_mp(
                self.h5_path,
                cfg,
                orbit_weights=orbit_weights,
                x0=x0_effective,
                tracker=tracker,
                ratio_cfg=ratio_cfg,
            )
        finally:
            try:
                reader.close()
            except Exception:
                pass

        with open_h5(self.h5_path, role="writer") as f_wr:
            if "/X_global" in f_wr:
                del f_wr["/X_global"]
            x1d = np.asarray(x_global, np.float64).ravel(order="C")
            f_wr.create_dataset("/X_global", data=x1d, dtype=np.float64,
                                chunks=(min(8192, x1d.size),),
                                compression="gzip", compression_opts=4,
                                shuffle=True)

        if tracker is not None:
            try: tracker.close()
            except Exception: pass

        return x_global, stats
