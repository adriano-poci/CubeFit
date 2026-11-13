# -*- coding: utf-8 -*-
r"""
    streaming_nnls_constrained.py
    Adriano Poci
    University of Oxford
    2026

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Implementation of a TRUE MONOLITHIC streaming active-set NNLS solver for the
        CubeFit problem, with hybrid parallelism (process-level over tile batches
        + BLAS threads inside workers).

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Forked from `streaming_nnls_augmented_rows` v1.2;
        A priori impose the orbit prior before the fit. 4 August 2026
v1.1:   Fixed ordering bug on `promoted_cols`;
        Ensure each non-zero orbit has at least one active column. 5 August 2026
v1.2:   Replaced fixed absolute orbit-mass targets with a hard a priori
            unit-sum orbit shape and a data-fitted nonnegative global amplitude;
        Extended the constrained KKT system to solve jointly for active
            coefficients, the global amplitude, and orbit Lagrange multipliers;
        Added constrained reduced-gradient promotion using the orbit
            multipliers;
        Added feasible fresh-start initialization and resume support for the
            fitted global amplitude and multipliers;
        Excluded zero-weight orbits from the active support and promotion pool;
        Preserved at least one active population in every positive-weight
            orbit during pruning;
        Fixed failed-promotion cleanup ordering so the active mask and
            coefficient vector remain consistent;
        Updated orbit-target, constraint, checkpoint, and final diagnostics
            for the flexible-amplitude hard prior. 6 August 2026
v1.3:   Switched checkpoint emission to atomic `save_checkpoint` and removed the
            split snapshot/state write path so resumable checkpoints preserve the
            full constrained solver state. 7 August 2026
v1.4:   Emit current `x` vector in the diagnostics to assist in debugging;
        Don't zero the solution in-place before returning, as this mutates the
            vector after the solver has finished;
        Make the final summary statistics use `best_x` as a total solver
            summary. 11 August 2026
v1.5:   Expanded exit diagnostics to include the reason, category, and
            convergence status of the solver termination. 12 August 2026
v1.6:   Dramatically expanded diagnostics for noop columns in
            `streamActiveSetNNLS`;
        Added explicit constrained KKT convergence diagnostics and termination
            in `streamActiveSetNNLS`;
        Distinguished active-positive stationarity from inactive-column KKT
            violations in `streamActiveSetNNLS`;
        Added detailed promotion-attempt and promotion-outcome diagnostics in
            `streamActiveSetNNLS`;
        Tightened near-noop detection to require negligible objective and
            solution-vector changes in `streamActiveSetNNLS`;
        Made checkpointing iteration-based and removed legacy epoch semantics in
            `streamActiveSetNNLS`;
        Made the monolithic solver the authoritative source of final
            checkpoint state in `streamActiveSetNNLS` and `solve_streaming_nnls`;
        Identified post-solve support pruning as requiring a constrained
            re-solve before the resulting state can be considered committed in
            `streamActiveSetNNLS`;
        No longer mutate `z` when rejecting failed promotions, restoring the
            previous feasible solver state instead in `streamActiveSetNNLS`;
        Removed probation pruning to prevent post-solve support changes from
            violating the hard orbit constraints in `streamActiveSetNNLS`. 26
            August 2026.
v1.7:   Removed residual probation-pruning state and bookkeeping so the
            implementation matches the committed-state semantics documented
            in v1.6;
        Added constrained KKT initialization for x-only warm starts, re-solving
            the supplied support before the first outer iteration to establish
            mutually consistent coefficients, global amplitude, orbit
            multipliers, and numerical ridge;
        Added the committed numerical ridge to checkpoint/resume state and to
            the full-space reduced gradient used for KKT convergence tests;
        Extended the constrained reduced NNLS active-set solve with dual
            feasibility checks and re-entry of fixed-zero variables that
            violate the KKT conditions;
        Added the inner KKT ridge and fixed-variable dual residual to
            constrained-solve diagnostics;
        Expanded outer KKT convergence to distinguish positive-active
            stationarity, zero-active dual feasibility, and inactive-column
            dual feasibility;
        Made KKT termination independent of diagnostic emission and moved
            promotion-score construction outside diagnostic conditionals;
        Separated mathematical promotion eligibility from heuristic candidate
            ranking, using the constrained reduced gradient for KKT decisions
            and the promotion score only to rank eligible columns;
        Preserved the KKT-violation filter when relaxing promotion cooldowns;
        Tightened failed-promotion zero thresholds to avoid rejecting small
            but valid newly active coefficients;
        Commit the fitted amplitude, orbit multipliers, and complete numerical
            ridge atomically with each accepted constrained solution
        Improved code formatting. 7 September 2026
v1.8:   Added unified outer-iteration progress accounting and persistent
            unresolved-dual stall detection to prevent repeated failed
            promotion cycles;
        Restricted normal promotions to genuine KKT-violating columns and
            removed obsolete forced-coverage promotion;
        Made failed and rejected promotions restore the previous feasible
            solver state before progress accounting;
        Enforced `known_zero_mask` as a hard solver constraint during
            initialization, resume, and inactive-column promotion, with
            explicit infeasibility detection for fully masked positive-weight
            orbits;
        Made the monolithic solution authoritative on return;
        Retained the numerical ridge as part of the solved regularized
            objective, with the committed total ridge used consistently in
            the reduced solve, checkpoint state, and outer KKT convergence
            test;
        Added adjustable `regularisation_scale` throughout the solver pathway.
            10 September 2026
v1.9:   Allow `regularisation_scale` to be zero to disable stabilisation ridge;
        Updated `monolithic_nnls_scipy` to have the same API as
            `solve_streaming_nnls` and solve the same mathematics;
        Added adjacent function `monolithicNNLS` computing a genuine
            monolithic non-negative least-squares solution. 11 September 2026
v1.10:  Switched `monolithicNNLS` to `OSQP` solver. 13 September 2026
v1.11:  Added diversity-aware round-robin promotion with a per-orbit batch cap
            to reduce redundant same-orbit candidates. 14 September 2026
v1.12:  Avoid computing full gradient when initialisation is `zeros`;
        Cache gradient in case of rejected solve, which avoids recomputing the
            full gradient of known columns. 25 September 2026
v1.13:  Added data-fit safeguards, consistent warm-start multipliers, retry
            protection, and robust checks for unstable reduced solves. 27
            September 2026
v1.14:  Separated effects of `regularisation_scale` into numerical
            stabilisation `ridge` and scientific regularisation. 28 September
            2026
v1.15:  Fixed bug in return assignments for `_solve_reduced_active`;
        Removed special promotion batches for `C <= 3`;
        Fixed the constrained-gradient calculation used for convergence checks
            in `streamActiveSetNNLS`;
        Removed the old exploratory-promotion logic and related bookkeeping in
            `streamActiveSetNNLS`;
        Remove coefficients that return to zero from the active set in
            `streamActiveSetNNLS`;
        Simplified promotion and stopping logic around the cleaned active set in
            `streamActiveSetNNLS`;
        Replaced gradient-based column promotion with joint-support exploration,
            allowing existing coefficients to re-adjust when testing new
            populations in `streamActiveSetNNLS`;
        Introduced `explore_obj_tol` to make the exploration tolerance
            scale-aware in `streamActiveSetNNLS`. 29 September 2026
v1.16:  Reworked support exploration to screen candidates by constrained
            gradient, data-fit gradient, and rotating population coverage before
            joint constrained re-solving in `streamActiveSetNNLS`;
        Made exploration acceptance consistent with the scientific regularisation objective in `streamActiveSetNNLS`. 30 September 2026
v1.17:  Changed `_canon_orbit_weights` to accept a `None` value for
            `orbit_weights` and return `None` in that case, rather than
            automatically reading the weights from the HDF5 file;
        Implemented a special case for when the solution is entirely zero --- 
            possible when `orbit_weights=None` --- to ensure termination
            proceeds, since there is no KKT support for all-zeros, in 
            `streamActiveSetNNLS`. 1 October 2026
v1.18:  Added `orbit_weights` to the checkpoint state and made it authoritative
            on resume, rather than reading the weights from the HDF5 file in
            `streamActiveSetNNLS`;
        Added a check to ensure that the resumed `orbit_weights` are consistent
            with the current `C` and `P` values, and raise an error if not in
            `streamActiveSetNNLS`;
        Allow convergence when KKT are satisfied alone, if `orbit_weights` is
            `None`, in `streamActiveSetNNLS`;
        Corrected column exploration; previous implementation ordered by orbit
            index, then truncated to `explore_batch_size` which arbitrarily
            affected higher orbit indices in `streamActiveSetNNLS`;
        Improved the rotation of the explored columns from sequential per orbit
            to coarse-to-fine interleaved sequence in `streamActiveSetNNLS`. 2
            October 2026
v2.0:   Fixed orbit-mass constraint implementation. The orbital weights are
            imposed during the contruction of the hypercube. The solver needs
            only to enforce the sum of the populations in each orbit to be equal
            to a global amplitude. `alpha` is already fit for by the solver and
            is now understood as the global amplitude between the parent
            Schwarzschild model and the observed spectral data. 3 October 2026
v2.1:   Added `_exploration_cycle_rounds` to terminate exploration cycles when
            the full candidate list has been cycled over. 4 October 2026
v2.2:   Changed blind cycling of candidates to tracking cached sets of active
            columns; don't retry non-improving successfully-resolved sets;
        Updated `monolithicNNLS` to solve the new mathematical formulation of
            the solver. 5 October 2026
v2.3:   Made `_worker_reduced` and `_worker_ATAz` storage-chunk-aware, grouping
            reads by the physical HDF5 C/P chunk to eliminate redundant
            hypercube I/O. 13 October 2026
v2.4:   Assemble all exploration proposals from one union-support Gram matrix,
            eliminating redundant HyperCube passes for exchange, data, and
            coverage trials;
        Removed redundant ATy computation from `_worker_reduced`. 15 October 2026
v2.5:   Replaced the fixed 50-percent exploration-coverage stopping criterion
            with consecutive stationary joint-support challenges;
        Made coverage proposals maximally separated from populations already
            tested within each orbit, so each stationary challenge probes new
            regions of SSP space;
        Reset stationary-challenge evidence after every accepted support change
            and recompute exact full-space KKT before counting further failures.
            16 October 2026
v2.6:   Changed the `regularisation_scale` to be an SSP nearest-neighbour
            smoothness penalty, rather than a classic L2 ridge. 17 October 2026
v2.7:   Made the constrained reduced solve more numerically robust;
        Added automatic smaller-batch retries when adding new populations;
        Compute normal equations only once in the orbit-constrained path of
            `monolithicNNLS`;
        Changed `scipy.optimize` to `trust-constr` from `SLSQP` and provide
            exact constant Hessian in orbit-constrained path of `monolithicNNLS`.
            27 October 2026
v2.8:   Removed expensive full RMSE computation after a successful run in
            `solve_streaming_nnls`; instead provide the relevant data cheaply
            directly from `streamActiveSetNNLS`. 28 October 2026
v2.9:   Changed `_ssp_smoothness_hessian` from first-difference regularisation
            to template-value-aware second-difference smoothness regularisation.
            30 October 2026
v2.10:  Added post-convergence minimum-L2 redistribution with targeted support
            expansion for less sparse solutions when `regularisation_scale=0`. 3
            November 2026
v2.11:  Output the pure-KKT unregularised solution before L2 minimisation. 11
            November 2026
v2.12:  Added population tying for poorly observed dynamical components while
            retaining their original spatial and kinematic model contributions;
        Solve tied populations in a reduced component space and expand solutions
            back to the original component layout;
        Added chunk-aware HyperCube operations for tied population fitting.
            13 November 2026
"""

from __future__ import annotations, print_function

import os, sys, traceback, json
import math
import time
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import osqp
try:
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, minimize, nnls, \
        lsq_linear
    _HAS_SCIPY_OPTIMIZE = True
except Exception:
    _HAS_SCIPY_OPTIMIZE = False

from CubeFit.hdf5_manager import open_h5
from CubeFit import cube_utils as cu

vprint = cu.vprint

SCIENTIFIC_REGULARISATION_MODE = 'ssp_second_difference_neumann'

def _init_worker(blas_threads: int):
    # called once per process; set BLAS env vars
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, blas_threads // 2))

# ------------------------------------------------------------------------------

def _diag_json_default(obj):
    """
    Make NumPy values JSON-serialisable.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)

def _diag_append_jsonl(path: str | None, record: dict) -> None:
    """
    Append one JSON record to a JSONL file.
    """
    if not path:
        return

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_diag_json_default,
            separators=(",", ":")) + "\n")

def _support_stats_1d(vec: np.ndarray) -> dict:
    """
    Summarise within-orbit coefficient diversity.
    """
    v = np.asarray(vec, dtype=np.float64).ravel(order="C")
    v = v[np.isfinite(v) & (v > 0.0)]

    if v.size == 0:
        return {
            "nnz": 0,
            "eff_support": 0.0,
            "entropy": 0.0,
            "top_share": 0.0,
            "gini_proxy": 0.0,
        }

    mass = float(np.sum(v))
    p = v / max(mass, 1e-30)
    eff_support = float(1.0 / np.sum(p * p))
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    top_share = float(np.max(p))
    gini_proxy = float(1.0 - np.sum(p * p))

    return {
        "nnz": int(v.size),
        "eff_support": eff_support,
        "entropy": entropy,
        "top_share": top_share,
        "gini_proxy": gini_proxy,
    }

# ------------------------------------------------------------------------------

def _worker_reduced(h5_path: str, batch: list, keep_idx,
    active_idx: np.ndarray, S_active: np.ndarray, CP: int, P: int,
    component_slot=None, C_physical=None):
    """
    Compute a partial tied reduced Gram matrix with chunk-aware HDF5 reads.

    Active solver columns refer to fitted dynamical components. Each fitted
    column is the sum of the corresponding physical HyperCube columns,

        A_fit[g, p] = sum_c A_phys[c, p],

    for all physical components ``c`` mapped to fitted component ``g``.

    Physical column requests are grouped by their native HDF5 ``(C, P)``
    chunks. Each required model chunk is therefore read only once per spatial
    tile, while all physical columns contained in that chunk are accumulated
    into their appropriate fitted active columns.

    Parameters
    ----------
    h5_path : str
        HyperCube HDF5 path.
    batch : list
        Spatial ``(s0, s1)`` tile ranges assigned to this worker.
    keep_idx : ndarray or None
        Wavelength indices retained by the spectral mask.
    active_idx : ndarray
        Active fitted-column indices in C-major, P-minor order.
    S_active : ndarray
        Column scalings corresponding to ``active_idx``.
    CP : int
        Total number of fitted model columns.
    P : int
        Number of SSP populations per fitted component.
    component_slot : ndarray, optional
        Fitted-component index associated with each physical component.
    C_physical : int, optional
        Number of physical components in the stored HyperCube.

    Returns
    -------
    ATA_loc : ndarray
        Partial scaled reduced Gram matrix with shape ``(k, k)``.
    ATy_loc : None
        Retained for API compatibility. The canonical global ``ATy_scaled``
        vector is used by the caller.

    Raises
    ------
    ValueError
        If active indices, scalings, or component mappings are inconsistent.
    RuntimeError
        If the stored HyperCube dimensions are inconsistent with the supplied
        physical component mapping.

    Examples
    --------
    Called internally by ``streamActiveSetNNLS``.
    """
    active_idx = np.asarray(active_idx, dtype=np.int64).ravel(order="C")
    S_active = np.asarray(S_active, dtype=np.float64).ravel(order="C")
    k = int(active_idx.size)

    if S_active.size != k:
        raise ValueError(f"S_active has size {S_active.size}; expected {k}.")
    if np.any(active_idx < 0) or np.any(active_idx >= CP):
        raise ValueError("active_idx contains invalid fitted-column indices.")
    if CP % P != 0:
        raise ValueError(f"CP={CP} is inconsistent with P={P}.")

    C_fit = CP//P
    gg_arr = active_idx//P
    pp_arr = active_idx % P
    ATA_loc = np.zeros((k, k), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]

        if M.ndim != 4:
            raise RuntimeError(
                f"Expected four-dimensional HyperCube; received {M.shape}.")

        C_disk = int(M.shape[1])
        if int(M.shape[2]) != P:
            raise RuntimeError(
                f"HyperCube P={M.shape[2]} does not match P={P}.")

        if C_physical is None:
            C_physical = C_disk
        else:
            C_physical = int(C_physical)

        if C_physical != C_disk:
            raise RuntimeError(f"HyperCube C={C_disk} does not match physical "
                f"C={C_physical}.")

        if component_slot is None:
            component_slot = np.arange(C_physical, dtype=np.int64)
        else:
            component_slot = np.asarray(component_slot,
                dtype=np.int64).ravel(order="C")

        if component_slot.size != C_physical:
            raise ValueError(f"component_slot has size {component_slot.size}; "
                f"expected {C_physical}.")
        if np.any(component_slot < 0) or np.any(component_slot >= C_fit):
            raise ValueError(
                "component_slot contains invalid fitted-component indices.")

        if M.chunks is None:
            C_chunk, P_chunk = 1, P
        else:
            C_chunk, P_chunk = int(M.chunks[1]), int(M.chunks[2])

        # Map physical column requests onto native HDF5 (C, P) chunks.
        requests = {}
        for jj, (gg, pp) in enumerate(zip(gg_arr, pp_arr)):
            members = np.flatnonzero(component_slot == gg)
            if members.size == 0:
                raise RuntimeError(
                    f"Fitted component {gg} has no physical members.")

            p_chunk_id = int(pp)//P_chunk
            for cc in members:
                cc = int(cc)
                key = (cc//C_chunk, p_chunk_id)
                requests.setdefault(key, []).append((jj, cc, int(pp)))

        # Read each required physical chunk once per spatial tile.
        for s0, s1 in batch:
            L_keep = int(keep_idx.size) if keep_idx is not None else M.shape[3]
            A_act = np.zeros(((s1-s0)*L_keep, k), dtype=np.float64)

            for (c_id, p_id), chunk_requests in requests.items():
                c0, p0 = c_id*C_chunk, p_id*P_chunk
                c1 = min(c0+C_chunk, C_physical)
                p1 = min(p0+P_chunk, P)

                slab = np.asarray(M[s0:s1, c0:c1, p0:p1, :],
                    dtype=np.float32, order="C")
                if keep_idx is not None:
                    slab = slab[:, :, :, keep_idx]

                for jj, cc, pp in chunk_requests:
                    lc, lp = cc-c0, pp-p0
                    A_act[:, jj] += S_active[jj]*slab[:, lc, lp, :].reshape(-1)

                del slab

            ATA_loc += A_act.T@A_act
            del A_act

    return ATA_loc, None

# ------------------------------------------------------------------------------

def _worker_ATAz(h5_path: str, batch: list, keep_idx, z: np.ndarray, CP: int,
    S_flat: np.ndarray, C: int, P: int, component_slot=None,
    C_physical=None):
    """
    Compute the exact quantity A^T A (S z) for a batch of tiles, but with
    much lower peak memory than the original implementation.

    This preserves the original mathematics:

        z_s = (S_flat * z).reshape(C, P)

        v = A @ z_s_flat
        g = A.T @ v
        partial += (S_flat.reshape(C, P) * g).reshape(-1)

    The only difference is that the model cube is read orbit-by-orbit instead
    of loading the full (Sblk, C, P, Lk) tile into memory.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    batch : list[tuple[int, int]]
        List of (s0, s1) tile ranges assigned to this worker.
    keep_idx : ndarray or None
        Wavelength indices to keep after masking, or None for full wavelength
        coverage.
    z : ndarray, shape (C * P,)
        Current reduced NNLS variable in z-space.
    CP : int
        Total number of columns, equal to C * P.
    S_flat : ndarray, shape (C * P,)
        Column scaling vector.
    C : int
        Number of orbit components.
    P : int
        Number of populations per orbit.

    Returns
    -------
    partial : ndarray, shape (C * P,)
        Contribution of this worker to A^T A (S z).
    """
    t_read = 0.0
    t_forward = 0.0
    t_transpose = 0.0

    z = np.asarray(z, dtype=np.float64).ravel(order="C")
    S_flat = np.asarray(S_flat, dtype=np.float64).ravel(order="C")

    if z.size != CP:
        raise ValueError(f"z has size {z.size}, expected CP={CP}.")
    if S_flat.size != CP:
        raise ValueError(f"S_flat has size {S_flat.size}, expected CP={CP}.")
    if CP != C * P:
        raise ValueError(f"CP={CP} is inconsistent with C*P={C * P}.")

    # Same scaling as the original code.
    z_cp = (S_flat * z).reshape(C, P)

    partial = np.zeros((CP,), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]

        if int(M.shape[2]) != P:
            raise RuntimeError(
                f"Model population dimension {M.shape[2]} != expected P={P}.")

        C_chunk = int(M.chunks[1]) if M.chunks else 1
        P_chunk = int(M.chunks[2]) if M.chunks else P
        Lk = int(keep_idx.size) if keep_idx is not None else int(M.shape[3])

        for s0, s1 in batch:
            Sblk = s1-s0
            v = np.zeros((Sblk, Lk), dtype=np.float64)

            # v = A_group @ z.
            for cc in range(C_physical):
                gg = component_slot[cc]
                for p0 in range(0, P, P_chunk):
                    p1 = min(p0+P_chunk, P)
                    slab = np.asarray(M[s0:s1, cc, p0:p1, :], dtype=np.float32)
                    if keep_idx is not None:
                        slab = slab[:, :, keep_idx]
                    v += np.tensordot(slab, z_cp[gg, p0:p1], axes=([1], [0]))

            # A_group.T @ v.
            partial_cp = partial.reshape(C, P)
            scale_cp = S_flat.reshape(C, P)
            for cc in range(C_physical):
                gg = component_slot[cc]
                for p0 in range(0, P, P_chunk):
                    p1 = min(p0+P_chunk, P)
                    slab = np.asarray(M[s0:s1, cc, p0:p1, :], dtype=np.float32)
                    if keep_idx is not None:
                        slab = slab[:, :, keep_idx]
                    g = np.tensordot(slab, v, axes=([0, 2], [0, 1]))
                    partial_cp[gg, p0:p1] += scale_cp[gg, p0:p1]*g

                    del slab

    print(f"[ATAz worker {os.getpid()}] read={t_read:.1f}s "
        f"forward={t_forward:.1f}s transpose={t_transpose:.1f}s",
        flush=True)
    return partial

# ------------------------------------------------------------------------------

def _worker_ATAz_from_tuple(args):
    """
    Thin wrapper to allow executor.map with tuple-packed arguments.
    Must be top-level to be pickleable.
    """
    return _worker_ATAz(*args)

# ------------------------------------------------------------------------------

def _worker_build_products(h5_path: str, batch: list, keep_idx, C: int,
    P: int, component_slot=None, C_fit=None):
    """
    Compute grouped A.T@y and exact grouped column energies for spatial tiles.

    The dominant HyperCube pass follows the native physical HDF5 component and
    population chunks. Physical A.T@y values and column energies are
    accumulated directly into their fitted component groups.

    For a tied fitted component ``g``, whose physical members are ``c``, the
    effective model column is

        A_g,p = sum_c A_c,p.

    Linearity means its data product requires no additional model pass,

        A_g,p.T @ y = sum_c A_c,p.T @ y.

    Its column energy additionally requires the physical-component cross
    terms,

        ||A_g,p||^2
            = sum_c ||A_c,p||^2
            + 2*sum_{c<d} A_c,p.T @ A_d,p.

    These cross terms are computed only for non-singleton tied groups by
    forming one memory-bounded grouped column at a time. The resulting
    products therefore correspond exactly to the reduced tied design matrix
    without materializing that matrix.

    Parameters
    ----------
    h5_path : str
        HyperCube HDF5 path.
    batch : list
        Spatial ``(s0, s1)`` tile ranges assigned to this worker.
    keep_idx : ndarray or None
        Wavelength indices retained by the spectral mask.
    C : int
        Number of physical dynamical components in the HyperCube.
    P : int
        Number of SSP populations per physical component.
    component_slot : ndarray, optional
        Fitted-component index associated with each physical component, with
        shape ``(C,)``. If None, every physical component is independent.
    C_fit : int, optional
        Number of independent fitted components. If None, defaults to ``C``.

    Returns
    -------
    ATy_loc : ndarray
        Partial grouped ``A.T@y`` vector with shape ``(C_fit*P,)``.
    D_loc : ndarray
        Partial exact grouped column energies with shape ``(C_fit, P)``.
    n_tiles : int
        Number of spatial tiles processed by this worker.

    Raises
    ------
    RuntimeError
        If the HyperCube dimensions, component mapping, or fitted component
        count are inconsistent.

    Examples
    --------
    Called internally by ``solve_streaming_nnls`` through
    ``_worker_build_products_from_tuple``.
    """
    C = int(C)
    P = int(P)

    if component_slot is None:
        component_slot = np.arange(C, dtype=np.int64)
    else:
        component_slot = np.asarray(
            component_slot, dtype=np.int64).ravel(order="C")

    if C_fit is None:
        C_fit = C

    C_fit = int(C_fit)

    if C <= 0 or P <= 0 or C_fit <= 0:
        raise RuntimeError(
            f"Invalid build dimensions: C={C}, P={P}, C_fit={C_fit}.")

    if component_slot.size != C:
        raise RuntimeError(
            f"component_slot has size {component_slot.size}; expected {C}.")

    if np.any(component_slot < 0) or np.any(component_slot >= C_fit):
        raise RuntimeError(
            "component_slot contains invalid fitted-component indices.")

    # Every fitted component must contain at least one physical component.
    group_counts = np.bincount(component_slot, minlength=C_fit)

    if group_counts.size != C_fit or np.any(group_counts == 0):
        missing = np.flatnonzero(group_counts == 0)
        raise RuntimeError(
            f"Fitted component groups without physical members: "
            f"{missing.tolist()}.")

    # Only these groups require the additional cross-term calculation.
    tied_groups = [np.flatnonzero(component_slot == gg)
        for gg in np.flatnonzero(group_counts > 1)]

    ATy_loc = np.zeros((C_fit, P), dtype=np.float64)
    D_loc = np.zeros((C_fit, P), dtype=np.float64)

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        if M.ndim != 4:
            raise RuntimeError(
                f"Expected four-dimensional HyperCube; received {M.shape}.")

        if int(M.shape[1]) != C or int(M.shape[2]) != P:
            raise RuntimeError(
                "HyperCube dimensions are inconsistent with "
                f"C={C}, P={P}: {M.shape}.")

        C_chunk = int(M.chunks[1]) if M.chunks else 1
        P_chunk = int(M.chunks[2]) if M.chunks else P

        # Cross-term construction uses float64 grouped columns. Bound this
        # independently of the native HDF5 population chunk to keep aggregate
        # worker memory modest on high-process-count nodes.
        P_cross = min(P_chunk, 16)

        for s0, s1 in batch:
            Y = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C")

            if keep_idx is not None:
                Y = Y[:, keep_idx]

            # ----------------------------------------------------------
            # Main physical pass.
            #
            # Preserve native HDF5 C/P chunk access. ATy is linear in the
            # physical columns, while the diagonal energy contribution is
            # accumulated before the tied-group cross-term correction below.
            # ----------------------------------------------------------
            for c0 in range(0, C, C_chunk):
                c1 = min(c0+C_chunk, C)
                slots = component_slot[c0:c1]

                for p0 in range(0, P, P_chunk):
                    p1 = min(p0+P_chunk, P)

                    slab = np.asarray(M[s0:s1, c0:c1, p0:p1, :],
                        dtype=np.float32, order="C")

                    if keep_idx is not None:
                        slab = slab[:, :, :, keep_idx]

                    aty = np.einsum("scpl,sl->cp", slab, Y,
                        dtype=np.float64, optimize=True)

                    energy = np.einsum("scpl,scpl->cp", slab, slab,
                        dtype=np.float64, optimize=True)

                    # C_chunk is normally small. np.add.at is important here
                    # because multiple physical components in the same chunk
                    # may map onto the same fitted component.
                    np.add.at(ATy_loc[:, p0:p1], slots, aty)

                    np.add.at(D_loc[:, p0:p1], slots, energy)

                    del slab, aty, energy

            # ----------------------------------------------------------
            # Exact tied-group energy correction.
            #
            # The main pass has already accumulated
            #
            #     sum_c ||A_c,p||^2.
            #
            # For each non-singleton group compute
            #
            #     ||sum_c A_c,p||^2 - sum_c ||A_c,p||^2,
            #
            # which is exactly the missing pairwise cross-term contribution.
            # ----------------------------------------------------------
            for members in tied_groups:
                gg = int(component_slot[members[0]])

                for p0 in range(0, P, P_cross):
                    p1 = min(p0+P_cross, P)

                    grouped = np.zeros((s1-s0, p1-p0, Y.shape[1]),
                        dtype=np.float64)

                    physical_energy = np.zeros(p1-p0, dtype=np.float64)

                    for cc in members:
                        cc = int(cc)

                        slab = np.asarray(M[s0:s1, cc, p0:p1, :],
                            dtype=np.float32, order="C")

                        if keep_idx is not None:
                            grouped += slab[:, :, keep_idx]

                            physical_energy += np.einsum("spl,spl->p",
                                slab[:, :, keep_idx], slab[:, :, keep_idx],
                                dtype=np.float64, optimize=True)
                        else:
                            grouped += slab

                            physical_energy += np.einsum("spl,spl->p",
                                slab, slab, dtype=np.float64, optimize=True)

                        del slab

                    grouped_energy = np.einsum("spl,spl->p", grouped,
                        grouped, dtype=np.float64, optimize=True)

                    correction = grouped_energy-physical_energy

                    # Roundoff can make a mathematically non-negative grouped
                    # energy correction slightly negative. Do not clip the
                    # correction itself: negative cross terms are physically
                    # legitimate. The final grouped energy is checked below.
                    D_loc[gg, p0:p1] += correction

                    del grouped, physical_energy, grouped_energy, correction

            del Y

    # The effective grouped column energy must itself be non-negative. Allow
    # only floating-point-scale excursions below zero.
    energy_scale = max(1.0, float(np.max(np.abs(D_loc))) if D_loc.size else 1.0)
    negative_tol = 100.0*np.finfo(np.float64).eps*energy_scale

    if np.any(D_loc < -negative_tol):
        idx = np.argwhere(D_loc < -negative_tol)[0]
        raise RuntimeError(
            "Grouped column energy became materially negative at "
            f"(component={int(idx[0])}, population={int(idx[1])}): "
            f"{D_loc[tuple(idx)]:.6e}.")

    D_loc[D_loc < 0.0] = 0.0

    if not np.all(np.isfinite(ATy_loc)):
        raise RuntimeError(
            "Grouped A.T@y contains non-finite values.")

    if not np.all(np.isfinite(D_loc)):
        raise RuntimeError(
            "Grouped column energy contains non-finite values.")

    return ATy_loc.ravel(order="C"), D_loc, int(len(batch))

# ------------------------------------------------------------------------------

def _worker_build_products_from_tuple(args):
    """
    Tuple wrapper for the parallel HyperCube build worker.
    """
    return _worker_build_products(*args)

# ------------------------------------------------------------------------------

@dataclass
class MPConfig:
    processes: int = 2
    blas_threads: int = 8
    apply_mask: bool = True
    dset_slots: int = 1_000_003
    dset_bytes: int = 256 * 1024**2
    dset_w0: float = 0.90
    s_tile_override: Optional[int] = None
    verbose: bool = True

# ---------------------- Small pool utilities --------------------------

def _pool_ping() -> int:
    return 1

def _pool_ok(pool, timeout: float = 5.0) -> bool:
    """
    Returns True if a trivial task round-trips within `timeout`.
    If it times out or raises, the pool is considered unhealthy.
    """
    try:
        res = pool.apply_async(_pool_ping)
        return res.get(timeout=timeout) == 1
    except Exception:
        return False

# ------------------------------------------------------------------------------

def rmse_proxy_subset(
    h5_path,
    x_CP,
    tile_ranges,
    keep_idx,
    inv_cp_flux_ref,
    w_lam_sqrt,
):
    """
    Compute RMSE proxy over a subset of tiles in the SAME weighted space
    used by the gradient (i.e. apply √w_λ if present). Returns sqrt(ssq/n).
    """

    ssq = 0.0
    nres = 0

    x_flat = x_CP.reshape(-1)

    # Only show tqdm if more than 1 tile
    use_bar = len(tile_ranges) > 1

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M  = f["/HyperCube/models"]

        iterator = tile_ranges
        if use_bar:
            iterator = tqdm(
                tile_ranges,
                desc="[RMSE-proxy]",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )

        for (s0, s1) in iterator:

            Y = np.asarray(DC[s0:s1, :], dtype=np.float64)
            if keep_idx is not None:
                Y = Y[:, keep_idx]

            A = np.asarray(M[s0:s1, :, :, :], dtype=np.float64)

            if keep_idx is not None:
                A = A[:, :, :, keep_idx]

            if inv_cp_flux_ref is not None:
                A *= inv_cp_flux_ref[None, :, :, None]

            Sblk, C, P, Lk = A.shape

            # reshape for single BLAS
            A2 = A.transpose(0, 3, 1, 2).reshape(Sblk * Lk, C * P)

            yhat_flat = A2 @ x_flat
            yhat = yhat_flat.reshape(Sblk, Lk)

            R = Y - yhat
            if not np.all(np.isfinite(R)):
                R = np.nan_to_num(R, copy=False)

            if w_lam_sqrt is not None:
                R *= w_lam_sqrt[None, :]
                ssq += float(np.sum(R * R))
            else:
                ssq += float(np.sum(R * R))

            nres += R.size

    return float(np.sqrt(ssq / max(nres, 1)))

# ------------------------------------------------------------------------------

def _canon_orbit_weights(orbit_weights, C: int, P: int) -> np.ndarray | None:
    """
    Return a unit-sum orbit-shape vector, or None if disabled.

    Parameters
    ----------
    orbit_weights : array-like or None
        Orbit weights. If ``None``, no orbit constraint is used.
        Accepted shapes are ``(C,)`` or ``(C * P,)``.
    C : int
        Number of components.
    P : int
        Number of population coefficients per component.

    Returns
    -------
    ndarray or None
        Unit-sum component-weight vector with shape ``(C,)``, or ``None``
        if ``orbit_weights`` is ``None``.

    Raises
    ------
    ValueError
        If the supplied weights have incompatible dimensions, contain
        invalid values, or have zero total mass.

    Examples
    --------
    >>> _canon_orbit_weights(None, C=20, P=360) is None
    True
    """
    if orbit_weights is None:
        return None

    w = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")

    if w.size == C:
        pass
    elif w.size == C * P:
        w = w.reshape(C, P).sum(axis=1)
    else:
        raise ValueError(f"orbit_weights length {w.size} incompatible with "
            f"C={C}, P={P}. Expected C or C*P.")

    if not np.all(np.isfinite(w)):
        raise ValueError("orbit_weights must contain only finite values.")

    if np.any(w < 0.0):
        raise ValueError("orbit_weights must be nonnegative.")

    w_sum = float(np.sum(w))

    if w_sum <= 0.0:
        raise ValueError("orbit_weights must have positive total mass.")

    return w / w_sum

# ------------------------------------------------------------------------------

def _canon_component_donor(component_donor, C: int) -> np.ndarray:
    """
    Validate and canonicalize component population ties.

    Parameters
    ----------
    component_donor : array-like or None
        Zero-based donor component for each physical component. Anchors point
        to themselves. None leaves all components independent.
    C : int
        Number of physical dynamical components.

    Returns
    -------
    donor : ndarray
        Canonical donor indices with shape ``(C,)``.

    Raises
    ------
    ValueError
        If the mapping has incorrect dimensions, invalid indices, or cycles.

    Examples
    --------
    >>> _canon_component_donor([0, 0, 2], 3)
    array([0, 0, 2])
    """
    if component_donor is None:
        return np.arange(C, dtype=np.int64)

    donor = np.asarray(component_donor, dtype=np.int64).ravel().copy()
    if donor.size != C:
        raise ValueError(f"component_donor has size {donor.size}; expected {C}.")
    if np.any(donor < 0) or np.any(donor >= C):
        raise ValueError("component_donor contains invalid component indices.")

    canonical = np.empty(C, dtype=np.int64)
    for cc in range(C):
        current = cc
        visited = set()
        while donor[current] != current:
            if current in visited:
                raise ValueError("component_donor contains a donor cycle.")
            visited.add(current)
            current = int(donor[current])
        canonical[cc] = current

    return canonical

# ------------------------------------------------------------------------------

def _get_known_zero_mask(h5_path: str, C: int, P: int) -> np.ndarray:
    """
    Return the structural zero-energy exclusion mask for the solver.

    The mask is derived directly from ``/HyperCube/col_energy``. A coefficient
    is excluded only when its global model-column energy is exactly zero.

    Parameters
    ----------
    h5_path : str
        Path to the CubeFit HDF5 file.
    C : int
        Number of orbit components.
    P : int
        Number of population columns per component.

    Returns
    -------
    known_zero : ndarray, shape (C, P)
        Boolean structural exclusion mask.

    Raises
    ------
    RuntimeError
        If column energy is absent, invalid, or dimensionally inconsistent.

    Examples
    --------
    >>> known_zero = _get_known_zero_mask(h5_path, C=20, P=245)
    """
    with open_h5(h5_path, role="reader") as f:
        if "/HyperCube/col_energy" not in f:
            raise RuntimeError(
                "Missing /HyperCube/col_energy; cannot determine structural "
                "known-zero columns.")

        energy = np.asarray(
            f["/HyperCube/col_energy"][...], dtype=np.float64)

    if energy.shape != (C, P):
        raise RuntimeError(
            f"col_energy has shape {energy.shape}; expected {(C, P)}.")

    if np.any(~np.isfinite(energy)):
        raise RuntimeError("col_energy contains non-finite values.")
    if np.any(energy < 0.0):
        raise RuntimeError("col_energy contains negative values.")

    return energy <= 0.0

# ------------------------------------------------------------------------------

def diffuse_seed_full_CP(
    seed_cp: np.ndarray,
    sigma_c: float,
    sigma_p: float,
    *,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Diffuse a sparse NNLS seed on the full (C,P) grid using a separable
    Gaussian kernel in orbit (c) and population (p).

    Parameters
    ----------
    seed_cp : ndarray (C,P)
        NNLS seed with zeros in non-sampled locations.
    sigma_c : float
        Gaussian width in orbit index units.
    sigma_p : float
        Gaussian width in population index units.

    Returns
    -------
    seed_support : ndarray (C,P)
        Smooth, positive field encoding proximity to NNLS seed support.
        Normalized per-row to max=1.
    """
    C, P = seed_cp.shape

    # Early exit: no seed information
    if not np.any(seed_cp > 0):
        return np.zeros_like(seed_cp)

    # Orbit kernel
    c_idx = np.arange(C, dtype=np.float64)
    dc = c_idx[:, None] - c_idx[None, :]
    Kc = np.exp(-0.5 * (dc / max(sigma_c, eps)) ** 2)

    # Population kernel
    p_idx = np.arange(P, dtype=np.float64)
    dp = p_idx[:, None] - p_idx[None, :]
    Kp = np.exp(-0.5 * (dp / max(sigma_p, eps)) ** 2)

    # Convolution: Kc @ seed @ Kp
    seed_support = Kc @ seed_cp @ Kp

    # Normalize per orbit so max=1 (scale-free gating)
    row_max = seed_support.max(axis=1, keepdims=True)
    seed_support = np.divide(
        seed_support,
        row_max,
        out=np.zeros_like(seed_support),
        where=row_max > eps,
    )

    return seed_support

# ------------------------------------------------------------------------------

def _projected_gradient_nnls(A, b, max_iter=200, tol=1e-8):
    A = np.asarray(A, dtype=np.float64, order="C")
    b = np.asarray(b, dtype=np.float64, order="C")
    m, n = A.shape
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    ATA = A.T @ A
    ATb = A.T @ b
    x = np.zeros((n,), dtype=np.float64)
    L = np.linalg.norm(ATA, 2)
    if not np.isfinite(L) or L <= 0:
        L = 1.0
    step = 1.0 / L
    for k in range(max_iter):
        grad = ATA @ x - ATb
        x_new = x - step * grad
        x_new[x_new < 0] = 0.0
        if np.linalg.norm(x_new - x) < tol * (1.0 + np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    return x

# ------------------------------------------------------------------------------

def _nnls_from_quadratic(
    ATA: np.ndarray,
    ATy: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Solve a small dense nonnegative quadratic problem.

    The problem is

        minimize

            0.5 * x.T @ ATA @ x - ATy.T @ x

        subject to

            x >= 0.

    When the quadratic matrix is positive definite, the problem is converted
    exactly to a conventional nonnegative least-squares problem,

        minimize ||R @ x - q||_2,

    where ``ATA = R.T @ R`` and ``R.T @ q = ATy``. This is then solved using
    ``scipy.optimize.nnls``.

    If Cholesky factorization fails because the quadratic is numerically
    positive-semidefinite or singular, the function falls back to projected
    gradient descent. The fallback uses the projected KKT residual rather than
    solution-step size alone to determine convergence.

    Parameters
    ----------
    ATA : ndarray, shape (n, n)
        Symmetric positive-semidefinite quadratic matrix.
    ATy : ndarray, shape (n,)
        Linear term of the quadratic objective.
    x0 : ndarray, optional
        Nonnegative initial solution used by the projected-gradient fallback.
        It is not required by the preferred SciPy NNLS path.
    max_iter : int, optional
        Maximum number of iterations for SciPy NNLS and the projected-gradient
        fallback.
    tol : float, optional
        Relative projected-KKT tolerance used by the fallback solver.

    Returns
    -------
    x : ndarray, shape (n,)
        Nonnegative solution of the quadratic problem.

    Raises
    ------
    ValueError
        If the supplied arrays have inconsistent dimensions or contain
        non-finite values.
    RuntimeError
        If the projected-gradient fallback fails to reach the requested KKT
        tolerance.

    Examples
    --------
    >>> x = _nnls_from_quadratic(
    ...     ATA_sub_reg, ATy_sub, x0=z_old_active,
    ...     max_iter=10000, tol=1e-12)
    """
    H = np.asarray(ATA, dtype=np.float64, order="C")
    b = np.asarray(ATy, dtype=np.float64).ravel(order="C")

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(
            "ATA must be a square two-dimensional array.")

    n = int(H.shape[0])

    if b.size != n:
        raise ValueError(
            f"ATy has size {b.size}; expected {n}.")

    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(b)):
        raise ValueError(
            "ATA and ATy must contain only finite values.")

    # Numerical accumulation can introduce tiny asymmetries in the streamed
    # reduced normal equations. Symmetrize before attempting factorization.
    H = 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    # Preferred path: exact conversion to a conventional NNLS problem.
    #
    # If H is positive definite,
    #
    #     H = R.T @ R
    #
    # and choosing q such that
    #
    #     R.T @ q = b
    #
    # gives
    #
    #     0.5 * ||R @ x - q||^2
    #
    #       = 0.5 * x.T @ H @ x - b.T @ x + constant.
    #
    # The nonnegative minimizer is therefore exactly the minimizer of the
    # original quadratic problem. This avoids the mathematically incorrect
    # operation of solving the unconstrained system and clipping x afterward.
    # ------------------------------------------------------------------
    if _HAS_SCIPY_OPTIMIZE:
        try:
            R = np.linalg.cholesky(H).T
            q = np.linalg.solve(R.T, b)

            x, _ = nnls(
                R, q, maxiter=max(3 * n, int(max_iter)))

            x = np.asarray(x, dtype=np.float64)

            if np.all(np.isfinite(x)) and np.all(x >= -1e-12):
                return np.maximum(x, 0.0)

        except np.linalg.LinAlgError:
            # A numerically singular or semidefinite reduced Hessian is
            # expected occasionally for highly degenerate SSP columns.
            pass
        except Exception:
            # Preserve the projected-gradient fallback if SciPy NNLS itself
            # cannot solve the factored problem.
            pass

    # ------------------------------------------------------------------
    # Fallback: projected gradient directly on the quadratic.
    #
    # Unlike the previous implementation, convergence is certified using
    # the projected KKT residual. A tiny change in x is not sufficient to
    # declare convergence in a severely ill-conditioned reduced system.
    # ------------------------------------------------------------------
    if x0 is None:
        x = np.zeros((n,), dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).ravel(order="C")

        if x.size != n:
            raise ValueError(
                f"x0 has size {x.size}; expected {n}.")

        if not np.all(np.isfinite(x)):
            raise ValueError(
                "x0 must contain only finite values.")

        x = np.maximum(x, 0.0)

    # Since H is small and dense, obtain the Lipschitz constant directly
    # rather than using a short power iteration. The projected-gradient
    # step 1/L is then guaranteed to be non-expansive for a PSD quadratic.
    try:
        eigvals = np.linalg.eigvalsh(H)
        L = float(np.max(eigvals))
    except np.linalg.LinAlgError:
        L = float(np.linalg.norm(H, ord=2))

    if not np.isfinite(L) or L <= 0.0:
        L = 1.0

    step = 1.0 / L

    # Scale the requested tolerance to the natural gradient scale. This is
    # analogous to the relative gradient criterion used by the outer solver.
    grad_scale = max(1.0, float(np.max(np.abs(b))))
    kkt_tol = float(tol) * grad_scale

    for _ in range(int(max_iter)):
        grad = H @ x - b
        x_new = x - step * grad
        np.maximum(x_new, 0.0, out=x_new)

        # For min f(x), x >= 0, the projected-gradient residual
        #
        #     x - P_+(x - grad)
        #
        # vanishes exactly at the NNLS KKT solution.
        grad_new = H @ x_new - b
        projected_grad = x_new - np.maximum(x_new - grad_new, 0.0)
        kkt = float(np.max(np.abs(projected_grad)))

        x[:] = x_new

        if kkt <= kkt_tol:
            return x

    # Never silently return a non-stationary reduced solution. Doing so can
    # make the outer active-set solver report an active-stationarity failure
    # only after another extremely expensive full HyperCube gradient pass.
    grad = H @ x - b
    projected_grad = x - np.maximum(x - grad, 0.0)
    final_kkt = float(np.max(np.abs(projected_grad)))

    if not np.all(np.isfinite(x)) or final_kkt > kkt_tol:
        raise RuntimeError(
            "Reduced NNLS failed to reach KKT tolerance: "
            f"kkt={final_kkt:.6e}, tol={kkt_tol:.6e}, "
            f"iterations={int(max_iter)}.")

    return np.maximum(x, 0.0)

# ------------------------------------------------------------------------------

def _ssp_smoothness_hessian(umetals: np.ndarray, uages: np.ndarray,
    ualphas: np.ndarray) -> np.ndarray:
    """
    Construct the N-dimensional SSP-grid curvature Hessian with
    zero-normal-gradient boundary closure.

    The regularisation penalizes second differences independently along
    metallicity, age, and alpha enhancement. The physical SSP coordinates
    are used explicitly, so non-uniform grid spacing is handled correctly.

    For three consecutive interior coordinates

        u0 < u1 < u2,

    the local curvature stencil is proportional to the change in first
    derivative,

        (x2 - x1)/(u2 - u1) - (x1 - x0)/(u1 - u0),

    giving coefficients

        [1/h0, -(1/h0 + 1/h1), 1/h1],

    where

        h0 = u1 - u0
        h1 = u2 - u1.

    At each lower and upper SSP-library boundary, the normal first
    derivative is additionally penalized,

        (x1 - x0)/(u1 - u0),

    and

        (xN - xN-1)/(uN - uN-1).

    This is a zero-normal-gradient, or Neumann, boundary closure. It does
    not penalize population mass merely for lying on an SSP-library
    boundary. Instead, it penalizes a coefficient field whose slope points
    through the finite edge of the library.

    Every stencil is normalized to unit Euclidean norm before being added
    to the Hessian. This retains the physical non-uniform-grid geometry
    while preventing the numerical units and ranges of age, metallicity,
    and alpha from setting arbitrary relative regularisation strengths.

    The complete objective for one dynamical component is

        0.5 * x.T @ Q @ x,

    where Q contains the sum of metallicity, age, and alpha curvature and
    boundary-normal-gradient contributions.

    The operator is symmetric positive semidefinite. A spatially uniform
    population remains an exact null mode. Linear SSP-coordinate fields are
    no longer null modes because a nonzero linear slope has a nonzero
    derivative at the finite SSP-library boundaries.

    Parameters
    ----------
    umetals : ndarray
        Strictly increasing SSP metallicity coordinates.
    uages : ndarray
        Strictly increasing SSP age coordinates.
    ualphas : ndarray
        Strictly increasing SSP alpha-enhancement coordinates.

    Returns
    -------
    Q : ndarray
        Dense SSP regularisation Hessian with shape ``(P, P)``, where

            P = n_metals * n_ages * n_alphas.

        Population indexing follows C-order,

            (metallicity, age, alpha).

    Raises
    ------
    ValueError
        If any SSP coordinate array is empty, contains non-finite values,
        or is not strictly increasing.
    RuntimeError
        If an invalid regularisation stencil is generated.

    Examples
    --------
    >>> Q = _ssp_smoothness_hessian(
    ...     np.array([-1.49, -0.96, -0.66, -0.35, 0.06, 0.15, 0.26]),
    ...     np.array([3., 6., 8., 9., 10., 11., 12., 13., 14.]),
    ...     np.array([-0.2, 0.0, 0.2, 0.4, 0.6]))
    >>> Q.shape
    (315, 315)
    """
    umetals = np.asarray(
        umetals, dtype=np.float64).ravel(order="C")
    uages = np.asarray(
        uages, dtype=np.float64).ravel(order="C")
    ualphas = np.asarray(
        ualphas, dtype=np.float64).ravel(order="C")

    for name, values in (
        ("umetals", umetals),
        ("uages", uages),
        ("ualphas", ualphas),
    ):
        if values.size == 0:
            raise ValueError(
                f"{name} must not be empty.")

        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"{name} contains non-finite values.")

        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError(
                f"{name} must be strictly increasing.")

    n_metals = int(umetals.size)
    n_ages = int(uages.size)
    n_alphas = int(ualphas.size)

    shape = (n_metals, n_ages, n_alphas)
    P = int(np.prod(shape))

    Q = np.zeros((P, P), dtype=np.float64)

    def _index(iz, ia, ialpha):
        return np.ravel_multi_index(
            (iz, ia, ialpha), shape, order="C")

    def _add_coefficients(indices, coeff):
        idx = np.asarray(
            indices, dtype=np.int64)

        coeff = np.asarray(
            coeff, dtype=np.float64)

        coeff_norm = float(np.linalg.norm(coeff))

        if not np.isfinite(coeff_norm) or coeff_norm <= 0.0:
            raise RuntimeError(
                "Invalid SSP regularisation stencil.")

        coeff /= coeff_norm

        Q[np.ix_(idx, idx)] += np.outer(
            coeff, coeff)

    def _add_curvature(p0, p1, p2, u0, u1, u2):
        h0 = float(u1-u0)
        h1 = float(u2-u1)

        if h0 <= 0.0 or h1 <= 0.0:
            raise ValueError(
                "SSP coordinates must be strictly increasing.")

        _add_coefficients(
            (p0, p1, p2),
            (
                1.0/h0,
                -(1.0/h0+1.0/h1),
                1.0/h1,
            ),
        )

    def _add_boundary(p0, p1, u0, u1):
        h = float(u1-u0)

        if h <= 0.0:
            raise ValueError(
                "SSP coordinates must be strictly increasing.")

        _add_coefficients(
            (p0, p1),
            (-1.0/h, 1.0/h),
        )

    # ------------------------------------------------------------------
    # Metallicity.
    # ------------------------------------------------------------------
    if n_metals >= 2:
        for ia in range(n_ages):
            for ialpha in range(n_alphas):
                _add_boundary(
                    _index(0, ia, ialpha),
                    _index(1, ia, ialpha),
                    umetals[0],
                    umetals[1])

                _add_boundary(
                    _index(n_metals-2, ia, ialpha),
                    _index(n_metals-1, ia, ialpha),
                    umetals[-2],
                    umetals[-1])

    if n_metals >= 3:
        for iz in range(1, n_metals-1):
            for ia in range(n_ages):
                for ialpha in range(n_alphas):
                    _add_curvature(
                        _index(iz-1, ia, ialpha),
                        _index(iz, ia, ialpha),
                        _index(iz+1, ia, ialpha),
                        umetals[iz-1],
                        umetals[iz],
                        umetals[iz+1])

    # ------------------------------------------------------------------
    # Age.
    # ------------------------------------------------------------------
    if n_ages >= 2:
        for iz in range(n_metals):
            for ialpha in range(n_alphas):
                _add_boundary(
                    _index(iz, 0, ialpha),
                    _index(iz, 1, ialpha),
                    uages[0],
                    uages[1])

                _add_boundary(
                    _index(iz, n_ages-2, ialpha),
                    _index(iz, n_ages-1, ialpha),
                    uages[-2],
                    uages[-1])

    if n_ages >= 3:
        for iz in range(n_metals):
            for ia in range(1, n_ages-1):
                for ialpha in range(n_alphas):
                    _add_curvature(
                        _index(iz, ia-1, ialpha),
                        _index(iz, ia, ialpha),
                        _index(iz, ia+1, ialpha),
                        uages[ia-1],
                        uages[ia],
                        uages[ia+1])

    # ------------------------------------------------------------------
    # Alpha enhancement.
    # ------------------------------------------------------------------
    if n_alphas >= 2:
        for iz in range(n_metals):
            for ia in range(n_ages):
                _add_boundary(
                    _index(iz, ia, 0),
                    _index(iz, ia, 1),
                    ualphas[0],
                    ualphas[1])

                _add_boundary(
                    _index(iz, ia, n_alphas-2),
                    _index(iz, ia, n_alphas-1),
                    ualphas[-2],
                    ualphas[-1])

    if n_alphas >= 3:
        for iz in range(n_metals):
            for ia in range(n_ages):
                for ialpha in range(1, n_alphas-1):
                    _add_curvature(
                        _index(iz, ia, ialpha-1),
                        _index(iz, ia, ialpha),
                        _index(iz, ia, ialpha+1),
                        ualphas[ialpha-1],
                        ualphas[ialpha],
                        ualphas[ialpha+1])

    Q = 0.5*(Q+Q.T)

    return Q

# ------------------------------------------------------------------------------

def _solve_equal_mass_nullspace(H: np.ndarray, b: np.ndarray,
    B: np.ndarray, constraint: np.ndarray) -> tuple:
    """
    Solve the equality-constrained quadratic system by null-space elimination.

    Solve

        minimize 0.5*z.T@H@z - b.T@z

    subject to

        B@z - constraint*alpha = 0.

    The equality constraints are eliminated before solving the quadratic,
    avoiding the indefinite saddle-point KKT system.

    Parameters
    ----------
    H : ndarray
        Symmetric coefficient Hessian with shape ``(n, n)``.
    b : ndarray
        Linear objective term with shape ``(n,)``.
    B : ndarray
        Equality-constraint matrix with shape ``(m, n)``.
    constraint : ndarray
        Alpha coefficient for each constraint with shape ``(m,)``.

    Returns
    -------
    z : ndarray
        Equality-constrained coefficient solution.
    alpha : float
        Fitted common component mass.
    lambda_eq : ndarray
        Equality-constraint Lagrange multipliers.
    info : dict
        Numerical diagnostics for the null-space solve.

    Raises
    ------
    np.linalg.LinAlgError
        If the equality constraints are rank deficient or the reduced solve
        cannot produce a finite solution.

    Examples
    --------
    Called internally by ``_orbit_hard_constraint_solve``.
    """
    H = np.asarray(H, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel(order="C")
    B = np.asarray(B, dtype=np.float64)
    constraint = np.asarray(
        constraint, dtype=np.float64).ravel(order="C")

    n = int(H.shape[0])
    m = int(B.shape[0])

    if H.shape != (n, n):
        raise ValueError("H must be square.")
    if b.size != n:
        raise ValueError("b has a size inconsistent with H.")
    if B.shape != (m, n):
        raise ValueError("B has a shape inconsistent with H.")
    if constraint.size != m:
        raise ValueError("constraint has a size inconsistent with B.")

    H = 0.5*(H+H.T)

    # Write the equality system as
    #
    #     E @ q = 0,
    #
    # with q = [z, alpha].
    E = np.empty((m, n+1), dtype=np.float64)
    E[:, :n] = B
    E[:, n] = -constraint

    # Full SVD gives an orthonormal basis for both the constrained directions
    # and the exact null space of E. Since the right-hand side of the
    # constraints is zero, every feasible q is q = N@y.
    _, singular, Vt = np.linalg.svd(E, full_matrices=True)

    if singular.size:
        smax = float(np.max(singular))
        rank_tol = max(E.shape)*np.finfo(np.float64).eps*max(smax, 1.0)
        rank = int(np.count_nonzero(singular > rank_tol))
    else:
        rank_tol = 0.0
        rank = 0

    if rank != m:
        raise np.linalg.LinAlgError(
            f"Equality constraints are rank deficient: rank={rank}, m={m}.")

    N = Vt[rank:, :].T

    if N.shape[1] == 0:
        raise np.linalg.LinAlgError(
            "Equality constraints leave no free null-space direction.")

    Nz = N[:n, :]

    # Alpha has no direct quadratic or linear objective term, so after
    # q=N@y the reduced objective is
    #
    #     0.5*y.T@(Nz.T@H@Nz)@y - (Nz.T@b).T@y.
    Hr = Nz.T @ H @ Nz
    Hr = 0.5*(Hr+Hr.T)
    br = Nz.T @ b

    # Prefer Cholesky. The numerical ridge already present in H should make
    # the reduced Hessian positive definite whenever the data constrain the
    # remaining feasible directions sufficiently.
    solve_method = "cholesky"

    try:
        chol = np.linalg.cholesky(Hr)
        y = np.linalg.solve(chol.T, np.linalg.solve(chol, br))
    except np.linalg.LinAlgError:
        # Degenerate SSP columns can leave essentially null directions. Use a
        # symmetric eigensolve rather than truncating an indefinite KKT matrix.
        solve_method = "eigh"
        evals, evecs = np.linalg.eigh(Hr)

        emax = float(np.max(np.abs(evals))) if evals.size else 0.0
        eig_tol = max(
            1e-14*max(emax, 1.0),
            np.finfo(np.float64).eps*max(Hr.shape)*max(emax, 1.0),
        )

        if np.any(evals < -eig_tol):
            raise np.linalg.LinAlgError(
                "Reduced null-space Hessian is materially indefinite: "
                f"emin={np.min(evals):.6e}, tol={eig_tol:.6e}.")

        keep = evals > eig_tol

        if not np.any(keep):
            raise np.linalg.LinAlgError(
                "Reduced null-space Hessian has no resolved curvature.")

        proj = evecs[:, keep].T @ br
        y = evecs[:, keep] @ (proj/evals[keep])

    q = N @ y
    z = np.asarray(q[:n], dtype=np.float64)
    alpha = float(q[n])

    if not np.all(np.isfinite(z)) or not np.isfinite(alpha):
        raise np.linalg.LinAlgError(
            "Null-space solve produced a non-finite primal solution.")

    # Recover lambda from the coefficient stationarity equation:
    #
    #     H z + B.T lambda = b.
    #
    # B has independent rows here, so solve B.T lambda ~= b-Hz in least
    # squares. This is only multiplier recovery; primal feasibility has
    # already been imposed exactly through the null-space parameterization.
    stationarity_rhs = b-H @ z
    lambda_eq = np.linalg.lstsq(
        B.T, stationarity_rhs, rcond=None)[0]

    equality_resid = B @ z-constraint*alpha
    stationarity_resid = stationarity_rhs-B.T @ lambda_eq

    info = {
        "method": solve_method,
        "constraint_rank": int(rank),
        "nullity": int(N.shape[1]),
        "constraint_rank_tol": float(rank_tol),
        "equality_linf": float(np.max(np.abs(equality_resid))),
        "stationarity_linf": float(np.max(np.abs(stationarity_resid))),
    }

    return z, alpha, lambda_eq, info

# ------------------------------------------------------------------------------

def _orbit_hard_constraint_solve(
    ATA_sub_reg: np.ndarray,
    ATy_sub: np.ndarray,
    active_idx: np.ndarray,
    S_active: np.ndarray,
    orbit_shape: np.ndarray,
    C: int,
    P: int,
    *,
    numerical_ridge: float = 0.0,
    scientific_hessian: np.ndarray | None = None,
    negative_tol: float = 1e-10,
    constraint_tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    Solve the reduced equality-constrained NNLS problem with a fixed
    orbit-prior shape and a data-fitted global amplitude.

    The problem is

        minimize

            0.5 * z.T @ ATA_sub_reg @ z - ATy_sub.T @ z

        subject to

            z >= 0
            alpha >= 0

            sum_j S_active[j] * z[j] = alpha

        for every positive-weight orbit ``c``.

    Orbit weights select participating components; their relative amplitudes
    are already encoded in the HyperCube. All participating components share
    one fitted coefficient mass ``alpha``.

    Parameters
    ----------
    ATA_sub_reg : ndarray, shape (k, k)
        Regularized reduced Hessian.
    ATy_sub : ndarray, shape (k,)
        Reduced linear term.
    active_idx : ndarray, shape (k,)
        Global indices of the active columns.
    S_active : ndarray, shape (k,)
        Column scaling factors for active columns.
    orbit_shape : ndarray, shape (C,)
        Nonnegative orbit weights. Positive values select components subject
        to the equal-mass constraint; their magnitudes do not set its target.
    C : int
        Number of orbits.
    P : int
        Number of populations per orbit.
    numerical_ridge : float, optional
        Numerical ridge added to the coefficient Hessian.
    negative_tol : float, optional
        Negative-coefficient tolerance.
    constraint_tol : float, optional
        Relative equality-constraint tolerance.

    Returns
    -------
    z_out : ndarray, shape (k,)
        Nonnegative constrained solution on the supplied active support.
    info : dict
        Solver status, fitted amplitude, absolute orbit targets,
        constraint residuals, and orbit-level KKT multipliers.

    Raises
    ------
    ValueError
        If the supplied arrays have inconsistent sizes or the orbit shape
        is invalid.

    Examples
    --------
    >>> z, info = _orbit_hard_constraint_solve(
    ...     ATA_sub_reg,
    ...     ATy_sub,
    ...     active_idx,
    ...     S_active,
    ...     orbit_shape,
    ...     C,
    ...     P,
    ... )
    >>> alpha = info["alpha"]
    """
    H = np.asarray(ATA_sub_reg, dtype=np.float64, order="C")
    b = np.asarray(ATy_sub, dtype=np.float64).ravel(order="C")
    active_idx = np.asarray(active_idx, dtype=np.int64).ravel(order="C")
    S_active = np.asarray(S_active, dtype=np.float64).ravel(order="C")
    shape = np.asarray(orbit_shape, dtype=np.float64).ravel(order="C")

    k = int(active_idx.size)

    # Scientific regularisation is supplied directly as its reduced z-space
    # Hessian. It may therefore contain off-diagonal couplings between
    # neighbouring SSP populations.
    if scientific_hessian is None:
        H_scientific = np.zeros((k, k), dtype=np.float64)
    else:
        H_scientific = np.asarray(
            scientific_hessian, dtype=np.float64, order="C")

        if H_scientific.shape != (k, k):
            raise ValueError(
                "scientific_hessian has a shape inconsistent with active_idx.")

        if not np.all(np.isfinite(H_scientific)):
            raise ValueError(
                "scientific_hessian contains non-finite values.")

    lambda_zero = np.zeros((C,), dtype=np.float64)
    target_zero = np.zeros((C,), dtype=np.float64)

    def _failure_info(
        reason: str,
        *,
        negative_count: int = 0,
        n_free: int = 0,
        removed_local_indices=None,
        missing_orbits=None,
        alpha=None,
        lambda_orbit=None,
    ) -> dict:
        if removed_local_indices is None:
            removed_local_indices = []
        if lambda_orbit is None:
            lambda_orbit = lambda_zero

        out = {
            "accepted": False,
            "reason": str(reason),
            "alpha": alpha,
            "orbit_target_mass": target_zero.copy(),
            "resid_l1": np.inf,
            "resid_l2": np.inf,
            "resid_linf": np.inf,
            "rel_resid_linf": np.inf,
            "negative_count": int(negative_count),
            "n_free": int(n_free),
            "n_removed": int(len(removed_local_indices)),
            "removed_local_indices": list(removed_local_indices),
            "lambda_orbit": np.asarray(lambda_orbit, dtype=np.float64).copy(),
        }

        if missing_orbits is not None:
            out["missing_orbits"] = [int(cc) for cc in missing_orbits]

        return out

    if H.shape != (k, k):
        raise ValueError(
            "ATA_sub_reg has a shape inconsistent with active_idx.")
    if b.size != k or S_active.size != k:
        raise ValueError(
            "ATy_sub or S_active has a size inconsistent with active_idx.")
    if shape.size != C:
        raise ValueError(
            "orbit_shape must have length C.")
    if not np.all(np.isfinite(shape)):
        raise ValueError(
            "orbit_shape contains non-finite values.")

    shape = np.maximum(shape, 0.0)
    shape_sum = float(np.sum(shape))

    if not np.isfinite(shape_sum) or shape_sum <= 0.0:
        raise ValueError(
            "orbit_shape must have positive finite total weight.")

    shape /= shape_sum
    positive_orbits = np.flatnonzero(shape > 0.0)
    constraint_shape = np.zeros((C,), dtype=np.float64)
    constraint_shape[positive_orbits] = 1.0

    if k == 0:
        return (np.zeros((0,), dtype=np.float64),
            _failure_info("empty_support", n_free=0))

    orbit_of_col = (active_idx // P).astype(np.int64)

    if np.any(orbit_of_col < 0) or np.any(orbit_of_col >= C):
        raise ValueError(
            "active_idx contains an invalid orbit index.")

    # Columns belonging to zero-shape orbits must not enter this helper.
    zero_shape_active = np.flatnonzero(shape[orbit_of_col] <= 0.0)

    if zero_shape_active.size > 0:
        return (np.zeros((k,), dtype=np.float64),
            _failure_info("zero_shape_orbit_active", n_free=k))

    support_count = np.bincount(orbit_of_col, minlength=C,)

    missing = positive_orbits[support_count[positive_orbits] == 0]

    if missing.size > 0:
        return (np.zeros((k,), dtype=np.float64),
            _failure_info("missing_orbit_support", n_free=k,
                missing_orbits=missing))

    free = np.ones((k,), dtype=bool)
    removed = []

    dual_scale = max(1.0, float(np.max(np.abs(b))))
    dual_tol = max(1e-10, 1e-10 * dual_scale)
    max_inner = max(10, 4 * k)

    for _ in range(max_inner):
        free_idx = np.flatnonzero(free)
        n_free = int(free_idx.size)

        if n_free == 0:
            break

        free_orbits = orbit_of_col[free_idx]

        # Every positive-shape orbit must retain at least one free variable.
        free_count = np.bincount(free_orbits, minlength=C)

        missing_free = positive_orbits[free_count[positive_orbits] == 0]

        if missing_free.size > 0:
            return (np.zeros((k,), dtype=np.float64),
                _failure_info(
                    "missing_free_orbit_support",
                    n_free=n_free,
                    removed_local_indices=removed,
                    missing_orbits=missing_free,
                ))

        # Only positive-shape orbits require constraint rows.
        constraint_orbits = positive_orbits.copy()
        n_constraints = int(constraint_orbits.size)

        B = np.zeros((n_constraints, n_free), dtype=np.float64)

        orbit_to_row = {int(cc): row
            for row, cc in enumerate(constraint_orbits)}

        for local_j, reduced_j in enumerate(free_idx):
            cc = int(orbit_of_col[reduced_j])
            B[orbit_to_row[cc], local_j] = S_active[reduced_j]

        constraint_sub = constraint_shape[constraint_orbits]

        Hff = H[np.ix_(free_idx, free_idx)]
        bff = b[free_idx]

        # ------------------------------------------------------------
        # Solve the equality-constrained quadratic in the null space of
        #
        #     B@z - constraint*alpha = 0.
        #
        # This avoids directly solving the indefinite saddle-point KKT
        # matrix. Hard equal-component-mass constraints are satisfied by
        # construction, while the remaining system contains only the
        # physically free directions.
        # ------------------------------------------------------------
        Hff_reg = Hff + H_scientific[np.ix_(free_idx, free_idx)]
        Hff_reg = Hff_reg + float(numerical_ridge)*np.eye(
            n_free, dtype=np.float64)

        try:
            z_free, alpha_trial, lambda_sub, linear_info = \
                _solve_equal_mass_nullspace(Hff_reg, bff, B, constraint_sub)

        except np.linalg.LinAlgError as exc:
            return (np.zeros((k,), dtype=np.float64),
                _failure_info("nullspace_solve_failed", n_free=n_free,
                    removed_local_indices=removed))

        lambda_orbit = np.zeros((C,), dtype=np.float64)
        lambda_orbit[constraint_orbits] = lambda_sub

        if (not np.all(np.isfinite(z_free))
            or not np.isfinite(alpha_trial)
            or not np.all(np.isfinite(lambda_sub))):
            return (np.zeros((k,), dtype=np.float64),
                _failure_info("nonfinite_kkt_solution", n_free=n_free,
                    removed_local_indices=removed))

        negative_local = np.flatnonzero(z_free < -negative_tol)

        if negative_local.size == 0:
            # With z >= 0, equal-mass feasibility implies alpha >= 0.
            # A materially negative alpha therefore indicates a failed solve.
            if alpha_trial < -negative_tol:
                return (np.zeros((k,), dtype=np.float64),
                    _failure_info("negative_alpha", n_free=n_free,
                        removed_local_indices=removed,
                        alpha=float(alpha_trial), lambda_orbit=lambda_orbit))

            alpha_out = max(0.0, alpha_trial)

            z_out = np.zeros((k,), dtype=np.float64)
            z_out[free_idx] = np.maximum(z_free, 0.0)

            full_mass = np.bincount(orbit_of_col,
                weights=S_active * z_out, minlength=C).astype(np.float64)

            target_mass = alpha_out * constraint_shape
            mass_resid = full_mass - target_mass
            scale = np.maximum(np.abs(target_mass), 1.0)
            rel_linf = float(np.max(np.abs(mass_resid) / scale))

            # A common alpha requires the constrained multipliers to sum to
            # zero.
            alpha_stationarity = float(
                np.dot(constraint_shape, lambda_orbit))
            dual_grad = (b - (H @ z_out) - (H_scientific @ z_out)
                - (float(numerical_ridge) * z_out)
                - (S_active * lambda_orbit[orbit_of_col]))
            free_dual = dual_grad[free]

            free_dual_linf = (float(np.max(np.abs(free_dual)))
                if free_dual.size else 0.0)

            free_dual_scale = max(1.0, float(np.max(np.abs(b))))
            free_dual_tol = max(1e-10, 1e-9*free_dual_scale)

            if (not np.isfinite(free_dual_linf)
                or free_dual_linf > free_dual_tol):
                return (np.zeros((k,), dtype=np.float64),
                    _failure_info("free_kkt_residual", n_free=n_free,
                        removed_local_indices=removed,
                        alpha=float(alpha_out), lambda_orbit=lambda_orbit))

            fixed = ~free
            dual_violators = np.flatnonzero(
                fixed & (dual_grad > dual_tol))

            if dual_violators.size > 0:
                reenter = int(dual_violators[np.argmax(
                    dual_grad[dual_violators])])
                free[reenter] = True

                if reenter in removed:
                    removed.remove(reenter)

                continue

            accepted = (np.all(np.isfinite(z_out)) and np.isfinite(alpha_out)
                and alpha_out >= 0.0 and np.all(np.isfinite(lambda_orbit))
                and rel_linf <= constraint_tol)

            return z_out, {
                "accepted": bool(accepted),
                "reason": ("accepted" if accepted else "constraint_residual"),
                "alpha": float(alpha_out),
                "orbit_target_mass": target_mass,
                "resid_l1": float(np.sum(np.abs(mass_resid))),
                "resid_l2": float(np.linalg.norm(mass_resid)),
                "resid_linf": float(np.max(np.abs(mass_resid))),
                "rel_resid_linf": float(rel_linf),
                "alpha_stationarity": alpha_stationarity,
                "numerical_ridge": float(numerical_ridge),
                "dual_linf": float(np.max(np.maximum(dual_grad[~free], 0.0))
                    ) if np.any(~free) else 0.0,
                "negative_count": 0,
                "n_free": int(n_free),
                "n_removed": int(len(removed)),
                "removed_local_indices": removed.copy(),
                "constraint_orbits": constraint_orbits.tolist(),
                "lambda_orbit": lambda_orbit,
                "free_dual_linf": float(free_dual_linf),
                "free_dual_tol": float(free_dual_tol),
            }

        # Remove the most negative coefficient that is not the final free
        # variable in its positive-shape orbit.
        order = negative_local[np.argsort(z_free[negative_local])]

        remove_reduced_idx = None

        for local_j in order:
            candidate = int(free_idx[int(local_j)])
            cc = int(orbit_of_col[candidate])

            remaining_in_orbit = int(np.count_nonzero(free
                & (orbit_of_col == cc)))

            if remaining_in_orbit <= 1:
                continue

            remove_reduced_idx = candidate
            break

        if remove_reduced_idx is None:
            z_fail = np.zeros((k,), dtype=np.float64)
            z_fail[free_idx] = z_free

            return z_fail, _failure_info(
                "negative_required_variable",
                negative_count=int(negative_local.size),
                n_free=n_free,
                removed_local_indices=removed,
                alpha=float(alpha_trial),
                lambda_orbit=lambda_orbit,
            )

        free[remove_reduced_idx] = False
        removed.append(int(remove_reduced_idx))

    return (np.zeros((k,), dtype=np.float64),
        _failure_info("inner_active_set_exhausted",
            n_free=int(np.count_nonzero(free)), removed_local_indices=removed))

# ------------------------------------------------------------------------------

def _compute_yty(
    h5_path: str,
    s_ranges: list,
    keep_idx,
) -> float:
    """Compute y.T @ y once in the fitted data space."""
    yty = 0.0

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]

        for s0, s1 in s_ranges:
            Yt = np.asarray(DC[s0:s1, :], dtype=np.float64, order="C",)

            if keep_idx is not None:
                Yt = Yt[:, keep_idx]
            yty += float(np.sum(Yt * Yt))

    return yty

# ------------------------------------------------------------------------------

def streamActiveSetNNLS(
    h5_path: str,
    s_ranges: list,
    keep_idx,
    ATy_flat: np.ndarray,
    inv_sqrt_energy_flat: np.ndarray,
    C: int,
    P: int,
    *,
    executor,
    component_slot: np.ndarray,
    C_physical: int,
    cfg: MPConfig,
    orbit_weights: Optional[np.ndarray] = None,
    known_zero_mask: Optional[np.ndarray] = None,
    x0_flat: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    max_active: int = 2000,
    tol_grad: float = 1e-8,
    max_iter: int = 5000,
    regularisation_scale: float = 1.0,
    minimum_l2_at_zero: bool = True,
    minimum_l2_rss_rtol: float = 1e-4,
    checkpoint_cb=None,
    checkpoint_every: int = 0,
):
    """
    Monolithic streaming active-set NNLS.
    """

    regularisation_scale = float(regularisation_scale)
    minimum_l2_at_zero = bool(minimum_l2_at_zero)
    minimum_l2_rss_rtol = float(minimum_l2_rss_rtol)

    if (not np.isfinite(minimum_l2_rss_rtol)
        or minimum_l2_rss_rtol < 0.0):
        raise ValueError('minimum_l2_rss_rtol must be nonnegative and finite.')

    use_minimum_l2 = (minimum_l2_at_zero and regularisation_scale == 0.0)
    # Fixed numerical ridge. Keeping this constant across all working supports
    # ensures that every active-set iteration solves the same convex quadratic
    # program rather than changing the objective as the support changes.
    numerical_ridge = 1e-12

    if (not np.isfinite(regularisation_scale)) or (regularisation_scale < 0.0):
        raise ValueError(
            "regularisation_scale must be nonnegative and finite.")

    def _robust_scale_ref(vec: np.ndarray, fallback: float = 1.0) -> float:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        arr = np.abs(arr[arr > 0.0])
        if arr.size == 0:
            return float(max(1.0, fallback))
        ref = float(np.median(arr))
        if (not np.isfinite(ref)) or (ref <= 0.0):
            return float(max(1.0, fallback))
        return ref


    component_slot = np.asarray(component_slot, dtype=np.int64).ravel(order="C")
    C_physical = int(C_physical)

    if component_slot.size != C_physical:
        raise ValueError('component_slot has inconsistent physical dimensions.')
    if np.any(component_slot < 0) or np.any(component_slot >= C):
        raise ValueError(
            "component_slot contains invalid fitted-component indices.")

    CP = int(C * P)
    orbit_of_col = (np.arange(CP, dtype=np.int64) // P).astype(np.int64)
    if known_zero_mask is None:
        known_zero_flat = np.zeros((CP,), dtype=bool)
    else:
        known_zero_flat = np.asarray(known_zero_mask, dtype=bool
            ).ravel(order="C")

        if known_zero_flat.size != CP:
            raise ValueError(
                "known_zero_mask has wrong size in monolithic solver")

    orbit_shape = _canon_orbit_weights(orbit_weights, C=C, P=P)

    S_flat = np.asarray(inv_sqrt_energy_flat, dtype=np.float64).ravel()
    ATy_scaled = S_flat * ATy_flat
    yty = _compute_yty(h5_path, s_ranges, keep_idx)
    # ------------------------------------------------------------------
    # Scientific SSP smoothness regularisation.
    #
    # Population ordering in the HyperCube is
    #
    #     metallicity x age x alpha
    #
    # flattened in C-order. The same P x P smoothness matrix applies
    # independently to every dynamical component.
    # ------------------------------------------------------------------
    with open_h5(h5_path, role="reader") as f:
        required_ssp = ("/SSPMetals", "/SSPAges", "/SSPAlphas")

        if not all(name in f for name in required_ssp):
            raise RuntimeError('SSP smoothness regularisation requires '
                '/SSPMetals, /SSPAges, and /SSPAlphas.')

        umetals = np.asarray(f["/SSPMetals"][...],
            dtype=np.float64).ravel(order="C")
        uages = np.asarray(f["/SSPAges"][...],
            dtype=np.float64).ravel(order="C")
        ualphas = np.asarray(f["/SSPAlphas"][...],
            dtype=np.float64).ravel(order="C")
    n_metals = int(umetals.size)
    n_ages = int(uages.size)
    n_alphas = int(ualphas.size)

    if n_metals * n_ages * n_alphas != P:
        raise RuntimeError("SSP grid dimensions are inconsistent with P: "
            f"{n_metals}*{n_ages}*{n_alphas} != {P}.")

    Q_ssp = _ssp_smoothness_hessian(umetals, uages, ualphas)

    # ------------------------------------------------------------------
    # Dimensionless scientific-regularisation scale.
    #
    # Q_ssp is the physical-coordinate second-difference curvature
    # Hessian. In solver z-space,
    #
    #     x = S z,
    #
    # so its diagonal curvature is
    #
    #     diag(S Q_ssp S).
    #
    # Normalize by the median positive diagonal curvature. Since the
    # solver's data columns are normalized to unit energy, the user-facing
    # regularisation_scale remains dimensionless and directly controls the
    # characteristic scientific curvature relative to the data curvature.
    #
    # This normalization changes only the overall strength of Q_ssp. It
    # does not alter its SSP-grid geometry or uniform null mode.
    # ------------------------------------------------------------------
    uniform_mode = np.ones(P, dtype=np.float64)
    uniform_null = float(np.max(np.abs(Q_ssp @ uniform_mode)))
    q_diag = np.diag(Q_ssp)

    print(f"[MONO][smoothness] diag=[{np.min(q_diag):.3e}, "
        f"{np.median(q_diag):.3e}, {np.max(q_diag):.3e}] "
        f"uniform_null={uniform_null:.3e}", flush=True)

    smooth_diag = np.tile(q_diag, C) * (S_flat * S_flat)
    positive_smooth = smooth_diag[smooth_diag > 0.0]

    if positive_smooth.size == 0:
        smooth_norm = 1.0
    else:
        smooth_norm = float(np.median(positive_smooth))

    smooth_scale = float(regularisation_scale) / smooth_norm

    grad_ref = max(1.0, float(np.max(np.abs(ATy_scaled))))
    data_scale_ref = _robust_scale_ref(ATy_scaled, fallback=grad_ref)
    tol_grad_rel = 1e-11

    # Number of inactive KKT violators admitted per exact active-set step.
    # Keeping this modest avoids unnecessarily large reduced systems while
    # allowing several strongly violating directions to enter together.
    kkt_promote_batch = max(1, min(32, C))


    # Orbit weights select constrained components. Their relative amplitudes
    # are already encoded in the HyperCube; coefficient masses are equal.
    has_hard_orbit_shape = orbit_shape is not None and np.any(orbit_shape > 0.0)
    positive_shape_orbits = (np.flatnonzero(orbit_shape > 0.0)
        if has_hard_orbit_shape else np.zeros((0,), dtype=np.int64))
    zero_shape_orbits = (np.flatnonzero(orbit_shape <= 0.0)
        if has_hard_orbit_shape else np.zeros((0,), dtype=np.int64))
    constraint_shape = np.zeros((C,), dtype=np.float64)
    constraint_shape[positive_shape_orbits] = 1.0
    if has_hard_orbit_shape:
        known_zero_cp = known_zero_flat.reshape(C, P)
        blocked_orbits = np.flatnonzero(np.all(known_zero_cp, axis=1)
            & (orbit_shape > 0.0))

        if blocked_orbits.size > 0:
            raise ValueError("known_zero_mask makes the hard orbit constraint "
                "infeasible for positive-weight orbits "
                f"{blocked_orbits.tolist()}")

    # Current accepted global amplitude. It is initialized below and then
    # replaced by the value fitted by each accepted KKT solve.
    alpha_current = 0.0

    # Orbit-level Lagrange multipliers from the most recently accepted
    # constrained reduced solve.
    lambda_orbit_current = np.zeros((C,), dtype=np.float64)
    ridge_current = float(numerical_ridge)

    diag_level = int(os.environ.get("CUBEFIT_DIAG_LEVEL", "1"))
    diag_stride = max(1, int(os.environ.get("CUBEFIT_DIAG_STRIDE", "1")))
    diag_jsonl_path = os.environ.get("CUBEFIT_DIAG_JSONL", "").strip() or None
    diag_topk = max(1, int(os.environ.get("CUBEFIT_DIAG_TOPK", "12")))
    diag_t0 = time.perf_counter()

    # Normalize and validate the resume metadata before it is used.
    resume_state = dict(resume_state or {})

    if resume_state:
        saved_regularisation_scale = float(
            resume_state.get("regularisation_scale", 1.0))

        if not np.isclose(saved_regularisation_scale, regularisation_scale,
            rtol=1e-12, atol=0.0):
            raise ValueError(
                "Cannot full-resume with a different regularisation_scale: "
                f"checkpoint={saved_regularisation_scale:.6e}, "
                f"requested={regularisation_scale:.6e}. "
                "Use warm_start='saved_x' instead.")
        saved_reg_mode = resume_state.get(
            "scientific_regularisation_mode", "legacy_l2")

        if saved_reg_mode != SCIENTIFIC_REGULARISATION_MODE:
            raise ValueError(
                "Cannot full-resume a checkpoint using a different scientific "
                "regularisation model. Use warm_start='saved_x' instead.")

        # Historical checkpoints predate the unconstrained orbit_weights=None
        # pathway, so a missing mode flag is interpreted as constrained.
        saved_hard_orbit_shape = bool(
            resume_state.get("hard_orbit_shape", True))

        if saved_hard_orbit_shape != has_hard_orbit_shape:
            raise ValueError(
                "Cannot full-resume with a different orbit constraint mode: "
                f"checkpoint hard_orbit_shape="
                f"{saved_hard_orbit_shape}, "
                f"requested={has_hard_orbit_shape}. "
                "Use warm_start='saved_x' instead.")
        if has_hard_orbit_shape:
            saved_mode = resume_state.get("orbit_constraint_mode", None)
            if saved_mode != "equal_component_mass":
                raise ValueError(
                    "Cannot full-resume a checkpoint using the old orbit "
                    "constraint. Use warm_start='saved_x' instead.")

        # New constrained checkpoints store the canonical normalized orbit
        # shape. Old constrained checkpoints may not have this field, in which
        # case retain backward compatibility and accept the resume.
        if (has_hard_orbit_shape and "orbit_shape" in resume_state
            and resume_state["orbit_shape"] is not None):
            saved_orbit_shape = np.asarray(
                resume_state["orbit_shape"],
                dtype=np.float64,
            ).ravel(order="C")

            if saved_orbit_shape.size != C:
                raise ValueError(
                    "Cannot full-resume with an incompatible orbit_shape: "
                    f"checkpoint size={saved_orbit_shape.size}, "
                    f"requested size={C}. "
                    "Use warm_start='saved_x' instead.")

            if not np.all(np.isfinite(saved_orbit_shape)):
                raise ValueError(
                    "Cannot full-resume from a checkpoint with non-finite "
                    "orbit_shape values. "
                    "Use warm_start='saved_x' instead.")

            if not np.allclose(
                saved_orbit_shape,
                np.asarray(orbit_shape, dtype=np.float64),
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "Cannot full-resume with different orbit_weights. "
                    "Use warm_start='saved_x' instead.")

    needs_kkt_initialise = bool(x0_flat is not None and not resume_state)

    start_iter = int(resume_state.get("iter", 0) or 0)

    if start_iter < 0:
        start_iter = 0

    def _emit_diag(record: dict) -> None:
        if diag_level <= 0 and ('error' not in str(record.get("kind"))):
            return
        rec = dict(record)
        rec.setdefault("t_sec", float(time.perf_counter() - diag_t0))
        _diag_append_jsonl(diag_jsonl_path, rec)

    _emit_diag({
        "kind": "mono_setup",
        "source": "streamActiveSetNNLS",
        "C": int(C),
        "P": int(P),
        "CP": int(CP),
        "processes": int(cfg.processes),
        "blas_threads": int(cfg.blas_threads),
        "data_scale_ref": float(data_scale_ref),
        "grad_ref": float(grad_ref),
        "tol_grad": float(tol_grad),
        "tol_grad_rel": float(tol_grad_rel),
        "regularisation_scale": float(regularisation_scale),
        "scientific_regularisation_mode": SCIENTIFIC_REGULARISATION_MODE,
        "smoothness_normalisation": float(smooth_norm),
        "effective_smoothness_scale": float(smooth_scale),
        "diag_level": int(diag_level),
        "diag_stride": int(diag_stride),
        "diag_topk": int(diag_topk),
        "hard_orbit_shape": bool(has_hard_orbit_shape),
        "orbit_constraint_mode": (
            "equal_component_mass" if has_hard_orbit_shape else "none"),
        "orbit_shape": (orbit_shape.tolist() if orbit_shape is not None
            else None),
        "orbit_shape_sum": (float(np.sum(orbit_shape))
            if orbit_shape is not None else None),
        "orbit_target_sum": None,
        "max_iter": int(max_iter),
        "start_iter": int(start_iter),
        "max_active": int(max_active),
        "kkt_promote_batch": int(kkt_promote_batch),
        "numerical_ridge": float(numerical_ridge),
    })

    if x0_flat is None:
        z = np.zeros((CP,), dtype=np.float64)
    else:
        x0_flat = np.asarray(x0_flat, dtype=np.float64).ravel(order="C")

        if x0_flat.size != CP:
            raise ValueError("x0_flat has wrong size in monolithic solver")

        z = np.divide(x0_flat, S_flat,
            out=np.zeros_like(x0_flat, dtype=np.float64),
            where=S_flat > 0.0)
        np.maximum(z, 0.0, out=z)

    seed_population = None
    # Build a feasible initial state for the fixed orbit shape. The initial
    # amplitude is only a starting value; it is not held fixed by the solver.
    if has_hard_orbit_shape:
        x_seed = (S_flat * z).reshape(C, P)

        # Strictly exclude zero-shape orbits.
        for cc in zero_shape_orbits:
            x_seed[int(cc), :] = 0.0

        if not resume_state:
            warm_orbit_mass = np.sum(x_seed, axis=1)
            warm_mass = float(np.mean(
                warm_orbit_mass[positive_shape_orbits]))

            if np.isfinite(warm_mass) and warm_mass > 0.0:
                alpha_current = warm_mass
            else:
                # This only constructs a feasible nonzero equal-mass seed.
                # The first constrained KKT solve refits alpha from the data.
                alpha_current = 1.0

            for cc in positive_shape_orbits:
                cc = int(cc)
                target_cc = alpha_current

                current_mass = float(np.sum(x_seed[cc, :]))

                if np.isfinite(current_mass) and current_mass > 0.0:
                    # Preserve the warm-start population mixture.
                    x_seed[cc, :] *= (target_cc / current_mass)
                else:
                    base = cc * P
                    orbit_cols = np.arange(base, base + P, dtype=np.int64)

                    allowed_p = np.flatnonzero(~known_zero_flat[orbit_cols])

                    if allowed_p.size == 0:
                        raise ValueError(
                            f"Orbit {cc} has no allowed population columns.")

                    # ------------------------------------------------------
                    # Construct a component-local multi-population seed.
                    #
                    # The old initialization placed the complete component mass
                    # on the single SSP with maximum normalized A.T@y. Because
                    # all dynamical components share the same SSP library, that
                    # can initialize many components at exactly the same SSP.
                    #
                    # Instead retain several locally preferred populations and
                    # distribute the component's required mass smoothly among
                    # them. This remains entirely component-local: no population
                    # information is shared between dynamical components.
                    # ------------------------------------------------------
                    local_score = np.asarray(ATy_scaled[base + allowed_p],
                        dtype=np.float64,)
                    n_seed = min(8, allowed_p.size)

                    # Select this component's own strongest candidate SSPs.
                    top_local = np.argpartition(local_score, -n_seed)[-n_seed:]

                    # Sort only for deterministic behaviour and diagnostics.
                    top_local = top_local[np.argsort(local_score[top_local],
                        kind="stable")[::-1]]
                    seed_p = allowed_p[top_local]
                    seed_score = local_score[top_local].copy()

                    # Robust component-local score scale. Only score differences
                    # matter for the softmax.
                    score_center = float(np.median(seed_score))
                    score_dev = np.abs(seed_score-score_center)
                    score_scale = float(np.median(score_dev))

                    if not np.isfinite(score_scale) or score_scale <= 0.0:
                        score_scale = float(np.std(seed_score))
                    if not np.isfinite(score_scale) or score_scale <= 0.0:
                        score_scale = max(float(np.max(np.abs(seed_score)
                            )), 1.0)

                    # Temperature > 1 deliberately avoids reproducing the
                    # near-delta-function behaviour of the old argmax seed.
                    temperature = 2.0
                    logits = ((seed_score-np.max(seed_score))
                        / (temperature*score_scale))

                    # Avoid pathological exponentials without changing the
                    # ordering of the candidate SSPs.
                    logits = np.clip(logits, -50.0, 0.0)
                    seed_weight = np.exp(logits)
                    weight_sum = float(np.sum(seed_weight))

                    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                        seed_weight = np.full(n_seed, 1.0/n_seed,
                            dtype=np.float64)
                    else:
                        seed_weight /= weight_sum

                    x_seed[cc, :] = 0.0
                    x_seed[cc, seed_p] = target_cc*seed_weight

            seed_population = np.full(C, -1, dtype=np.int64)

            for cc in positive_shape_orbits:
                positive_p = np.flatnonzero(x_seed[int(cc)] > 0.0)

                if positive_p.size:
                    seed_population[int(cc)] = int(positive_p[np.argmax(
                        x_seed[int(cc), positive_p])])

            print("[MONO][seed] population indices: " + " ".join(
                f"{cc}:{seed_population[cc]}" for cc in positive_shape_orbits),
                flush=True)
            x_seed_flat = x_seed.ravel(order="C")

            z = np.divide(x_seed_flat, S_flat,
                out=np.zeros_like(x_seed_flat, dtype=np.float64),
                where=S_flat > 0.0)
            np.maximum(z, 0.0, out=z)

        else:
            # Infer fallback alpha as the mean constrained component mass.
            x_resume = (S_flat * z).reshape(C, P)
            resume_mass = np.sum(x_resume, axis=1)
            alpha_current = float(np.mean(
                resume_mass[positive_shape_orbits]))

            if (not np.isfinite(alpha_current)
                or alpha_current < 0.0):
                alpha_current = 0.0
    z[known_zero_flat] = 0.0
    z_scale0 = np.max(z) + 1e-30
    active = z > 0.0
    active[known_zero_flat] = False

    if has_hard_orbit_shape and zero_shape_orbits.size > 0:
        for cc in zero_shape_orbits:
            base = int(cc) * P
            z[base:base + P] = 0.0
            active[base:base + P] = False

    if resume_state:

        if "active_mask" in resume_state:
            am = np.asarray(resume_state["active_mask"], dtype=bool).ravel(
                order="C")
            if am.size == CP:
                active = am.copy()

        if "alpha_current" in resume_state:
            alpha_saved = float(resume_state["alpha_current"])

            if (np.isfinite(alpha_saved)
                and alpha_saved >= 0.0):
                alpha_current = alpha_saved
        if "ridge_current" in resume_state:
            ridge_saved = float(resume_state["ridge_current"])
            if not np.isclose(ridge_saved, numerical_ridge, rtol=1e-12,
                atol=0.0):
                raise ValueError('Cannot full-resume a checkpoint using a '
                    'different numerical ridge. Use warm_start=\'saved_x\'' 
                    'instead.')
        ridge_current = float(numerical_ridge)
        if "lambda_orbit" in resume_state:
            lambda_saved = np.asarray(resume_state["lambda_orbit"],
                dtype=np.float64).ravel(order="C")

            if (lambda_saved.size == C
                and np.all(np.isfinite(lambda_saved))):
                lambda_orbit_current = lambda_saved.copy()

        if has_hard_orbit_shape and zero_shape_orbits.size > 0:
            for cc in zero_shape_orbits:
                base = int(cc) * P
                z[base:base + P] = 0.0
                active[base:base + P] = False

    z[known_zero_flat] = 0.0
    active[known_zero_flat] = False

    def _current_x_from_z(z_vec: np.ndarray) -> np.ndarray:
        return S_flat * z_vec
    def _data_quad_obj(A, b, x):
        """
        Evaluate the unregularised data objective on a reduced support.

        Parameters
        ----------
        A : ndarray
            Unregularised reduced normal matrix.
        b : ndarray
            Reduced data vector.
        x : ndarray
            Reduced scaled coefficient vector.

        Returns
        -------
        float
            Unregularised data objective,
            ``0.5 * x.T @ A @ x - b.T @ x``.
        """
        return 0.5 * float(x @ (A @ x)) - float(b @ x)
    def _scientific_gradient_x(x_vec: np.ndarray) -> np.ndarray:
        """
        Return the gradient of the SSP smoothness penalty in physical x-space.
        """
        x_cp = np.asarray(x_vec, dtype=np.float64).reshape(C, P)

        return (smooth_scale * (x_cp @ Q_ssp)).ravel(order="C")

    def _scientific_objective_x(x_vec: np.ndarray) -> float:
        """
        Return the dimensionless-scaled SSP smoothness penalty.
        """
        x_cp = np.asarray(x_vec, dtype=np.float64).reshape(C, P)

        return 0.5 * smooth_scale * float(np.sum((x_cp @ Q_ssp) * x_cp))

    def _scientific_hessian_reduced(active_idx: np.ndarray,
        S_active: np.ndarray) -> np.ndarray:
        """
        Return the SSP smoothness Hessian on a reduced support in z-space.
        """
        active_idx = np.asarray(active_idx, dtype=np.int64).ravel(order="C")
        S_active = np.asarray(S_active, dtype=np.float64).ravel(order="C")

        orbit_idx = active_idx // P
        pop_idx = active_idx % P

        H_reg = np.zeros((active_idx.size, active_idx.size), dtype=np.float64)

        same_orbit = orbit_idx[:, None] == orbit_idx[None, :]

        H_reg[same_orbit] = (smooth_scale
            * Q_ssp[pop_idx[:, None], pop_idx[None, :]][same_orbit]
            * (S_active[:, None] * S_active[None, :])[same_orbit])

        return H_reg

    if start_iter >= max_iter:
        print("[MONO] resume_state already reached max_iter; returning the "
            "committed solution without entering the outer loop.", flush=True)
        return _current_x_from_z(z)

    checkpoint_error_logged = False

    def _log_checkpoint_error(where: str, exc: Exception) -> None:
        nonlocal checkpoint_error_logged
        if not checkpoint_error_logged:
            print(f"[MONO][checkpoint] {where}: {exc}", flush=True)
            checkpoint_error_logged = True
    
    def _pack_resume_state(it: int, phase: str, final: bool) -> dict:
        return {
            "iter": int(it + 1),
            "max_iter": int(max_iter),
            "phase": str(phase),
            "final": bool(final),
            "active_mask": active.astype(bool).tolist(),
            "lambda_orbit": lambda_orbit_current.astype(np.float64).tolist(),
            "alpha_current": float(alpha_current),
            "ridge_current": float(ridge_current),
            "regularisation_scale": float(regularisation_scale),
            "scientific_regularisation_mode": SCIENTIFIC_REGULARISATION_MODE,
            "hard_orbit_shape": bool(has_hard_orbit_shape),
            "orbit_constraint_mode": (
                "equal_component_mass" if has_hard_orbit_shape else "none"),
            "orbit_shape": (
                orbit_shape.astype(np.float64).tolist()
                if orbit_shape is not None else None),
            "stop_reason": str(stop_reason),
            "stop_category": str(stop_category),
            "stop_converged": bool(stop_converged),
            "stop_details": dict(stop_details),
        }
    
    def _emit_checkpoint(
        it: int,
        *,
        final: bool = False,
        phase: str = "solve",
    ) -> None:
        if checkpoint_cb is None:
            return

        every = int(checkpoint_every)
        if (not final) and (every > 0) and ((it + 1) % every != 0):
            return

        try:
            checkpoint_cb(
                _current_x_from_z(z),
                {
                    "iter": int(it + 1),
                    "max_iter": int(max_iter),
                    "phase": str(phase),
                    "final": bool(final),
                    "active": int(np.count_nonzero(active)),
                    "resume_state": _pack_resume_state(it, phase, final),
                },
            )
        except Exception as exc:
            _log_checkpoint_error("checkpoint callback failed", exc)

    def _orbit_mass_and_deficit(z_vec: np.ndarray,
        alpha_value: float | None = None):
        """
        Compute current orbit masses and residuals relative to the fixed
        orbit shape at the supplied global amplitude.
        """
        x_vec = _current_x_from_z(z_vec).reshape(C, P)

        s_orbit = np.sum(x_vec, axis=1)

        if not has_hard_orbit_shape:
            t_orbit = np.zeros((C,), dtype=np.float64)
        else:
            if alpha_value is None:
                alpha_use = float(alpha_current)
            else:
                alpha_use = float(alpha_value)

            if (not np.isfinite(alpha_use)
                or alpha_use < 0.0):
                alpha_use = 0.0

            t_orbit = alpha_use * constraint_shape

        deficit = t_orbit - s_orbit

        return s_orbit, t_orbit, deficit

    # ------------------------------------------------------------
    # Helper: compute ATAz in parallel using provided executor
    # ------------------------------------------------------------
    cached_ATAz_z = None
    cached_ATAz = None

    def _compute_ATAz_scaled(z_vec):
        nonlocal cached_ATAz_z, cached_ATAz
        x_cand = np.asarray(z_vec, dtype=np.float64).ravel(order="C")

        if not np.any(x_cand):
            ATAz = np.zeros((CP,), dtype=np.float64)
            cached_ATAz_z = x_cand.copy()
            cached_ATAz = ATAz.copy()
            return ATAz

        if (cached_ATAz_z is not None
            and np.array_equal(x_cand, cached_ATAz_z)):
            print("[MONO] reusing cached ATAz", flush=True,)
            return cached_ATAz.copy()

        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]
        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        ATAz = np.zeros((CP,), dtype=np.float64)

        worker_args = [(h5_path, batch, keep_idx, x_cand, CP, S_flat, C, P,
            component_slot, C_physical) for batch in batches if len(batch) > 0]
        it = executor.map(_worker_ATAz_from_tuple, worker_args)

        for r in tqdm(it, total=len(worker_args), desc="[MONO] ATAz tiles",
            leave=False):
            try:
                ATAz += r
            except Exception as e:
                print("[ERROR] worker returned exception:", e, flush=True)
                raise

        cached_ATAz_z = x_cand.copy()
        cached_ATAz = ATAz.copy()

        return ATAz
    
    def _assemble_reduced(active_idx):
        """
        Assemble reduced normal equations on the supplied support.

        The global ``ATy_scaled`` vector is the canonical source for the
        reduced linear term. This guarantees that the reduced solve and the
        outer KKT gradient optimize exactly the same objective.
        """
        active_idx = np.asarray(active_idx, dtype=np.int64).ravel(order="C")

        k = int(active_idx.size)
        S_active = S_flat[active_idx]

        ATA_sub = np.zeros((k, k), dtype=np.float64)

        n_workers = max(1, int(cfg.processes))
        batches = [[] for _ in range(n_workers)]

        for i, tile in enumerate(s_ranges):
            batches[i % n_workers].append(tile)

        futures = [executor.submit(_worker_reduced, h5_path, batch, keep_idx,
                active_idx, S_active, CP, P, component_slot, C_physical)
            for batch in batches if batch]

        for future in as_completed(futures):
            ATA_loc, _ = future.result()
            ATA_sub += ATA_loc

        # IMPORTANT:
        # Use exactly the same A^T y vector as the outer gradient.
        ATy_sub = ATy_scaled[active_idx].copy()

        return ATA_sub, ATy_sub, S_active
    def _solve_reduced_active(active_idx, assembled=None,
        extra_hessian=None):
        """
        Assemble and solve the constrained reduced problem.

        If ``assembled`` is supplied, use the provided reduced normal
        equations instead of rereading the HyperCube.

        ``extra_hessian`` is an optional secondary Hessian in solver z-space.
        It is used only by the post-convergence minimum-L2 solve when the
        scientific regularisation is exactly zero.
        """
        active_idx = np.asarray(active_idx, dtype=np.int64)
        k_loc = int(active_idx.size)

        if k_loc == 0:
            raise RuntimeError(
                "Cannot solve reduced problem with empty active support.")

        if assembled is None:
            ATA_loc, ATy_loc, S_loc = _assemble_reduced(active_idx)
        else:
            ATA_loc, ATy_loc, S_loc = assembled

            ATA_loc = np.asarray(ATA_loc, dtype=np.float64, order="C")
            ATy_loc = np.asarray(ATy_loc, dtype=np.float64).ravel(order="C")
            S_loc = np.asarray(S_loc, dtype=np.float64).ravel(order="C")

            if ATA_loc.shape != (k_loc, k_loc):
                raise ValueError(
                    "Preassembled ATA shape is inconsistent with "
                    "active_idx.")

            if ATy_loc.size != k_loc:
                raise ValueError(
                    "Preassembled ATy size is inconsistent with "
                    "active_idx.")

            if S_loc.size != k_loc:
                raise ValueError(
                    "Preassembled scaling size is inconsistent with "
                    "active_idx.")

        H_scientific_loc = _scientific_hessian_reduced(active_idx, S_loc)

        if extra_hessian is not None:
            H_extra_loc = np.asarray(extra_hessian, dtype=np.float64, order="C")

            if H_extra_loc.shape != (k_loc, k_loc):
                raise ValueError(
                    "extra_hessian shape is inconsistent with active_idx.")
            H_scientific_loc = H_scientific_loc+H_extra_loc

        ATA_scientific_loc = ATA_loc+H_scientific_loc
        numerical_ridge_loc = float(numerical_ridge)

        try:
            eigs_loc = np.linalg.eigvalsh(ATA_scientific_loc)
            emin_loc = float(eigs_loc[0])
            emax_loc = float(eigs_loc[-1])
            cond_loc = emax_loc/emin_loc if emin_loc > 0.0 else np.inf
        except np.linalg.LinAlgError:
            emin_loc = np.nan
            emax_loc = np.nan
            cond_loc = np.inf

        if has_hard_orbit_shape:
            z_raw_loc, info_loc = _orbit_hard_constraint_solve(
                ATA_sub_reg=ATA_loc, ATy_sub=ATy_loc, active_idx=active_idx,
                S_active=S_loc,
                orbit_shape=np.asarray(orbit_shape, dtype=np.float64),
                C=C, P=P, numerical_ridge=numerical_ridge_loc,
                scientific_hessian=H_scientific_loc)

            if not info_loc["accepted"]:
                return (active_idx, None, info_loc, None, None, ATA_loc,
                    ATy_loc, S_loc, numerical_ridge_loc, cond_loc)

            z_loc = np.asarray(z_raw_loc, dtype=np.float64).copy()
            alpha_loc = float(info_loc["alpha"])
            lambda_loc = np.asarray(info_loc["lambda_orbit"],
                dtype=np.float64).ravel(order="C")

        else:
            ATA_reg_loc = (ATA_loc+H_scientific_loc + numerical_ridge_loc*np.eye(
                k_loc, dtype=np.float64))

            z_loc = _nnls_from_quadratic(ATA_reg_loc, ATy_loc,
                x0=np.maximum(z[active_idx], 0.0), max_iter=10000, tol=1e-11)

            alpha_loc = None
            lambda_loc = None

            info_loc = {
                "accepted": True,
                "mu": 0.0,
                "lambda_orbit": np.zeros((C,), dtype=np.float64),
            }

        info_loc = dict(info_loc)
        info_loc["numerical_ridge"] = float(numerical_ridge_loc)
        info_loc["condition_number"] = float(cond_loc)

        return (active_idx, z_loc, info_loc, alpha_loc, lambda_loc, ATA_loc,
            ATy_loc, S_loc, numerical_ridge_loc, cond_loc)

    def _prune_zero_support(
        z_vec: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remove zero coefficients from the working support.

        Parameters
        ----------
        z_vec : ndarray
            Current scaled coefficient vector.
        active_mask : ndarray
            Current working-support mask.

        Returns
        -------
        dropped : ndarray
            Global indices removed from the working support.

        Raises
        ------
        None

        Examples
        --------
        >>> dropped = _prune_zero_support(z, active)
        """
        z_scale = max(1.0, float(np.max(np.abs(z_vec))))

        support_tol = 1e-12 * z_scale

        drop_mask = (active_mask & (z_vec <= support_tol))

        dropped = np.flatnonzero(drop_mask).astype(np.int64)

        if dropped.size > 0:
            active_mask[dropped] = False
            z_vec[dropped] = 0.0

        return dropped

    def _expand_minimum_l2_support(active_mask: np.ndarray,
        reduced_gradient: np.ndarray, n_kkt_per_orbit: int = 8) -> np.ndarray:
        """
        Expand the raw NNLS support for the secondary minimum-L2 solve.

        Retain the raw support, its direct SSP-grid neighbours, and the
        inactive populations with the smallest physical-space constrained KKT
        slack in each dynamical component.

        Parameters
        ----------
        active_mask : ndarray
            Raw converged active support with shape ``(C*P,)``.
        reduced_gradient : ndarray
            Full constrained reduced gradient in physical x-space.
        n_kkt_per_orbit : int, optional
            Maximum number of KKT-selected inactive populations added per
            dynamical component.

        Returns
        -------
        expanded : ndarray
            Expanded candidate-support mask with shape ``(C*P,)``.

        Raises
        ------
        ValueError
            If supplied arrays have incorrect dimensions.

        Examples
        --------
        Called internally after raw KKT convergence.
        """
        active_mask = np.asarray(active_mask, dtype=bool).ravel(order="C")
        reduced_gradient = np.asarray(reduced_gradient,
            dtype=np.float64).ravel(order="C")

        if active_mask.size != C*P:
            raise ValueError("active_mask has incorrect size.")
        if reduced_gradient.size != C*P:
            raise ValueError("reduced_gradient has incorrect size.")

        expanded = active_mask.copy()
        shape = (int(n_metals), int(n_ages), int(n_alphas))

        offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
            (0, 1, 0), (0, 0, -1), (0, 0, 1))

        # Direct physical SSP neighbours of the raw solution.
        for jj in np.flatnonzero(active_mask):
            cc = int(jj//P)
            pp = int(jj % P)
            iz, ia, ialpha = np.unravel_index(pp, shape, order="C")

            for dz, dt, da in offsets:
                jz = iz+dz
                jt = ia+dt
                ja = ialpha+da

                if (jz < 0 or jz >= n_metals or jt < 0 or jt >= n_ages
                    or ja < 0 or ja >= n_alphas):
                    continue

                qq = int(np.ravel_multi_index((jz, jt, ja), shape,
                    order="C"))
                kk = cc*P+qq

                if not known_zero_flat[kk]:
                    expanded[kk] = True

        # Add the inactive populations closest to the raw NNLS KKT boundary.
        # At the converged maximum-form NNLS solution:
        #
        #     x_j > 0  -> g_j = 0
        #     x_j = 0  -> g_j <= 0.
        #
        # Therefore the largest inactive reduced gradients are the columns
        # that can enter with the smallest first-order data penalty.
        n_kkt_per_orbit = max(0, int(n_kkt_per_orbit))

        for cc in range(C):
            if has_hard_orbit_shape and orbit_shape[cc] <= 0.0:
                continue

            i0 = cc*P
            i1 = i0+P

            candidate = ~active_mask[i0:i1]
            candidate &= ~known_zero_flat[i0:i1]

            if not np.any(candidate) or n_kkt_per_orbit == 0:
                continue

            local = np.flatnonzero(candidate)
            score = reduced_gradient[i0:i1][local]

            finite = np.isfinite(score)
            local = local[finite]
            score = score[finite]

            if local.size == 0:
                continue

            # Descending gradient: zero is the KKT boundary, while increasingly
            # negative values are progressively more expensive to activate.
            n_take = min(n_kkt_per_orbit, local.size)

            if local.size > n_take:
                select = np.argpartition(score, -n_take)[-n_take:]
                local = local[select]
                score = score[select]

            order = np.argsort(score, kind="stable")[::-1]
            local = local[order]

            expanded[i0+local] = True

        expanded[known_zero_flat] = False

        if has_hard_orbit_shape and zero_shape_orbits.size > 0:
            for cc in zero_shape_orbits:
                i0 = int(cc)*P
                expanded[i0:i0+P] = False

        return expanded

    # --------------------------------------------------------------------------
    
    if needs_kkt_initialise and has_hard_orbit_shape:
        active_idx = np.flatnonzero(active).astype(np.int64)

        if active_idx.size == 0:
            raise RuntimeError(
                "Cannot initialise constrained KKT state from empty x support.")

        ATA_init, ATy_init, S_init = _assemble_reduced(active_idx)
        diag_init = np.diag(ATA_init)
        diag_med_init = (
            float(np.median(diag_init)) if diag_init.size else 0.0)

        try:
            eigs_init = np.linalg.eigvalsh(ATA_init)
            emin_init = float(np.min(eigs_init))
            emax_init = float(np.max(eigs_init))
        except Exception:
            emin_init = 0.0
            emax_init = (
                float(np.max(diag_init)) if diag_init.size else 1.0)

        H_scientific_init = _scientific_hessian_reduced(active_idx, S_init)
        ATA_scientific_init = ATA_init + H_scientific_init

        numerical_ridge_init = float(numerical_ridge)
        try:
            eigs_init = np.linalg.eigvalsh(ATA_scientific_init)
            emin_init = float(eigs_init[0])
            emax_init = float(eigs_init[-1])
            cond_init = (
                emax_init / emin_init if emin_init > 0.0 else np.inf
            )
        except np.linalg.LinAlgError:
            emin_init = np.nan
            emax_init = np.nan
            cond_init = np.inf

        z_init, info_init = _orbit_hard_constraint_solve(
            ATA_sub_reg=ATA_init, ATy_sub=ATy_init,
            active_idx=active_idx, S_active=S_init,
            orbit_shape=np.asarray(orbit_shape, dtype=np.float64), C=C, P=P,
            numerical_ridge=numerical_ridge_init,
            scientific_hessian=H_scientific_init)

        if not info_init["accepted"]:
            raise RuntimeError("Initial constrained KKT solve failed: "
                f"{info_init['reason']}.")
        
        z[:] = 0.0
        z[active_idx] = z_init
        alpha_current = float(info_init["alpha"])
        lambda_orbit_current = np.asarray(info_init["lambda_orbit"],
            dtype=np.float64,).copy()
        ridge_current = float(numerical_ridge_init)

        print("[MONO][init] established constrained KKT state: "
            f"active={active_idx.size} alpha={alpha_current:.6e} "
            f"ridge={ridge_current:.6e}", flush=True)


    # ------------------------------------------------------------
    # Outer-loop termination diagnostics
    # ------------------------------------------------------------
    stop_reason = "max_iter"
    stop_category = "limit"
    stop_details = {}
    stop_iter = None
    stop_converged = False

    def _set_stop(
        reason: str,
        category: str,
        it: int,
        *,
        converged: bool = False,
        **details,
    ) -> None:
        """
        Record and report the reason for terminating the outer solver.

        Parameters
        ----------
        reason : str
            Machine-readable termination reason.
        category : str
            Broad termination category, such as ``"converged"``,
            ``"limit"``, ``"stall"``, ``"active_set"``, or
            ``"numerical"``.
        it : int
            Zero-based outer iteration index.
        converged : bool, optional
            Whether the termination represents solver convergence.
        **details
            Additional diagnostic values associated with the exit.

        Returns
        -------
        None

        Raises
        ------
        None

        Examples
        --------
        >>> _set_stop(
        ...     "max_active",
        ...     "active_set",
        ...     it,
        ...     active=982,
        ...     requested=32,
        ...     max_active=1000,
        ... )
        """
        nonlocal stop_reason
        nonlocal stop_category
        nonlocal stop_details
        nonlocal stop_iter
        nonlocal stop_converged

        stop_reason = str(reason)
        stop_category = str(category)
        stop_details = dict(details)
        stop_iter = int(it)
        stop_converged = bool(converged)

        detail_text = " ".join(f"{key}={value}"
            for key, value in stop_details.items())

        print(f"[MONO][STOP] reason={stop_reason} "
            f"category={stop_category} converged={stop_converged} "
            f"iter={it + 1}/{max_iter}"
            + (f" {detail_text}" if detail_text else ""), flush=True)

        _emit_diag({
            "kind": "termination",
            "reason": stop_reason,
            "category": stop_category,
            "converged": stop_converged,
            "iter": int(it + 1),
            "iter_zero_based": int(it),
            "start_iter": int(start_iter),
            "max_iter": int(max_iter),
            "n_active": int(np.count_nonzero(active)),
            "max_active": int(max_active),
            "alpha": (float(alpha_current) if has_hard_orbit_shape
                else None),
            "details": stop_details,
        })

    # ------------------------------------------------------------
    # Active-set outer loop
    # ------------------------------------------------------------
    for it in range(start_iter, max_iter):
        t_iter = time.perf_counter()
        z_iter_start = z.copy()
        active_iter_start = active.copy()

        ATAz_scaled = _compute_ATAz_scaled(z)
        data_objective_current = (0.5 * float(np.dot(z, ATAz_scaled))
            - float(np.dot(ATy_scaled, z)))

        grad_data = ATy_scaled - ATAz_scaled

        # Scientific smoothness gradient. The penalty is defined in physical
        # x-space, so transform its gradient back to z-space with dx/dz = S.
        x_current = S_flat * z
        grad_regularisation_x = _scientific_gradient_x(x_current)
        grad_regularisation = S_flat * grad_regularisation_x
        # Numerical ridge is deliberately separate from the scientific
        # penalty and exists only to stabilise the reduced linear algebra.
        grad_numerical_ridge = float(ridge_current) * z
        grad_objective = (grad_data - grad_regularisation
            - grad_numerical_ridge)

        if has_hard_orbit_shape:
            constraint_gradient = S_flat * lambda_orbit_current[orbit_of_col]
            grad_total = grad_objective - constraint_gradient
        else:
            constraint_gradient = np.zeros(CP, dtype=np.float64)
            grad_total = grad_data - grad_regularisation - grad_numerical_ridge
        grad_x = np.divide(grad_objective, S_flat,
            out=np.full(CP, -np.inf, dtype=np.float64),
            where=S_flat > 0.0)

        candidate_mask = ~active
        candidate_mask &= ~known_zero_flat
        if has_hard_orbit_shape:
            candidate_mask &= (orbit_shape[orbit_of_col] > 0.0)
        not_active = np.flatnonzero(candidate_mask)
        gvals_data = grad_data[not_active]
        gvals_constraint = constraint_gradient[not_active]
        gvals_total = grad_total[not_active]

        max_grad_all = (
            float(np.max(grad_total))
            if grad_total.size else 0.0)
        max_grad_data = (
            float(np.max(gvals_data))
            if gvals_data.size else 0.0)
        max_grad_orbit = (
            float(np.max(np.abs(gvals_constraint)))
            if gvals_constraint.size else 0.0)

        tol_here = max(tol_grad, tol_grad_rel * grad_ref)

        # Check the full KKT conditions before attempting any promotion.
        #
        # For the maximization-form reduced gradient used here,
        #
        #     z_j > 0  ->  g_j = 0
        #     z_j = 0  ->  g_j <= 0
        #
        # up to the numerical KKT tolerance. The historical active-set mask
        # is deliberately not part of this mathematical classification.
        z_scale = max(1.0, float(np.max(np.abs(z))))
        positive_tol = 1e-12 * z_scale

        positive_mask = z > positive_tol
        positive_mask &= ~known_zero_flat

        free_mask = ~known_zero_flat

        if has_hard_orbit_shape:
            positive_shape_mask = orbit_shape[orbit_of_col] > 0.0
            positive_mask &= positive_shape_mask
            free_mask &= positive_shape_mask

        zero_free_mask = free_mask & ~positive_mask

        max_grad_active = (float(np.max(np.abs(grad_total[positive_mask])))
            if np.any(positive_mask) else 0.0)

        max_grad_inactive = (float(np.max(grad_total[zero_free_mask]))
            if np.any(zero_free_mask) else -np.inf)
        # Exact dual-KKT violators. These columns must be admitted to the
        # working set before the current point can be a solution of the full
        # convex constrained NNLS problem.
        dual_violation_mask = zero_free_mask & (grad_total > tol_here)
        dual_violators = np.flatnonzero(dual_violation_mask).astype(np.int64)

        if dual_violators.size:
            order = np.argsort(grad_total[dual_violators], kind="stable")[::-1]
            dual_violators = dual_violators[order]

        max_grad_dual = max(max_grad_inactive, 0.0)
        kkt_violation = max(max_grad_active, max_grad_dual)
        kkt_tol = float(tol_here)

        active_ok = max_grad_active <= kkt_tol
        dual_ok = max_grad_dual <= kkt_tol
        support_stationary = active_ok and dual_ok

        if np.any(zero_free_mask):
            zero_free_idx = np.flatnonzero(zero_free_mask)
            jj = int(np.argmax(grad_total[zero_free_idx]))
            best_dual_col = int(zero_free_idx[jj])
            best_dual_value = float(grad_total[best_dual_col])
        else:
            best_dual_col = -1
            best_dual_value = -np.inf

        # --------------------------------------------------------------
        # Per-iteration objective and primal-constraint diagnostics.
        #
        # ATAz has already been computed for the current committed state,
        # so all objective terms below are cheap scalar/vector operations.
        # No additional HyperCube pass is required.
        # --------------------------------------------------------------
        data_objective_iter = float(data_objective_current)
        residual_objective_iter = data_objective_iter + 0.5 * float(yty)
        scientific_regularisation_objective_iter = _scientific_objective_x(
            x_current)
        numerical_ridge_objective_iter = (0.5 * float(ridge_current) * float(np.dot(z, z)))
        scientific_objective_iter = (data_objective_iter
            + scientific_regularisation_objective_iter)
        total_objective_iter = (scientific_objective_iter
            + numerical_ridge_objective_iter)
        smooth_to_residual_iter = (scientific_regularisation_objective_iter
            / max(residual_objective_iter, 1e-300))

        if has_hard_orbit_shape:
            orbit_mass_iter, orbit_target_iter, orbit_deficit_iter = \
                _orbit_mass_and_deficit(z, alpha_value=alpha_current)
            orbit_resid_l1_iter = float(np.sum(np.abs(orbit_deficit_iter)))
            orbit_resid_l2_iter = float(np.linalg.norm(orbit_deficit_iter))
            orbit_resid_linf_iter = float(np.max(np.abs(orbit_deficit_iter)))
            orbit_resid_rel_linf_iter = float(orbit_resid_linf_iter
                / max(abs(float(alpha_current)), 1.0))
        else:
            orbit_mass_iter = np.sum(x_current.reshape(C, P), axis=1)
            orbit_target_iter = np.zeros((C,), dtype=np.float64)
            orbit_deficit_iter = orbit_target_iter - orbit_mass_iter

            orbit_resid_l1_iter = None
            orbit_resid_l2_iter = None
            orbit_resid_linf_iter = None
            orbit_resid_rel_linf_iter = None
        if diag_level >= 1 and ((it % diag_stride) == 0):
            _emit_diag({
                "kind": "kkt",
                "source": "streamActiveSetNNLS",
                "iter": int(it + 1),

                # ------------------------------------------------------
                # KKT convergence.
                # ------------------------------------------------------
                "tol": float(tol_here),
                "positive_tol": float(positive_tol),
                "n_working_active": int(np.count_nonzero(active)),
                "n_positive": int(np.count_nonzero(positive_mask)),
                "n_zero_free": int(np.count_nonzero(zero_free_mask)),
                "max_grad_active": float(max_grad_active),
                "max_grad_inactive": float(max_grad_inactive),
                "max_grad_dual": float(max_grad_dual),
                "max_grad_data": float(np.max(np.abs(grad_data))),
                "max_grad_regularisation": float(
                    np.max(np.abs(grad_regularisation))),
                "max_grad_numerical_ridge": float(
                    np.max(np.abs(grad_numerical_ridge))),
                "max_grad_constraint": float(np.max(np.abs(
                    constraint_gradient))),
                "kkt_violation": float(kkt_violation),
                "kkt_tol": float(kkt_tol),
                "active_ok": bool(active_ok),
                "dual_ok": bool(dual_ok),
                "support_stationary": bool(support_stationary),

                # ------------------------------------------------------
                # Worst inactive KKT violation.
                # ------------------------------------------------------
                "best_dual_col": int(best_dual_col),
                "best_dual_value": float(best_dual_value),
                "best_dual_orbit": (int(best_dual_col // P)
                    if best_dual_col >= 0 else -1),
                "best_dual_population": (int(best_dual_col % P)
                    if best_dual_col >= 0 else -1),
                "n_dual_violators": int(dual_violators.size),
                "kkt_promote_batch": int(kkt_promote_batch),

                # ------------------------------------------------------
                # Current constrained state.
                # ------------------------------------------------------
                "ridge": float(ridge_current),
                "alpha": (float(alpha_current) if has_hard_orbit_shape
                    else None),
                "constraint_dot_lambda": (float(np.dot(constraint_shape,
                        lambda_orbit_current,))
                    if has_hard_orbit_shape else None),

                # ------------------------------------------------------
                # Primal equality-constraint feasibility.
                # ------------------------------------------------------
                "orbit_resid_l1": orbit_resid_l1_iter,
                "orbit_resid_l2": orbit_resid_l2_iter,
                "orbit_resid_linf": orbit_resid_linf_iter,
                "orbit_resid_rel_linf": orbit_resid_rel_linf_iter,

                # ------------------------------------------------------
                # Objective decomposition.
                # ------------------------------------------------------
                "data_objective": float(data_objective_iter),
                "residual_objective": float(residual_objective_iter),
                "scientific_regularisation_objective": float(
                    scientific_regularisation_objective_iter),
                "scientific_objective": float(scientific_objective_iter),
                "numerical_ridge_objective": float(
                    numerical_ridge_objective_iter),
                "total_objective": float(total_objective_iter),
                "smooth_to_residual": float(smooth_to_residual_iter),
            })

        print(f"[MONO][iter {it + 1}] "
            f"active_stat={max_grad_active:.3e} "
            f"dual={max_grad_dual:.3e} tol={tol_here:.3e} "
            f"violators={dual_violators.size} "
            f"working={np.count_nonzero(active)} "
            f"positive={np.count_nonzero(positive_mask)}", flush=True)
        
        if support_stationary:
            # The complete convex quadratic problem satisfies the full-space
            # KKT conditions, which are sufficient for global optimality.
            _set_stop(
                "kkt_stationary",
                "converged",
                it,
                converged=True,
                n_active=int(np.count_nonzero(active)),
                max_grad_active=float(max_grad_active),
                max_grad_dual=float(max_grad_dual),
                tol_here=float(tol_here),
            )
            _emit_checkpoint(
                it,
                final=True,
                phase="kkt_stationary",
            )
            break


        # ------------------------------------------------------------
        # Exact primal-dual active-set expansion.
        #
        # For the convex quadratic objective with linear equality constraints
        # and z >= 0, a positive reduced gradient at a zero coefficient is a
        # genuine KKT violation. Admit those columns directly, re-solve the
        # complete working set, and recompute the full-space KKT conditions on
        # the next outer iteration.
        #
        # Do not send known KKT violators through heuristic exploration.
        # ------------------------------------------------------------
        if dual_violators.size:
            room = int(max_active) - int(np.count_nonzero(active))

            if room <= 0:
                _set_stop(
                    "max_active_with_dual_violation",
                    "numerical",
                    it,
                    converged=False,
                    n_active=int(np.count_nonzero(active)),
                    n_dual_violators=int(dual_violators.size),
                    max_grad_dual=float(max_grad_dual),
                    tol_here=float(tol_here),
                )
                break
            # KKT conditions, which are sufficient for global optimality.

            # --------------------------------------------------------
            # Adaptive KKT-promotion batch.
            #
            # Strongly violating SSP columns can be extremely correlated.
            # Admitting many of them simultaneously may therefore make the
            # enlarged constrained reduced problem numerically difficult even
            # though smaller subsets are perfectly admissible.
            #
            # Start with the normal promotion batch. If its reduced solve
            # fails, repeatedly halve the number of promoted columns while
            # preserving their KKT-gradient ordering:
            #
            #     32 -> 16 -> 8 -> 4 -> 2 -> 1
            #
            # The current committed state is never modified during these
            # attempts. Only a successful reduced solve is committed below.
            # Failure of the single strongest dual violator is therefore the
            # only promotion failure that terminates the outer solver.
            # --------------------------------------------------------
            promote_n = min(
                int(kkt_promote_batch), int(room), int(dual_violators.size))

            promotion_attempts = []
            promotion_solved = False

            while promote_n >= 1:
                promote = dual_violators[:promote_n]

                active_trial = active.copy()
                active_trial[promote] = True
                active_idx = np.flatnonzero(active_trial).astype(np.int64)

                active_idx, z_trial, info_trial, alpha_trial, lambda_trial, \
                    ATA_trial, ATy_trial, S_trial, ridge_trial, cond_trial = \
                    _solve_reduced_active(active_idx)

                solve_accepted = (info_trial is not None
                and bool(info_trial.get("accepted", False)))

                solve_reason = (str(info_trial.get("reason", "unknown"))
                    if info_trial is not None else "reduced_solve_failed")

                promotion_attempts.append({
                    "batch_size": int(promote_n),
                    "promoted": promote.tolist(),
                    "accepted": bool(solve_accepted),
                    "solve_reason": solve_reason,
                })

                if solve_accepted:
                    promotion_solved = True
                    break

                next_n = promote_n // 2
                if next_n < 1:
                    break

                print(f"[MONO][iter {it + 1}] KKT promotion solve failed "
                    f"for batch={promote_n}; retrying batch={next_n} "
                    f"reason={solve_reason}", flush=True)
                promote_n = next_n

            if not promotion_solved:
                _set_stop(
                    "kkt_promotion_solve_failed",
                    "numerical",
                    it,
                    converged=False,
                    n_active=int(np.count_nonzero(active)),
                    promoted=promote.tolist(),
                    attempted_batch_sizes=[int(attempt["batch_size"])
                        for attempt in promotion_attempts],
                    n_dual_violators=int(dual_violators.size),
                    max_grad_dual=float(max_grad_dual),
                    tol_here=float(tol_here),
                    solve_reason=solve_reason,
                )
                break

            if len(promotion_attempts) > 1:
                print(f"[MONO][iter {it + 1}] KKT promotion recovered "
                    f"with batch={promote.size} after "
                    f"{len(promotion_attempts)-1} retries", flush=True)

            # Commit the exact constrained solution. The reduced solver may
            # return some promoted or historical columns at zero; remove those
            # from the working mask after the solve.
            active[:] = False
            active[active_idx] = True
            z[:] = 0.0
            z[active_idx] = z_trial
            dropped_zero = _prune_zero_support(z, active)

            if has_hard_orbit_shape:
                alpha_current = float(alpha_trial)
                lambda_orbit_current = np.asarray(lambda_trial, dtype=np.float64
                    ).copy()

            ridge_current = float(ridge_trial)

            _emit_diag({
                "kind": "kkt_promotion",
                "source": "streamActiveSetNNLS",
                "iter": int(it + 1),
                "n_dual_violators": int(dual_violators.size),
                "n_promoted": int(promote.size),
                "requested_promote_batch": int(kkt_promote_batch),
                "attempted_batch_sizes": [int(attempt["batch_size"])
                    for attempt in promotion_attempts],
                "promotion_retries": int(len(promotion_attempts)-1),
                "promoted": promote.tolist(),
                "promoted_grad": grad_total[promote].tolist(),
                "worst_grad_before": float(max_grad_dual),
                "n_active_after": int(np.count_nonzero(active)),
                "n_zero_dropped": int(dropped_zero.size),
                "zero_dropped": dropped_zero.tolist(),
                "alpha": (float(alpha_current)
                    if has_hard_orbit_shape else None),
            })

            _emit_checkpoint(
                it,
                final=False,
                phase="kkt_promotion",
            )

            # ATAz and all multipliers/gradients must be recomputed from the
            # newly committed solution before making another KKT decision.
            continue
        # Reaching this point means that the full-space KKT conditions are
        # not satisfied, but there is no inactive coefficient with a positive
        # dual violation to promote. This is internally inconsistent and must
        # terminate as a numerical failure rather than falling through to any
        # heuristic support search.
        _set_stop(
            "nonstationary_without_promotable_dual",
            "numerical",
            it,
            converged=False,
            n_active=int(np.count_nonzero(active)),
            n_dual_violators=0,
            max_grad_active=float(max_grad_active),
            max_grad_dual=float(max_grad_dual),
            tol_here=float(tol_here),
        )
        break

    try:
        final_it = int(it)
    except Exception:
        final_it = 0

    pure_kkt_x = None
    if bool(stop_converged):
        pure_kkt_x = _current_x_from_z(z).copy()
    minimum_l2_gradient_x = None
    if use_minimum_l2 and bool(stop_converged):
        minimum_l2_gradient_x = np.divide(grad_total, S_flat,
            out=np.full(CP, -np.inf, dtype=np.float64),
            where=S_flat > 0.0)
    minimum_l2_used = False
    minimum_l2_raw_residual = None
    minimum_l2_final_residual = None
    minimum_l2_raw_norm = None
    minimum_l2_final_norm = None
    minimum_l2_strength = 0.0
    minimum_l2_raw_active = None
    minimum_l2_expanded_active = None
    minimum_l2_final_active = None

    if use_minimum_l2 and bool(stop_converged) and np.any(active):
        # ------------------------------------------------------------------
        # Lexicographic minimum-L2 solution.
        #
        # The primary constrained NNLS problem has already converged. The
        # following stage selects the least-concentrated physical coefficient
        # vector whose data residual remains within minimum_l2_rss_rtol of
        # that raw optimum.
        #
        # Assemble the expanded-support normal equations exactly once. Every
        # L2 trial, including its data objective, is then evaluated entirely
        # in this reduced space. No additional full HyperCube ATAz operation
        # is required by the L2 stage.
        # ------------------------------------------------------------------
        minimum_l2_max_trials = 5
        minimum_l2_refine = 5

        z_raw = np.asarray(z, dtype=np.float64).copy()
        raw_active_idx = np.flatnonzero(active).astype(np.int64)
        minimum_l2_raw_active = int(raw_active_idx.size)

        # Expand the converged raw support by one direct neighbour along
        # metallicity, age, and alpha within each dynamical component.
        #
        # Keep this candidate support fixed throughout the continuation so
        # initially zero neighbours remain available to enter the solution.
        l2_support = _expand_minimum_l2_support(active,
            minimum_l2_gradient_x, n_kkt_per_orbit=8)

        l2_active_idx = np.flatnonzero(l2_support).astype(np.int64)

        minimum_l2_expanded_active = int(l2_active_idx.size)
        n_l2_added = int(np.count_nonzero(l2_support & ~active))

        print(f"[MONO][minimum-L2] raw_active={minimum_l2_raw_active} "
            f"expanded_active={minimum_l2_expanded_active} "
            f"added={n_l2_added} rss_rtol={minimum_l2_rss_rtol:.3e}", flush=True)

        if l2_active_idx.size > max_active:
            raise RuntimeError('Minimum-L2 expanded support exceeds max_active: '
                f"{l2_active_idx.size} > {max_active}.")

        print(f"[MONO][minimum-L2] raw_active={minimum_l2_raw_active} "
            f"expanded_active={minimum_l2_expanded_active} "
            f"rss_rtol={minimum_l2_rss_rtol:.3e}", flush=True)

        # This is the only additional HyperCube Gram-matrix assembly required
        # by the complete minimum-L2 stage.
        ATA_l2, ATy_l2, S_l2 = _assemble_reduced(l2_active_idx)

        assembled_l2 = (ATA_l2, ATy_l2, S_l2)

        # Embed the raw converged solution into the expanded reduced support.
        # Since l2_active_idx contains the original active support, this is
        # exactly the same physical solution represented in reduced space.
        z_raw_l2 = z_raw[l2_active_idx].copy()

        # Evaluate the raw data objective using the same reduced normal
        # equations that will be used for every L2 trial.
        data_objective_raw = _data_quad_obj(ATA_l2, ATy_l2, z_raw_l2)

        residual_objective_raw = (data_objective_raw+0.5*yty)

        minimum_l2_raw_residual = float(residual_objective_raw)

        # Physical-space norm on the reduced support. Since all coefficients
        # outside this support are zero in the raw solution, this is exactly
        # the full ||x||_2.
        x_raw_l2 = S_l2*z_raw_l2
        minimum_l2_raw_norm = float(np.linalg.norm(x_raw_l2))
        residual_limit = residual_objective_raw * (1.0+minimum_l2_rss_rtol)

        # Physical L2 objective:
        #
        #     0.5 * beta * x.T @ x
        #
        # with x = S*z, hence
        #
        #     H_L2,z = beta * diag(S**2).
        #
        # Normalize the secondary curvature by its median positive diagonal
        # so the continuation parameter is dimensionless.
        l2_diag = S_l2*S_l2

        positive_l2_diag = l2_diag[l2_diag > 0.0]

        if positive_l2_diag.size == 0:
            l2_norm = 1.0
        else:
            l2_norm = float(np.median(positive_l2_diag))

        # Store the best solution only in reduced coordinates during the
        # continuation. Construct a full CP vector only once, when committing
        # the selected solution.
        best_z_l2 = z_raw_l2.copy()
        best_alpha = float(alpha_current)
        best_lambda = np.asarray(lambda_orbit_current, dtype=np.float64).copy()

        best_residual = float(residual_objective_raw)
        best_l2_norm = float(minimum_l2_raw_norm)
        best_strength = 0.0

        lower_strength = None
        upper_strength = None

        def _minimum_l2_trial(strength):
            effective_strength = strength/l2_norm
            H_l2 = np.diag(effective_strength*l2_diag)

            active_idx_trial, z_trial_l2, info_trial, alpha_trial,\
                lambda_trial, _, _, _, _, _ = _solve_reduced_active(
                l2_active_idx, assembled=assembled_l2, extra_hessian=H_l2)

            if z_trial_l2 is None or not info_trial["accepted"]:
                return None

            if not np.array_equal(active_idx_trial, l2_active_idx):
                raise RuntimeError('Minimum-L2 reduced solve changed the '
                    'candidate support ordering.')

            z_trial_l2 = np.asarray(z_trial_l2, dtype=np.float64).ravel(
                order="C")
            data_objective_trial = _data_quad_obj(ATA_l2, ATy_l2,
                z_trial_l2)
            residual_objective_trial = data_objective_trial+0.5*yty
            x_norm_trial = float(np.linalg.norm(S_l2*z_trial_l2))
            residual_relative = ((residual_objective_trial
                - residual_objective_raw)/max(residual_objective_raw, 1e-300))

            return {
                "z": z_trial_l2,
                "alpha": float(alpha_trial),
                "lambda": np.asarray(lambda_trial, dtype=np.float64).copy(),
                "residual": float(residual_objective_trial),
                "relative": float(residual_relative),
                "norm": float(x_norm_trial),
            }

        def _accept_minimum_l2_trial(result, strength):
            nonlocal best_z_l2, best_alpha, best_lambda, best_residual,\
                best_l2_norm, best_strength

            best_z_l2 = result["z"].copy()
            best_alpha = float(result["alpha"])
            best_lambda = result["lambda"].copy()
            best_residual = float(result["residual"])
            best_l2_norm = float(result["norm"])
            best_strength = float(strength)

        # Start at the requested RSS scale. If this is too strong, search
        # downward; if acceptable, search upward until the RSS limit is crossed.
        trial_strength = max(1e-12, minimum_l2_rss_rtol)
        result = _minimum_l2_trial(trial_strength)

        if result is None:
            acceptable = False
            residual_relative = np.inf
            norm_ratio = 1.0
        else:
            acceptable = result["residual"] <= residual_limit
            residual_relative = result["relative"]
            norm_ratio = result["norm"]/max(minimum_l2_raw_norm, 1e-300)

        print(f"[MONO][minimum-L2] trial= 0 "
            f"strength={trial_strength:.3e} "
            f"dRSS/RSS={residual_relative:.3e} "
            f"||x||/||x0||={norm_ratio:.6e}", flush=True)

        if acceptable:
            lower_strength = float(trial_strength)
            _accept_minimum_l2_trial(result, trial_strength)

            for l2_iter in range(1, minimum_l2_max_trials):
                trial_strength *= 10.0
                result = _minimum_l2_trial(trial_strength)

                if result is None:
                    acceptable = False
                    residual_relative = np.inf
                    norm_ratio = 1.0
                else:
                    acceptable = result["residual"] <= residual_limit
                    residual_relative = result["relative"]
                    norm_ratio = (result["norm"]
                        / max(minimum_l2_raw_norm, 1e-300))

                print(f"[MONO][minimum-L2] trial={l2_iter:2d} "
                    f"strength={trial_strength:.3e} "
                    f"dRSS/RSS={residual_relative:.3e} "
                    f"||x||/||x0||={norm_ratio:.6e}", flush=True)

                if acceptable:
                    lower_strength = float(trial_strength)
                    _accept_minimum_l2_trial(result, trial_strength)
                else:
                    upper_strength = float(trial_strength)
                    break

        else:
            upper_strength = float(trial_strength)

            for l2_iter in range(1, minimum_l2_max_trials):
                trial_strength *= 0.1
                result = _minimum_l2_trial(trial_strength)

                if result is None:
                    acceptable = False
                    residual_relative = np.inf
                    norm_ratio = 1.0
                else:
                    acceptable = result["residual"] <= residual_limit
                    residual_relative = result["relative"]
                    norm_ratio = (result["norm"]
                        / max(minimum_l2_raw_norm, 1e-300))

                print(f"[MONO][minimum-L2] trial={l2_iter:2d} "
                    f"strength={trial_strength:.3e} "
                    f"dRSS/RSS={residual_relative:.3e} "
                    f"||x||/||x0||={norm_ratio:.6e}", flush=True)

                if acceptable:
                    lower_strength = float(trial_strength)
                    _accept_minimum_l2_trial(result, trial_strength)
                    break

                upper_strength = float(trial_strength)

        # Refine the acceptable/unacceptable boundary logarithmically.
        if (lower_strength is not None and upper_strength is not None
            and upper_strength > lower_strength):
            for l2_iter in range(minimum_l2_refine):
                trial_strength = float(np.sqrt(
                    lower_strength*upper_strength))
                result = _minimum_l2_trial(trial_strength)

                if result is None:
                    upper_strength = float(trial_strength)
                    continue

                print(f"[MONO][minimum-L2] refine={l2_iter:2d} "
                    f"strength={trial_strength:.3e} "
                    f"dRSS/RSS={result['relative']:.3e}", flush=True)

                if result["residual"] <= residual_limit:
                    lower_strength = float(trial_strength)
                    _accept_minimum_l2_trial(result, trial_strength)
                else:
                    upper_strength = float(trial_strength)

        elif lower_strength is None:
            print("[MONO][minimum-L2] WARNING: no acceptable nonzero L2 "
                "strength found.", flush=True)

        elif upper_strength is None:
            print("[MONO][minimum-L2] WARNING: RSS boundary not bracketed; "
                f"strongest tested strength={lower_strength:.3e}", flush=True)

        # ------------------------------------------------------------------
        # Commit the selected lexicographic solution.
        #
        # This is the only point at which the reduced minimum-L2 solution is
        # expanded back into the complete CP coefficient vector.
        # ------------------------------------------------------------------
        if best_strength > 0.0:
            z[:] = 0.0
            z[l2_active_idx] = best_z_l2

            active[:] = False

            z_scale_l2 = max(1.0, float(np.max(np.abs(z))))
            l2_positive_tol = (1e-12*z_scale_l2)

            active[z > l2_positive_tol] = True
            z[~active] = 0.0

            alpha_current = float(best_alpha)
            lambda_orbit_current = (best_lambda.copy())

            minimum_l2_used = True

        minimum_l2_strength = float(best_strength)
        minimum_l2_final_residual = float(best_residual)
        minimum_l2_final_norm = float(best_l2_norm)
        minimum_l2_final_active = int(np.count_nonzero(active))

        residual_relative = ((best_residual - residual_objective_raw) / max(
            residual_objective_raw, 1e-300))

        norm_reduction = (1.0- best_l2_norm / max(minimum_l2_raw_norm, 1e-300))

        print(f"[MONO][minimum-L2] accepted={minimum_l2_used} "
            f"strength={best_strength:.6e} dRSS/RSS={residual_relative:.6e} "
            f"norm_reduction={norm_reduction:.3%} "
            f"active={np.count_nonzero(active)}", flush=True)

    # Final objective summary in the solver's own scaled space.
    x_final = _current_x_from_z(z)
    x_final_cp = x_final.reshape(C, P)
    final_population = np.argmax(x_final_cp, axis=1)
    curvature_raw = 0.5*float(np.sum((x_final_cp @ Q_ssp)*x_final_cp))
    smooth_objective = smooth_scale*curvature_raw

    if seed_population is not None:
        same_as_seed = (final_population[positive_shape_orbits]
            == seed_population[positive_shape_orbits])
        print('[MONO][seed-vs-final] dominant population indices:', flush=True)
        for cc in positive_shape_orbits:
            cc = int(cc)
            print(f"  component={cc:3d} seed={seed_population[cc]:3d} "
                f"final={final_population[cc]:3d} "
                f"same={bool(final_population[cc] == seed_population[cc])}",
                flush=True)

        print(f"[MONO][seed-vs-final] same={np.count_nonzero(same_as_seed)}/"
            f"{positive_shape_orbits.size} ({np.mean(same_as_seed):.1%})",
            flush=True)
    elif resume_state:
        print("[MONO][seed-vs-final] unavailable for full-resume run",
            flush=True)

    if minimum_l2_used:
        residual_objective = float(minimum_l2_final_residual)
        data_objective = (residual_objective-0.5*yty)
    else:
        ATAz_final = _compute_ATAz_scaled(z)
        data_objective = (0.5 * float(np.dot(z, ATAz_final))
            - float(np.dot(ATy_scaled, z)))
        residual_objective = data_objective+0.5*yty
    
    scientific_regularisation_objective = _scientific_objective_x(x_final)
    numerical_ridge_objective = 0.5 * float(ridge_current) * float(np.dot(z, z))
    # Scientific objective actually being optimized conceptually
    scientific_objective = data_objective + scientific_regularisation_objective
    total_objective = scientific_objective + numerical_ridge_objective
    # Report the scientific regularization strength relative to the actual
    # residual objective. Unlike data_objective, residual_objective includes
    # the constant 0.5*y.T@y and is therefore 0.5*||A*x-y||^2.
    smooth_to_residual = (scientific_regularisation_objective/
        max(residual_objective, 1e-300))
    # Get fitted wavelength pixels
    with open_h5(h5_path, role="reader") as f:
        n_wave_fit = (int(keep_idx.size) if keep_idx is not None
            else int(f["/DataCube"].shape[1]))
    n_fit_pixels = int(sum(s1-s0 for s0, s1 in s_ranges)*n_wave_fit)

    residual_ss = 2.0*float(residual_objective)
    rmse_final = float(np.sqrt(
        max(residual_ss, 0.0)/max(n_fit_pixels, 1)))
    print(f"[MONO][objective] residual={residual_objective:.6e} "
        f"rmse={rmse_final:.6e} "
        f"curvature={curvature_raw:.6e} "
        f"smooth={scientific_regularisation_objective:.6e} "
        f"smooth/residual={smooth_to_residual:.6e}", flush=True)

    if has_hard_orbit_shape:
        orbit_mass, orbit_target, orbit_deficit = _orbit_mass_and_deficit(z,
            alpha_value=alpha_current)
        orbit_resid_l1 = float(np.sum(np.abs(orbit_deficit)))
        orbit_resid_l2 = float(np.linalg.norm(orbit_deficit))
        orbit_resid_linf = float(np.max(np.abs(orbit_deficit)))
    else:
        orbit_mass = np.sum(x_final.reshape(C, P), axis=1)
        orbit_target = np.zeros((C,), dtype=np.float64)
        orbit_deficit = orbit_target - orbit_mass
        orbit_resid_l1 = None
        orbit_resid_l2 = None
        orbit_resid_linf = None

    oL1 = (orbit_resid_l1 if orbit_resid_l1 is not None else float('nan'))
    oLInf = (orbit_resid_linf if orbit_resid_linf is not None else float('nan'))
    print(f"[MONO][final] primary_stop={stop_reason} "
        f"primary_converged={stop_converged} iter="
        f"{((stop_iter + 1) if stop_iter is not None else 0)}/"
        f"{max_iter} active={int(np.count_nonzero(active))}/{int(max_active)} "
        f"minimum_L2={minimum_l2_used} "
        f"data_obj={data_objective:.3e} total_obj={total_objective:.3e} "
        f"alpha={alpha_current:.6e} orbit_L1={oL1:.3e} "
        f"orbit_Linf={oLInf:.3e}", flush=True)

    _emit_diag({
        "kind": "objective_summary",
        "source": "streamActiveSetNNLS",
        "data_objective": float(data_objective),
        "residual_objective": float(residual_objective),
        "smooth_to_residual": float(smooth_to_residual),
        "residual_ss": float(residual_ss),
        "rmse": float(rmse_final),
        "n_fit_pixels": int(n_fit_pixels),
        "scientific_regularisation_objective": float(
            scientific_regularisation_objective),
        "scientific_objective": float(scientific_objective),
        "numerical_ridge_objective": float(numerical_ridge_objective),
        "total_objective": float(total_objective),
        "scientific_regularisation": float(regularisation_scale),
        "scientific_regularisation_mode": SCIENTIFIC_REGULARISATION_MODE,
        "smoothness_normalisation": float(smooth_norm),
        "effective_smoothness_scale": float(smooth_scale),
        "numerical_ridge": float(ridge_current),
        "orbit_resid_l1": orbit_resid_l1,
        "orbit_resid_l2": orbit_resid_l2,
        "orbit_resid_linf": orbit_resid_linf,
        "orbit_mass": orbit_mass.tolist(),
        "orbit_target": orbit_target.tolist(),
        "orbit_resid": orbit_deficit.tolist(),
        "alpha": float(alpha_current) if has_hard_orbit_shape else None,
        "orbit_shape": orbit_shape.tolist() if orbit_shape is not None else \
            None,
        "orbit_prior_present": bool(has_hard_orbit_shape),
        "orbit_constraint_mode": (
            "equal_component_mass" if has_hard_orbit_shape else "none"),
        "x": x_final.ravel(order="C").tolist(),
        "stop_reason": str(stop_reason),
        "stop_category": str(stop_category),
        "stop_converged": bool(stop_converged),
        "stop_iter": int(stop_iter + 1) if stop_iter is not None else None,
        "max_iter": int(max_iter),
        "n_active_final": int(np.count_nonzero(active)),
        "max_active": int(max_active),
        "stop_details": dict(stop_details),
        "minimum_l2_used": bool(minimum_l2_used),
        "minimum_l2_strength": float(minimum_l2_strength),
        "minimum_l2_rss_rtol": (float(minimum_l2_rss_rtol)
            if use_minimum_l2 else None),
        "minimum_l2_raw_residual": minimum_l2_raw_residual,
        "minimum_l2_final_residual": minimum_l2_final_residual,
        "minimum_l2_raw_norm": minimum_l2_raw_norm,
        "minimum_l2_final_norm": minimum_l2_final_norm,
        "minimum_l2_raw_active": minimum_l2_raw_active,
        "minimum_l2_expanded_active": minimum_l2_expanded_active,
        "minimum_l2_final_active": minimum_l2_final_active,
    })
    _emit_diag({
        "kind": "final_solution",
        "source": "streamActiveSetNNLS",
        "iter": int(final_it + 1),
        "x": x_final.ravel(order="C").tolist(),
        "x_sum": float(np.sum(x_final)),
        "x_norm": float(np.linalg.norm(x_final)),
        "x_max": float(np.max(x_final)),
        "x_nnz": int(np.count_nonzero(x_final > 0.0)),
        "alpha": (float(alpha_current) if has_hard_orbit_shape else None),
        "orbit_mass": orbit_mass.tolist(),
        "orbit_target": orbit_target.tolist(),
        "orbit_resid": orbit_deficit.tolist(),
        "orbit_resid_linf": (float(orbit_resid_linf)
            if orbit_resid_linf is not None else None),
    })

    _emit_checkpoint(final_it, final=True, phase="complete")

    final_stats = {
        "residual_objective": float(residual_objective),
        "residual_ss": float(residual_ss),
        "rmse": float(rmse_final),
        "n_fit_pixels": int(n_fit_pixels),
        "data_objective": float(data_objective),
        "scientific_regularisation_objective": float(
            scientific_regularisation_objective),
        "smooth_to_residual": float(smooth_to_residual),
        "scientific_objective": float(scientific_objective),
        "numerical_ridge_objective": float(
            numerical_ridge_objective),
        "total_objective": float(total_objective),
        "stop_reason": str(stop_reason),
        "stop_category": str(stop_category),
        "converged": bool(stop_converged),
        "iterations": (
            int(stop_iter+1) if stop_iter is not None else None),
        "pure_kkt_x": pure_kkt_x,
    }

    return S_flat*z, final_stats

# ------------------------------------------------------------------------------

def _build_literal_Ab(h5_path: str, cfg: MPConfig):
    """
    Build the literal physical-space CubeFit design matrix and data vector.

    Parameters
    ----------
    h5_path : str
        Path to the CubeFit HDF5 file.
    cfg : MPConfig
        Solver configuration controlling masking and spatial tile size.

    Returns
    -------
    A : ndarray
        Physical-space design matrix with shape (N, K), where K contains all
        columns not excluded by ``known_zero_mask``.
    y : ndarray
        Flattened observed spectra with shape (N,).
    free_idx : ndarray
        Full ``C*P`` indices represented by the columns of ``A``.
    known_zero : ndarray
        Persistent structural exclusion mask with shape (C, P).
    dims : tuple
        ``(S, L, C, P)`` dimensions of the original problem.

    Raises
    ------
    RuntimeError
        If the HDF5 dimensions are inconsistent or no columns remain.
    ValueError
        If the assembled matrix or data contain non-finite values.

    Examples
    --------
    >>> A, y, free_idx, known_zero, dims = _build_literal_Ab(h5_path, cfg)
    """
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]
        S, L = map(int, DC.shape)
        Sm, C, P, Lm = map(int, M.shape)

        if DC.ndim != 2 or M.ndim != 4:
            raise RuntimeError(
                "Expected /DataCube (S,L) and /HyperCube/models (S,C,P,L).")
        if Sm != S or Lm != L:
            raise RuntimeError("Model and data dimensions are inconsistent.")

        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        chunks = M.chunks

    CP = int(C*P)
    known_zero = _get_known_zero_mask(h5_path, C, P)
    free_idx = np.flatnonzero(~known_zero.ravel(order="C")).astype(np.int64)

    if free_idx.size == 0:
        raise RuntimeError("No allowed columns remain for monolithic solve.")

    Lk = int(keep_idx.size) if keep_idx is not None else L
    n_rows = int(S*Lk)
    s_tile = int(cfg.s_tile_override or
        (chunks[0] if chunks and chunks[0] else 128))

    # A is intentionally materialized: this is the monolithic reference.
    A = np.empty((n_rows, free_idx.size), dtype=np.float64, order="F")
    y = np.empty(n_rows, dtype=np.float64)

    row0 = 0
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        iterator = range(0, S, s_tile)
        iterator = tqdm(iterator, desc="[MONOLITHIC] building A",
            disable=not getattr(cfg, "verbose", True))

        for s0 in iterator:
            s1 = min(S, s0+s_tile)
            Y = np.asarray(DC[s0:s1, :], dtype=np.float64)

            if keep_idx is not None:
                Y = Y[:, keep_idx]

            rows = Y.size
            row1 = row0+rows
            y[row0:row1] = Y.ravel(order="C")

            # Form the literal rows (s,l) x columns (c,p).
            slab = np.asarray(M[s0:s1, :, :, :], dtype=np.float64)
            if keep_idx is not None:
                slab = slab[:, :, :, keep_idx]

            A_tile = slab.transpose(0, 3, 1, 2).reshape(rows, CP)
            A[row0:row1, :] = A_tile[:, free_idx]
            row0 = row1

    if not np.all(np.isfinite(A)):
        raise ValueError("Literal A contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Literal y contains non-finite values.")

    return A, y, free_idx, known_zero, (S, L, C, P)

# ------------------------------------------------------------------------------

def monolithicNNLS(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    tracker: Optional[object] = None,
    block_size: Optional[int] = None,
    monolithic_max_active: int = 1000,
    regularisation_scale: float = 1.0,
    cache_path: str | None = None,
    reuse_cache: bool = True,
):
    """
    Solve the raw, unregularised CubeFit NNLS problem using literal A and y.

    The complete masked physical-space design matrix and data vector are
    materialized. No scientific regularisation, column scaling, streamed
    normal equations, support heuristics, or reduced active-set machinery are
    used.

    Without orbit constraints, this solves

        min ||A @ x - y||_2

        subject to x >= 0.

    With orbit constraints, it solves the same raw least-squares problem with
    the additional hard requirement that every participating dynamical
    component has equal total population coefficient mass.

    This function is intended as an independent unregularised reference
    solution for testing the streamed solver.

    Parameters
    ----------
    h5_path : str
        Path to the CubeFit HDF5 file.
    cfg : MPConfig
        HDF5 and masking configuration.
    orbit_weights : ndarray, optional
        Orbit weights. Positive entries select components participating in the
        equal-component-mass constraint. If None, conventional NNLS is used.
    x0 : ndarray, optional
        Physical-space initial solution with shape (C, P).
    resume_state : dict, optional
        Unsupported; accepted for API compatibility.
    tracker : object, optional
        Accepted for API compatibility and ignored.
    block_size : int, optional
        Accepted for API compatibility and ignored.
    monolithic_max_active : int, optional
        Accepted for API compatibility and ignored.
    regularisation_scale : float, optional
        Accepted for API compatibility and ignored. The monolithic reference
        solve is always completely unregularised.
    cache_path : str, optional
        Accepted for API compatibility and ignored.
    reuse_cache : bool, optional
        Accepted for API compatibility and ignored.

    Returns
    -------
    x : ndarray
        Physical-space solution with shape (C, P).
    stats : dict
        Monolithic solver diagnostics.

    Raises
    ------
    RuntimeError
        If the monolithic optimizer fails.
    ValueError
        If the inputs or orbit constraints are invalid.

    Examples
    --------
    >>> x, stats = monolithicNNLS(h5_path, cfg, orbit_weights=None)
    >>> x, stats = monolithicNNLS(
    ...     h5_path, cfg, orbit_weights=weights)
    """
    del tracker, block_size, monolithic_max_active
    del cache_path, reuse_cache, regularisation_scale

    if resume_state:
        raise ValueError("monolithicNNLS does not support full-state resume.")

    if not _HAS_SCIPY_OPTIMIZE:
        raise RuntimeError("SciPy optimize is required for monolithicNNLS.")

    t0 = time.perf_counter()
    A, y, free_idx, known_zero, dims = _build_literal_Ab(h5_path, cfg)
    S, L, C, P = dims
    CP = int(C*P)

    orbit_shape = _canon_orbit_weights(orbit_weights, C=C, P=P)
    constrained = orbit_shape is not None

    if constrained:
        positive_orbits = np.flatnonzero(orbit_shape > 0.0)

        blocked = np.flatnonzero(
            np.all(known_zero, axis=1) & (orbit_shape > 0.0))
        if blocked.size:
            raise ValueError(
                "known_zero_mask makes the orbit constraint infeasible for "
                f"components {blocked.tolist()}.")

        # Zero-weight components do not participate in the fitted model.
        use = orbit_shape[free_idx//P] > 0.0
        A = A[:, use]
        free_idx = free_idx[use]
    else:
        positive_orbits = np.zeros(0, dtype=np.int64)

    n_free = int(free_idx.size)
    n_rows = int(A.shape[0])
    if n_free == 0:
        raise RuntimeError("No free columns remain for monolithic solve.")

    print(f"[MONOLITHIC] A={A.shape[0]}x{A.shape[1]} "
        f"constrained={constrained}", flush=True)

    # ------------------------------------------------------------------
    # Classical unconstrained-orbit NNLS: literal A and y.
    # ------------------------------------------------------------------
    if not constrained:
        print("[MONOLITHIC] solving raw NNLS with scipy.optimize.nnls",
            flush=True)

        x_free, rnorm = nnls(A, y, maxiter=max(3*n_free, 10000))

        x_free = np.asarray(x_free, dtype=np.float64)

        solver_name = "scipy.optimize.nnls"
        iterations = None
        optimality = None

    # ------------------------------------------------------------------
    # Constrained literal least squares.
    #
    # Equal component masses are imposed without introducing alpha:
    #
    #   sum_p x[c,p] - sum_p x[c0,p] = 0
    #
    # for every positive orbit c != c0.
    # ------------------------------------------------------------------
    else:
        free_c = free_idx//P

        print("[MONOLITHIC] forming normal equations", flush=True)
        ATA = A.T @ A
        ATy = A.T @ y
        yty = float(np.dot(y, y))
        ATA = 0.5*(ATA+ATA.T)
        # Build the equal-component-mass constraints:
        #
        #   sum_p x[c, p] - sum_p x[c0, p] = 0
        #
        # for every participating component c != c0.
        free_c = free_idx//P
        n_constraints = max(0, int(positive_orbits.size)-1)
        E = np.zeros((n_constraints, n_free), dtype=np.float64)
        if n_constraints:
            reference = int(positive_orbits[0])
            for rr, cc in enumerate(positive_orbits[1:]):
                E[rr, free_c == int(cc)] = 1.0
                E[rr, free_c == reference] = -1.0

        del A

        # --------------------------------------------------------------
        # Raw unregularised constrained NNLS as a convex quadratic
        # program:
        #
        #   min 0.5*x.T@ATA@x - ATy.T@x
        #
        # subject to
        #
        #   x >= 0
        #   E@x = 0
        #
        # The omitted constant 0.5*y.T@y has no effect on the solution.
        # --------------------------------------------------------------
        scale = max(1.0, float(np.max(np.abs(ATy))) /
            max(float(np.max(np.diag(ATA))), 1e-300))
        diag_max = float(np.max(np.diag(ATA)))
        aty_max = float(np.max(np.abs(ATy)))
        x_scale = max(1.0, aty_max/max(diag_max, 1e-300))
        print("[MONOLITHIC] OSQP coefficient scale x_scale={x_scale:.6e}",
            flush=True)
        P_qp = sparse.csc_matrix(ATA*(x_scale*x_scale))
        q_qp = -np.asarray(ATy*x_scale, dtype=np.float64)

        I_qp = sparse.eye(n_free, format="csc", dtype=np.float64)

        if n_constraints:
            A_qp = sparse.vstack((I_qp, sparse.csc_matrix(E)), format="csc")

            lower = np.concatenate((np.zeros(n_free, dtype=np.float64),
                np.zeros(n_constraints, dtype=np.float64)))

            upper = np.concatenate((np.full(n_free, np.inf, dtype=np.float64),
                np.zeros(n_constraints, dtype=np.float64)))
        else:
            A_qp = I_qp
            lower = np.zeros(n_free, dtype=np.float64)
            upper = np.full(n_free, np.inf, dtype=np.float64)

        print("[MONOLITHIC] solving raw constrained NNLS with OSQP", flush=True)

        solver = osqp.OSQP()

        solver.setup(P=P_qp, q=q_qp, A=A_qp, l=lower, u=upper, eps_abs=1e-7,
            eps_rel=1e-7, max_iter=100000, polish=True, adaptive_rho=True,
            scaling=20, verbose=bool(getattr(cfg, "verbose", True)))

        result = solver.solve()
        status = str(result.info.status).lower()
        if result.x is None or status not in ("solved", "solved inaccurate"):
            raise RuntimeError("Monolithic constrained NNLS failed: "
                f"{result.info.status}")

        u_free = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(u_free)):
            raise RuntimeError("OSQP returned non-finite coefficients.")
        # Validate the solution in the scaled coordinates actually solved by
        # OSQP. Do not project or otherwise modify the reference solution.
        u_scale = max(1.0, float(np.max(np.abs(u_free))))
        min_u = float(np.min(u_free))
        negative_rel = max(0.0, -min_u)/u_scale
        if n_constraints:
            equality_resid = E @ u_free
            equality_rel = float(np.max(np.abs(equality_resid)))/u_scale
        else:
            equality_rel = 0.0
        feasibility_tol = 5e-7
        if negative_rel > feasibility_tol:
            raise RuntimeError("OSQP solution materially violates non-negativity: "
                f"relative violation={negative_rel:.6e}.")
        if equality_rel > feasibility_tol:
            raise RuntimeError("OSQP solution materially violates the equal-component-mass "
                f"constraint: relative violation={equality_rel:.6e}.")
        print("[MONOLITHIC] OSQP feasibility: "
            f"negative_rel={negative_rel:.3e} "
            f"mass_rel={equality_rel:.3e}",
            flush=True)
        # Transform the unmodified OSQP solution back to physical coefficient
        # units. Tiny negative entries are retained here because this is an
        # independent numerical reference solution, not the production fit.
        x_free = x_scale*u_free

        solver_name = "OSQP"
        iterations = int(result.info.iter)
        optimality = float(getattr(result.info, "dual_res", np.nan))

    # ------------------------------------------------------------------
    # Restore full physical coefficient vector and calculate diagnostics.
    # ------------------------------------------------------------------

    if constrained:
        residual_ss = (float(x_free @ (ATA @ x_free))
            - 2.0*float(np.dot(ATy, x_free)) + yty)
        # Normal-equation roundoff can produce a tiny negative value.
        residual_ss = max(residual_ss, 0.0)
    else:
        # Raw scipy.optimize.nnls already returns ||A*x-y||_2.
        residual_ss = float(rnorm*rnorm)
    data_objective = 0.5*residual_ss
    reg_objective = 0.0
    total_objective = data_objective

    # Remove only the tiny negative numerical leakage from the reference
    # QP solution before returning it as an NNLS solution.
    x_free_return = np.maximum(x_free, 0.0)
    x_flat = np.zeros(CP, dtype=np.float64)
    x_flat[free_idx] = x_free_return
    x_flat[known_zero.ravel(order="C")] = 0.0
    x = x_flat.reshape(C, P)
    orbit_mass = np.sum(x, axis=1)
    if constrained:
        alpha = float(np.mean(orbit_mass[positive_orbits]))
        orbit_target = alpha*(orbit_shape > 0.0)
        orbit_resid = orbit_mass-orbit_target
        orbit_linf = float(np.max(np.abs(orbit_resid[positive_orbits])))
    else:
        alpha = None
        orbit_target = None
        orbit_resid = None
        orbit_linf = None

    elapsed = float(time.perf_counter()-t0)
    stats = {
        "solver": solver_name,
        "success": True,
        "elapsed_sec": elapsed,
        "rows": int(n_rows),
        "cols": int(CP),
        "n_free": int(n_free),
        "n_active": int(np.count_nonzero(x_free > 0.0)),
        "x_nnz": int(np.count_nonzero(x > 0.0)),
        "x_sum": float(np.sum(x)),
        "x_norm": float(np.linalg.norm(x)),
        "x_max": float(np.max(x)),
        "residual_ss": residual_ss,
        "rmse": float(np.sqrt(residual_ss/max(1, n_rows))),
        "data_objective": data_objective,
        "total_objective": total_objective,
        "scientific_regularisation_objective": 0.0,
        "scientific_regularisation_mode": "none",
        "regularisation_scale": 0.0,
        "alpha": alpha,
        "orbit_mass": orbit_mass,
        "orbit_target": orbit_target,
        "orbit_resid": orbit_resid,
        "orbit_resid_linf": orbit_linf,
        "iterations": iterations,
        "optimality": optimality,
        "known_zero_mask": known_zero.copy(),
    }

    print(f"[MONOLITHIC] done in {elapsed:.1f}s "
        f"solver={solver_name} active={stats['n_active']}/{CP} "
        f"rmse={stats['rmse']:.6e} orbit_Linf={orbit_linf}", flush=True)

    return x, stats

# ------------------------------------------------------------------------------

def solve_streaming_nnls(
    h5_path: str,
    cfg: MPConfig,
    *,
    orbit_weights: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    tracker: Optional[object] = None,
    monolithic_max_active: int = 1000,
    cache_path: str | None = None,
    reuse_cache: bool = True,
    **active_set_kwargs,
):
    """
    Fused single-pass block-coordinate NNLS solver (BLAS-friendly).

    - Accumulates ATA and ATy for each block during one streaming pass (no repeated HDF5 reads per block).
    - Solves small quadratic NNLS problems per block using a projected-gradient
      solver on the quadratic form ATA/ATy.
    - Supports a final constrained orbit-mass correction on the final active
      set using the reduced Hessian.
    """
    t0 = time.perf_counter()

    component_donor = active_set_kwargs.pop("component_donor", None)

    # ---------------------------- metadata -----------------------------
    with open_h5(h5_path, role="reader") as f:
        S, L = map(int, f["/DataCube"].shape)
        _, C, P, Lm = map(int, f["/HyperCube/models"].shape)
        if Lm != L:
            raise RuntimeError("Model / data wavelength mismatch")
        mask = cu._get_mask(f) if cfg.apply_mask else None
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        chunks = f["/HyperCube/models"].chunks
        s_tile = int(chunks[0]) if (chunks and chunks[0]) else 128
        if cfg.s_tile_override is not None:
            s_tile = int(cfg.s_tile_override)

    Lk_guess = None
    if keep_idx is not None:
        Lk_guess = int(keep_idx.size)

    s_ranges = [(s0, min(S, s0+s_tile)) for s0 in range(0, S, s_tile)]

    # Physical HyperCube dimensions and population-tying reduction.
    C_physical = C
    CP_physical = C_physical*P
    component_donor = _canon_component_donor(component_donor, C_physical)
    anchors = np.flatnonzero(component_donor == np.arange(C_physical))
    C_fit = int(anchors.size)

    anchor_slot = np.full(C_physical, -1, dtype=np.int64)
    anchor_slot[anchors] = np.arange(C_fit, dtype=np.int64)
    component_slot = anchor_slot[component_donor]

    if np.any(component_slot < 0):
        raise RuntimeError("Invalid component donor reduction.")

    print(f"[MONO] population tying: {C_physical} physical -> "
        f"{C_fit} independent components", flush=True)

    # Preserve the physical prior for final diagnostics, then reduce the solver.
    orbit_shape_outer = _canon_orbit_weights(
        orbit_weights, C=C_physical, P=P)

    if orbit_weights is not None:
        orbit_weights_physical = np.asarray(
            orbit_weights, dtype=np.float64).ravel(order="C")
        if orbit_weights_physical.size == CP_physical:
            orbit_weights_physical = orbit_weights_physical.reshape(
                C_physical, P).sum(axis=1)
        if orbit_weights_physical.size != C_physical:
            raise ValueError("orbit_weights has incompatible physical dimensions.")

        orbit_weights = np.empty(C_fit, dtype=np.float64)
        for gg in range(C_fit):
            orbit_weights[gg] = np.sum(
                orbit_weights_physical[component_slot == gg])

    # Everything below this point is in reduced solver space unless explicitly
    # labelled physical.
    C = C_fit
    CP = C*P

    # ---------------------------- diagnostics --------------------------
    diag_level = int(os.environ.get("CUBEFIT_DIAG_LEVEL", "1"))
    diag_stride = max(1, int(os.environ.get("CUBEFIT_DIAG_STRIDE", "1")))
    diag_jsonl_path = os.environ.get("CUBEFIT_DIAG_JSONL", "").strip() or None
    diag_topk = max(1, int(os.environ.get("CUBEFIT_DIAG_TOPK", "12")))
    diag_t0 = time.perf_counter()

    def _emit_diag(record: dict) -> None:
        if diag_level <= 0:
            return
        rec = dict(record)
        rec.setdefault("t_sec", float(time.perf_counter() - diag_t0))
        _diag_append_jsonl(diag_jsonl_path, rec)

    _emit_diag(
        {
            "kind": "setup",
            "source": "solve_streaming_nnls",
            "S": int(S),
            "L": int(L),
            "C_physical": int(C_physical),
            "C_fit": int(C),
            "P": int(P),
            "CP_fit": int(CP),
            "keep_idx": int(keep_idx.size) if keep_idx is not None else None,
            "s_tile": int(s_tile),
            "n_ranges": int(len(s_ranges)),
            "processes": int(cfg.processes),
            "blas_threads": int(cfg.blas_threads),
            "diag_level": int(diag_level),
            "diag_stride": int(diag_stride),
            "diag_topk": int(diag_topk),
        }
    )

    # ---------------------------- initial x ----------------------------
    # Public warm starts remain physical (C_physical, P); retain anchor rows.
    if x0 is None:
        x = np.zeros((C, P), dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64).ravel(order="C")
        if x0.size != CP_physical:
            raise ValueError(f"x0 has size {x0.size}; expected {CP_physical}.")
        x = x0.reshape(C_physical, P)[anchors].copy()

    if resume_state and C != C_physical:
        raise ValueError('Full-state resume is incompatible with component '
            'tying; use warm_start=\'saved_x\' or a fresh solve.')

    verbose = getattr(cfg, "verbose", True)

    best_x = x.copy()
    best_proxy = np.inf 

    if cache_path is None:
        cache_path = f"{h5_path}.bcfused.npz"

    keep_idx_sig = (np.asarray(keep_idx, dtype=np.int64) if keep_idx is not None
        else np.asarray([], dtype=np.int64))

    cache_npz = None
    cache_hit = False

    if reuse_cache and cache_path and os.path.exists(cache_path):
        try:
            cache_npz = np.load(cache_path, allow_pickle=False)

            ok = (int(cache_npz["S"]) == int(S) and int(cache_npz["L"]) == int(L)
                and int(cache_npz["C_physical"]) == int(C_physical)
                and int(cache_npz["C_fit"]) == int(C)
                and int(cache_npz["P"]) == int(P)
                and np.array_equal(np.asarray(cache_npz["component_donor"],
                    dtype=np.int64), component_donor)
                and int(cache_npz["s_tile"]) == int(s_tile)
                and int(cache_npz["apply_mask"]) == int(bool(cfg.apply_mask))
                and np.array_equal(np.asarray(
                    cache_npz["keep_idx"], dtype=np.int64), keep_idx_sig))

            if ok:
                cache_hit = True
                print(f"[MONO][build] cache hit: {cache_path}", flush=True)
            else:
                cache_npz.close()
                cache_npz = None
        except Exception as _e:
            print(f"[MONO][build] cache load failed: {_e}", flush=True)
            cache_hit = False
            if cache_npz is not None:
                try:
                    cache_npz.close()
                except Exception:
                    pass
                cache_npz = None

    if cache_hit:
        print("[MONO][build] reusing cached ATy and column energies",
            flush=True)

        ATy_flat = np.asarray(cache_npz["ATy_flat"], dtype=np.float64).copy()
        D_tot = np.asarray(cache_npz["D_tot"], dtype=np.float64).copy()
        inv_sqrt_energy_flat = np.asarray(cache_npz["inv_sqrt_energy_flat"],
            dtype=np.float64).copy()

        try:
            cache_npz.close()
        except Exception:
            pass

    else:
        # ------------------------------------------------------------------
        # Parallel HyperCube build.
        #
        # Compute the two global quantities required by the monolithic solver:
        #
        #     ATy_flat = A.T @ y
        #     D_tot    = diag(A.T @ A)
        #
        # Spatial tiles are distributed across persistent worker processes.
        # Each worker streams only the physical HDF5 (C,P) chunks required for
        # its tiles, avoiding a complete (Sblk,C,P,L) model-tile allocation.
        # ------------------------------------------------------------------
        n_workers = max(1, min(int(cfg.processes), len(s_ranges)))
        batches = [[] for _ in range(n_workers)]

        for ii, tile in enumerate(s_ranges):
            batches[ii % n_workers].append(tile)

        worker_args = [(h5_path, batch, keep_idx, C_physical, P, component_slot,
            C_fit) for batch in batches if batch]

        print(f"[MONO][build] constructing ATy and column energies: "
            f"tiles={len(s_ranges)} workers={len(worker_args)}", flush=True)

        build_t0 = time.perf_counter()
        ATy_flat = np.zeros(CP, dtype=np.float64)
        D_tot = np.zeros((C, P), dtype=np.float64)

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=len(worker_args),
            mp_context=ctx, initializer=_init_worker,
            initargs=(cfg.blas_threads,)) as build_executor:

            results = build_executor.map(
                _worker_build_products_from_tuple, worker_args)

            iterator = results

            if verbose:
                iterator = tqdm(results, total=len(worker_args),
                    desc="[MONO] build", dynamic_ncols=True)

            tiles_done = 0

            for ATy_loc, D_loc, n_tiles_loc in iterator:
                ATy_flat += ATy_loc
                D_tot += D_loc
                tiles_done += int(n_tiles_loc)

        build_elapsed = time.perf_counter()-build_t0

        print(f"[MONO][build] finished: tiles={tiles_done} "
            f"elapsed={build_elapsed:.1f}s", flush=True)

        # Exact global physical column energy.
        col_energy = D_tot.copy()

        # Exact column-energy scaling.
        if np.any(~np.isfinite(col_energy)):
            raise RuntimeError("Column energy contains non-finite values.")

        inv_sqrt_energy = np.zeros_like(col_energy, dtype=np.float64)

        positive = col_energy > 0.0
        inv_sqrt_energy[positive] = 1.0 / np.sqrt(col_energy[positive])

        inv_sqrt_energy_flat = inv_sqrt_energy.ravel(order="C")

        E_pos = col_energy[col_energy > 0.0]
        S_pos = inv_sqrt_energy[inv_sqrt_energy > 0.0]

        print("[MONO][build] exact column-energy scaling:",
            f"E[min/median/max]={np.min(E_pos):.6e}/"
            f"{np.median(E_pos):.6e}/{np.max(E_pos):.6e}",
            f"E_range={np.max(E_pos) / np.min(E_pos):.6e}", flush=True)

        print("[MONO][build] exact S scaling:",
            f"S[min/median/max]={np.min(S_pos):.6e}/"
            f"{np.median(S_pos):.6e}/{np.max(S_pos):.6e}",
            f"S_range={np.max(S_pos) / np.min(S_pos):.6e}", flush=True)

        if cache_path:
            try:
                np.savez_compressed(cache_path, ATy_flat=ATy_flat,
                    inv_sqrt_energy_flat=inv_sqrt_energy_flat, D_tot=D_tot,
                    keep_idx=keep_idx_sig,
                    S=int(S), L=int(L), C_physical=int(C_physical),
                    C_fit=int(C), P=int(P), component_donor=component_donor,
                    s_tile=int(s_tile), apply_mask=int(bool(cfg.apply_mask)))
                print(f"[MONO][build] cache saved: {cache_path}", flush=True)
            except Exception as _e:
                print(f"[MONO][build] cache save failed: {_e}", flush=True)

    # Structural zeros refer to the actual grouped solver columns.
    known_zero = D_tot <= 0.0
    known_zero_orbit = np.all(known_zero, axis=1)

    if ATy_flat.size != CP or D_tot.shape != (C, P):
        raise RuntimeError('Grouped solver products have inconsistent '
            f"dimensions: ATy={ATy_flat.size}, D={D_tot.shape}, "
            f"expected CP={CP}, C={C}, P={P}.")

    if inv_sqrt_energy_flat.size != CP:
        raise RuntimeError('Column scaling has size '
            f"{inv_sqrt_energy_flat.size}; expected {CP}.")

    # DIAGNOSTIC: report S statistics (helps find extreme scalings)
    S_sample = inv_sqrt_energy_flat
    try:
        S_min = float(np.min(S_sample))
        S_p50 = float(np.median(S_sample))
        S_p90 = float(np.percentile(S_sample, 90.0))
        S_p99 = float(np.percentile(S_sample, 99.0))
        S_max = float(np.max(S_sample))
        print('[DIAG] inv_sqrt_energy after per-tile avg + floor/cap: '
            f"min/med/p90/p99/max = {S_min:.4e}/{S_p50:.4e}/{S_p90:.4e}/"
            f"{S_p99:.4e}/{S_max:.4e}", flush=True)
    except Exception as _e:
        print(f"[DIAG] error printing S stats: {_e}", flush=True)

    # run streaming active-set NNLS (monolithic) using persistent executor
    print('[MONO] starting streaming active-set NNLS (mono)',
        flush=True)

    # --------------------------
    # Create persistent executor
    # --------------------------
    n_workers = max(1, int(cfg.processes))
    # Use spawn to be safe with HDF5 + forking
    ctx = mp.get_context("spawn")

    executor = ProcessPoolExecutor(max_workers=n_workers,
        mp_context=ctx, initializer=_init_worker, initargs=(cfg.blas_threads,))
    # `initargs` must be one-lement tuple

    checkpoint_every = int(os.environ.get("CUBEFIT_SOLVER_CHECKPOINT_EVERY",
        "50"))

    latest_ckpt = {
        "x": x[component_slot].ravel(order="C").copy(),
        "stats": {
            "iter": 0,
            "max_iter": 0,
            "phase": "init",
            "final": False,
            "active": int(np.count_nonzero(x > 0.0)),
            "stall_count": 0,
    }}

    checkpoint_error_logged = False

    def _checkpoint_cb(x_vec: np.ndarray, stats: dict) -> None:
        nonlocal checkpoint_error_logged

        x_fit_ckpt = np.asarray(x_vec, dtype=np.float64).reshape(C, P)
        x_physical_ckpt = x_fit_ckpt[component_slot]

        latest_ckpt["x"] = x_physical_ckpt.ravel(order="C").copy()
        latest_ckpt["stats"] = dict(stats)

        if tracker is None:
            return

        state = dict(stats.get("resume_state", {}))
        if not state:
            state = {
                "iter": int(stats.get("iter", -1)),
                "max_iter": int(stats.get("max_iter", -1)),
                "phase": str(stats.get("phase", "solve")),
                "final": bool(stats.get("final", False)),
                "active": int(stats.get("active", -1)),
                "stall_count": int(stats.get("stall_count", -1)),
            }

        try:
            tracker.save_checkpoint(latest_ckpt["x"], state, block=True)
        except Exception as exc:
            if not checkpoint_error_logged:
                print(f"[MONO][checkpoint] save_checkpoint failed: {exc}",
                    flush=True)
                checkpoint_error_logged = True

    try:
        # ----------------------------------------------------------------------
        # Call to the streaming active-set solver
        #
        # `streamActiveSetNNLS` handles active-set management, block-wise
        # reduced Hessian solves, and orbit-mass constraints.
        # ----------------------------------------------------------------------
        active_set_kwargs.setdefault('max_active', monolithic_max_active)
        active_set_kwargs.setdefault('tol_grad', 1e-8,)
        active_set_kwargs.setdefault('max_iter', 5 * monolithic_max_active,)
        active_set_kwargs.setdefault('regularisation_scale', 1.0)

        x_flat_unscaled, mono_stats = streamActiveSetNNLS(h5_path=h5_path,
            s_ranges=s_ranges, keep_idx=keep_idx, ATy_flat=ATy_flat,
            inv_sqrt_energy_flat=inv_sqrt_energy_flat, C=C, P=P,
            executor=executor, component_slot=component_slot,
            C_physical=C_physical, cfg=cfg, orbit_weights=orbit_weights,
            known_zero_mask=known_zero,x0_flat=x.ravel(order="C"),
            resume_state=resume_state,checkpoint_cb=_checkpoint_cb,
            checkpoint_every=checkpoint_every, **active_set_kwargs)

        print("[MONO] finished streaming active-set NNLS", flush=True)

        x_fit = x_flat_unscaled.reshape(C, P).copy()
        x = x_fit[component_slot].copy()

        best_x = x.copy()
        best_proxy = np.inf
        pure_kkt_x = mono_stats.pop("pure_kkt_x", None)
        if pure_kkt_x is not None:
            pure_kkt_x = np.asarray(pure_kkt_x, dtype=np.float64).reshape(C, P)
            pure_kkt_x = pure_kkt_x[component_slot].copy()

        mono_stats["component_donor"] = component_donor.copy()
        mono_stats["n_components_physical"] = int(C_physical)
        mono_stats["n_components_fit"] = int(C_fit)

    finally:
        try:
            # shut down worker pool once per monolithic solve
            executor.shutdown(wait=True)
        except Exception:
            pass

    # --- DIAG: after all blocks solved (before projection) ---
    try:
        x_flat = x.ravel(order="C")
        x_nonzero = int(np.count_nonzero(x_flat > 0))
        x_sparsity = 1.0 - (x_nonzero / float(x_flat.size)) if x_flat.size \
            else 1.0
        x_l1 = float(np.sum(x_flat))
        x_max = float(np.max(x_flat)) if x_flat.size else 0.0
        x_min = float(np.min(x_flat)) if x_flat.size else 0.0
        D_tot_finite = np.isfinite(D_tot)
        dpos = float(np.sum(D_tot[D_tot_finite]))
        print(f"[MONO][solution] x_l1={x_l1:.3e} max={x_max:.3e} "
            f"min={x_min:.3e} nonzero={x_nonzero}/{x_flat.size} "
            f"sparsity={x_sparsity:.3f} D_tot_sum={dpos:.3e}", flush=True)

        # check for non-finite in x
        nbad = int(np.count_nonzero(~np.isfinite(x_flat)))
        if nbad:
            print(f"[MONO][solution] WARNING: non-finite entries in x: "
                f"{nbad}", flush=True)

    except Exception as _e:
        print("[MONO][solution] diag error:", _e, flush=True)
    # --- end after-blocks diagnostics ---

    # The monolithic active-set solution is authoritative. The exact residual
    # norm was already obtained from the final full-space quadratic objective.
    best_x = x.copy()
    rmse_curr = float(mono_stats["rmse"])
    residual_ss = float(mono_stats["residual_ss"])
    # Preserve the proxy definition.
    data_proxy = 0.5*rmse_curr*rmse_curr
    best_proxy = float(data_proxy)

    # --- DIAG: fianl summary (timings & numerical health) ---
    try:
        elapsed = time.perf_counter() - t0
        # D_tot stats per-orbit
        D_tot_flat = D_tot.ravel(order="C")
        D_finite = D_tot_flat[np.isfinite(D_tot_flat) & (D_tot_flat > 0)]
        D_median = float(np.median(D_finite)) if D_finite.size else 0.0
        D_min = float(np.min(D_finite)) if D_finite.size else 0.0
        D_max = float(np.max(D_finite)) if D_finite.size else 0.0

        x_flat = best_x.ravel(order="C")
        x_nonfinite = int(np.count_nonzero(~np.isfinite(x_flat)))
        x_nonzero = int(np.count_nonzero(x_flat > 0))
        x_sum = float(np.sum(x_flat))
        x_norm = float(np.linalg.norm(x_flat)) if x_flat.size else 0.0

        print("=== [MONO][summary] ===", flush=True)
        print(f"[MONO][summary] elapsed_total={elapsed:.1f}s", flush=True)
        print(f"[MONO][summary] data_proxy={data_proxy:.3e} rmse="
            f"{rmse_curr if np.isfinite(rmse_curr) else float('nan'):.3e}",
            flush=True)
        print(f"[MONO][summary] best_proxy={best_proxy:.3e}", flush=True)
        print(f"[MONO][summary] x_sum={x_sum:.3e} x_norm={x_norm:.3e} "
            f"nonzero={x_nonzero}/{x_flat.size} nonfinite={x_nonfinite}",
            flush=True)
        print(f"[MONO][summary] D_tot diag med/min/max = {D_median:.3e} / "
            f"{D_min:.3e} / {D_max:.3e}", flush=True)

        # sanity checks that likely indicate major numerical problems
        if x_nonzero == 0:
            print("[MONO][summary] ALERT: x is entirely zero!", flush=True)
        if x_nonfinite > 0:
            print("[MONO][summary] ALERT: non-finite elements in x!",
                flush=True)
        if not np.isfinite(data_proxy) or data_proxy <= 0.0:
            print("[MONO][summary] ALERT: data_proxy non-finite or <= 0.0",
                flush=True)

        # show top contributors (per-orbit totals)
        try:
            s_full = np.sum(best_x, axis=1)  # per-orbit
            top_idx = np.argsort(s_full)[::-1][:10]
            top_vals = s_full[top_idx]
            print("[MONO][summary] top orbits (index:mass): " + ", ".join(
                [f"{int(i)}:{v:.3e}" for i, v in zip(top_idx.tolist(),
                top_vals.tolist())]), flush=True)
        except Exception:
            pass
        # --- orbit_weights residual diagnostic (if prior provided) -----
        try:
            if (orbit_shape_outer is not None
                and np.sum(orbit_shape_outer) > 0.0):

                positive_outer = orbit_shape_outer > 0.0
                alpha_outer = float(np.mean(s_full[positive_outer]))
                s_proj = alpha_outer * positive_outer.astype(np.float64)

                print("[DIAG][orbit_weights]", flush=True,)

                for cc in range(C_physical):
                    st = _support_stats_1d(best_x[cc, :])
                    resid = float(s_full[cc] - s_proj[cc])
                    ratio = float(s_full[cc] / max(s_proj[cc], 1e-30))

                    print(f"orbit {cc:2d}: mass={s_full[cc]:.3e} "
                        f"target={s_proj[cc]:.3e} resid={resid:+.3e} "
                        f"ratio={ratio:7.3f} nz="
                        f"{int(np.count_nonzero(best_x[cc, :] > 0.0)):3d} "
                        f"eff={st['eff_support']:.2f} top={st['top_share']:.2f}",
                        flush=True)

                _emit_diag({
                    "kind": "final_orbit_table",
                    "source": "solve_streaming_nnls",
                    "alpha": float(alpha_outer),
                    "orbit_shape": orbit_shape_outer.tolist(),
                    "orbit_mass": s_full.tolist(),
                    "orbit_target": s_proj.tolist(),
                    "orbit_resid": (s_full - s_proj).tolist(),
                    "orbit_ratio": (s_full / np.maximum(s_proj, 1e-30)
                        ).tolist(),
                    "orbit_nz": (np.count_nonzero(best_x > 0.0, axis=1
                        ).astype(int).tolist()),
                })
            else:
                print('[DIAG][orbit_weights] no w_target present; skipping '
                    'orbit table.', flush=True)
        except Exception as _e:
            print("[DIAG][orbit_weights] diagnostic failed:", _e, flush=True)

        print("=== [MONO][final-summary] end ===", flush=True)
    except Exception as _e:
        print("[MONO][final-summary] error:", _e, flush=True)
    # --- end final summary diagnostics ---

    # final diagnostics
    # ------------------------------------------------------------


    # Final NNLS feasibility check.
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Final NNLS feasibility validation.
    #
    # If the constrained solver has produced an infeasible solution, preserve and
    # diagnose the exact offending vector before failing.
    # ------------------------------------------------------------
    x_final_flat = np.asarray(best_x, dtype=np.float64).ravel(order="C")
    bad = np.flatnonzero(~np.isfinite(x_final_flat))
    if bad.size:
        _emit_diag({
            "kind": "error",
            "source": "solve_streaming_nnls",
            "error": "nonfinite_final_x",
            "message": (
                "Final NNLS solution contains non-finite coefficients."),
            "n_bad": int(bad.size),
            "bad_indices": bad.tolist(),
            "x": x_final_flat.tolist(),
        })
        raise RuntimeError("Final NNLS solution contains non-finite "
            f"coefficients: n_bad={bad.size}, "
            f"first_indices={bad[:10].tolist()}")
    neg = np.flatnonzero(x_final_flat < 0.0)
    if neg.size:
        _emit_diag({
            "kind": "error",
            "source": "solve_streaming_nnls",
            "error": "negative_final_x",
            "message": ("Final NNLS solution violates non-negativity."),
            "n_negative": int(neg.size),
            "min_x": float(np.min(x_final_flat)),
            "negative_indices": neg.tolist(),
            "negative_values": x_final_flat[neg].tolist(),
            "x": x_final_flat.tolist(),
        })
        raise RuntimeError("Final NNLS solution violates non-negativity: "
            f"n_negative={neg.size}, "
            f"min={float(np.min(x_final_flat)):.17e}, "
            f"first_indices={neg[:10].tolist()}, "
            f"first_values={x_final_flat[neg[:10]].tolist()}")
    elapsed = time.perf_counter() - t0

    stats = dict(mono_stats)
    stats.update(
        elapsed_sec=elapsed,
        rmse_proxy_best=float(best_proxy),
        component_donor=component_donor.copy(),
        n_components_physical=int(C_physical),
        n_components_fit=int(C),
        known_zero_mask=known_zero[component_slot].copy(),
        pure_kkt_x=pure_kkt_x)

    return best_x, stats

# ------------------------------------------------------------------------------
