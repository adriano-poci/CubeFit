# -*- coding: utf-8 -*-
r"""
    nnls_patch.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Patch-scale non-negative least-squares (NNLS) utilities for CubeFit.

    This module solves a *reduced* NNLS problem on a representative subset of
    spaxels and a restricted set of populations per component, producing a
    compact seed solution ``x_CP`` (shape ``(C, P)``) that can be:

    - written to the main HDF5 file (typically ``/Seeds/x0_nnls_patch``), and
    - used as a warm-start for global solvers (e.g. Kaczmarz-based fits).

    The patch fit can optionally:
    - apply the global wavelength mask (``/Mask``),
    - apply λ-weights (typically ``/HyperCube/lambda_weights``),
    - normalize model columns consistently with the HyperCube normalization,
    - compare population “usage” against an optional orbit-weight prior, and
    - generate quick-look diagnostics (per-spaxel overlays, RMSE/χ² summaries,
      and usage-vs-prior plots).

    The main entry point is :func:`run_patch`, and a small CLI wrapper is
    provided for interactive use and debugging.

    Inputs
    ------
    The HDF5 file is expected to contain, at minimum:
    - ``/DataCube`` (S, L)
    - ``/HyperCube/models`` (S, C, P, L)

    Optional datasets used when enabled:
    - ``/Mask`` (L,)
    - ``/HyperCube/lambda_weights`` (L,)
    - ``/HyperCube/col_energy`` or equivalent global column-energy storage

    Outputs
    -------
    - ``x_CP``: seed weights (C, P) in float64
    - optional HDF5 seed write (e.g. ``/Seeds/x0_nnls_patch``)
    - optional diagnostic PNGs in ``out_dir``

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   19 December 2025
v1.1:   Universally removed all ad-hoc scalings;
        Added smooth regularisation using orbit prior in
            `_global_nnls_workingset`. 25 January 2026
v2.0:   Refactored to use OSQP QP solver for exact equality constraints. 31
            January 2026
v2.1:   Implemented `known_zero_mask` to exclude components from the patch fit
            in `run_patch`;
        Normalize each spaxel spectrum independently before stacking to ensure
            patch NNLS solves for spectral shape rather than absolute flux;
        Remove all orbit-prior constraints from the NNLS patch to avoid over
            constraining an underdetermined local problem;
        Record `seed_support_mask` for `(c,p)` bins activated by the patch NNLS
            solve;
        Record `seed_tested_mask` for `(c,p)` bins explicitly evaluated in the
            patch design matrix;
        Initialize `known_zero_mask` conservatively from patch-tested but
            inactive bins to reduce SPG nullspace exploration;
        Stop projecting patch outputs onto per-orbit simplices, delegating all
            mass normalization to the SPG solver. 3 February 2026
v2.2:   Scaled out the normalisation  of `x_CP` before returning it to ensure
            consistent physical basis with the solver, and persistent the
            normalisation. 6 February 2026
"""


from __future__ import annotations
import os, math, pathlib as plp, builtins
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.sparse as sps
import osqp
from tqdm import tqdm
import argparse

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import read_global_column_energy
from CubeFit.cube_utils import read_lambda_weights, ensure_lambda_weights,\
    compare_usage_to_orbit_weights

# ----------------------------- helpers ---------------------------------

def _predict_spaxel_sparse_from_models(M,
                                      s_idx,
                                      x_cp,
                                      picks,
                                      keep_idx=None):
    """
    Predict one spaxel spectrum by streaming /HyperCube/models in chunk-aligned
    slabs, avoiding eager materialization of (C, P, L).

    This is intended for diagnostics/plotting (e.g. the NNLS seed patch plots),
    where we only need y_fit for a small number of spaxels and want to avoid:

        np.asarray(M[s, :, :, :])  # huge (C,P,L) read per spaxel.

    The implementation respects the dataset chunking: it reads blocks
    (1, C_chunk, P_block, L) and accumulates contributions in float64.

    Notes
    -----
    - If the dataset is chunked with L_chunk == L (common in our files),
      masking wavelengths (keep_idx) reduces compute but may not reduce I/O
      much, because HDF5 still has to decompress whole L-chunks.
    - We still avoid the large temporary (C,P,Lk) allocation and instead
      stream blocks and accumulate into y_fit.

    Parameters
    ----------
    M : h5py.Dataset
        The /HyperCube/models dataset of shape (S, C, P, L).
    s_idx : int
        Spaxel index.
    x_cp : ndarray
        Weights shaped (C, P), float64 recommended.
    picks : list[list[int]]
        For each component c, a list of population indices to include.
        This is the same structure already used in nnls_patch.py.
    keep_idx : ndarray[int] or None
        Optional wavelength indices (length Lk) to evaluate on. If None,
        predicts the full L grid.

    Returns
    -------
    y_fit : ndarray
        Predicted spectrum for spaxel s_idx with shape (Lk,) if keep_idx is
        not None, else (L,). dtype float64.

    Raises
    ------
    ValueError
        If shapes are inconsistent with (C, P).

    Examples
    --------
    >>> with open_h5(h5_path, role="reader") as f:
    ...     M = f["/HyperCube/models"]
    ...     y_fit = _predict_spaxel_sparse_from_models(
    ...         M, s_idx=0, x_cp=x_CP, picks=picks, keep_idx=keep_idx
    ...     )
    """
    s_idx = int(s_idx)
    C = int(M.shape[1])
    P = int(M.shape[2])
    L = int(M.shape[3])

    x_cp = np.asarray(x_cp, dtype=np.float64, order="C")
    if x_cp.shape != (C, P):
        raise ValueError(f"x_cp shape {x_cp.shape} != (C,P)=({C},{P})")

    if keep_idx is None:
        Lk = L
        keep_idx_local = None
    else:
        keep_idx_local = np.asarray(keep_idx, dtype=np.int64).ravel()
        Lk = int(keep_idx_local.size)

    # Use storage chunking to align reads
    chunks = M.chunks or (1, 1, P, L)
    C_chunk = int(max(1, chunks[1]))
    P_chunk = int(max(1, chunks[2]))

    # Work in float64 accumulator; multiply in float32 for speed.
    y_fit = np.zeros(Lk, dtype=np.float64)

    # Iterate in storage order: components in C_chunk blocks
    for c0 in range(0, C, C_chunk):
        c1 = min(C, c0 + C_chunk)

        # Most of our files are C_chunk == 1, so this is one component.
        for c in range(c0, c1):
            plist = picks[c]
            # Avoid ambiguous truth-value for numpy arrays: check emptiness explicitly
            if (plist is None) or (np.size(plist) == 0):
                continue

            # Determine which P-chunks contain any of the picked populations.
            plist_arr = np.asarray(plist, dtype=np.int64)
            pj = np.unique((plist_arr // P_chunk) * P_chunk)

            for p0 in pj:
                p0 = int(p0)
                p1 = int(min(P, p0 + P_chunk))

                # Select only the picked populations that lie in [p0, p1)
                in_block = (plist_arr >= p0) & (plist_arr < p1)
                if not np.any(in_block):
                    continue
                local_idx = (plist_arr[in_block] - p0).astype(np.int64,
                    copy=False)

                # Read one storage-aligned block: (1, 1, Pb, L) float32
                A32 = M[s_idx:s_idx + 1, c:c + 1, p0:p1, :][...]
                A32 = np.asarray(A32, dtype=np.float32, order="C")[
                    0, 0, :, :]  # (Pb, L)

                # Slice to selected populations in this P-block
                A32 = A32[local_idx, :]  # (Kb, L)

                # Optional wavelength masking (compute reduction; I/O depends on L-chunk)
                if keep_idx_local is not None:
                    A32 = A32[:, keep_idx_local]  # (Kb, Lk)

                # Weights for this block
                w64 = x_cp[c, p0:p1][local_idx]
                w32 = w64.astype(np.float32, copy=False)

                y_fit += (A32.T @ w32).astype(np.float64, copy=False)

    return y_fit

def _get_mask_local(f) -> np.ndarray:
    """Return a 1D boolean λ-mask (L,), True = keep."""
    if "/Mask" in f:
        m = np.asarray(f["/Mask"][...], dtype=bool).ravel()
        # Accept only if it matches L
        L = int(f["/DataCube"].shape[1])
        if m.size == L:
            return m
    # fallback: keep all
    return np.ones(int(f["/DataCube"].shape[1]), dtype=bool)

def _pick_spaxels(S: int,
                  s_sel: Optional[str],
                  Ns_default: int = 32) -> np.ndarray:
    """
    Return a 1-D array of spaxel indices to use in the patch NNLS.

    If `s_sel` is None or empty, we pick `Ns_default` spaxels that are
    approximately evenly spaced across [0, S-1], so the seed sees the whole
    field instead of just the first Ns spaxels.

    If `s_sel` is of the form "start:count", we honour that exactly.
    Otherwise we treat it as a comma-separated list of explicit indices.
    """
    if s_sel is None or s_sel.strip() == "":
        Ns = int(builtins.min(Ns_default, S))
        if S <= Ns:
            return np.arange(S, dtype=np.int64)

        # Evenly spaced indices across [0, S-1]
        idx = np.linspace(0, S - 1, Ns, dtype=np.int64)
        # Just to be safe, enforce uniqueness and sort
        idx = np.unique(idx)
        return idx.astype(np.int64)

    s_sel = s_sel.strip()
    if ":" in s_sel:
        start_str, cnt_str = s_sel.split(":", 1)
        start = int(start_str)
        cnt   = int(cnt_str)
        stop  = int(builtins.min(S, start + cnt))
        if start < 0 or start >= S or cnt <= 0:
            raise ValueError(f"Bad s_sel='{s_sel}' for S={S}")
        return np.arange(start, stop, dtype=np.int64)

    # Explicit comma-separated list
    parts = [p for p in s_sel.split(",") if p.strip()]
    idx = np.array([int(p) for p in parts], dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"Empty s_sel='{s_sel}' for S={S}")
    if np.any(idx < 0) or np.any(idx >= S):
        raise ValueError(f"s_sel indices out of range for S={S}: {idx}")
    return np.unique(idx)

def _choose_pops(#E: np.ndarray,
    C: int, P: int, K_per_comp: int, mode: str = "random") -> List[np.ndarray]:
    # C, P = map(int, E.shape)
    K = int(builtins.max(1, builtins.min(K_per_comp, P)))
    out: List[np.ndarray] = []
    for c in range(C):
        # if mode == "energy":
        #     order = np.argsort(E[c, :])[::-1]
        # elif mode == "random":
        # print(mode)
        # if "random" in mode:
        rng = np.random.default_rng(1234 + c)
        order = np.arange(P); rng.shuffle(order)
        # else:
        #     raise ValueError("mode must be 'energy' or 'random'")
        out.append(np.sort(order[:K].astype(np.int64)))
    return out

def _weighted_system(B: np.ndarray, y: np.ndarray, sqrt_w_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return B * sqrt_w_rows[:, None], y * sqrt_w_rows

def _topk_by_corr_per_comp(Bw, yw, col_map, k_per_comp_init=32):
    # cosine similarity on the weighted system
    colnorm = np.linalg.norm(Bw, axis=0)
    score = (Bw.T @ yw) / np.maximum(colnorm, 1e-30)
    # group by component
    C_guess = int(np.max([c for c, _ in col_map])) + 1
    by_c = [[] for _ in range(C_guess)]
    for j, (c, p) in enumerate(col_map):
        by_c[c].append((score[j], j))
    keep = []
    for c, items in enumerate(by_c):
        if (items is None) or (len(items) == 0):
            continue
        items.sort(key=lambda t: t[0], reverse=True)
        k_c = int(np.minimum(k_per_comp_init, len(items)))
        keep.extend([items[i][1] for i in range(k_c)])
    keep = np.array(sorted(set(keep)), dtype=np.int64)
    return keep

def _filter_by_coherence(Bw, colnorm_all, keep_idx, cand_idx, rho_max=0.995):
    """
    Keep only candidates whose cosine similarity to ALL already-kept columns
    is <= rho_max. Vectorized; cheap because kincr_per_comp is small.
    """
    if len(cand_idx) == 0 or len(keep_idx) == 0:
        return np.array(cand_idx, dtype=np.int64)
    A_keep = Bw[:, keep_idx] / colnorm_all[keep_idx]          # (rows, Kkeep)
    V_cand = Bw[:, cand_idx] / colnorm_all[cand_idx]          # (rows, Kcand)
    coh = np.abs(A_keep.T @ V_cand)                           # (Kkeep, Kcand)
    ok = np.all(coh <= float(rho_max), axis=0)                # (Kcand,)
    return np.array(cand_idx, dtype=np.int64)[ok]

# ------------------------------------------------------------------------------

def _global_nnls_workingset(
    Bw, yw, col_map, orbit_weights=None,
    k0_per_comp=32, kincr_per_comp=16, rounds=4, rho_max=0.995
):
    """
    Working-set NNLS patch solver with exact orbit-ratio constraints
    enforced via OSQP.

    Returns
    -------
    x : ndarray, shape (K,)
        Full coefficient vector (zeros outside working set).
    keep : ndarray
        Indices of columns retained in the working set.
    """

    # --- prepare orbit target (ratios only) ---
    if orbit_weights is not None:
        w_target = np.asarray(orbit_weights, dtype=np.float64).ravel()
        if np.sum(w_target) <= 0:
            raise ValueError("orbit_weights sum must be > 0")
        w_target = w_target / np.sum(w_target)
    else:
        w_target = None

    K = int(Bw.shape[1])
    colnorm_all = np.linalg.norm(Bw, axis=0)
    comp_of_j = np.array([c for (c, p) in col_map], dtype=np.int64)
    C = int(np.max(comp_of_j)) + 1

    # --- persistent column-norm accumulator (full grid) ---
    # Default = 1.0 means "never normalized / untouched"
    col_norm_full = np.ones((C, np.max([p for (_, p) in col_map]) + 1),
        dtype=np.float64)

    # --- initial working set ---
    keep0 = _topk_by_corr_per_comp(Bw, yw, col_map, k_per_comp_init=int(k0_per_comp))
    keep0 = np.intersect1d(
        keep0, np.flatnonzero(colnorm_all > 1e-12), assume_unique=False
    )

    keep = []
    for c in range(C):
        cand_c = keep0[comp_of_j[keep0] == c]
        chosen_c = []
        for j in cand_c:
            ok = _filter_by_coherence(
                Bw, colnorm_all, keep + chosen_c, [j], rho_max=rho_max
            )
            if ok.size == 1:
                chosen_c.append(j)
        keep.extend(chosen_c)
    keep = np.array(sorted(set(keep)), dtype=np.int64)

    x = np.zeros(K, dtype=np.float64)

    # --- working-set expansion loop ---
    for r in range(int(rounds)):
        Bw_sub = Bw[:, keep]
        yw_sub = yw
        col_map_sub = [col_map[j] for j in keep]

        x_sub, col_norm_1d = _solve_nnls(
            Bw_sub,
            yw_sub,
            solver="nnls",
            ridge=0.0,
            normalize_columns=True,
        )
        # --- map normalized column scales back to full (C,P) grid ---
        for jj, j in enumerate(keep):
            c, p = col_map[j]     # original (C,P) index
            col_norm_full[c, p] = col_norm_1d[jj]

        info = dict(status="unconstrained")

        print(
            "[ws-NNLS]",
            "round:", r + 1,
            "status:", info["status"],
        )

        x.fill(0.0)
        x[keep] = x_sub

        # KKT check for expansion
        r_w = Bw @ x - yw
        g = Bw.T @ r_w
        viol = (x == 0.0) & (g < -1e-6)
        if not np.any(viol):
            break

        add = []
        for c in range(C):
            mask = (comp_of_j == c) & viol & (colnorm_all > 1e-12)
            if not np.any(mask):
                continue
            j_c = np.flatnonzero(mask)
            j_c = j_c[np.argsort(g[j_c])]
            j_c = _filter_by_coherence(
                Bw, colnorm_all, keep, j_c, rho_max=rho_max
            )
            if j_c.size:
                k_c = int(np.minimum(kincr_per_comp, j_c.size))
                add.extend(j_c[:k_c].tolist())

        if len(add) == 0:
            break
        keep = np.array(sorted(set(keep.tolist() + add)), dtype=np.int64)
        if keep.size >= K:
            break

    res = float(np.linalg.norm(Bw @ x - yw))
    print(
        f"[ws-NNLS] rounds={r+1} support={keep.size}/{K} "
        f"||Bw x - yw||={res:.3g}"
    )

    return x, keep, col_norm_full

# ------------------------------------------------------------------------------

def _solve_nnls_with_orbit_equality_qp_osqp(
    Bw_sub: np.ndarray,
    yw_sub: np.ndarray,
    col_map_sub: list[tuple[int, int]],
    w_target: np.ndarray,
    *,
    eps_diag: float = 1e-8,
    osqp_max_iter: int = 200000,
    osqp_verbose: bool = False,
):
    """
    One-shot NNLS patch solve with exact orbit *ratio* constraints.

    Solves:
        min_{x >= 0} ||Bw_sub x - yw_sub||^2
        subject to:
            sum_p x[c,p] / sum_p x[c_ref,p] = w_c / w_c_ref
        for all usable components c != c_ref.

    This enforces orbit ratios exactly on the patch basis, while allowing
    the data to determine the global normalization.

    No fallback to unconstrained NNLS is provided: infeasible problems
    raise a RuntimeError.
    """

    Bw_sub = np.asarray(Bw_sub, dtype=np.float64, order="C")
    yw_sub = np.asarray(yw_sub, dtype=np.float64).ravel()
    Ksub = Bw_sub.shape[1]

    if Ksub == 0:
        raise RuntimeError("NNLS patch: empty working set")

    # --- orbit prior (ratios only) ---
    w = np.asarray(w_target, dtype=np.float64).ravel()
    if w.ndim != 1 or np.sum(w) <= 0:
        raise ValueError("Invalid orbit_weights")
    w = w / np.sum(w)

    # --- component bookkeeping ---
    comp_list = np.array([c for (c, p) in col_map_sub], dtype=np.int64)
    unique_comps = np.unique(comp_list)

    # usable components: at least one non-zero column
    col_norms = np.linalg.norm(Bw_sub, axis=0)
    usable = []
    for c in unique_comps:
        cols_c = np.where(comp_list == c)[0]
        if np.any(col_norms[cols_c] > 0):
            usable.append(c)
    usable = np.asarray(usable, dtype=np.int64)

    if usable.size <= 1:
        raise RuntimeError(
            "NNLS patch: not enough usable components to enforce ratios"
        )

    # normalize prior over usable components only
    w_use = w[usable]
    w_use = w_use / np.sum(w_use)

    # choose reference component with largest prior weight (stable)
    ref_i = int(np.argmax(w_use))
    c_ref = int(usable[ref_i])
    w_ref = float(w_use[ref_i])

    # map component -> column indices
    comp_to_cols = {}
    for j, c in enumerate(comp_list):
        comp_to_cols.setdefault(int(c), []).append(j)

    cols_ref = np.asarray(comp_to_cols[c_ref], dtype=np.int64)

    # --- build quadratic objective ---
    Qx = Bw_sub.T @ Bw_sub
    if eps_diag > 0:
        Qx = Qx + eps_diag * np.eye(Ksub)
    P = sps.csc_matrix(0.5 * (Qx + Qx.T))
    q = -(Bw_sub.T @ yw_sub)

    # --- equality constraints: ratios ---
    rows, cols, data = [], [], []
    eq = 0

    for i, c in enumerate(usable):
        if c == c_ref:
            continue
        cols_c = np.asarray(comp_to_cols.get(int(c), []), dtype=np.int64)
        if cols_c.size == 0:
            continue
        beta = float(w_use[i] / w_ref)

        # +1 * sum_p x[c,p]
        for j in cols_c:
            rows.append(eq)
            cols.append(j)
            data.append(1.0)

        # -beta * sum_p x[c_ref,p]
        for j in cols_ref:
            rows.append(eq)
            cols.append(j)
            data.append(-beta)

        eq += 1

    if eq == 0:
        raise RuntimeError("NNLS patch: no valid ratio constraints constructed")

    Aeq = sps.csr_matrix((data, (rows, cols)), shape=(eq, Ksub))

    # --- inequality constraints: x >= 0 ---
    A = sps.vstack([Aeq, sps.identity(Ksub, format="csr")], format="csc")
    l = np.concatenate([np.zeros(eq), np.zeros(Ksub)])
    u = np.concatenate([np.zeros(eq), np.full(Ksub, np.inf)])

    # --- OSQP solve ---
    prob = osqp.OSQP()
    prob.setup(
        P=P,
        q=q,
        A=A,
        l=l,
        u=u,
        verbose=osqp_verbose,
        polish=True,
        max_iter=int(osqp_max_iter),
        eps_abs=1e-8,
        eps_rel=1e-8,
    )

    res = prob.solve()
    if res.info.status not in ("solved", "solved_inaccurate"):
        raise RuntimeError(f"OSQP failed: {res.info.status}")

    x = np.asarray(res.x, dtype=np.float64)

    return x, dict(
        status=res.info.status,
        iter=res.info.iter,
        usable_components=usable,
        reference_component=c_ref,
    )

# ------------------------------------------------------------------------------

def _solve_nnls(Bw: np.ndarray,
                yw: np.ndarray,
                solver: str = "nnls",
                max_iter: int = 200,
                ridge: float = 0.0,
                normalize_columns: bool = True) -> np.ndarray:
    """
    Solve the non-negative least-squares problem

        min_x 0.5 * ||Bw x - yw||_2^2 + 0.5 * ridge * ||x||_2^2
        s.t.  x >= 0

    The implementation solves on a column-normalized system when
    ``normalize_columns=True`` for numerical stability and speed, and maps
    the solution back to the original column scaling.

    Parameters
    ----------
    Bw : ndarray (R, K)
        Weighted design matrix (rows already incorporate row-weights).
    yw : ndarray (R,)
        Weighted right-hand side vector.
    solver : {'nnls','lsq','pg'}
        Backend solver to use. 'nnls' prefers scipy.optimize.nnls on a
        normalized system (fast); 'lsq' uses lsq_linear (bounds); 'pg'
        uses an internal projected-gradient solver (robust fallback).
    max_iter : int
        Maximum iterations for iterative solvers (pg or lsq_linear).
    ridge : float
        L2 ridge regularizer applied to the quadratic term.
    normalize_columns : bool
        If True, solve on Bw / col_norm to remove per-column scale and
        then return x in the original scale.

    Returns
    -------
    x : ndarray (K,)
        Non-negative solution vector in the original column scaling.
    col_norm_safe : ndarray (K,)
        The safe column norms used to normalize columns (so callers can
        persist them or undo/redo scalings consistently).

    Raises
    ------
    ValueError
        If input shapes are inconsistent or solver argument is unknown.

    Examples
    --------
    >>> x = _solve_nnls(Bw, yw, solver='nnls', ridge=1e-4)
    """

    Bw = np.asarray(Bw, dtype=np.float64, order="C")
    yw = np.asarray(yw, dtype=np.float64).ravel()
    if Bw.ndim != 2:
        raise ValueError("Bw must be 2-D (R,K)")
    R, K = Bw.shape
    if yw.size != R:
        raise ValueError("yw length must match Bw.shape[0]")

    # ---- Normalise columns if requested ---------------------------------
    if normalize_columns:
        col_norm = np.linalg.norm(Bw, axis=0)
        # avoid division by zero -> set tiny norms to 1.0 for scaling only
        col_norm_safe = np.where(col_norm > 0.0, col_norm, 1.0)
        Bn = Bw / col_norm_safe[None, :]
    else:
        col_norm_safe = np.ones(K, dtype=np.float64)
        Bn = Bw.copy()

    # incorporate ridge in objective: for normalized solver we add it to grad
    # For lsq we augment; for nnls we rely on PG fallback if needed.

    # ---- NNLS via scipy.nnls on normalized columns -----------------------
    if solver == "nnls":
        try:
            from scipy.optimize import nnls as _scipy_nnls
            # nnls expects 2-D (M,N), b of length M. Use Bn and yw.
            z, _ = _scipy_nnls(Bn, yw)
            x = z / col_norm_safe
            # ensure non-negative and finite
            x = np.maximum(0.0, np.asarray(x, dtype=np.float64))
            return x, col_norm_safe
        except Exception:
            # fallback: use projected gradient on normalized system
            solver = "pg"

    # ---- LSQ_LINEAR (bounded least-squares) ------------------------------
    if solver == "lsq":
        try:
            from scipy.optimize import lsq_linear as _lsq_linear
            if ridge > 0.0:
                # augment to implement ridge: [Bn; sqrt(ridge)*I] x = [yw; 0]
                sqrt_r = np.sqrt(float(ridge))
                B_aug = np.vstack([Bn, sqrt_r * np.eye(K)])
                y_aug = np.concatenate([yw, np.zeros(K, dtype=np.float64)])
                res = _lsq_linear(B_aug, y_aug, bounds=(0.0, np.inf),
                                  max_iter=int(max_iter), verbose=0, lsmr_tol='auto')
            else:
                res = _lsq_linear(Bn, yw, bounds=(0.0, np.inf),
                                  max_iter=int(max_iter), verbose=0, lsmr_tol='auto')
            z = np.asarray(res.x, dtype=np.float64)
            x = z / col_norm_safe
            x = np.maximum(0.0, x)
            return x, col_norm_safe
        except Exception as e:
            # fallback to PG if lsq_linear is not available or fails
            solver = "pg"

    # ---- Projected Gradient (works on normalized system) ---------------
    if solver == "pg":
        z = np.zeros(K, dtype=np.float64)   # solution in normalized space
        # estimate Lipschitz (per-column quadratic diag)
        L_est = float(np.max(np.sum(Bn * Bn, axis=0))) + float(ridge)
        if not np.isfinite(L_est) or L_est <= 0.0:
            L_est = 1.0
        step = 1.0 / L_est
        for it in range(int(max_iter)):
            r = Bn @ z - yw
            g = Bn.T @ r + float(ridge) * z
            # gradient step + projection
            z_try = np.maximum(0.0, z - step * g)
            # backtracking on quadratic objective (sufficient decrease)
            r_try = Bn @ z_try - yw
            f_old = 0.5 * float(r @ r) + 0.5 * float(ridge) * float(z @ z)
            f_new = 0.5 * float(r_try @ r_try) + 0.5 * float(ridge) * float(z_try @ z_try)
            bt = 0
            while f_new > f_old and bt < 10:
                step *= 0.5
                z_try = np.maximum(0.0, z - step * g)
                r_try = Bn @ z_try - yw
                f_new = 0.5 * float(r_try @ r_try) + 0.5 * float(ridge) * float(z_try @ z_try)
                bt += 1
            z = z_try
            # stopping criterion (inf-norm of projected gradient)
            pg = z - np.maximum(0.0, z - g * step)
            if np.linalg.norm(pg, ord=np.inf) * step < 1e-9:
                break
        x = z / col_norm_safe
        x = np.maximum(0.0, x)
        return x, col_norm_safe

    # ---- Unknown solver option ------------------------------------------
    raise ValueError(f"Unknown solver='{solver}'")

# ------------------------------------------------------------------------------

def run_patch(h5_path: str,
              s_sel: Optional[str] = None,
              k_per_comp: int = 12,
              pick_mode: str = "random",
              use_mask: bool = True,
              use_lambda: bool = True,
              out_dir: Optional[str | os.PathLike] = None,
              write_seed: bool = False,
              seed_path: str = "/Seeds/x0_nnls_patch",
              orbit_weights: np.ndarray | None = None):

    out_dir = None if out_dir is None else plp.Path(out_dir)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- read dims, global mask, λ-weights, picks
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]; M = f["/HyperCube/models"]
        S, L = map(int, DC.shape)
        _, C, P, Lm = map(int, M.shape)
        if Lm != L:
            raise RuntimeError(f"models L={Lm} != data L={L}")
        norm_mode = str(f["/HyperCube"].attrs.get("norm.mode", "model")).lower()
        mask = _get_mask_local(f) if use_mask else np.ones(L, bool)
        keep_idx = np.flatnonzero(mask) if mask is not None else None
        Lk = int(keep_idx.size) if keep_idx is not None else L
        # observed λ grid for plotting
        obs = f["/ObsPix"][...] if "/ObsPix" in f else np.arange(L, dtype=np.float64)
        lam_plot = obs[keep_idx] if keep_idx is not None else obs

    s_idx = _pick_spaxels(S, s_sel, Ns_default=32)
    s_idx = np.sort(np.asarray(s_idx, dtype=int))
    if s_idx.size < 1:
        raise RuntimeError("Empty spaxel selection.")

    # ---- quick triage (paste right after w_lam and s_idx) ----
    print(f"[triage] S_sel={s_idx.size}")

    # Lk = int(keep_idx.size) if keep_idx is not None else int(len(w_full))
    Lk = int(keep_idx.size) if keep_idx is not None else int(L)
    print(f"[triage] Lk (masked wavelengths) = {Lk}")
    if Lk == 0:
        raise SystemExit("[triage] /Mask removes all wavelengths for this run "
            "(Lk=0). Try --no-mask or fix /Mask.")

    # finite counts per spaxel under the mask
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        finite_counts = []
        for s in s_idx:
            spec = np.asarray(DC[s, :], dtype=np.float64)
            spec = spec[keep_idx] if keep_idx is not None else spec
            finite_counts.append(int(np.count_nonzero(np.isfinite(spec))))
        # binCounts = np.asarray(f['/BinCounts'][...], dtype=int)
    finite_counts = np.asarray(finite_counts, int)
    print(f"[triage] finite pixels per spaxel (min/med/max): "
        f"{finite_counts.min()}/{int(np.median(finite_counts))}/{finite_counts.max()}")

    if finite_counts.min() == 0:
        print("[triage] Every selected spaxel with 0 finite pixels under the mask "
            "will contribute zero rows. If they are ALL zero, "
            "Bw==0 and you’ll see the 'all columns ~zero' message. "
            "Try --no-mask or pick spaxels with finite data.")

    # Subsample the selected spaxels for this expensive check
    if s_idx.size > 12:
        rng = np.random.default_rng(12345)
        tri_s = np.sort(
            rng.choice(s_idx, size=12, replace=False)
        )
    else:
        tri_s = s_idx

    with open_h5(h5_path, role="reader") as f:
        mode = str(f["/HyperCube"].attrs.get("norm.mode", "model"))
        S, L = map(int, f["/DataCube"].shape)

    print(pick_mode)
    picks = _choose_pops(C, P, k_per_comp, 'random')

    # --- Build RHS y and per-row weights (spaxel-wise finite mask)
    rows = int(s_idx.size * Lk)
    y = np.empty(rows, dtype=np.float64)
    sqrt_w_rows = np.empty(rows, dtype=np.float64)

    print(f"[patch] Building RHS and row-weights: S_sel={s_idx.size}, Lk={Lk}")
    finite_counts = []
    pos = 0
    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        it = tqdm(s_idx, desc="[patch] RHS", mininterval=0.5, dynamic_ncols=True)
        for s in it:
            spec = np.asarray(DC[s, :], dtype=np.float64)
            if keep_idx is not None:
                spec = spec[keep_idx]

            good = np.isfinite(spec)
            spec_sanit = np.where(good, spec, 0.0)

            # -------- NEW: per-spaxel normalization (shape only) --------
            norm = np.linalg.norm(spec_sanit)
            if norm > 0:
                spec_sanit /= norm
            # ------------------------------------------------------------

            y[pos:pos+Lk] = spec_sanit
            sqrt_w_rows[pos:pos+Lk] = good.astype(np.float64)
            pos += Lk
            finite_counts.append(int(np.count_nonzero(good)))
    finite_counts = np.asarray(finite_counts, int)
    print(f"[patch] finite pixels per spaxel (min/median/max): "
          f"{finite_counts.min()}/{int(np.median(finite_counts))}/{finite_counts.max()}")

    # --- Build design B: concat columns (one per chosen (c,p))
    K = int(sum(len(pc) for pc in picks))
    B = np.empty((rows, K), dtype=np.float64)
    col_map: List[Tuple[int, int]] = []

    # Map (c, p) -> column index in B
    cp_to_col: dict[Tuple[int, int], int] = {}
    j = 0
    for c, plist in enumerate(picks):
        for p_idx in plist:
            p_int = int(p_idx)
            cp_to_col[(c, p_int)] = j
            col_map.append((c, p_int))
            j += 1

    print(f"[patch] Building design (blockwise): rows={rows}, K={K}")

    # Reshape row weights once; they depend only on (spaxel, λ)
    # sqrt_w_rows has length rows = Ns * Lk
    Ns = int(s_idx.size)
    sqrt_w_2d = sqrt_w_rows.reshape(Ns, Lk)
    row_mask = (sqrt_w_2d > 0.0)  # True where data are finite / used

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]  # (S, C, P, L)

        for c, plist in enumerate(
            tqdm(
                picks,
                desc="[patch] comps",
                mininterval=0.5,
                dynamic_ncols=True,
            )
        ):
            # plist can be a list or a numpy array; in both cases len(...) works.
            if plist is None or len(plist) == 0:
                continue

            # (optional) normalize to a 1D iterable of ints
            plist_iter = [int(p) for p in plist]

            # Block read: all selected spaxels, all pops for this component
            A_c = np.asarray(M[s_idx, c, :, :], dtype=np.float32)

            if keep_idx is not None:
                A_c = A_c[:, :, keep_idx]  # (Ns, P, Lk)

            A_c = np.nan_to_num(
                A_c, nan=0.0, posinf=0.0, neginf=0.0, copy=False
            )

            A_c *= row_mask[:, None, :]   # (Ns, P, Lk)

            for p_idx in plist_iter:
                p_int = int(p_idx)
                j = cp_to_col[(c, p_int)]
                B[:, j] = A_c[:, p_int, :].reshape(rows, order="C")


    # final sanitation guard (paranoia)
    if not np.isfinite(B).all():
        bad_cols = np.any(~np.isfinite(B), axis=0).nonzero()[0]
        print(
            f"[patch] WARNING: non-finite values in {bad_cols.size} columns; "
            "sanitizing to 0."
        )
        B[:, bad_cols] = np.nan_to_num(B[:, bad_cols], copy=False)

    # quick column-norm diagnostics on the weighted system
    colnorm_w = np.linalg.norm(B * sqrt_w_rows[:, None], axis=0)
    print("[patch] column norms (weighted) min/median/max: "
          f"{colnorm_w.min():.3g}/{np.median(colnorm_w):.3g}/{colnorm_w.max():.3g}")
    if np.all(colnorm_w < 1e-12):
        print("[patch] All columns are ~zero after masking/weighting → check "
              "hypercube normalization and LOSVD amplitudes for these spaxels.")

    # --- Apply row weights (λ and finite mask) to (B,y)
    Bw, yw = _weighted_system(B, y, sqrt_w_rows)

    # --- column/target sanity on the weighted system
    colnorm_all = np.linalg.norm(Bw, axis=0)
    nz_cols = int(np.count_nonzero(colnorm_all > 0))
    print(f"[patch] weighted col-norms: nz={nz_cols}/{colnorm_all.size}")

    # raw correlations; if all <= 0 then NNLS optimum is x=0
    corr = Bw.T @ yw
    pos_corr = int(np.count_nonzero(corr > 0))
    print(f"[patch] corr>0 = {pos_corr}/{corr.size}  min/med/max = "
        f"{float(np.min(corr)):.3g}/{float(np.median(corr)):.3g}/{float(np.max(corr)):.3g}")

    if nz_cols == 0:
        raise RuntimeError("[patch] All columns are zero after masking/weighting for these spaxels.")

    # --- Fast working-set GLOBAL NNLS on a pruned basis
    print(f"[patch] Working-set global NNLS (init=32/comp, +16/round, rounds=4)...")
    x_full, keep, col_norm_full = _global_nnls_workingset(
        Bw, yw, col_map, orbit_weights=None,
        k0_per_comp=32, kincr_per_comp=16, rounds=4,
    )
    # keep col_map aligned with x_patch for the unpack loop
    x_patch = x_full[keep].copy()                # compress to working set
    col_map  = [col_map[j] for j in keep]        # now 1:1 aligned
    K = int(len(col_map))  # (optional) if you print/return K later

    # --- Unpack to full (C,P)
    x_CP = np.zeros((C, P), dtype=np.float64)
    for j, (c, p) in enumerate(col_map):
        x_CP[c, p] = x_patch[j]
    # --- map NNLS solution back to physical column scaling ---
    x_CP /= col_norm_full
    # ------------------------------------------------------------
    # Interpret NNLS patch result as SUPPORT information
    # ------------------------------------------------------------
    EPS_PATCH = 1e-12

    # Zero-out numerical noise
    x_CP[np.abs(x_CP) < EPS_PATCH] = 0.0

    # 1) Support mask: bins that were non-zero in patch
    seed_support_mask = x_CP > 0.0    # shape (C,P)

    # 2) Tested mask: bins that were included in NNLS design
    seed_tested_mask = np.zeros((C, P), dtype=bool)
    for (c, p) in col_map:
        seed_tested_mask[c, p] = True

    # 3) Initial KNOWN_ZERO candidates:
    #    tested by patch but found unnecessary
    initial_known_zero_mask = seed_tested_mask & (~seed_support_mask)

    # --- Reconstruct + diagnostics per spaxel
    rmse = np.zeros(s_idx.size, dtype=np.float64)
    chi2 = np.zeros_like(rmse)
    pos = 0
    print("[patch] Reconstructing and plotting spectra...")
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        for i, s in enumerate(tqdm(s_idx, desc="[patch] plots", mininterval=0.5, dynamic_ncols=True)):
            y_fit = np.zeros(Lk, dtype=np.float64)
            # weighted least-squares sense: compare on masked/log grid unweighted
            y_fit = _predict_spaxel_sparse_from_models(
                M=M,
                s_idx=int(s),
                x_cp=x_CP,
                picks=picks,
                keep_idx=keep_idx,
            )

            y_obs = y[pos:pos+Lk]
            #  Renormalize to observed spectrum norm
            scale = np.dot(y_obs, y_fit) / (np.dot(y_fit, y_fit) + 1e-30)
            y_fit *= scale

            wrow  = sqrt_w_rows[pos:pos+Lk]
            pos += Lk
            r = y_obs - y_fit
            # RMSE (unweighted) and χ² (λ-weighted on finite)
            w2 = wrow * wrow
            den = float(np.sum(w2)) or 1.0
            chi2[i] = float(np.sum((r * wrow)**2) / den)
            rmse[i] = np.sqrt(chi2[i])

            if out_dir:
                fig = plt.figure(figsize=(9.5, 3.2))
                ax = fig.add_subplot(111)
                ax.plot(lam_plot, y_obs, lw=1.0, label="data")
                ax.plot(lam_plot, y_fit, lw=1.0, label="model")
                ax.set_title(f"spaxel {int(s)}  rmse={rmse[i]:.3f}")
                ax.set_xlabel("λ (ObsPix)")
                ax.set_ylabel("flux")
                ax.legend(loc="best", fontsize=8)
                fig.savefig(out_dir/\
                    f"patch_spax{int(s):05d}.png", dpi=120)
                plt.close(fig)

    if out_dir:
        flat = x_CP.ravel(order="C")
        nnz = int(np.count_nonzero(flat))
        if nnz == 0:
            print("[plot] x_CP is all zeros (no bars to draw).")
        else:
            fig = plt.figure(figsize=plt.figaspect(0.5)*0.9)
            gs = gridspec.GridSpec(1, 2, hspace=0.0, wspace=0.1)

            ax = fig.add_subplot(gs[0, 0])
            ax.bar(np.arange(flat.size), flat)
            ax.set_xlabel('component populations')
            ax.set_ylabel('weight')

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(seed_support_mask, aspect="auto", interpolation="nearest")
            ax2.set_xlabel("population p")
            ax2.set_ylabel("component c")
            ax2.set_title("NNLS patch support mask")

            fig.savefig(out_dir / 'xcp.png', dpi=120)
            plt.close(fig)

    if write_seed:
        with open_h5(h5_path, role="writer") as f:
            grp = f.require_group("/Seeds")
            if seed_path in f:
                del f[seed_path]
            ds = f.create_dataset(seed_path, data=x_CP.astype(np.float64),
                dtype="f8")
            ds.attrs["origin"] = "nnls_patch"
            ds.attrs["s_idx"]  = s_idx
            ds.attrs["k_per_comp"] = int(k_per_comp)
            ds.attrs["pick_mode"]  = str(pick_mode)
            ds.attrs["norm.mode_at_fit"] = norm_mode

            # ------------------------------------------------------------
            # Write seed support metadata for SPG
            # ------------------------------------------------------------

            if "seed_support_mask" in grp:
                del grp["seed_support_mask"]
            grp.create_dataset(
                "seed_support_mask",
                data=seed_support_mask.astype(bool),
                dtype="bool",
            )

            if "seed_tested_mask" in grp:
                del grp["seed_tested_mask"]
            grp.create_dataset(
                "seed_tested_mask",
                data=seed_tested_mask.astype(bool),
                dtype="bool",
            )

            # ------------------------------------------------------------
            # Seed initial KNOWN_ZERO mask
            # ------------------------------------------------------------
            hg = f.require_group("/HyperCube")

            if "known_zero_mask" in hg:
                # Merge with existing mask (never un-freeze)
                kz_old = np.asarray(hg["known_zero_mask"][...], dtype=bool)
                known_zero_mask = kz_old | initial_known_zero_mask
                del hg["known_zero_mask"]
            else:
                known_zero_mask = initial_known_zero_mask.copy()

            hg.create_dataset(
                "known_zero_mask",
                data=known_zero_mask.astype(bool),
                dtype="bool",
            )
    
    else:
        import tempfile
        tmp_sidecar = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                prefix=os.path.basename(h5_path)+".fit.",
                suffix=".h5", delete=False
            )
            tmp_sidecar = tmp.name
            tmp.close()

            import h5py
            with h5py.File(tmp_sidecar, "w") as G:
                G.create_dataset("/Fit/x_best", data=x_CP.astype("f8"),
                    dtype="f8",)

        except Exception as e:
            print(f"[nnls_patch] usage-vs-prior (temp sidecar) failed: {e}")
        finally:
            if tmp_sidecar and os.path.exists(tmp_sidecar):
                try:
                    os.remove(tmp_sidecar)
                except OSError:
                    pass
    
    # NOTE:
    # x_CP amplitudes from the NNLS patch are NOT physically meaningful.
    # They encode support only; SPG will overwrite amplitudes completely.

    # --- persist column normalization for downstream solvers (SPG) ---
    # This guarantees NNLS and SPG operate in the same physical basis.
    with open_h5(h5_path, role="writer") as f:
        grp = f.require_group("/HyperCube")
        if "col_norm" in grp:
            del grp["col_norm"]
        grp.create_dataset(
            "col_norm",
            data=col_norm_full,
            dtype="f8",
        )

    return dict(
        x_CP=x_CP, picks=picks, s_idx=s_idx, rmse=rmse, chi2=chi2,
        meta=dict(norm_mode=norm_mode, rows=int(rows), cols=int(K),
                  mask_used=bool(use_mask), lambda_used=bool(use_lambda),
                  finite_min=int(finite_counts.min()),
                  finite_med=int(np.median(finite_counts)),
                  finite_max=int(finite_counts.max()))
    )

# -------------------------------- CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Patch-scale NNLS diagnostic fitter (self-contained)")
    ap.add_argument("h5", help="Path to CubeFit HDF5")
    ap.add_argument("--spax", default=None, help="spaxel selection: 'start:count' or 'i,j,k'")
    ap.add_argument("--k-per-comp", type=int, default=12, help="populations per component")
    ap.add_argument("--pick", default="energy", choices=["energy","random"], help="how to pick pops")
    ap.add_argument("--no-mask", action="store_true", help="ignore /Mask")
    ap.add_argument("--no-lambda", action="store_true", help="ignore /HyperCube/lambda_weights")
    ap.add_argument("--lam-dset", default="/HyperCube/lambda_weights", help="λ-weights dataset")
    ap.add_argument("--out", default=None, help="directory for diagnostic plots")
    ap.add_argument("--write-seed", action="store_true", help="write /Seeds/x0_nnls_patch")
    ap.add_argument("--seed-path", default="/Seeds/x0_nnls_patch", help="dataset to write seed")
    args = ap.parse_args()

    res = run_patch(
        h5_path=args.h5,
        s_sel=args.spax,
        k_per_comp=int(args.k_per_comp),
        pick_mode=args.pick,
        use_mask=not args.no_mask,
        use_lambda=not args.no_lambda,
        out_dir=args.out,
        write_seed=bool(args.write_seed),
        seed_path=args.seed_path,
    )
    small = {k: (f"array{v.shape}" if isinstance(v, np.ndarray) else v) for k, v in res.items() if k != "picks"}
    print("[nnls_patch] done.", small)
    print(res['x_CP'])

if __name__ == "__main__":
    main()