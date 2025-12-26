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
"""


from __future__ import annotations
import os, math, pathlib as plp
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
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
            if not plist:
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
                local_idx = (plist_arr[in_block] - p0).astype(np.int64, copy=False)

                # Read one storage-aligned block: (1, 1, Pb, L) float32
                # (This avoids the full (C,P,L) slab.)
                A32 = M[s_idx:s_idx + 1, c:c + 1, p0:p1, :][...]
                A32 = np.asarray(A32, dtype=np.float32, order="C")[0, 0, :, :]  # (Pb, L)

                # Slice to selected populations in this P-block
                A32 = A32[local_idx, :]  # (Kb, L)

                # Optional wavelength masking (compute reduction; I/O depends on L-chunk)
                if keep_idx_local is not None:
                    A32 = A32[:, keep_idx_local]  # (Kb, Lk)

                # Weights for this block
                w64 = x_cp[c, p0:p1][local_idx]                 # (Kb,) float64
                w32 = w64.astype(np.float32, copy=False)        # float32 GEMV

                # Accumulate: y += A^T w
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
        Ns = int(min(Ns_default, S))
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
        stop  = min(S, start + cnt)
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

def _choose_pops(E: np.ndarray, K_per_comp: int, mode: str = "energy") -> List[np.ndarray]:
    C, P = map(int, E.shape)
    K = int(max(1, min(K_per_comp, P)))
    out: List[np.ndarray] = []
    for c in range(C):
        if mode == "energy":
            order = np.argsort(E[c, :])[::-1]
        elif mode == "random":
            rng = np.random.default_rng(1234 + c)
            order = np.arange(P); rng.shuffle(order)
        else:
            raise ValueError("mode must be 'energy' or 'random'")
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
        if not items: 
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

def _prune_and_resolve_if_needed(Bw, yw, keep, x, colnorm_all, rho_max,
                                 solver, ridge, normalize_columns):
    """
    If the kept set is still highly coherent, prune greedily (keep big coeffs,
    drop new ones that exceed rho_max w.r.t. already-kept), then re-solve.
    """
    if len(keep) <= 1:
        return x, keep

    A = Bw[:, keep] / colnorm_all[keep]
    G = np.abs(A.T @ A)
    np.fill_diagonal(G, 0.0)
    mu = float(np.max(G))
    print(f"[ws-NNLS] max coherence among kept = {mu:.6f}")
    if mu <= float(rho_max):
        return x, keep

    # Greedy prune by descending coefficient magnitude
    order = np.argsort(x[keep])[::-1]
    keep_sorted = np.array(keep, dtype=np.int64)[order]
    new_keep = []
    for j in keep_sorted:
        if len(new_keep) == 0:
            new_keep.append(j)
            continue
        # test coherence vs new_keep
        j_ok = _filter_by_coherence(Bw, colnorm_all, new_keep, [j], rho_max=rho_max)
        if j_ok.size == 1:
            new_keep.append(j)

    new_keep = np.array(sorted(set(new_keep)), dtype=np.int64)
    if new_keep.size == keep.size:
        return x, keep

    print(f"[ws-NNLS] prune: {keep.size} → {new_keep.size} (re-solve)")
    x_new = np.zeros_like(x)
    x_sub = _solve_nnls(Bw[:, new_keep], yw, solver=solver, max_iter=400,
                        ridge=float(ridge), normalize_columns=bool(normalize_columns))
    x_new[new_keep] = x_sub
    return x_new, new_keep

def _global_nnls_workingset(Bw, yw, col_map,
                            k0_per_comp=32, kincr_per_comp=16, rounds=4,
                            solver="nnls", ridge=1e-2, normalize_columns=True,
                            rho_max=0.995):
    """
    Fast global NNLS via a small working set with coherence control.
    Returns (x_full, keep_idx).
    """
    K = int(Bw.shape[1])
    colnorm_all = np.linalg.norm(Bw, axis=0)
    comp_of_j = np.array([c for (c, p) in col_map], dtype=np.int64)
    C = int(np.max(comp_of_j)) + 1

    # initial keep by local correlation, then de-duplicate by coherence
    keep0 = _topk_by_corr_per_comp(Bw, yw, col_map, k_per_comp_init=int(k0_per_comp))
    keep0 = np.intersect1d(keep0, np.flatnonzero(colnorm_all > 1e-12), assume_unique=False)
    # per-component coherence throttling
    keep = []
    for c in range(C):
        cand_c = keep0[comp_of_j[keep0] == c]
        chosen_c = []
        for j in cand_c:
            ok = _filter_by_coherence(Bw, colnorm_all, keep + chosen_c, [j], rho_max=rho_max)
            if ok.size == 1:
                chosen_c.append(j)
        keep.extend(chosen_c)
    keep = np.array(sorted(set(keep)), dtype=np.int64)

    x = np.zeros(K, dtype=np.float64)

    for r in range(int(rounds)):
        # solve on current working set
        Bw_sub = Bw[:, keep]
        x_sub = _solve_nnls(Bw_sub, yw, solver=solver, max_iter=400,
                            ridge=float(ridge), normalize_columns=bool(normalize_columns))
        x.fill(0.0); x[keep] = x_sub

        # residual & gradient
        r_w = Bw @ x - yw
        g = Bw.T @ r_w  # weighted gradient

        # KKT satisfied?
        viol = (x == 0.0) & (g < -1e-6)
        if not np.any(viol):
            break

        # candidate adds per component, with coherence gate
        add = []
        for c in range(C):
            mask = (comp_of_j == c) & viol & (colnorm_all > 1e-12)
            if not np.any(mask): 
                continue
            j_c = np.flatnonzero(mask)
            # sort by most negative gradient
            j_c = j_c[np.argsort(g[j_c])]
            # coherence filter against current keep
            j_c = _filter_by_coherence(Bw, colnorm_all, keep, j_c, rho_max=rho_max)
            if j_c.size:
                k_c = int(np.minimum(kincr_per_comp, j_c.size))
                add.extend(j_c[:k_c].tolist())

        if not add:
            break
        keep = np.array(sorted(set(keep.tolist() + add)), dtype=np.int64)
        if keep.size >= K:
            break  # safety

    # optional prune and re-solve if coherence still high
    x, keep = _prune_and_resolve_if_needed(Bw, yw, keep, x, colnorm_all,
                                           rho_max, solver, ridge, normalize_columns)

    res_w = float(np.linalg.norm(Bw @ x - yw))
    print(f"[ws-NNLS] rounds={r+1} support={keep.size}/{K}  ||Bw x - yw||={res_w:.3g}")
    return x, keep

# ------------------------------------------------------------------------------

def apply_orbit_prior_to_seed(Xcp: np.ndarray,
                              orbit_weights,
                              *,
                              E_cp: np.ndarray | None = None,
                              preserve_total: bool = True,
                              min_w_frac: float = 1e-4) -> np.ndarray:
    """
    Rescale rows of Xcp (C,P) so row usage matches orbit_weights fractions.

    If E_cp is provided (C,P), usage is energy-weighted: s_c = sum_p X[c,p]*E[c,p].
    Otherwise usage is plain row sums: s_c = sum_p X[c,p].

    This is a fast, closed-form, mass-preserving projection; O(C*P).
    """
    X = np.nan_to_num(np.asarray(Xcp, np.float64), nan=0.0, posinf=0.0,
                      neginf=0.0, copy=True)
    if X.ndim != 2:
        raise ValueError("Xcp must be 2-D (C,P).")
    C, P = X.shape

    w = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")
    if w.size not in (C, C * P):
        raise ValueError(f"orbit_weights has size {w.size}, expected {C} or {C*P}")
    if w.size == C * P:
        w = w.reshape(C, P, order="C").sum(axis=1)

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    Wsum = float(w.sum())
    if Wsum <= 0.0:
        return X  # no prior to enforce
    t = w / Wsum  # target fractions per component in [0,1]

    # Current usage per component
    if E_cp is not None:
        E = np.asarray(E_cp, np.float64, order="C")
        if E.shape != (C, P):
            raise ValueError(f"E_cp shape {E.shape} != (C,P) {(C,P)}")
        s = (X * E).sum(axis=1)  # energy-weighted usage
    else:
        s = X.sum(axis=1)        # plain usage
    S = float(s.sum())

    # If a component has zero usage but positive target, sprinkle tiny mass
    need = (s <= 0.0) & (t > 0.0)
    if np.any(need):
        eps = 1e-12 * (S if S > 0.0 else 1.0)
        X[need, :] = (eps / P)
        # recompute s/S in the same metric
        if E_cp is not None:
            s = (X * E).sum(axis=1)
        else:
            s = X.sum(axis=1)
        S = float(s.sum())

    # Row scale factors so that new usage ∝ target t
    s_safe = np.maximum(s, 1e-30)
    scale = t / s_safe
    X *= scale[:, None]

    if preserve_total:
        # Keep overall mass in the same metric as before
        if E_cp is not None:
            Sp = float((X * E).sum())
        else:
            Sp = float(X.sum())
        if Sp > 0.0 and S > 0.0:
            X *= (S / Sp)

    np.maximum(X, 0.0, out=X)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

# ------------------------------------------------------------------------------

def _solve_nnls(Bw: np.ndarray, yw: np.ndarray, solver="nnls", max_iter=200, ridge=0.0,
                normalize_columns: bool = True) -> np.ndarray:
    """
    Solve min ||Bw x - yw||_2 subject to x>=0.
    If normalize_columns, solve on Bn = Bw / col_norm, then map back via x = z / col_norm.
    """
    K = int(Bw.shape[1])
    if normalize_columns:
        col_norm = np.linalg.norm(Bw, axis=0)
        col_norm = np.where(col_norm > 0.0, col_norm, 1.0)
        Bn = Bw / col_norm
    else:
        col_norm = np.ones(K, dtype=np.float64)
        Bn = Bw

    if solver == "nnls":
        from scipy.optimize import nnls
        try:
            z, _ = nnls(Bn, yw)              # solve on normalized system
            return z / col_norm               # map back: x = z / ||Bw[:,j]||
        except Exception as e:
            # Lawson–Hanson hit a singular AtA[P,P]. Fall back to a stable PG solve on Bn.
            # Lightweight projected gradient on normalized system -> z, then x = z/col_norm
            # (same stopping as your PG branch, but inline to avoid re-normalizing twice)
            K = int(Bn.shape[1])
            z = np.zeros(K, dtype=np.float64)
            L_est = float(np.max(np.sum(Bn*Bn, axis=0))) + float(ridge)
            L_est = L_est if np.isfinite(L_est) and L_est > 0.0 else 1.0
            step = 1.0 / L_est
            for _ in range(int(np.minimum(max_iter, 400))):
                r = Bn @ z - yw
                g = Bn.T @ r + float(ridge) * z
                z_try = np.maximum(0.0, z - step * g)
                r_try = Bn @ z_try - yw
                f_old = 0.5 * float(r @ r)     + 0.5 * float(ridge) * float(z @ z)
                f_new = 0.5 * float(r_try @ r_try) + 0.5 * float(ridge) * float(z_try @ z_try)
                bt = 0
                while f_new > f_old and bt < 8:
                    step *= 0.5
                    z_try = np.maximum(0.0, z - step * g)
                    r_try = Bn @ z_try - yw
                    f_new = 0.5 * float(r_try @ r_try) + 0.5 * float(ridge) * float(z_try @ z_try)
                    bt += 1
                z = z_try
                if np.linalg.norm(g, ord=np.inf) * step < 1e-9:
                    break
            return z / col_norm

    if solver == "lsq":
        from scipy.optimize import lsq_linear
        if ridge > 0.0:
            I = np.sqrt(ridge) * np.eye(K, dtype=np.float64)
            Bw_aug = np.vstack([Bn, I])
            yw_aug = np.concatenate([yw, np.zeros(K, dtype=np.float64)], axis=0)
            res = lsq_linear(Bw_aug, yw_aug, bounds=(0.0, np.inf), method="trf",
                             max_iter=int(max_iter), verbose=0)
        else:
            res = lsq_linear(Bn, yw, bounds=(0.0, np.inf), method="trf",
                             max_iter=int(max_iter), verbose=0)
        return np.maximum(0.0, res.x) / col_norm

    # simple PGD with backtracking (supports ridge)
    xz = np.zeros(K, dtype=np.float64)  # this is 'z' in the normalized space
    L_est = float((Bn**2).sum(axis=0).max() + ridge) or 1.0
    step = 1.0 / L_est
    for _ in range(int(max_iter)):
        r = Bn @ xz - yw
        g = Bn.T @ r + ridge * xz
        z_try = np.maximum(0.0, xz - step * g)
        r_try = Bn @ z_try - yw
        f_old = 0.5 * float(r @ r)     + 0.5 * ridge * float(xz @ xz)
        f_new = 0.5 * float(r_try @ r_try) + 0.5 * ridge * float(z_try @ z_try)
        bt = 0
        while f_new > f_old and bt < 8:
            step *= 0.5
            z_try = np.maximum(0.0, xz - step * g)
            r_try = Bn @ z_try - yw
            f_new = 0.5 * float(r_try @ r_try) + 0.5 * ridge * float(z_try @ z_try)
            bt += 1
        xz = z_try
        if np.linalg.norm(g, ord=np.inf) * step < 1e-9:
            break
    return xz / col_norm

# -------------------------- main routine -------------------------------------

def run_patch(h5_path: str,
              s_sel: Optional[str] = None,
              k_per_comp: int = 12,
              pick_mode: str = "energy",
              solver: str = "nnls",
              ridge: float = 0.0,
              use_mask: bool = True,
              use_lambda: bool = True,
              lam_dset: str = "/HyperCube/lambda_weights",
              out_dir: Optional[str | os.PathLike] = None,
              write_seed: bool = False,
              seed_path: str = "/Seeds/x0_nnls_patch",
              normalize_columns: bool = True,
              orbit_weights: np.ndarray | None = None):

    out_dir = None if out_dir is None else str(out_dir)
    if out_dir:
        plp.Path(out_dir).mkdir(parents=True, exist_ok=True)

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

    if use_lambda:
        try:
            w_full = read_lambda_weights(h5_path, dset_name=lam_dset, floor=1e-6)
        except Exception:
            w_full = ensure_lambda_weights(h5_path, dset_name=lam_dset)
    else:
        w_full = np.ones(L, dtype=np.float64)
    w_lam = w_full if keep_idx is None else w_full[keep_idx]

    s_idx = _pick_spaxels(S, s_sel, Ns_default=32)
    s_idx = np.sort(np.asarray(s_idx, dtype=int))
    if s_idx.size < 1:
        raise RuntimeError("Empty spaxel selection.")

    # ---- quick triage (paste right after w_lam and s_idx) ----
    print(f"[triage] S_sel={s_idx.size}")

    Lk = int(keep_idx.size) if keep_idx is not None else int(len(w_full))
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
    finite_counts = np.asarray(finite_counts, int)
    print(f"[triage] finite pixels per spaxel (min/med/max): "
        f"{finite_counts.min()}/{int(np.median(finite_counts))}/{finite_counts.max()}")

    if finite_counts.min() == 0:
        print("[triage] Every selected spaxel with 0 finite pixels under the mask "
            "will contribute zero rows. If they are ALL zero, "
            "Bw==0 and you’ll see the 'all columns ~zero' message. "
            "Try --no-mask or pick spaxels with finite data.")

    # λ-weights sanity
    w_masked = w_lam
    print(f"[triage] lambda weights on mask min/med/max: "
        f"{w_masked.min():.3g}/{np.median(w_masked):.3g}/{w_masked.max():.3g}")

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
        df   = (np.asarray(f["/HyperCube/data_flux"][s_idx], float)
                if "/HyperCube/data_flux" in f else None)
        la   = (np.asarray(f["/HyperCube/norm/losvd_amp_sum"][s_idx], float)
                if "/HyperCube/norm/losvd_amp_sum" in f else None)

        E_cp = np.asarray(f["/HyperCube/col_energy"][...], float)  # (C,P)
        S, L = map(int, f["/DataCube"].shape)
        rms_cp = np.sqrt(E_cp / max(1, S * L)) # RMS amplitude

    print(f"[triage] norm.mode={mode}")
    print(
        "[triage] model RMS amplitude per (c,p) column (min/med/max): "
        f"{rms_cp.min():.3g}/{np.median(rms_cp):.3g}/{rms_cp.max():.3g}"
    )
    if df is not None:
        print(
            "[triage] data_flux (min/med/max): "
            f"{df.min():.3g}/{np.median(df):.3g}/{df.max():.3g}"
        )
    if la is not None:
        print(
            "[triage] losvd_amp_sum (min/med/max): "
            f"{la.min():.3g}/{np.median(la):.3g}/{la.max():.3g}"
        )
    
    E = read_global_column_energy(h5_path)  # (C,P)
    picks = _choose_pops(E, k_per_comp, pick_mode)

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
            # per-spaxel finite mask
            good = np.isfinite(spec)
            # store y with NaNs → 0 (they get zero weight)
            spec_sanit = np.where(good, spec, 0.0)
            y[pos:pos+Lk] = spec_sanit
            # per-row weights: √w_lambda * 1[finite]
            sqrt_w_rows[pos:pos+Lk] = np.sqrt(w_lam) * good.astype(np.float64)
            finite_counts.append(int(np.count_nonzero(good)))
            pos += Lk
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
    x_full, keep = _global_nnls_workingset(
        Bw, yw, col_map,
        k0_per_comp=32, kincr_per_comp=16, rounds=4,
        solver=solver, ridge=float(ridge), normalize_columns=bool(normalize_columns)
    )
    # keep col_map aligned with x_patch for the unpack loop
    x_patch = x_full[keep].copy()                # compress to working set
    col_map  = [col_map[j] for j in keep]        # now 1:1 aligned
    K = int(len(col_map))  # (optional) if you print/return K later

    # --- Unpack to full (C,P)
    x_CP = np.zeros((C, P), dtype=np.float64)
    for j, (c, p) in enumerate(col_map):
        x_CP[c, p] = x_patch[j]
    # --- Enforce orbit prior on the seed (energy metric) --- FAST and strict
    if orbit_weights is not None:
        x_CP = apply_orbit_prior_to_seed(x_CP, orbit_weights,
            E_cp=E, preserve_total=True, min_w_frac=1e-4)

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
            wrow  = sqrt_w_rows[pos:pos+Lk]
            pos += Lk
            r = y_obs - y_fit
            # RMSE (unweighted) and χ² (λ-weighted on finite)
            rmse[i] = float(np.sqrt(np.mean((r)**2)))
            w2 = wrow * wrow
            den = float(np.sum(w2)) or 1.0
            chi2[i] = float(np.sum((r * wrow)**2) / den)

            if out_dir:
                fig = plt.figure(figsize=(9.5, 3.2))
                ax = fig.add_subplot(111)
                ax.plot(lam_plot, y_obs, lw=1.0, label="data")
                ax.plot(lam_plot, y_fit, lw=1.0, label="model")
                ax.set_title(f"spaxel {int(s)}  rmse={rmse[i]:.3f}")
                ax.set_xlabel("λ (ObsPix)")
                ax.set_ylabel("flux")
                ax.legend(loc="best", fontsize=8)
                fig.savefig(os.path.join(out_dir,
                    f"patch_spax{int(s):05d}.png"), dpi=120)
                plt.close(fig)

    if out_dir:
        flat = x_CP.ravel(order="C")
        nnz = int(np.count_nonzero(flat))
        if nnz == 0:
            print("[plot] x_CP is all zeros (no bars to draw).")
        else:
            # (optional) plot only the top 1000 by magnitude to make the bars visible
            order = np.argsort(flat)[::-1][:int(np.minimum(1000, nnz))]
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
            ax.bar(np.arange(order.size), flat[order])
            ax.set_title("x_CP top coefficients (sorted)")
            ax.set_xlabel("rank"); ax.set_ylabel("weight")
            fig.savefig(os.path.join(out_dir, "xcp_bar.png"), dpi=120)
            plt.close(fig)

    if write_seed:
        with open_h5(h5_path, role="writer") as f:
            g = f.require_group("/Seeds")
            if seed_path in f:
                del f[seed_path]
            ds = f.create_dataset(seed_path, data=x_CP.astype(np.float64),
                dtype="f8")
            ds.attrs["origin"] = "nnls_patch"
            ds.attrs["s_idx"]  = s_idx
            ds.attrs["k_per_comp"] = int(k_per_comp)
            ds.attrs["pick_mode"]  = str(pick_mode)
            ds.attrs["solver"]     = str(solver)
            ds.attrs["ridge"]      = float(ridge)
            ds.attrs["norm.mode_at_fit"] = norm_mode
    
    usage_png = os.path.join(out_dir, "nnlsPatchUsage.png") if out_dir else None

    metrics = None
    if write_seed:
        # Seed is in the main HDF5 at `seed_path`
        try:
            metrics = compare_usage_to_orbit_weights(
                h5_path,
                sidecar=None,
                x_dset=seed_path, # e.g. "/Seeds/x0_nnls_patch"
                normalize="unit_sum",
                out_png=usage_png,
                usage_metric="energy", # use energy–weighted usage
                E_cp=E, # (C,P) from read_global_column_energy
            )
        except Exception as e:
            print(f"[nnls_patch] usage-vs-prior (seed) failed: {e}")

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

            metrics = compare_usage_to_orbit_weights(
                h5_path,
                sidecar=tmp_sidecar,
                x_dset="/Fit/x_best",
                normalize="unit_sum",
                out_png=usage_png,
                usage_metric="energy",
                E_cp=E,
            )
        except Exception as e:
            print(f"[nnls_patch] usage-vs-prior (temp sidecar) failed: {e}")
        finally:
            if tmp_sidecar and os.path.exists(tmp_sidecar):
                try:
                    os.remove(tmp_sidecar)
                except OSError:
                    pass

    if metrics is not None:
        print(
            "[nnls_patch] usage-vs-prior:"
            f" L1={metrics.get('l1', float('nan')):.3e}"
            f"  L∞={metrics.get('linf', float('nan')):.3e}"
            f"  cos={metrics.get('cosine', float('nan')):.4f}"
            f"  r={metrics.get('pearson_r', float('nan')):.4f}"
            f"  plot={metrics.get('plot_path')}"
        )

    return dict(
        x_CP=x_CP, picks=picks, s_idx=s_idx, rmse=rmse, chi2=chi2,
        meta=dict(norm_mode=norm_mode, rows=int(rows), cols=int(K),
                  mask_used=bool(use_mask), lambda_used=bool(use_lambda),
                  solver=solver, ridge=float(ridge),
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
    ap.add_argument("--solver", default="nnls", choices=["nnls","lsq","pg"], help="NNLS solver")
    ap.add_argument("--ridge", type=float, default=0.0, help="L2 ridge (lsq/pg)")
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
        solver=args.solver,
        ridge=float(args.ridge),
        use_mask=not args.no_mask,
        use_lambda=not args.no_lambda,
        lam_dset=args.lam_dset,
        out_dir=args.out,
        write_seed=bool(args.write_seed),
        seed_path=args.seed_path,
    )
    small = {k: (f"array{v.shape}" if isinstance(v, np.ndarray) else v) for k, v in res.items() if k != "picks"}
    print("[nnls_patch] done.", small)
    print(res['x_CP'])

if __name__ == "__main__":
    main()