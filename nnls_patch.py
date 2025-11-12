from __future__ import annotations
import os, math, pathlib as plp
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import read_global_column_energy
from CubeFit.cube_utils import read_lambda_weights, ensure_lambda_weights

# ----------------------------- helpers ---------------------------------

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

def _pick_spaxels(S: int, s_sel: Optional[str]) -> np.ndarray:
    if s_sel is None:
        n = min(32, S)
        return np.arange(n, dtype=np.int64)
    if ":" in s_sel:
        a, b = s_sel.split(":")
        start = max(0, min(S-1, int(a)))
        count = max(1, int(b))
        end   = max(start, min(S, start + count))
        return np.arange(start, end, dtype=np.int64)
    idx = np.array([int(x) for x in s_sel.split(",") if x.strip() != ""], dtype=np.int64)
    idx = np.unique(idx)
    return idx[(idx >= 0) & (idx < S)]

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
              normalize_columns: bool = True) -> dict:

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

    s_idx = _pick_spaxels(S, s_sel)
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

    # hypercube scale sanity for your spaxels
    with open_h5(h5_path, role="reader") as f:
        mode = str(f["/HyperCube"].attrs.get("norm.mode", "model"))
        df   = f["/HyperCube/data_flux"][s_idx] if "/HyperCube/data_flux" in f else None
        la   = f["/HyperCube/norm/losvd_amp_sum"][s_idx] if "/HyperCube/norm/losvd_amp_sum" in f else None

        mmax = np.empty(s_idx.size, float)
        for i, s in enumerate(s_idx):
            Ms = np.asarray(f["/HyperCube/models"][s, :, :, :], float)  # (C,P,L)
            mmax[i] = np.nanmax(np.abs(Ms))
    print(f"[triage] norm.mode={mode}")
    print(f"[triage] max |model| per spaxel (min/med/max): "
        f"{mmax.min():.3g}/{np.median(mmax):.3g}/{mmax.max():.3g}")
    if df is not None:
        print(f"[triage] data_flux[s] (min/med/max): "
            f"{df.min():.3g}/{np.median(df):.3g}/{df.max():.3g}")
    if la is not None:
        print(f"[triage] losvd_amp_sum[s] (min/med/max): "
            f"{la.min():.3g}/{np.median(la):.3g}/{la.max():.3g}")

    with open_h5(h5_path, role="reader") as f:
        amp_ok = None
        if "/HyperCube/norm/losvd_amp_sum" in f:
            amp_ok = np.asarray(f["/HyperCube/norm/losvd_amp_sum"][s_idx], float)
        data_flux = None
        if "/HyperCube/data_flux" in f:
            data_flux = np.asarray(f["/HyperCube/data_flux"][s_idx], float)

        # Max model value per selected spaxel across all (c,p,λ) — cheap sanity check
        mmax = np.empty(s_idx.size, float)
        for i, s in enumerate(s_idx):
            # read just this spaxel; mask later
            Ms = np.asarray(f["/HyperCube/models"][s, :, :, :], dtype=np.float64)
            mmax[i] = np.nanmax(np.abs(Ms))

    print(f"[patch] norm.mode={norm_mode} | max(|model|) per spaxel "
          f"(min/med/max): {mmax.min():.3g}/{np.median(mmax):.3g}/{mmax.max():.3g}")
    if amp_ok is not None:
        print(f"[patch] losvd_amp_sum (min/med/max): "
              f"{amp_ok.min():.3g}/{np.median(amp_ok):.3g}/{amp_ok.max():.3g}")
    if data_flux is not None:
        print(f"[patch] data_flux (min/med/max): "
              f"{data_flux.min():.3g}/{np.median(data_flux):.3g}/{data_flux.max():.3g}")
    
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
    col_map: List[Tuple[int,int]] = []
    j = 0
    print(f"[patch] Building design: K={K} columns from chosen pops…")
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        for c, plist in enumerate(tqdm(picks, desc="[patch] comps", mininterval=0.5, dynamic_ncols=True)):
            for p_idx in plist:
                col = np.empty(rows, dtype=np.float64)
                pos = 0
                for s in s_idx:
                    A_sp = np.asarray(M[s, c, p_idx, :], dtype=np.float64)
                    if keep_idx is not None:
                        A_sp = A_sp[keep_idx]
                    # NEW: sanitize model values to avoid NaN/Inf propagation
                    # and zero out rows where the DATA are bad for this spaxel.
                    A_sp = np.where(np.isfinite(A_sp), A_sp, 0.0)
                    good_rows = (sqrt_w_rows[pos:pos+Lk] > 0.0)
                    if not np.all(good_rows):
                        # zero-out model rows that correspond to non-finite data
                        A_sp = A_sp * good_rows.astype(np.float64)
                    col[pos:pos+Lk] = A_sp
                    pos += Lk
                B[:, j] = col
                col_map.append((c, int(p_idx)))
                j += 1

    # NEW: final sanitation guard (paranoia)
    if not np.isfinite(B).all():
        bad_cols = np.any(~np.isfinite(B), axis=0).nonzero()[0]
        print(f"[patch] WARNING: non-finite values in {bad_cols.size} columns; "
              f"sanitizing to 0.")
        B[:, bad_cols] = np.nan_to_num(B[:, bad_cols], copy=False)

    # NEW: quick column-norm diagnostics on the weighted system
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
            A_sp = np.asarray(M[s, :, :, :], dtype=np.float32)  # (C,P,L)
            if keep_idx is not None:
                A_sp = A_sp[:, :, keep_idx]                       # (C,P,Lk)
            # only chosen pops contribute
            for c, plist in enumerate(picks):
                if len(plist) == 0:
                    continue
                coeff = x_CP[c, plist]                           # (|plist|,)
                if coeff.sum() != 0.0:
                    y_fit += coeff @ A_sp[c, plist, :]
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
                fig = plt.figure(figsize=(9.5, 3.2)); ax = fig.add_subplot(111)
                ax.plot(lam_plot, y_obs, lw=1.0, label="data")
                ax.plot(lam_plot, y_fit, lw=1.0, label="model")
                ax.set_title(f"spaxel {int(s)}  rmse={rmse[i]:.3f}")
                ax.set_xlabel("λ (ObsPix)")
                ax.set_ylabel("flux")
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"patch_spax{int(s):05d}.png"), dpi=120)
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
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, "xcp_bar.png"), dpi=120)
            plt.close(fig)

    if write_seed:
        with open_h5(h5_path, role="writer") as f:
            g = f.require_group("/Seeds")
            if seed_path in f: del f[seed_path]
            ds = f.create_dataset(seed_path, data=x_CP.astype(np.float64), dtype="f8", compression="gzip")
            ds.attrs["origin"] = "nnls_patch"
            ds.attrs["s_idx"]  = s_idx
            ds.attrs["k_per_comp"] = int(k_per_comp)
            ds.attrs["pick_mode"]  = str(pick_mode)
            ds.attrs["solver"]     = str(solver)
            ds.attrs["ridge"]      = float(ridge)
            ds.attrs["norm.mode_at_fit"] = norm_mode

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