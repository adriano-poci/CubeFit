#!/usr/bin/env python3
"""
CubeFit diagnostics: include /HyperCube/col_norms and compare
pre-normalization (X) vs post-normalization (Y = X * col_norms).

Edit only H5_PATH. Uses repo helpers; performs only small reads.
"""
import sys
import json
import numpy as np
from pathlib import Path

# ---------------------- USER: set this path -------------------------------
H5_PATH = "/data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/NGC4365_207_12.h5"
# -------------------------------------------------------------------------

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import read_global_column_energy
from CubeFit import cube_debug


def read_dims(h5_path):
    with open_h5(h5_path, role="reader", swmr=True) as f:
        if "/HyperCube/models" not in f:
            raise RuntimeError("Missing /HyperCube/models dataset")
        S, C, P, L = map(int, f["/HyperCube/models"].shape)
        dims_json = f["/"].attrs.get("dims_json", None)
        dims = json.loads(dims_json) if dims_json is not None else {}
    return C, P, dims


def read_X(h5_path, C, P):
    candidates = ("/X_global", "/Fit/x_last", "/Fit/x_best", "/Fit/x_epoch_last")
    with open_h5(h5_path, role="reader", swmr=True) as f:
        for name in candidates:
            if name in f:
                raw = np.asarray(f[name][...], dtype=np.float64)
                Xcp = cube_debug._row_or_vec_to_CP(raw, C, P)
                return Xcp, name
        for name in ("/Fit/orbit_weights", "/CompWeights"):
            if name in f:
                raw = np.asarray(f[name][...], dtype=np.float64).ravel(order="C")
                if raw.size == C * P:
                    Xcp = raw.reshape(C, P, order="C")
                    return Xcp, name + " (C*P)"
                elif raw.size == C:
                    Xcp = np.zeros((C, P), dtype=np.float64)
                    Xcp[:, 0] = raw
                    return Xcp, name + " (C->P=1 proxy)"
    raise RuntimeError("No X found in known locations")


def load_or_compute_col_norms(h5_path, C, P):
    """Return col_norms shaped (C,P) float64.

    Priority:
      1) /HyperCube/col_norms
      2) sqrt(max(/HyperCube/col_energy,0))
      3) ones fallback
    """
    with open_h5(h5_path, role="reader", swmr=True) as f:
        if "/HyperCube/col_norms" in f:
            cn = np.asarray(f["/HyperCube/col_norms"][...], dtype=np.float64)
            if cn.shape != (C, P):
                raise RuntimeError("col_norms shape mismatch: expected (%d,%d)" % (C, P))
            return cn, "col_norms"
    # fallback to col_energy
    Ecp = read_global_column_energy(h5_path, dset_name="/HyperCube/col_energy",
                                    strict=False)
    if Ecp is not None:
        Ecp = np.maximum(Ecp, 0.0)
        cn = np.sqrt(Ecp, dtype=np.float64)
        if cn.shape != (C, P):
            raise RuntimeError("computed col_norms shape mismatch: expected (%d,%d)" % (C, P))
        return cn, "sqrt(col_energy)"
    # final fallback
    return np.ones((C, P), dtype=np.float64), "ones_fallback"


def summarize_matrix_rows(Xcp, name):
    """Compute and print compact row/PC/correlation summaries for Xcp."""
    C, P = Xcp.shape
    print(f"\n--- Summary for {name} (shape {C}x{P}) ---")
    Xflat = Xcp.ravel(order="C")
    print(" entries: dtype=%s  count=%d" % (Xflat.dtype, Xflat.size))
    for q in (0.0, 1, 5, 25, 50, 75, 95, 99, 100):
        print("  %3dth pct: % .6e" % (q, np.percentile(Xflat, q)))
    # per-orbit norms
    l1 = np.sum(np.abs(Xcp), axis=1)
    l2 = np.sqrt(np.sum(Xcp**2, axis=1))
    nz = np.count_nonzero(Xcp, axis=1)
    print("  per-orbit L1: min/med/max/std: %.3e / %.3e / %.3e / %.3e"
          % (l1.min(), np.median(l1), l1.max(), l1.std()))
    print("  per-orbit L2: min/med/max/std: %.3e / %.3e / %.3e / %.3e"
          % (l2.min(), np.median(l2), l2.max(), l2.std()))
    # population variance
    pop_var = np.var(Xcp, axis=0)
    print("  pop_var: min/med/max: %.3e / %.3e / %.3e"
          % (pop_var.min(), np.median(pop_var), pop_var.max()))
    # row correlations (summary)
    # normalize rows to unit std (avoid divide-by-zero)
    row_means = Xcp.mean(axis=1, keepdims=True)
    row_std = Xcp.std(axis=1, keepdims=True)
    safe_std = np.where(row_std == 0.0, 1.0, row_std)
    X_norm = (Xcp - row_means) / safe_std
    corr = (X_norm @ X_norm.T) / float(P)
    offdiag = corr[np.triu_indices(C, k=1)]
    print("  corr off-diag mean/med/90p/99p: %.6f / %.6f / %.6f / %.6f"
          % (offdiag.mean(), np.median(offdiag),
             np.percentile(offdiag, 90), np.percentile(offdiag, 99)))
    # PCA / SVD on rows (economy)
    X_centered = Xcp - Xcp.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    sv2 = S**2
    if sv2.sum() == 0:
        explained = np.zeros_like(sv2)
    else:
        explained = sv2 / sv2.sum()
    cum = np.cumsum(explained)
    k90 = int(np.searchsorted(cum, 0.90)) + 1
    k99 = int(np.searchsorted(cum, 0.99)) + 1
    print("  PCA: comps for 90%% var: %d ; for 99%% var: %d" % (k90, k99))
    print("  largest singulars (first 6):", ["%.6g" % s for s in S[:6]])
    return {
        "pctiles": [np.percentile(Xflat, q) for q in (0,1,5,25,50,75,95,99,100)],
        "row_l1_med": float(np.median(l1)),
        "row_l2_med": float(np.median(l2)),
        "pop_var_med": float(np.median(pop_var)),
        "corr_offdiag_med": float(np.median(offdiag)),
        "k90": int(k90),
        "k99": int(k99),
        "svals": S[:6].tolist()
    }


def main(h5_path):
    h5_path = str(Path(h5_path))
    print("Normalized-diagnostics run on:", h5_path)
    C, P, dims = read_dims(h5_path)
    print("Detected dims: C=%d, P=%d" % (C, P))

    Xcp, src = read_X(h5_path, C, P)
    print("Read X from:", src, "shape:", Xcp.shape)

    col_norms_cp, cn_src = load_or_compute_col_norms(h5_path, C, P)
    print("col_norms source:", cn_src)
    cn_flat = col_norms_cp.ravel(order="C")
    # stats of col_norms
    pos = cn_flat > 0.0
    print("\ncol_norm stats (flat):")
    print("  dtype:", cn_flat.dtype, "count:", cn_flat.size)
    print("  min/median/max (pos entries): %.6e / %.6e / %.6e"
          % (float(np.min(cn_flat[pos])) if np.any(pos) else 0.0,
             float(np.median(cn_flat[pos])) if np.any(pos) else 0.0,
             float(np.max(cn_flat))))
    zeros = np.count_nonzero(cn_flat == 0.0)
    print("  zero-count:", zeros, " (%.2f%%)" % (100.0 * zeros / cn_flat.size))

    # create normalized variable Y = X * col_norms (elementwise)
    Ycp = Xcp * col_norms_cp  # shape (C,P); broadcasting elementwise
    # Summarize X and Y
    summary_X = summarize_matrix_rows(Xcp, "X (original)")
    summary_Y = summarize_matrix_rows(Ycp, "Y = X * col_norms (normalized)")

    # Compact compare table
    print("\n--- Compact comparison ---")
    print(" metric                    X (orig)        Y (normalized)")
    print(" median row L2         :  %12.6e    %12.6e" %
          (summary_X["row_l2_med"], summary_Y["row_l2_med"]))
    print(" median pop variance   :  %12.6e    %12.6e" %
          (summary_X["pop_var_med"], summary_Y["pop_var_med"]))
    print(" median corr off-diag  :  %12.6e    %12.6e" %
          (summary_X["corr_offdiag_med"], summary_Y["corr_offdiag_med"]))
    print(" comps for 90%% var     :  %6d         %6d" %
          (summary_X["k90"], summary_Y["k90"]))
    print(" comps for 99%% var     :  %6d         %6d" %
          (summary_X["k99"], summary_Y["k99"]))

    # Additional quick diagnostic: count near-identical row pairs after norm
    def top_identical_pairs(Xm, topk=10):
        C = Xm.shape[0]
        # small C (207) allows full pairwise check cheaply
        Xm_n = (Xm - Xm.mean(axis=1, keepdims=True))
        # normalize rows to unit L2 to compare shape similarity
        norms = np.linalg.norm(Xm_n, axis=1)
        safe = np.where(norms == 0.0, 1.0, norms)
        Xm_unit = Xm_n / safe[:, None]
        corrmat = Xm_unit @ Xm_unit.T
        iu = np.triu_indices(C, k=1)
        vals = corrmat[iu]
        idx = np.argsort(-vals)[:topk]
        pairs = []
        tri_i, tri_j = iu
        for k in idx:
            pairs.append((float(vals[k]), int(tri_i[k]), int(tri_j[k])))
        return pairs

    print("\nTop correlated orbit pairs BEFORE normalization (rho,i,j):")
    for rho, i, j in top_identical_pairs(Xcp, topk=8):
        print("  %.6f   %3d   %3d" % (rho, i, j))
    print("Top correlated orbit pairs AFTER normalization (rho,i,j):")
    for rho, i, j in top_identical_pairs(Ycp, topk=8):
        print("  %.6f   %3d   %3d" % (rho, i, j))

    print("\nNormalized diagnostics complete.")


if __name__ == "__main__":
    main(H5_PATH)
