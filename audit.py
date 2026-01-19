import numpy as np
from CubeFit.hdf5_manager import open_h5

h5_path = "/data/phys-gal-dynamics/phys2603/CubeFit/FCC170/FCC170_3_12.h5"

# -------------------------------
# Open once, keep alive
# -------------------------------
with open_h5(h5_path, role="reader") as f:
    # Required datasets
    X = np.asarray(f["/X_global"][...], dtype=np.float64)
    M = f["/HyperCube/models"]                  # h5py dataset handle
    DataCube = np.asarray(f["/DataCube"][...], dtype=np.float64)
    ModelCube = np.asarray(f["/ModelCube"][...], dtype=np.float64) if "/ModelCube" in f else None
    mask = np.asarray(f["/Mask"][...], bool).ravel() if "/Mask" in f else None
    binCounts = np.asarray(f["/BinCounts"][...], dtype=np.float64) if "/BinCounts" in f else None

    # -------------------------------
    # Canonicalise X -> (C,P)
    # -------------------------------
    X_flat = X.ravel(order="C")
    
    # -------------------------------
    # Detect HyperCube layout
    # -------------------------------
    if M.ndim == 4:
        # (S, C, P, L)
        S, C, P, L = map(int, M.shape)

        def model_spatial_vector(c, p, lam):
            return np.asarray(M[:, c, p, lam], dtype=np.float64)

    elif M.ndim == 5:
        # (nB, B, C, P, L)
        nB, B, C, P, L = map(int, M.shape)
        S = nB * B

        def model_spatial_vector(c, p, lam):
            out = np.empty(S, dtype=np.float64)
            for s in range(S):
                b = s // B
                i = s % B
                out[s] = M[b, i, c, p, lam]
            return out

    else:
        raise RuntimeError(f"Unexpected /HyperCube/models rank {M.ndim}")

    print(f"Hypercube layout: S={S}, C={C}, P={P}, L={L}")

    # -------------------------------
    # Pick a representative wavelength
    # -------------------------------
    if mask is not None and np.any(mask):
        lam0 = int(np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2])
    else:
        lam0 = L // 2

    print(f"Using wavelength index lam0={lam0}")

    # -------------------------------
    # 1) Is ModelCube flat across space?
    # -------------------------------
    if ModelCube is not None:
        v = ModelCube[:, lam0]
        r = np.std(v) / (np.mean(v) + 1e-30)
        print(f"ModelCube[:,lam0] std/mean = {r:.3e}")
    else:
        print("No /ModelCube found.")

    # -------------------------------
    # 2) Do HyperCube templates vary with s?
    #    (test a few (c,p) with nonzero X)
    # -------------------------------
    if X_flat.size != C * P:
        raise RuntimeError("X_global incompatible with HyperCube shape")

    Xcp = X_flat.reshape(C, P)
    nonzero_cp = np.argwhere(Xcp > 0)

    if nonzero_cp.size == 0:
        raise RuntimeError("X_global has no positive entries")

    # pick up to 5 representative (c,p)
    pick = nonzero_cp[np.linspace(0, len(nonzero_cp)-1,
                                  min(5, len(nonzero_cp)),
                                  dtype=int)]

    print("\nSpatial variation of HyperCube models at lam0:")
    for (c, p) in pick:
        col = model_spatial_vector(int(c), int(p), lam0)
        r = np.std(col) / (np.mean(col) + 1e-30)
        print(f"  models[:,c={c},p={p},lam0] std/mean = {r:.3e}")

    # -------------------------------
    # 3) Is X_global collapsed (rank-1)?
    # -------------------------------
    U, svals, Vt = np.linalg.svd(Xcp, full_matrices=False)
    frac = svals[0] / (np.sum(svals) + 1e-30)

    print("\nX_global SVD diagnostics:")
    print("  singular values (first 6):", svals[:6])
    print(f"  s1 / sum(s) = {frac:.6f}")

    # -------------------------------
    # 4) Sanity: does raw model vary with binCounts?
    # -------------------------------
    if binCounts is not None and ModelCube is not None:
        model_raw = np.sum(ModelCube[:, mask], axis=1) if mask is not None else np.sum(ModelCube, axis=1)
        valid = np.isfinite(binCounts) & np.isfinite(model_raw)
        if np.any(valid):
            corr = np.corrcoef(binCounts[valid], model_raw[valid])[0, 1]
            print(f"\nCorrelation corr(binCounts, model_raw) = {corr:.3f}")
        else:
            print("\nNot enough valid bins to compute corr(binCounts, model_raw).")
