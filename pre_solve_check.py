#!/usr/bin/env python3
# Pre-solve sanity checks for CubeFit HDF5 files.

import sys, argparse, json, random
from typing import List
try:
    import h5py
    import numpy as np
except Exception:
    print("ERROR: This script requires h5py and numpy. Please install them.", file=sys.stderr)
    sys.exit(2)

def validate_schema(h5_path: str):
    issues: List[str] = []
    warnings: List[str] = []
    with h5py.File(h5_path, "r", swmr=True) as f:
        # Required datasets
        required = [
            "/DataCube", "/Templates", "/LOSVD", "/TemPix", "/ObsPix", "/VelPix",
            "/R_T", "/HyperCube/models", "/HyperCube/_done"
        ]
        missing = [p for p in required if p not in f]
        if missing:
            issues += [f"missing dataset: {p}" for p in missing]
            return False, issues, warnings

        # Shapes
        S, L = f["/DataCube"].shape
        P, T = f["/Templates"].shape
        S2, V, C = f["/LOSVD"].shape
        T2 = f["/TemPix"].shape[0]
        L2 = f["/ObsPix"].shape[0]
        V2 = f["/VelPix"].shape[0]

        if S2 != S: issues.append(f"/LOSVD first dim S={S2} ≠ /DataCube S={S}")
        if V2 != V: issues.append(f"/VelPix V={V2} ≠ /LOSVD V={V}")
        if T2 != T: issues.append(f"/TemPix T={T2} ≠ /Templates T={T}")
        if L2 != L: issues.append(f"/ObsPix L={L2} ≠ /DataCube L={L}")

        # Rebin operators
        if f["/R_T"].shape != (T, L):
            issues.append(f"/R_T shape {f['/R_T'].shape} != (T,L)=({T},{L})")
        if "/RebinMatrix" in f and f["/RebinMatrix"].shape != (L, T):
            issues.append(f"/RebinMatrix shape {f['/RebinMatrix'].shape} != (L,T)=({L},{T})")
        if f["/R_T"].dtype != np.float32:
            warnings.append(f"/R_T dtype {f['/R_T'].dtype} (expected float32)")

        # FFT caches (optional)
        if "/TemplatesFFT" in f:
            tfft = f["/TemplatesFFT"]
            if tfft.shape != (P, T//2 + 1):
                issues.append(f"/TemplatesFFT shape {tfft.shape} != (P, T//2+1)=({P},{T//2+1})")
            if tfft.dtype.kind != "c":
                warnings.append(f"/TemplatesFFT dtype {tfft.dtype} (expected complex)")
        if "/TemplatesFFT_R" in f:
            tfftr = f["/TemplatesFFT_R"]
            if tfftr.shape != (P, T//2 + 1):
                issues.append(f"/TemplatesFFT_R shape {tfftr.shape} != (P, T//2+1)")

        # Mask (optional)
        if "/Mask" in f and f["/Mask"].shape[0] != L:
            issues.append(f"/Mask length {f['/Mask'].shape[0]} != L={L}")

        # HyperCube dataset
        HC = f["/HyperCube/models"]
        if HC.shape != (S, C, P, L):
            issues.append(f"/HyperCube/models shape {HC.shape} != (S,C,P,L)=({S},{C},{P},{L})")
        if HC.dtype != np.float32:
            warnings.append(f"/HyperCube/models dtype {HC.dtype} (expected float32)")

        # Root attrs
        if "dims" in f["/"].attrs:
            try:
                d = f["/"].attrs["dims"]
                d = json.loads(d) if isinstance(d, (bytes, str)) else d
                check = dict(nSpat=S, nLSpec=L, nTSpec=T, nVel=V, nComp=C, nPop=P)
                for k, v in check.items():
                    key = str(k)
                    if key in d and int(d[key]) != v:
                        issues.append(f"dims[{k}]={d[key]} ≠ {v}")
            except Exception as e:
                warnings.append(f"failed to parse root attr 'dims': {e}")
        else:
            warnings.append("root attr 'dims' missing")

        # Grid monotonicity
        for name in ["/TemPix", "/ObsPix", "/VelPix"]:
            v = f[name][...]
            if not np.all(np.diff(v) > 0):
                issues.append(f"{name} is not strictly increasing")

    return (len(issues) == 0), issues, warnings

def transpose_cache_delta(f) -> float:
    if "/RebinMatrix" not in f:
        return float("nan")
    RM = f["/RebinMatrix"][...]
    RTt = f["/R_T"][...].T
    return float(np.max(np.abs(RM - RTt)))

def sample_neg_nan_stats(f, samples=3, seed=0):
    rng = random.Random(seed)
    M = f["/HyperCube/models"]; S,C,P,L = M.shape
    picks = [(0,0), (S-1, C-1)]
    for _ in range(max(0, samples - len(picks))):
        picks.append((rng.randrange(S), rng.randrange(C)))
    results = []
    for s,c in picks:
        slab = np.asarray(M[s,c,:,:], np.float32)  # (P,L)
        neg_frac = float((slab < 0).mean())
        has_nan = bool(np.isnan(slab).any())
        maxabs = float(np.max(np.abs(slab))) if slab.size else 0.0
        results.append(((s,c), neg_frac, has_nan, maxabs))
    return results

def main():
    ap = argparse.ArgumentParser(description="Pre-solve sanity checks for CubeFit HDF5 files.")
    ap.add_argument("h5_path", help="Path to HDF5 file")
    ap.add_argument("--samples", type=int, default=3, help="How many (s,c) slabs to probe")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for slab sampling")
    ap.add_argument("--max-transpose-delta", type=float, default=1e-6, help="Tolerance for /RebinMatrix vs /R_T.T")
    ap.add_argument("--fail-on-warn", action="store_true", help="Exit nonzero if warnings are present")
    args = ap.parse_args()

    ok, issues, warnings = validate_schema(args.h5_path)

    with h5py.File(args.h5_path, "r", swmr=True) as f:
        delta = transpose_cache_delta(f)
        if not np.isnan(delta):
            print(f"transpose cache max|Δ| = {delta:.3e}")
            if delta > args.max_transpose_delta:
                issues.append(f"/RebinMatrix differs from /R_T.T by {delta:.3e} > {args.max_transpose_delta}")
        else:
            print("transpose cache: /RebinMatrix not present (skip)")

        stats = sample_neg_nan_stats(f, samples=args.samples, seed=args.seed)
        for (s,c), neg_frac, has_nan, maxabs in stats:
            print(f\"(s={s}, c={c}) neg%={100*neg_frac:.4f}%  nan?={has_nan}  max|model|={maxabs:.3e}\")

        # Try optional edge diagnostics if available
        try:
            import hdf5_manager as H5M  # user's module
            if hasattr(H5M, "diagnose_rebin_edges"):
                safe = H5M.diagnose_rebin_edges(args.h5_path, quiet=True)
                print(f"diagnose_rebin_edges: Safe? {safe}")
                if not safe:
                    warnings.append("diagnose_rebin_edges reported Safe=False")
        except Exception as e:
            print(f"diagnose_rebin_edges: unavailable ({e.__class__.__name__})")

    # Print summary
    if issues:
        print("\\nSchema INVALID:")
        for i in issues:
            print(" -", i)
    else:
        print("\\nSchema OK")

    if warnings:
        print("\\nWarnings:")
        for w in warnings:
            print(" -", w)

    # Exit code
    if issues or (warnings and args.fail_on_warn):
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
