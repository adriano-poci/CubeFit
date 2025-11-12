#!/usr/bin/env python3
# diag_losvd_models.py
from __future__ import annotations
import argparse, os, pathlib as plp
import numpy as np
import matplotlib.pyplot as plt

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import read_global_column_energy

def _pick_spaxels(S: int, s_sel: str | None) -> np.ndarray:
    if s_sel is None:
        return np.arange(min(1, S), dtype=np.int64)
    if ":" in s_sel:
        a, b = s_sel.split(":")
        start = max(0, min(S - 1, int(a)))
        count = max(1, int(b))
        end = max(start, min(S, start + count))
        return np.arange(start, end, dtype=np.int64)
    idx = np.array([int(x) for x in s_sel.split(",") if x.strip() != ""],
                   dtype=np.int64)
    idx = np.unique(idx)
    return idx[(idx >= 0) & (idx < S)]

def _choose_pops(E: np.ndarray, K: int, comps: np.ndarray | None) -> list[np.ndarray]:
    C, P = map(int, E.shape)
    if comps is None:
        comps = np.arange(C, dtype=np.int64)
    picks: list[np.ndarray] = []
    for c in range(C):
        if c in set(comps):
            order = np.argsort(E[c, :])[::-1]
            picks.append(order[:min(K, P)].astype(np.int64))
        else:
            picks.append(np.array([], dtype=np.int64))
    return picks

def main():
    ap = argparse.ArgumentParser(
        description="Inspect LOSVD, templates, and stored model columns."
    )
    ap.add_argument("h5", help="Path to CubeFit HDF5")
    ap.add_argument("--spax", default="0", help="'i' or 'start:count'")
    ap.add_argument("--comps", default=None,
                    help="components like '0,1' (default: all)")
    ap.add_argument("--k-per-comp", type=int, default=3,
                    help="top-K populations per component by global energy")
    ap.add_argument("--pops", default=None,
                    help="explicit pops e.g. '5,12,42' (overrides K)")
    ap.add_argument("--use-mask", action="store_true",
                    help="plot only masked λ")
    ap.add_argument("--out", default="diag_out", help="output directory")
    args = ap.parse_args()

    plp.Path(args.out).mkdir(parents=True, exist_ok=True)

    with open_h5(args.h5, role="reader") as f:
        S, L = map(int, f["/DataCube"].shape)
        T = int(f["/Templates"].shape[1])
        C = int(f["/LOSVD"].shape[2])
        P = int(f["/Templates"].shape[0])

        s_idx = _pick_spaxels(S, args.spax)
        if s_idx.size != 1:
            raise SystemExit("Please select exactly one spaxel (e.g., --spax 24).")
        s = int(s_idx[0])

        comps = None
        if args.comps:
            comps = np.array([int(x) for x in args.comps.split(",")], np.int64)

        # grids
        vgrid = np.asarray(f["/VelPix"][...], dtype=np.float64)           # (V,)
        tempix = np.asarray(f["/TemPix"][...], dtype=np.float64)          # (T,)
        obspix = np.asarray(f["/ObsPix"][...], dtype=np.float64)          # (L,)

        # mask (optional)
        if args.use_mask and "/Mask" in f:
            mask = np.asarray(f["/Mask"][...], dtype=bool).ravel()
            if mask.size != L:
                mask = np.ones(L, bool)
        else:
            mask = np.ones(L, bool)

        # picks
        if args.pops:
            pops = np.array([int(x) for x in args.pops.split(",")], np.int64)
            picks = [pops if (comps is None or c in set(comps)) else
                     np.array([], np.int64) for c in range(C)]
        else:
            E = read_global_column_energy(args.h5)                        # (C,P)
            picks = _choose_pops(E, int(args.k_per_comp), comps)

        # pull arrays once
        LOS = np.asarray(f["/LOSVD"][s, :, :], dtype=np.float64)          # (V,C)
        TPL = np.asarray(f["/Templates"][...], dtype=np.float64)          # (P,T)
        MOD = np.asarray(f["/HyperCube/models"][s, :, :, :], dtype=np.float32) # (C,P,L)

        # quick stats
        los_sums = LOS.sum(axis=0)
        los_trapz = np.trapz(LOS, vgrid, axis=0)
        print(f"[diag] spaxel {s}")
        print(f"[diag] LOSVD sum per comp:   min/med/max "
              f"{los_sums.min():.3g}/{np.median(los_sums):.3g}/"
              f"{los_sums.max():.3g}")
        print(f"[diag] LOSVD trapz per comp: min/med/max "
              f"{los_trapz.min():.3g}/{np.median(los_trapz):.3g}/"
              f"{los_trapz.max():.3g}")

        # per-(c,p) column norms on stored models (masked and unmasked)
        colnorm = {}
        for c in range(C):
            if picks[c].size == 0:
                continue
            for p in picks[c]:
                A = MOD[c, p, :]
                colnorm[(c, int(p))] = (
                    float(np.linalg.norm(A)),
                    float(np.linalg.norm(A[mask]))
                )

        if len(colnorm):
            u = np.array([v[0] for v in colnorm.values()])
            m = np.array([v[1] for v in colnorm.values()])
            print(f"[diag] ||model||2 (unmasked)  min/med/max "
                  f"{u.min():.3g}/{np.median(u):.3g}/{u.max():.3g}")
            print(f"[diag] ||model||2 (masked)    min/med/max "
                  f"{m.min():.3g}/{np.median(m):.3g}/{m.max():.3g}")

        # ----- plots -----
        # 1) LOSVD per component
        fig = plt.figure(figsize=(9, 3.2))
        ax = fig.add_subplot(111)
        for c in range(C):
            if comps is not None and c not in set(comps):
                continue
            ax.plot(vgrid, LOS[:, c], lw=1.0, label=f"c={c}")
        ax.set_title(f"LOSVD histograms @ spaxel {s}")
        ax.set_xlabel("velocity (km/s)")
        ax.set_ylabel("count / weight")
        ax.legend(fontsize=8, ncol=min(C, 4))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"losvd_spax{s:05d}.png"), dpi=130)
        plt.close(fig)

        # 2) Raw templates (selected pops)
        for c in range(C):
            if picks[c].size == 0:
                continue
            plist = picks[c]
            fig = plt.figure(figsize=(9.5, 3.2))
            ax = fig.add_subplot(111)
            for p in plist:
                ax.plot(tempix, TPL[p, :], lw=0.9, label=f"p={int(p)}")
            ax.set_title(f"Raw templates (P)  —  comps used: c={c}")
            ax.set_xlabel("log λ (TemPix)")
            ax.set_ylabel("template flux")
            ax.legend(fontsize=7, ncol=5)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"templates_c{c}_spax{s:05d}.png"),
                        dpi=130)
            plt.close(fig)

        # 3) Stored model columns (convolved+rebinned)
        for c in range(C):
            if picks[c].size == 0:
                continue
            plist = picks[c]
            fig = plt.figure(figsize=(9.5, 3.2))
            ax = fig.add_subplot(111)
            for p in plist:
                y = MOD[c, p, :]
                y_plot = y[mask] if mask is not None else y
                x_plot = obspix[mask] if mask is not None else obspix
                ax.plot(x_plot, y_plot, lw=0.9,
                        label=f"c={c}, p={int(p)}, "
                              f"||·||2={np.linalg.norm(y_plot):.2g}")
            ax.set_title(f"Stored models @ spaxel {s}  (convolved + rebinned)")
            ax.set_xlabel("log λ (ObsPix)")
            ax.set_ylabel("model flux")
            ax.legend(fontsize=7, ncol=3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"models_c{c}_spax{s:05d}.png"),
                        dpi=130)
            plt.close(fig)

        print(f"[diag] wrote figures to: {plp.Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
