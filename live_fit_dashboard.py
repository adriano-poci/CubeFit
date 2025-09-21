#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, h5py, numpy as np, matplotlib.pyplot as plt

def _minutes_since_start(ts):
    if ts.size == 0: return ts
    t0 = float(ts[0]); return (ts - t0) / 60.0

def load_metrics(f):
    m = f["/Fit/metrics"]; g = lambda n: m[n][...] if n in m else np.array([], float)
    return (g("epoch").astype(int), g("time_sec"), g("train_rmse_ewma"), g("val_rmse"),
            g("delta_x_rel"), g("nnz"), g("l1_norm"), g("l2_norm"), g("mass_err_L1"), g("mass_err_Linf"))

def render_dashboard(h5_path: str, out_png: str, n_spaxels: int = 8, downsample: int = 512, figsize=(14, 10)):
    with h5py.File(h5_path, "r", swmr=True, libver="latest") as f:
        epoch, time_sec, tr_ewma, val_rmse, dx_rel, nnz, l1, l2, me_L1, me_Linf = load_metrics(f)
        x = f["/Fit/x_latest"][...].astype(np.float64, copy=True)
        S, C, P, L = f["/HyperCube/models"].shape
        spaxels = f["/Fit/sample/spaxels"][...].astype(int)
        ow = f["/Fit/orbit_weights"][...].astype(np.float64) if "/Fit/orbit_weights" in f else None

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0,0]
        tmin = _minutes_since_start(time_sec)
        if tr_ewma.size: ax.plot(tmin, tr_ewma, label="train_rmse_ewma")
        if val_rmse.size: ax.plot(tmin, val_rmse, marker="o", linestyle="none", label="val_rmse")
        ax.set_xlabel("minutes since start"); ax.set_ylabel("RMSE"); ax.set_title("Convergence"); ax.legend(loc="best")

        ax = axes[0,1]
        if l1.size: ax.plot(tmin, l1, label="||x||1")
        if l2.size: ax.plot(tmin, l2, label="||x||2")
        if nnz.size: ax.plot(tmin, nnz, label="nnz(x)")
        if dx_rel.size: ax.plot(tmin, dx_rel, label="delta_x_rel")
        if me_L1.size: ax.plot(tmin, me_L1, label="mass_err_L1")
        if me_Linf.size: ax.plot(tmin, me_Linf, label="mass_err_Linf")
        ax.set_xlabel("minutes since start"); ax.set_title("Params & mass errors"); ax.legend(loc="best")

        ax = axes[1,0]
        # small multiples collapsed (overlay) for speed
        lam_idx = np.linspace(0, L - 1, min(L, downsample), dtype=int)
        X = x.reshape(C, P)
        for s in spaxels[:min(n_spaxels, spaxels.size)]:
            # predict fast by stacking A[:,:,k]
            blocks = [f["/HyperCube/models"][s, :, :, k][...] for k in lam_idx]
            A = np.stack(blocks, axis=-1)
            yhat = np.tensordot(X, A, axes=([0,1],[0,1]))
            y = f["/DataCube"][s, lam_idx][...].astype(np.float64, copy=False)
            ax.plot(lam_idx, y, alpha=0.5); ax.plot(lam_idx, yhat, alpha=0.8)
        ax.set_xlabel("pixel"); ax.set_title("Observed vs model (subset)")

        ax = axes[1,1]
        usage = X.sum(axis=1)
        ax.bar(np.arange(C), usage, label="usage")
        if ow is not None:
            ax.plot(np.arange(C), ow, marker="o", linestyle="--", label="target w_c")
            maxerr = float(np.max(np.abs(usage - ow))) if ow.size and usage.size else np.nan
            ax.set_title(f"Component usage vs target (max|e|={maxerr:.3g})")
        else:
            ax.set_title("Component usage (sum over P)")
        ax.set_xlabel("component c"); ax.legend(loc="best")

        fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Render live Kaczmarz fit dashboard (SWMR reader).")
    ap.add_argument("h5_path"); ap.add_argument("--out", default="fit_live.png")
    ap.add_argument("--n-spaxels", type=int, default=8); ap.add_argument("--downsample", type=int, default=512)
    args = ap.parse_args(); render_dashboard(args.h5_path, args.out, n_spaxels=args.n_spaxels, downsample=args.downsample); print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
