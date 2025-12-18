#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob, os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from CubeFit.hypercube_reader import HyperCubeReader, ReaderCfg
from CubeFit.hdf5_manager import open_h5
from CubeFit.logger import get_logger

logger = get_logger()

divcmap = 'GECKOSdr'
moncmap = 'inferno'
moncmapr = 'inferno_r'

# ------------------------------------------------------------------------------

def render_aperture_fits_separate(h5_path: str,
                                  out_dir: str,
                                  apertures: list[int],
                                  sidecar: str | None = None,
                                  show_residual: bool = True,
                                  title_prefix: str | None = None,
                                  dpi: int = 150):
    """
    Write one figure per spaxel (isolated), using the same masked reader the solver uses.
    File names: <out_dir>/fit_spax<NNNN>.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # pick x (same priority order as dashboard)
    with open_h5(h5_path, role="reader") as F:
        if sidecar and os.path.exists(sidecar):
            with open_h5(sidecar, role="reader", swmr=True) as G:
                x = _read_latest_x(G, F)
        else:
            x = _read_latest_x(None, F)
    X_vec = np.asarray(x, float).ravel()

    reader = HyperCubeReader(h5_path, ReaderCfg(dtype_models="float32", apply_mask=True))
    try:
        for s in apertures:
            A, y = reader.read_spaxel_plane(int(s))  # masked A:(N,L_eff), y:(L_eff,)
            yhat = A.T @ X_vec
            lam = np.arange(y.size, dtype=int)
            good = np.isfinite(y)  # mirror solver's finite-row rule

            # choose layout
            if show_residual:
                fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
                ax0, ax1 = axes
            else:
                fig, ax0 = plt.subplots(1, 1, figsize=(12, 3.8))
                ax1 = None

            # top: data & model
            ax0.plot(lam[good], y[good],    lw=1.0, alpha=0.75, label="observed")
            ax0.plot(lam[good], yhat[good], lw=1.0, alpha=0.95, ls="--", label="model")
            ax0.set_ylabel("flux")
            ax0.legend(loc="best")
            t = f"Spaxel {s}"
            if title_prefix:
                t = f"{title_prefix} — {t}"
            ax0.set_title(t)

            # bottom: residual (optional)
            if show_residual and ax1 is not None:
                r = yhat - y
                ax1.plot(lam[good], r[good], lw=0.9)
                ax1.axhline(0.0, ls="--", lw=0.8)
                ax1.set_xlabel("pixel")
                ax1.set_ylabel("model - data")

            try:
                fig.tight_layout()
            except Exception:
                pass

            out_path = os.path.join(out_dir, f"fit_spax{int(s):05d}.png")
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            print(f"[Aperture] wrote {out_path}")
    finally:
        reader.close()

# --- Jacobi diagnostics helpers ----------------------------------------------

def alpha_star_stats(h5_path, x_vec, n_spax=32, tile=None, swmr_main: bool | None = None):
    """
    Compute alpha* over a small sample. For main file reads during diagnostics,
    keep swmr_main=None (no SWMR) to avoid flag conflicts. Use SWMR only for
    sidecar reads where a writer is active.
    """
    def _compute(F):
        M = F["/HyperCube/models"]; Y = F["/DataCube"]
        Mask = np.asarray(F["/Mask"][...], bool).ravel() if "/Mask" in F else None
        S, C, P, L = map(int, M.shape)
        keep = np.flatnonzero(Mask) if Mask is not None else None

        if tile is None:
            s0, s1 = 0, int(np.min((S, n_spax)))
        else:
            s0, s1 = int(tile[0]), int(np.min((tile[1], S)))
        idx = np.linspace(s0, s1 - 1, int(np.min((n_spax, s1 - s0))), dtype=int)

        X = np.asarray(x_vec, float).reshape(C, P)
        alphas = []
        for s in idx:
            slab = np.asarray(M[s, :, :, :], float)
            if keep is not None:
                slab = slab[:, :, keep]
            yhat = np.tensordot(slab, X, axes=([0, 1], [0, 1]))
            y    = np.asarray(Y[s, :], float)
            if keep is not None:
                y = y[keep]
            num = float(np.dot(y, yhat))
            den = float(np.dot(yhat, yhat) + 1e-300)
            alphas.append(num / den)
        a = np.array(alphas, float)
        return {
            "median": float(np.median(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "n": int(a.size),
        }

    # Preferred: no SWMR for the main H5 during diagnostics
    try:
        with open_h5(h5_path, role="reader", swmr=(True if swmr_main else None)) as f:
            return _compute(f)
    except OSError as e:
        # If someone left a conflicting handle open with a different SWMR flag,
        # retry without SWMR to avoid “SWMR read access flag … already open”.
        if "SWMR read access flag" in str(e):
            with open_h5(h5_path, role="reader") as f:
                return _compute(f)
        raise

def render_aperture_fits_with_x(
    h5_path,
    x_vec,
    out_png,
    apertures,
    show_residual=True,
    title=None,
):
    """
    Render fits for selected spaxels using a global solution x_vec, but
    stream the HyperCube per-component to avoid a large (C, P, L) slab
    in memory.
    """
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]   # (S, C, P, L)
        Y = f["/DataCube"]           # (S, L)

        Mask = (
            np.asarray(f["/Mask"][...], bool).ravel()
            if "/Mask" in f
            else None
        )

        S, C, P, L = map(int, M.shape)

        X = np.asarray(x_vec, float).reshape(C, P)   # same basis as x_vec

        # Precompute λ indices once
        if Mask is not None:
            keep = np.flatnonzero(Mask)
        else:
            keep = None

        fig, axes = plt.subplots(
            len(apertures),
            1,
            figsize=(12, 2.7 * len(apertures)),
        )
        if len(apertures) == 1:
            axes = [axes]

        for ax, s in zip(axes, apertures):
            if not (0 <= s < S):
                raise IndexError(f"spaxel index {s} out of range [0, {S})")

            # Observed spectrum
            y = np.asarray(Y[s, :], float)   # (L,)

            # Model spectrum: stream over components c
            yhat = np.zeros(L, float)
            for c in range(C):
                # A_c has shape (P, L) for this spaxel / component
                A_c = np.asarray(M[s, c, :, :], float)  # (P, L)
                # dot over P: (P,) @ (P, L) -> (L,)
                yhat += X[c, :].astype(float, copy=False) @ A_c

            if keep is not None:
                y_plot = y[keep]
                yhat_plot = yhat[keep]
                good = np.isfinite(y_plot)
                lam_idx = np.flatnonzero(good)
                ax.plot(lam_idx, y_plot[good], label="data")
                ax.plot(lam_idx, yhat_plot[good], label="model")
                if show_residual:
                    ax.plot(
                        lam_idx,
                        (y_plot - yhat_plot)[good],
                        label="residual",
                    )
            else:
                good = np.isfinite(y)
                lam_idx = np.flatnonzero(good)
                ax.plot(lam_idx, y[good], label="data")
                ax.plot(lam_idx, yhat[good], label="model")
                if show_residual:
                    ax.plot(
                        lam_idx,
                        (y - yhat)[good],
                        label="residual",
                    )

            ax.set_title(f"spaxel {s}")
            ax.set_xlabel("λ (log space)")
            ax.set_ylabel("flux")
            ax.legend(loc="upper right")

        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
        fig.savefig(out_png, dpi=130)
        plt.close(fig)

def render_sfh_from_x(h5_path: str,
                      x_flat: np.ndarray,
                      out_png: str):
    """
    Simple Age×Z panels (per α) from x. Uses /Templates.attrs['pop_shape'].
    """

    x_flat = np.asarray(x_flat, dtype=np.float64).ravel()
    with open_h5(h5_path, role="reader") as f:
        C = int(f["/HyperCube/models"].shape[1])
        T = f["/Templates"]
        pop_shape = T.attrs.get("pop_shape", None)
        if pop_shape is None:
            # fallback: treat templates as 1-D population axis
            P = int(T.shape[0])
            pop_shape = (P,)
        pop_shape = tuple(int(v) for v in pop_shape)
        P = int(np.prod(pop_shape))
        if x_flat.size != C * P:
            raise ValueError(f"x has length {x_flat.size}, expected C*P={C*P}.")

        # reshape to (C, nZ, nT, nA) or (C, P) degenerate
        X = x_flat.reshape((C,) + pop_shape, order="C")
        if len(pop_shape) == 1:
            # nothing to panel; just a single heatmap
            W = X.sum(axis=0)  # (P,)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            ax.plot(np.arange(P), W, lw=1)
            ax.set_title("Population weights (flattened)")
            fig.savefig(out_png, dpi=130); plt.close(fig); return

        # assume (nZ, nT, nA) order
        nZ, nT, nA = pop_shape
        W = X.sum(axis=0) # (nZ, nT, nA)
        vmax = float(np.max(np.log10(W))) if np.isfinite(W).any() else 1.0

        fig, axes = plt.subplots(1, nA, figsize=(3.0*nA, 3.4), squeeze=False)
        axes = axes[0]
        for a in range(nA):
            ax = axes[a]
            im = ax.imshow(np.log10(W[:, :, a]), origin="lower", aspect="auto",
                cmap=moncmapr, norm=Normalize(vmin=0.0, vmax=vmax))
            ax.set_title(f"α index {a}")
            ax.set_xlabel("Age index")
            if a == 0:
                ax.set_ylabel("Metal index")
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.02)
        fig.savefig(out_png, dpi=130); plt.close(fig)

    logger.log(f"[JacobiDiag] wrote {out_png}")

# ---------- helpers -----------------------------------------------------------

def _minutes_since_start(ts):
    if ts.size == 0:
        return ts
    t0 = float(ts[0])
    return (ts - t0) / 60.0

def _find_latest_sidecar(main_path: str) -> str | None:
    """
    Look for sidecars created by the new FitTracker:
      <main>.fit.<pid>.<ts>.h5
    Return the newest one by mtime, or None if not found.
    """
    pat = f"{os.fspath(Path(main_path))}.fit.*.h5"
    cand = glob.glob(pat)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p))
    return cand[-1]

def _ewma(x, alpha=0.2):
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=float)
    s = 0.0
    for i, v in enumerate(np.asarray(x, float)):
        s = alpha * v + (1.0 - alpha) * (s if i else v)
        y[i] = s
    return y

def _load_metrics_new(side_f: h5py.File):
    """
    New tracker: histories live as flat datasets. Time stamps are not
    guaranteed; we synthesize a monotonic axis using indices (minutes).
    """
    gfit = side_f["/Fit"]
    epoch = side_f["/Fit/epoch_hist"][...] if "/Fit/epoch_hist" in side_f else np.array([], int)
    rmse  = side_f["/Fit/rmse_hist"][...]  if "/Fit/rmse_hist"  in side_f else np.array([], float)

    # monotone "time" in minutes (index-based)
    if epoch.size:
        time_sec = np.linspace(0.0, float(epoch.size), epoch.size, dtype=float)
    else:
        time_sec = np.array([], float)

    train_rmse_ewma = _ewma(rmse, alpha=0.2) if rmse.size else np.array([], float)

    # placeholders for fields the old dashboard expects
    val_rmse  = np.array([], float)
    dx_rel    = np.array([], float)
    nnz       = np.array([], float)
    l1        = np.array([], float)
    l2        = np.array([], float)
    me_L1     = np.array([], float)
    me_Linf   = np.array([], float)
    return (epoch.astype(int), time_sec, train_rmse_ewma, val_rmse,
            dx_rel, nnz, l1, l2, me_L1, me_Linf)

def _load_metrics_old(main_f: h5py.File):
    """
    Legacy tracker: metrics group with multiple 1D series.
    """
    if "/Fit/metrics" not in main_f:
        z = np.array([], float); zi = np.array([], int)
        return (zi, z, z, z, z, z, z, z, z, z)
    m = main_f["/Fit/metrics"]
    g = lambda n: m[n][...] if n in m else np.array([], float)
    return (g("epoch").astype(int), g("time_sec"), g("train_rmse_ewma"),
            g("val_rmse"), g("delta_x_rel"), g("nnz"), g("l1_norm"),
            g("l2_norm"), g("mass_err_L1"), g("mass_err_Linf"))

def _choose_sample_spaxels(main_f: h5py.File, n: int) -> np.ndarray:
    # Try new/old sample lists, else pick evenly spaced indices
    if "/Fit/sample/spaxels" in main_f:
        v = np.asarray(main_f["/Fit/sample/spaxels"][...], int)
        if v.size:
            return v[:min(n, v.size)]
    S = int(main_f["/DataCube"].shape[0])
    if n >= S:
        return np.arange(S, dtype=int)
    return np.linspace(0, S - 1, n, dtype=int)

def _read_latest_x(side_f: h5py.File | None, main_f: h5py.File) -> np.ndarray:
    """
    Preference order (sidecar first):
      1) /Fit/x_best
      2) /Fit/x_last
      3) /Fit/x_epoch_last
      4) last row of /Fit/x_snapshots  (ring/snapshots)
      5) last row of /Fit/x_hist       (legacy ring)
    Fallbacks (main file):
      6) /Fit/x_latest
      7) /X_global
    """
    def _row_or_vec(dset) -> np.ndarray:
        arr = np.asarray(dset[...], float)
        # accept 1-D (N,) or 2-D (T,N); if 2-D, take last row
        if arr.ndim == 2 and arr.shape[0] > 0:
            return arr[-1, :].astype(float, copy=False)
        return arr.ravel().astype(float, copy=False)

    # --- Sidecar first (if present)
    if side_f is not None:
        if "/Fit/x_best" in side_f:
            print("[Dashboard] Using /Fit/x_best from sidecar.")
            return _row_or_vec(side_f["/Fit/x_best"])

        if "/Fit/x_last" in side_f:
            print("[Dashboard] Using /Fit/x_last from sidecar.")
            return _row_or_vec(side_f["/Fit/x_last"])

        if "/Fit/x_epoch_last" in side_f:
            print("[Dashboard] Using /Fit/x_epoch_last from sidecar.")
            return _row_or_vec(side_f["/Fit/x_epoch_last"])

        if "/Fit/x_snapshots" in side_f and side_f["/Fit/x_snapshots"].shape[0] > 0:
            print("[Dashboard] Using last row of /Fit/x_snapshots from sidecar.")
            return _row_or_vec(side_f["/Fit/x_snapshots"])

        if "/Fit/x_hist" in side_f and side_f["/Fit/x_hist"].shape[0] > 0: # legacy
            print("[Dashboard] Using last row of /Fit/x_hist from sidecar.")
            return _row_or_vec(side_f["/Fit/x_hist"])

    # --- Main file fallbacks (legacy)
    if "/Fit/x_latest" in main_f:
        print("[Dashboard] Using /Fit/x_latest from main file.")
        return _row_or_vec(main_f["/Fit/x_latest"])

    if "/X_global" in main_f:
        print("[Dashboard] Using /X_global from main file.")
        return _row_or_vec(main_f["/X_global"])

    raise RuntimeError("No solution vector found in sidecar or main file.")

def _read_orbit_weights(side_f: h5py.File | None, main_f: h5py.File) -> np.ndarray | None:
    if side_f is not None and "/CompWeights" in side_f:
        return np.asarray(side_f["/CompWeights"][...], float)
    if "/Fit/orbit_weights" in main_f:
        return np.asarray(main_f["/Fit/orbit_weights"][...], float)
    if "/CompWeights" in main_f:
        return np.asarray(main_f["/CompWeights"][...], float)
    return None

# ---------- main API: dashboard ------------------------------------------------

def render_dashboard(h5_path: str, out_png: str,
                     n_spaxels: int = 8, downsample: int = 512,
                     figsize=(14, 10), sidecar: str | None = None):
    """
    Render the live dashboard (new sidecar or legacy). Uses HyperCubeReader with
    apply_mask=True so A and y match the solver exactly.
    """
    sidecar = sidecar or _find_latest_sidecar(h5_path)

    print(f"[Dashboard] Rendering {out_png} from {h5_path} "+
          (f"and sidecar {sidecar}" if sidecar else "(legacy)"))

    with open_h5(h5_path, role="reader", swmr=True) as F:
        # ---- metrics + solution vector ---------------------------------------
        if sidecar and os.path.exists(sidecar):
            with open_h5(sidecar, role="reader", swmr=True) as G:
                epoch, time_sec, tr_ewma, val_rmse, dx_rel, nnz, l1, l2, me1, meinf = _load_metrics_new(G)
                x = _read_latest_x(G, F)
                ow = _read_orbit_weights(G, F)
        else:
            epoch, time_sec, tr_ewma, val_rmse, dx_rel, nnz, l1, l2, me1, meinf = _load_metrics_old(F)
            x = _read_latest_x(None, F)
            ow = _read_orbit_weights(None, F)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # ---- (0,0) Convergence ------------------------------------------------
        ax = axes[0, 0]
        tmin = _minutes_since_start(time_sec) if time_sec.size else np.arange(tr_ewma.size, dtype=float)
        if tr_ewma.size:
            ax.plot(tmin, tr_ewma, label="train_rmse_ewma")
        if val_rmse.size:
            n = min(val_rmse.size, tmin.size if tmin.size else val_rmse.size)
            x_ax = tmin[:n] if tmin.size else np.arange(n, dtype=float)
            ax.plot(x_ax, val_rmse[:n], marker="o", linestyle="none", label="val_rmse")
        ax.set_xlabel("minutes since start" if time_sec.size else "updates")
        ax.set_ylabel("RMSE")
        ax.set_title("Convergence")
        if tr_ewma.size or val_rmse.size:
            ax.legend(loc="best")

        # ---- (0,1) Params & mass errors (plot only if present) ---------------
        ax = axes[0, 1]
        def _plot_series(series: np.ndarray, label: str) -> bool:
            if series.size == 0:
                return False
            if time_sec.size:
                n = min(series.size, time_sec.size)
                ax.plot(_minutes_since_start(time_sec)[:n], series[:n], label=label)
            else:
                ax.plot(np.arange(series.size, dtype=float), series, label=label)
            return True

        any_line = False
        any_line |= _plot_series(l1,     "||x||₁")
        any_line |= _plot_series(l2,     "||x||₂")
        any_line |= _plot_series(nnz,    "nnz(x)")
        any_line |= _plot_series(dx_rel, "Δx_rel")
        any_line |= _plot_series(me1,    "mass_err_L1")
        any_line |= _plot_series(meinf,  "mass_err_L∞")
        ax.set_xlabel("minutes since start" if time_sec.size else "updates")
        ax.set_title("Params & mass errors")
        if any_line:
            ax.legend(loc="best")

        # ---- (1,0) Observed vs model (subset) with spectral mask -------------
        ax = axes[1, 0]
        reader = HyperCubeReader(h5_path, ReaderCfg(dtype_models="float32", apply_mask=True))
        try:
            C, P = reader.nComp, reader.nPop
            X_vec = np.asarray(x, float).ravel()  # length N=C*P
            spaxels = _choose_sample_spaxels(F, n_spaxels)
            for s in spaxels:
                A, y = reader.read_spaxel_plane(int(s))  # A:(N,L_eff), y:(L_eff,)
                L_eff = y.size

                if downsample and L_eff > downsample:
                    idx = np.linspace(0, L_eff - 1, downsample, dtype=int)
                    lam_plot = idx
                    y_plot = y[idx]
                    yhat_plot = (A[:, idx].T @ X_vec)
                else:
                    lam_plot = np.arange(L_eff, dtype=int)
                    y_plot = y
                    yhat_plot = (A.T @ X_vec)

                # mirror solver: only data finiteness defines the mask
                good = np.isfinite(y_plot)

                ax.plot(lam_plot[good], y_plot[good],    alpha=0.6,  lw=1.0, label=None)
                ax.plot(lam_plot[good], yhat_plot[good], alpha=0.95, lw=1.0, ls="--", label=None)
        finally:
            reader.close()
        ax.set_xlabel("pixel")
        ax.set_title("Observed vs model (subset)")

        # ---- (1,1) Component usage vs target (needs X reshaped C×P) ----------
        ax = axes[1, 1]
        # Use reader's C,P to reshape safely
        X_cp = np.asarray(x, float).ravel().reshape(C, P)
        usage = X_cp.sum(axis=1)
        ax.bar(np.arange(C), usage, label="usage", color='r')
        if ow is not None and ow.size == usage.size:
            ax.plot(np.arange(C), ow, marker="o", linestyle="--", label="target w_c", color='k')
            maxerr = float(np.max(np.abs(usage - ow))) if usage.size else np.nan
            ax.set_title(f"Component usage vs target (max|e|={maxerr:.3g})")
            ax.set_ylim(0.0, float(1.05 * np.max(ow)) if ow.size else 1.0)
        else:
            ax.set_title("Component usage (sum over P)")
            ax.set_ylim(0.0, float(1.05 * np.max(usage)) if usage.size else 1.0)
        ax.set_xlabel("component c")
        ax.legend(loc="best")

        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"[Dashboard] Wrote {out_png}  [{Path(sidecar).name if sidecar else 'legacy'}]")

def render_aperture_fits(h5_path: str, out_png: str,
                         apertures: list[int],
                         sidecar: str | None = None,
                         show_residual: bool = True,
                         title: str | None = None):
    """
    Full-resolution observed vs model (and residual) for specific spaxels.
    Uses the masked reader so A,y match the solver.
    """
    sidecar = sidecar or _find_latest_sidecar(h5_path)

    # pick x (same priority order as dashboard)
    with open_h5(h5_path, role="reader") as F:
        if sidecar and os.path.exists(sidecar):
            with open_h5(sidecar, role="reader", swmr=True) as G:
                x = _read_latest_x(G, F)
        else:
            x = _read_latest_x(None, F)
    X_vec = np.asarray(x, float).ravel()

    reader = HyperCubeReader(h5_path, ReaderCfg(dtype_models="float32", apply_mask=True))
    try:
        n = len(apertures)
        ncols = 2 if show_residual else 1
        figsize = (14, 3.5 * n) if ncols == 1 else (14, 5.5 * n)
        fig, axes = plt.subplots(n, ncols, figsize=figsize, squeeze=False)

        for row, s in enumerate(apertures):
            A, y = reader.read_spaxel_plane(int(s))  # masked A:(N,L_eff), y:(L_eff,)
            yhat = A.T @ X_vec
            lam = np.arange(y.size, dtype=int)

            ax0 = axes[row, 0]
            ax0.plot(lam, y,    lw=1.0, alpha=0.7, label="observed")
            ax0.plot(lam, yhat, lw=1.0, alpha=0.9, label="model")
            ax0.set_ylabel(f"spax {s}")
            ax0.legend(loc="best")
            ax0.set_xlabel("pixel")

            if show_residual:
                r = y - yhat
                ax1 = axes[row, 1]
                ax1.plot(lam, r, lw=0.9)
                ax1.axhline(0.0, ls="--", lw=0.8)
                ax1.set_ylabel("residual")
                ax1.set_xlabel("pixel")

        if title:
            fig.suptitle(title, y=0.995, fontsize=12)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}")
    finally:
        reader.close()

# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render live Kaczmarz fit dashboard (new sidecar or legacy) "
                    "and optional full-resolution per-aperture overlays."
    )
    ap.add_argument("h5_path", help="Main HDF5 path (contains /HyperCube and /DataCube)")
    ap.add_argument("--out", default="fit_live.png")
    ap.add_argument("--n-spaxels", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=512)
    ap.add_argument("--sidecar", default=None,
                    help="Optional explicit sidecar path; otherwise auto-detects")
    ap.add_argument("--apertures", default="",
                    help="Comma-separated spaxel indices for detailed full-L fits")
    ap.add_argument("--apertures-separate-dir", default="",
                help="If set (and --apertures provided), also write one PNG per spaxel to this directory.")
    args = ap.parse_args()

    # 1) Always write the main dashboard
    render_dashboard(args.h5_path, args.out, n_spaxels=args.n_spaxels,
        downsample=args.downsample, sidecar=args.sidecar)

    # 2) Optionally also write the per-aperture full-resolution fits (combined)
    if args.apertures.strip():
        aps = [int(t) for t in args.apertures.replace(" ", "").split(",") if t]
        base = Path(args.out)
        fits_out = str(base.with_name(base.stem + "_fits" + base.suffix))
        render_aperture_fits(args.h5_path, fits_out, apertures=aps,
                            sidecar=args.sidecar, show_residual=True,
                            title=f"Observed vs model (spaxels={aps})")

        # 3) If requested, write separate figures per spaxel
        if args.apertures_separate_dir.strip():
            render_aperture_fits_separate(
                args.h5_path,
                args.apertures_separate_dir,
                apertures=aps,
                sidecar=args.sidecar,
                show_residual=True,
                title_prefix="Observed vs model"
            )

if __name__ == "__main__":
    main()

# ipython -- /data/phys-gal-dynamics/phys2603/CubeFit/live_fit_dashboard.py /data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/NGC4365_12.h5 --out /data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/figures/fit_live.png --apertures 0,2000,1400,740 --apertures-separate-dir /data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/figures/apertures/