# -*- coding: utf-8 -*-
r"""
    plotting.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Diagnostic and summary plotting for CubeFit results: spectra, white-light
    images, residual maps, and convergence history.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Spectrum/white-light/residual plotting. 2025
v1.1:   Integrated convergence and comparison plots. 2025
v1.2:   Added `plot_diagnostic_jsonl_dashboard` for streaming solver diagnostics.
            4 August 2026
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import gridspec

from CubeFit.logger import get_logger
from dynamics.IFU.Constants import UnitStr

UTS = UnitStr()
logger = get_logger()

# ------------------------------------------------------------------------------

def plot_aperture_fit(
    y_obs: np.ndarray,
    y_model: np.ndarray,
    obs_pix: np.ndarray,
    aperture_index: int | str = 0,
    mask: np.ndarray | None = None,
    show_residual: bool = True,
    wavelength_str: str | None = None,
) -> None:
    """
    Plot observed and model spectra (with optional mask), handling both single
    and stacked (multiple) spectra.

    Notes
    -----
    - If inputs are stacked (len(y_obs) is a multiple of len(obs_pix)),
      spectra are drawn as separate curves with NaN breaks (no cross-aperture
      line joins).
    - `mask` should match the shape of the stacked data (i.e., tiled).
    - Residuals are shown on a separate panel if requested.

    Parameters
    ----------
    y_obs : ndarray
        Observed flux, shape (nLSpec,) or (N_stack * nLSpec,).
    y_model : ndarray
        Model flux, same shape as y_obs.
    obs_pix : ndarray
        Wavelength array, shape (nLSpec,).
    aperture_index : int | str
        Title label (e.g., aperture index or "stacked").
    mask : ndarray or None
        Boolean mask for good pixels. If stacking, tile to (N_stack * nLSpec,).
    show_residual : bool
        Whether to include residuals panel.
    wavelength_str : str or None
        X-axis label. Defaults to "Wavelength [$\\AA$]".
    """
    if wavelength_str is None:
        wavelength_str = r"Wavelength [$\AA$]"

    n_spec = obs_pix.size
    n_total = y_obs.size
    is_stacked = (n_total % n_spec == 0) and (n_total > n_spec)

    print(is_stacked, n_total, n_spec)

    if show_residual:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        fig, ax0 = plt.subplots(figsize=(10, 4))

    if is_stacked:
        N_stack = n_total // n_spec
        y_obs_2d   = y_obs.reshape(N_stack, n_spec)
        y_model_2d = y_model.reshape(N_stack, n_spec)

        if mask is not None and mask.size == n_total:
            mask_2d = mask.reshape(N_stack, n_spec)
        else:
            mask_2d = None

        # Build segments for LineCollection: one segment per spectrum
        # Observed
        obs_segs = []
        mod_segs = []
        for i in range(N_stack):
            if mask_2d is not None:
                mg = mask_2d[i]
                x_i = obs_pix[mg]
                yo  = y_obs_2d[i, mg]
                ym  = y_model_2d[i, mg]
            else:
                x_i = obs_pix
                yo  = y_obs_2d[i]
                ym  = y_model_2d[i]
            if x_i.size >= 2:  # need at least two points to draw a line
                obs_segs.append(np.column_stack([x_i, yo]))
                mod_segs.append(np.column_stack([x_i, ym]))

        # Draw without cross-spectrum joins
        if obs_segs:
            lc_obs = LineCollection(obs_segs, linewidths=1.2, alpha=0.7, label="Observed")
            ax0.add_collection(lc_obs)
        if mod_segs:
            lc_mod = LineCollection(mod_segs, linewidths=1.2, alpha=0.7, label="Model")
            ax0.add_collection(lc_mod)

        # Axis limits
        x_min, x_max = obs_pix[0], obs_pix[-1]
        x_pad = 0.01 * (x_max - x_min)
        ax0.set_xlim(x_min - x_pad, x_max + x_pad)

        # y-limits from the data we plotted
        def _stack_minmax(segs):
            if not segs: return (0.0, 1.0)
            y_all = np.concatenate([s[:, 1] for s in segs])
            return float(np.nanmin(y_all)), float(np.nanmax(y_all))

        y_min_obs, y_max_obs = _stack_minmax(obs_segs)
        y_min_mod, y_max_mod = _stack_minmax(mod_segs)
        y_min = min(y_min_obs, y_min_mod)
        y_max = max(y_max_obs, y_max_mod)
        yr = y_max - y_min
        y_pad = 0.05 * yr if yr > 0 else 0.05 * abs(y_min if y_min else 1.0)
        ax0.set_ylim(y_min - y_pad, y_max + y_pad)

        # Residual panel: plot each stacked residual as gray line segments too
        if show_residual:
            resid_segs = []
            for i in range(N_stack):
                if mask_2d is not None:
                    mg = mask_2d[i]
                    x_i = obs_pix[mg]
                    r_i = (y_obs_2d[i, mg] - y_model_2d[i, mg])
                else:
                    x_i = obs_pix
                    r_i = (y_obs_2d[i] - y_model_2d[i])
                if x_i.size >= 2:
                    resid_segs.append(np.column_stack([x_i, r_i]))
            if resid_segs:
                lc_res = LineCollection(resid_segs, linewidths=1.0, alpha=0.6, color="gray")
                ax1.add_collection(lc_res)
                ax1.set_xlim(x_min - x_pad, x_max + x_pad)
                r_all = np.concatenate([s[:, 1] for s in resid_segs]) if resid_segs else np.array([0.0])
                rmin, rmax = float(np.nanmin(r_all)), float(np.nanmax(r_all))
                rr = rmax - rmin
                rpad = 0.05 * rr if rr > 0 else 0.05 * abs(rmin if rmin else 1.0)
                ax1.set_ylim(rmin - rpad, rmax + rpad)
                ax1.set_ylabel("Residual")
                ax1.set_xlabel(wavelength_str)

    else:
        if mask is not None:
            mask = mask.astype(bool, copy=False)
            x_good      = obs_pix[mask]
            y_obs_good  = y_obs[mask]
            y_mod_good  = y_model[mask]
            x_masked    = obs_pix[~mask]
            y_obs_masked = y_obs[~mask]
            y_mod_masked = y_model[~mask]
        else:
            x_good = obs_pix
            y_obs_good = y_obs
            y_mod_good = y_model
            x_masked = y_obs_masked = y_mod_masked = None

        ax0.plot(x_good, y_obs_good, label="Observed", alpha=0.7)
        ax0.plot(x_good, y_mod_good, label="Model", alpha=0.7)

        x_pad = 0.01 * (x_good[-1] - x_good[0])
        y_min = min(np.min(y_obs_good), np.min(y_mod_good))
        y_max = max(np.max(y_obs_good), np.max(y_mod_good))
        y_rng = y_max - y_min
        y_pad = 0.05 * y_rng if y_rng > 0 else 0.05 * abs(y_min if y_min else 1.0)
        ax0.set_xlim(x_good[0] - x_pad, x_good[-1] + x_pad)
        ax0.set_ylim(y_min - y_pad, y_max + y_pad)

        # Masked dots, clipped to y-limits
        if mask is not None and np.any(~mask):
            ylim = ax0.get_ylim()
            in_y_obs = (y_obs_masked > ylim[0]) & (y_obs_masked < ylim[1])
            in_y_mod = (y_mod_masked > ylim[0]) & (y_mod_masked < ylim[1])
            if np.any(in_y_obs):
                ax0.plot(x_masked[in_y_obs], y_obs_masked[in_y_obs],
                         '.', color="gray", alpha=0.3, markersize=6,
                         label="Masked (data)", zorder=1)
            if np.any(in_y_mod):
                ax0.plot(x_masked[in_y_mod], y_mod_masked[in_y_mod],
                         '.', color="orange", alpha=0.2, markersize=6,
                         label="Masked (model)", zorder=1)

        if show_residual:
            resid = y_obs_good - y_mod_good
            r_min, r_max = np.min(resid), np.max(resid)
            r_rng = r_max - r_min
            r_pad = 0.05 * r_rng if r_rng > 0 else 0.05 * abs(r_min if r_min else 1.0)
            ax1.plot(x_good, resid, color="gray")
            ax1.set_xlim(x_good[0] - x_pad, x_good[-1] + x_pad)
            ax1.set_ylim(r_min - r_pad, r_max + r_pad)
            ax1.set_ylabel("Residual")
            ax1.set_xlabel(wavelength_str)
            if mask is not None and np.any(~mask):
                ylim = ax1.get_ylim()
                resid_masked = (y_obs - y_model)[~mask]
                x_masked_resid = obs_pix[~mask]
                in_ylim = (resid_masked > ylim[0]) & (resid_masked < ylim[1])
                if np.any(in_ylim):
                    ax1.plot(x_masked_resid[in_ylim], resid_masked[in_ylim],
                             '.', color="gray", alpha=0.15, markersize=4,
                             zorder=1, clip_on=False)

    ax0.set_title(f"Aperture {aperture_index}")
    ax0.set_ylabel("Flux")
    ax0.legend()

# ------------------------------------------------------------------------------

def plot_white_light_images(
    data_cube: np.ndarray,
    model_cube: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Summation along spectral axis → "white-light" images.

    Parameters
    ----------
    data_cube  : ndarray, (ny, nx, nPix)
    model_cube : ndarray, same shape
    save_path  : str | None
        If provided, PNG is written to this path; otherwise shown on screen.
    """
    wl_data   = data_cube.sum(-1)
    wl_model  = model_cube.sum(-1)
    wl_resid  = wl_data - wl_model

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        (wl_data, wl_model, wl_resid),
        ("Data (white-light)", "Model (white-light)", "Residual"),
    ):
        im = ax.imshow(img, origin="lower", cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# ------------------------------------------------------------------------------

def plot_model_decomposition(
    y_obs: np.ndarray,          # Observed spectrum (full, not masked)
    obs_pix: np.ndarray,        # Wavelength array
    mask: np.ndarray,           # Boolean mask array (same shape)
    x_ref: np.ndarray,          # Fit solution vector
    A0: np.ndarray,             # Full design matrix (all columns)
    nComp: int, nPop: int,      # Number of components and populations
    C: np.ndarray | None = None,# Continuum matrix (nLSpec, nContinuum) or None
    aperture_index: int | str = "stacked",
    wavelength_str: str | None = None,
    show_residual: bool = True,
) -> None:
    """
    Plot observed, main model, velocity shift, continuum, and residual.

    This is primarily for *reference NNLS* diagnostics (single or stacked
    spectrum). It assumes your A0 column order:
        [templates | velshift | continuum]

    Parameters
    ----------
    y_obs, obs_pix, mask : arrays
        Full (unmasked) spectrum, wavelengths, and boolean mask.
    x_ref : ndarray
        Full solution vector from NNLS (including velshift/continuum if present).
    A0 : ndarray
        Design matrix, shape (nLSpec_total, nCols).
    nComp, nPop : int
        Numbers of components and populations.
    C : ndarray or None
        Continuum basis used in the fit. If None, continuum is omitted.
    """
    if wavelength_str is None:
        wavelength_str = r"Wavelength [$\AA$]"

    nTemplates = nComp * nPop
    # Block order: [templates | velshift | continuum]
    w_templates = x_ref[:nTemplates]
    w_velshift  = x_ref[nTemplates:2 * nTemplates]
    has_cont    = (C is not None and C.shape[1] > 0)
    w_cont      = x_ref[2 * nTemplates:] if has_cont else np.array([])

    model_main   = A0[:, :nTemplates] @ w_templates
    model_vshift = A0[:, nTemplates:2 * nTemplates] @ w_velshift
    model_cont   = (A0[:, 2 * nTemplates:] @ w_cont) if has_cont else 0.0
    model_total  = model_main + model_vshift + model_cont

    # Figure
    if show_residual:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
    else:
        plt.figure(figsize=(10, 5))
        ax0 = plt.gca()

    # Plot only unmasked in solid lines
    mask = mask.astype(bool, copy=False)
    x_good = obs_pix[mask]
    ax0.plot(x_good, y_obs[mask], label="Observed", color="k", lw=1)
    ax0.plot(x_good, model_main[mask], label="Model (no vshift/cont)",
             color="b", lw=1)
    ax0.plot(x_good, (model_main + model_vshift)[mask],
             label="Model + vshift", color="r", lw=1)
    if has_cont:
        ax0.plot(x_good, model_total[mask],
                 label="Full model (+ continuum)", color="g", lw=1)

    # Velocity/continuum trends across all pixels (dashed)
    ax0.plot(obs_pix, model_vshift, '--', color="orange", lw=1,
             label="Velocity shift (all pixels)", alpha=0.7, zorder=2)
    if has_cont:
        ax0.plot(obs_pix, model_cont, '--', color="purple", lw=1,
                 label="Continuum (all pixels)", alpha=0.7, zorder=3)

    # Limits
    x_pad = 0.01 * (x_good[-1] - x_good[0])
    ax0.set_xlim(x_good[0] - x_pad, x_good[-1] + x_pad)
    lines = [y_obs[mask], model_main[mask], (model_main + model_vshift)[mask]]
    if has_cont:
        lines.append(model_total[mask])
    y_min = np.min([np.min(l) for l in lines])
    y_max = np.max([np.max(l) for l in lines])
    y_rng = y_max - y_min
    y_pad = 0.05 * y_rng if y_rng > 0 else 0.05 * abs(y_min if y_min else 1.0)
    ax0.set_ylim(y_min - y_pad, y_max + y_pad)
    ylim = ax0.get_ylim()

    # Masked dots, clipped
    if np.any(~mask):
        x_masked = obs_pix[~mask]
        y_obs_masked = y_obs[~mask]
        y_model_masked = model_total[~mask] if has_cont else \
                         (model_main + model_vshift)[~mask]
        in_y_obs = (y_obs_masked > ylim[0]) & (y_obs_masked < ylim[1])
        in_y_mod = (y_model_masked > ylim[0]) & (y_model_masked < ylim[1])
        if np.any(in_y_obs):
            ax0.plot(x_masked[in_y_obs], y_obs_masked[in_y_obs],
                     '.', color="gray", alpha=0.3, markersize=6,
                     label="Masked (data)", zorder=1)
        if np.any(in_y_mod):
            ax0.plot(x_masked[in_y_mod], y_model_masked[in_y_mod],
                     '.', color="lime", alpha=0.2, markersize=6,
                     label="Masked (model)", zorder=1)

    ax0.set_title(f"Aperture {aperture_index}")
    ax0.set_ylabel("Flux")
    ax0.legend(fontsize=8, loc="best")

    # Residuals panel
    if show_residual:
        ax1 = plt.subplot(gs[1])
        resid = y_obs[mask] - model_total[mask]
        ax1.plot(x_good, resid, color="gray")
        r_min, r_max = np.min(resid), np.max(resid)
        r_rng = r_max - r_min
        r_pad = 0.05 * r_rng if r_rng > 0 else 0.05 * abs(r_min if r_min else 1.0)
        ax1.set_xlim(x_good[0] - x_pad, x_good[-1] + x_pad)
        ax1.set_ylim(r_min - r_pad, r_max + r_pad)
        ax1.set_ylabel("Residual")
        ax1.set_xlabel(wavelength_str)
        if np.any(~mask):
            resid_masked = (y_obs - model_total)[~mask]
            ylim_res = ax1.get_ylim()
            in_ylim = (resid_masked > ylim_res[0]) & (resid_masked < ylim_res[1])
            if np.any(in_ylim):
                ax1.plot(obs_pix[~mask][in_ylim], resid_masked[in_ylim],
                         '.', color="gray", alpha=0.15, markersize=4, zorder=1)
    else:
        ax0.set_xlabel(wavelength_str)

# ------------------------------------------------------------------------------

def plot_diagnostic_jsonl_dashboard(
    jsonl_path: str,
    *,
    max_points: int | None = 5000,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (16.0, 11.0),
):
    """
    Plot streaming diagnostics from a solver JSONL file.

    The function understands the current solver records emitted by
    `streamActiveSetNNLS(...)` / `solve_streaming_nnls(...)`, including:

    - iter / n_active / support / k_active
    - rmse / residual_norm
    - max_grad_total / max_grad_data / max_grad_orbit / max_grad_promotable
    - n_promoted / n_failed / n_dropped
    - orbit_mass / orbit_target / orbit_resid / orbit_ratio
    - orbit_nz / orbit_eff_support
    - norm_old / norm_new
    - orbit objective summary fields when present

    Parameters
    ----------
    jsonl_path
        Path to the diagnostics JSONL file.
    live
        If True, repeatedly rereads the file and refreshes the figure.
    refresh_sec
        Delay between refreshes in live mode.
    max_points
        Keep only the most recent N records. Set to None to use all records.
    save_path
        If provided, save the current figure to this path.
    show
        If True, call plt.show() in non-live mode and at the end of live mode.
    figsize
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : dict[str, matplotlib.axes.Axes]
        Dictionary of subplot handles.
    records : list[dict]
        Parsed JSONL records used for the final draw.
    """

    eps = 1e-300

    def _load_records() -> list[dict]:
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)

        if max_points is not None and len(records) > max_points:
            records = records[-max_points:]
        return records

    def _scalar(rec: dict, *keys: str, default=np.nan) -> float:
        for key in keys:
            if key in rec and rec[key] is not None:
                try:
                    return float(rec[key])
                except Exception:
                    continue
        return default

    def _vec(rec: dict, *keys: str) -> np.ndarray | None:
        for key in keys:
            if key in rec and rec[key] is not None:
                try:
                    arr = np.asarray(rec[key], dtype=np.float64).ravel()
                    if arr.size:
                        return arr
                except Exception:
                    pass
        return None

    def _extract(records: list[dict]) -> dict:
        iters = []
        active = []
        support = []
        mean_orbit_nz = []
        mean_eff_support = []
        rmse = []
        resid = []
        orbit_l1 = []
        orbit_linf = []
        norm_old = []
        norm_new = []
        max_grad_total = []
        max_grad_data = []
        max_grad_orbit = []
        max_grad_promo = []
        n_promoted = []
        n_failed = []
        n_dropped = []

        orbit_heat = []
        orbit_last_mass = None
        orbit_last_target = None
        orbit_last_resid = None
        orbit_heat_C = None

        for rec in records:
            it = _scalar(rec, "iter")
            if not np.isfinite(it):
                continue

            iters.append(int(it))
            active.append(_scalar(rec, "n_active", "active", "k_active"))
            support.append(_scalar(rec, "support", "support_n", "x_nonzero"))

            rmse.append(_scalar(rec, "rmse", "rmse_proxy"))
            resid.append(_scalar(rec, "residual_norm", "resid"))
            orbit_l1.append(
                _scalar(rec, "orbit_l1", "orbit_L1", "orbit_resid_l1")
            )
            orbit_linf.append(
                _scalar(rec, "orbit_linf", "orbit_Linf", "orbit_resid_linf")
            )
            norm_old.append(_scalar(rec, "norm_old"))
            norm_new.append(_scalar(rec, "norm_new"))
            max_grad_total.append(_scalar(rec, "max_grad_total"))
            max_grad_data.append(_scalar(rec, "max_grad_data"))
            max_grad_orbit.append(_scalar(rec, "max_grad_orbit"))
            max_grad_promo.append(
                _scalar(rec, "max_grad_promotable", "max_grad_promo")
            )
            n_promoted.append(_scalar(rec, "n_promoted", default=0.0))
            n_failed.append(_scalar(rec, "n_failed", default=0.0))
            n_dropped.append(_scalar(rec, "n_dropped", default=0.0))

            orbit_nz = _vec(rec, "orbit_nz")
            if orbit_nz is not None:
                mean_orbit_nz.append(float(np.nanmean(orbit_nz)))
            else:
                mean_orbit_nz.append(np.nan)

            orbit_eff = _vec(rec, "orbit_eff_support")
            if orbit_eff is not None:
                mean_eff_support.append(float(np.nanmean(orbit_eff)))
            else:
                mean_eff_support.append(np.nan)

            orbit_mass = _vec(rec, "orbit_mass")
            orbit_target = _vec(rec, "orbit_target")
            orbit_resid = _vec(rec, "orbit_resid")
            orbit_ratio = _vec(rec, "orbit_ratio")

            if orbit_mass is not None:
                if orbit_heat_C is None:
                    orbit_heat_C = int(orbit_mass.size)

                if orbit_mass.size == orbit_heat_C:
                    if (
                        orbit_ratio is None
                        and orbit_target is not None
                        and orbit_target.size == orbit_heat_C
                    ):
                        orbit_ratio = orbit_mass / np.maximum(orbit_target, eps)

                    if orbit_ratio is not None and orbit_ratio.size == orbit_heat_C:
                        orbit_heat.append(np.log10(np.maximum(orbit_ratio, eps)))
                        orbit_last_mass = orbit_mass.copy()
                        orbit_last_target = (
                            orbit_target.copy()
                            if orbit_target is not None
                            and orbit_target.size == orbit_heat_C
                            else None
                        )
                        if orbit_resid is not None and orbit_resid.size == orbit_heat_C:
                            orbit_last_resid = orbit_resid.copy()
                        elif orbit_last_target is not None:
                            orbit_last_resid = orbit_last_mass - orbit_last_target

        return {
            "iters": np.asarray(iters, dtype=np.int64),
            "active": np.asarray(active, dtype=np.float64),
            "support": np.asarray(support, dtype=np.float64),
            "mean_orbit_nz": np.asarray(mean_orbit_nz, dtype=np.float64),
            "mean_eff_support": np.asarray(mean_eff_support, dtype=np.float64),
            "rmse": np.asarray(rmse, dtype=np.float64),
            "resid": np.asarray(resid, dtype=np.float64),
            "orbit_l1": np.asarray(orbit_l1, dtype=np.float64),
            "orbit_linf": np.asarray(orbit_linf, dtype=np.float64),
            "norm_old": np.asarray(norm_old, dtype=np.float64),
            "norm_new": np.asarray(norm_new, dtype=np.float64),
            "max_grad_total": np.asarray(max_grad_total, dtype=np.float64),
            "max_grad_data": np.asarray(max_grad_data, dtype=np.float64),
            "max_grad_orbit": np.asarray(max_grad_orbit, dtype=np.float64),
            "max_grad_promo": np.asarray(max_grad_promo, dtype=np.float64),
            "n_promoted": np.asarray(n_promoted, dtype=np.float64),
            "n_failed": np.asarray(n_failed, dtype=np.float64),
            "n_dropped": np.asarray(n_dropped, dtype=np.float64),
            "orbit_heat": (
                np.asarray(orbit_heat, dtype=np.float64) if orbit_heat else None
            ),
            "orbit_last_mass": orbit_last_mass,
            "orbit_last_target": orbit_last_target,
            "orbit_last_resid": orbit_last_resid,
        }

    def _plot_series(ax, x, y, label, *, logy=False, **kwargs):
        xx = np.asarray(x, dtype=np.float64).ravel()
        yy = np.asarray(y, dtype=np.float64).ravel()
        m = np.isfinite(xx) & np.isfinite(yy)
        if not np.any(m):
            return

        xx = xx[m]
        yy = yy[m]

        if logy:
            yy = np.abs(yy)
            m2 = yy > eps
            if not np.any(m2):
                return
            ax.semilogy(xx[m2], np.maximum(yy[m2], eps), label=label, **kwargs)
        else:
            ax.plot(xx, yy, label=label, **kwargs)

    def _draw(data: dict, fig=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)
        else:
            fig.clf()

        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        ax20 = fig.add_subplot(gs[2, 0])
        ax21 = fig.add_subplot(gs[2, 1])

        it = data["iters"]
        if it.size == 0:
            for ax in (ax00, ax01, ax10, ax11, ax20, ax21):
                ax.set_axis_off()
            ax00.text(0.5, 0.5, "No diagnostics yet", ha="center", va="center")
            return fig, {
                "active": ax00,
                "norms": ax01,
                "gradients": ax10,
                "counts": ax11,
                "orbit_heat": ax20,
                "orbit_final": ax21,
            }

        _plot_series(ax00, it, data["active"], "Active", lw=1.5)
        _plot_series(ax00, it, data["support"], "Support", lw=1.5)
        _plot_series(
            ax00, it, data["mean_orbit_nz"], "Mean orbit nz", lw=1.2, alpha=0.85
        )
        _plot_series(
            ax00,
            it,
            data["mean_eff_support"],
            "Mean eff support",
            lw=1.2,
            alpha=0.85,
        )
        ax00.set_title("Active set / orbit occupation")
        ax00.set_xlabel("Iteration")
        ax00.set_ylabel("Count / support")
        ax00.legend(fontsize=8, ncol=2)

        _plot_series(ax01, it, data["rmse"], "RMSE", logy=True, lw=1.5)
        _plot_series(ax01, it, data["resid"], "Residual norm", logy=True, lw=1.5)
        _plot_series(ax01, it, data["orbit_l1"], "Orbit L1", logy=True, lw=1.2)
        _plot_series(
            ax01, it, data["orbit_linf"], "Orbit Linf", logy=True, lw=1.2
        )
        _plot_series(ax01, it, data["norm_old"], "Norm old", logy=True, lw=1.0)
        _plot_series(ax01, it, data["norm_new"], "Norm new", logy=True, lw=1.0)
        ax01.set_title("Fit norms")
        ax01.set_xlabel("Iteration")
        ax01.set_ylabel("Value")
        ax01.legend(fontsize=8, ncol=2)

        _plot_series(
            ax10, it, data["max_grad_total"], "Max grad total", logy=True, lw=1.5
        )
        _plot_series(
            ax10, it, data["max_grad_data"], "Max grad data", logy=True, lw=1.2
        )
        _plot_series(
            ax10, it, data["max_grad_orbit"], "Max grad orbit", logy=True, lw=1.2
        )
        _plot_series(
            ax10,
            it,
            data["max_grad_promo"],
            "Max grad promotable",
            logy=True,
            lw=1.0,
            alpha=0.85,
        )
        ax10.set_title("Promotion gradients")
        ax10.set_xlabel("Iteration")
        ax10.set_ylabel("Gradient")
        ax10.legend(fontsize=8, ncol=2)

        _plot_series(ax11, it, data["n_promoted"], "Promoted", lw=1.3)
        _plot_series(ax11, it, data["n_failed"], "Failed", lw=1.3)
        _plot_series(ax11, it, data["n_dropped"], "Dropped", lw=1.3)
        ax11.set_title("Batch bookkeeping")
        ax11.set_xlabel("Iteration")
        ax11.set_ylabel("Count")
        ax11.legend(fontsize=8)

        heat = data["orbit_heat"]
        if heat is not None and heat.size > 0:
            im = ax20.imshow(
                heat.T,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[float(it.min()), float(it.max()), 0.0, float(heat.shape[1])],
                cmap="viridis",
            )
            ax20.set_title(r"Orbit occupation heatmap: $\log_{10}(M/T)$")
            ax20.set_xlabel("Iteration")
            ax20.set_ylabel("Orbit")
            cbar = fig.colorbar(im, ax=ax20, fraction=0.046, pad=0.04)
            cbar.set_label(r"$\log_{10}(M/T)$")
        else:
            ax20.set_axis_off()
            ax20.text(0.5, 0.5, "No orbit ratio history", ha="center", va="center")

        mass = data["orbit_last_mass"]
        target = data["orbit_last_target"]
        resid_last = data["orbit_last_resid"]
        if mass is not None:
            idx = np.arange(mass.size, dtype=np.int64)
            w = 0.38
            ax21.bar(idx - w / 2, mass, width=w, label="Mass")
            if target is not None and target.size == mass.size:
                ax21.bar(idx + w / 2, target, width=w, label="Target")
            ax21.set_title("Final orbit masses")
            ax21.set_xlabel("Orbit")
            ax21.set_ylabel("Mass")
            ax21.legend(fontsize=8, loc="best")

            ax21r = ax21.twinx()
            if resid_last is not None and resid_last.size == mass.size:
                ax21r.plot(idx, resid_last, "k.-", lw=1.0, label="Residual")
                ax21r.axhline(0.0, color="k", lw=0.8, alpha=0.6)
                ax21r.set_ylabel("Residual")
                l1 = float(np.sum(np.abs(resid_last)))
                linf = float(np.max(np.abs(resid_last)))
                ax21.set_title(
                    f"Final orbit masses  |  L1={l1:.3e}, Linf={linf:.3e}"
                )
        else:
            ax21.set_axis_off()
            ax21.text(0.5, 0.5, "No orbit masses found", ha="center", va="center")

        fig.tight_layout()
        return fig, {
            "active": ax00,
            "norms": ax01,
            "gradients": ax10,
            "counts": ax11,
            "orbit_heat": ax20,
            "orbit_final": ax21,
        }

    records = _load_records()
    data = _extract(records)
    fig, axes = _draw(data, None)

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, axes, records

# ------------------------------------------------------------------------------
