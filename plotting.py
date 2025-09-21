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
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 1, height_ratios=[3, 1])
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
