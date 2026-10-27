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
v1.3:   Reworked `plot_diagnostic_jsonl_dashboard` for the constrained
            solver's fixed orbit-shape and fitted global-amplitude formulation. 6
            August 2026
v1.4:   Added explicit `color` and `linestyle` keyword arguments to
            `_plot_finite` calls in `plot_diagnostic_jsonl_dashboard` to manually
            differentiate through `twinx` calls. 9 August 2026
v1.5:   Re-worked most panels to reflect updated diagnostics around KKT
            convergence, active-set size, and promotion eligibility in
            `plot_diagnostic_jsonl_dashboard`. 26 August 2026
v1.6:   Expanded and improved `plot_diagnostic_jsonl_dashboard` for richer
            diagnostics. 14 September 2026
v1.7:   Made `plot_diagnostic_jsonl_dashboard` able to accept multiple file
            paths and automatically merge run diagnostics, to support 
            `resume` runs. 23 September 2026
v1.8:   Updated `plot_diagnostic_jsonl_dashboard` to latest diagnostic outputs.
            29 September 2026
v1.9:   Reworked `plot_diagnostic_jsonl_dashboard` for joint-support
            exploration, replacing obsolete promotion and stall diagnostics. 30
            September 2026
v2.0:   Updated `plot_diagnostic_jsonl_dashboard` to reflect the new definition
            of orbital weights. 3 October 2026
v2.1:   Rebuilt `plot_diagnostic_jsonl_dashboard` for the exact constrained
            KKT active-set solver, replacing obsolete exploration and coverage
            diagnostics with KKT, promotion, gradient, support, constraint,
            and runtime health diagnostics. 25 October 2026
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import gridspec
import matplotlib.ticker as mticker

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

def _fmt_compact(x, _pos=None):
    """
    Compact numeric tick formatter.
    """
    if not np.isfinite(x):
        return ""
    if abs(x) < 1e-12:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.3g}".replace("e+0", "e+").replace("e-0", "e-")
    if abs(x - round(x)) < 1e-8:
        return f"{int(round(x))}"
    return f"{x:.3g}"

def _homogenise_ticks(ax, *, nbins: int = 4) -> None:
    """
    Apply consistent, compact y-axis tick formatting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose y ticks are formatted.
    nbins : int, optional
        Approximate maximum number of major y ticks.

    Returns
    -------
    None
        The axis is modified in place.

    Raises
    ------
    TypeError
        If ``nbins`` cannot be converted to an integer.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> _homogenise_ticks(ax)
    """
    nbins = int(nbins)

    if ax.get_yscale() == "log":
        ymin, ymax = ax.get_ylim()
        lo, hi = sorted((ymin, ymax))
        decades = np.log10(hi) - np.log10(lo)

        if lo > 0.0 and decades < 1.0:
            locator = mticker.MaxNLocator(nbins=nbins)
            ticks = locator.tick_values(lo, hi)
            ticks = ticks[(ticks >= lo) & (ticks <= hi)]

            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(False)

            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.yaxis.set_major_locator(
                mticker.LogLocator(base=10.0, numticks=nbins))
            ax.yaxis.set_major_formatter(
                mticker.LogFormatterMathtext(base=10.0))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        formatter.set_useOffset(True)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins))
        ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="y", which="major", labelsize=9)

# ------------------------------------------------------------------------------

def plot_diagnostic_jsonl_dashboard(jsonl_paths: str | list[str], *,
    max_points: int | None = 5000, save_path: str | None = None,
    show: bool = False, figsize: tuple[float, float] = (22.0, 10.5),
) -> list[dict]:
    """
    Plot diagnostics for the exact constrained KKT active-set solver.

    The dashboard combines ``kkt`` and ``kkt_promotion`` records by outer
    iteration. It is designed for the exact convex-QP solver in which
    full-space KKT violations are deterministically promoted into the working
    set and the complete constrained reduced problem is then re-solved.

    The panels show:

    1. Full-space KKT convergence relative to tolerance.
    2. Data, regularisation, constraint, and ridge gradient magnitudes.
    3. Number and fraction of unresolved inactive dual violations.
    4. Working and strictly positive support sizes.
    5. Promotion survival and zero-return behaviour.
    6. Fitted common component mass and hard-constraint stationarity.
    7. Per-iteration and cumulative runtime.
    8. Current solver state and final objective summary.

    Parameters
    ----------
    jsonl_paths : str or list of str
        One or more diagnostics JSONL paths. Multiple files are merged in the
        supplied order, allowing resumed runs to be plotted continuously.
    max_points : int or None, optional
        Maximum number of merged outer iterations to retain. ``None`` keeps
        all iterations.
    save_path : str or None, optional
        Output image path. If None, the figure is not written.
    show : bool, optional
        If True, display the figure.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    raw_records : list of dict
        Raw JSONL records successfully parsed from all supplied files.

    Raises
    ------
    FileNotFoundError
        If an input diagnostics file does not exist.
    ValueError
        If ``max_points`` is non-positive or no usable KKT records exist.
    OSError
        If an input file cannot be read or the output cannot be written.

    Examples
    --------
    >>> records = plot_diagnostic_jsonl_dashboard(
    ...     "diagnostics.jsonl", save_path="diagnostics_dashboard.png")
    """
    eps = 1e-300

    if max_points is not None and int(max_points) <= 0:
        raise ValueError("max_points must be positive or None.")

    def _finite_scalar(record: dict, *keys: str,
        default: float = np.nan) -> float:
        """Return the first finite scalar found under the supplied keys."""
        for key in keys:
            value = record.get(key)

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(value):
                return value

        return float(default)

    def _latest_finite(values: np.ndarray,
        default: float = np.nan) -> float:
        """Return the latest finite value in an array."""
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return float(default)

        return float(values[-1])

    def _series(records: list[dict], *keys: str,
        default: float = np.nan) -> np.ndarray:
        """Extract a scalar history from merged iteration records."""
        return np.asarray([
            _finite_scalar(record, *keys, default=default)
            for record in records
        ], dtype=np.float64)

    def _plot_finite(axis, x_values: np.ndarray, y_values: np.ndarray,
        label: str, *, positive_log: bool = False, **kwargs) -> None:
        """Plot only finite values, optionally on a logarithmic y-axis."""
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)
        valid = np.isfinite(x_values) & np.isfinite(y_values)

        if positive_log:
            valid &= y_values > 0.0

        if not np.any(valid):
            return

        if positive_log:
            axis.semilogy(x_values[valid],
                np.maximum(y_values[valid], eps), label=label, **kwargs)
        else:
            axis.plot(x_values[valid], y_values[valid], label=label,
                **kwargs)

    def _fmt_sci(value: float) -> str:
        """Return compact scientific-notation mathtext."""
        if not np.isfinite(value):
            return r"\mathrm{n/a}"

        if value == 0.0:
            return "0"

        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value/(10.0**exponent)

        return rf"{mantissa:.2f}\times10^{{{exponent}}}"

    paths = [jsonl_paths] if isinstance(jsonl_paths, str) else list(jsonl_paths)
    raw_records = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()

                if not text:
                    continue

                try:
                    record = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue

                if isinstance(record, dict):
                    raw_records.append(record)

    # ------------------------------------------------------------------
    # Merge records by outer iteration.
    #
    # KKT is the pre-promotion state. kkt_promotion describes the exact
    # support transition committed from that state. We deliberately do not
    # copy final_solution.x into the merged records because it can be huge.
    # ------------------------------------------------------------------
    merged_by_iter = {}
    setup_records = []
    final_record = {}

    for record in raw_records:
        kind = str(record.get("kind", ""))

        if kind == "mono_setup":
            setup_records.append(dict(record))
            continue

        if kind == "objective_summary":
            final_record = dict(record)
            continue

        if kind not in {"kkt", "kkt_promotion"}:
            continue

        try:
            iteration = int(record["iter"])
        except (KeyError, TypeError, ValueError):
            continue

        current = merged_by_iter.setdefault(iteration, {"iter": iteration})

        if kind == "kkt":
            for key, value in record.items():
                if key not in {"kind", "iter"}:
                    current[key] = value

            current["_has_kkt"] = True

        else:
            for key, value in record.items():
                if key not in {"kind", "iter"}:
                    current[f"promotion_{key}"] = value

            current["_has_promotion"] = True

    merged = [merged_by_iter[key] for key in sorted(merged_by_iter)]

    if max_points is not None:
        merged = merged[-int(max_points):]

    if not merged:
        raise ValueError("No KKT iteration diagnostics found.")

    setup = setup_records[-1] if setup_records else {}

    iterations = np.asarray(
        [int(record["iter"]) for record in merged], dtype=np.int64)

    # ------------------------------------------------------------------
    # KKT state.
    # ------------------------------------------------------------------
    tol = _series(merged, "kkt_tol", "tol")
    grad_active = _series(merged, "max_grad_active")
    grad_dual = _series(merged, "max_grad_dual")
    kkt_violation = _series(merged, "kkt_violation")

    kkt_ratio = np.divide(
        kkt_violation, tol, out=np.full_like(tol, np.nan), where=tol > 0.0)

    active_ratio = np.divide(
        np.abs(grad_active), tol, out=np.full_like(tol, np.nan),
        where=tol > 0.0)

    dual_ratio = np.divide(
        np.maximum(grad_dual, 0.0), tol, out=np.full_like(tol, np.nan),
        where=tol > 0.0)

    # ------------------------------------------------------------------
    # Gradient decomposition.
    # ------------------------------------------------------------------
    grad_data = _series(merged, "max_grad_data")
    grad_regularisation = _series(merged, "max_grad_regularisation")
    grad_constraint = _series(merged, "max_grad_constraint")
    grad_ridge = _series(merged, "max_grad_numerical_ridge")

    # ------------------------------------------------------------------
    # Support and dual-feasibility state.
    # ------------------------------------------------------------------
    n_active = _series(merged, "n_working_active")
    n_positive = _series(merged, "n_positive")
    n_zero_free = _series(merged, "n_zero_free")
    n_violators = _series(merged, "n_dual_violators")

    violator_fraction = np.divide(
        n_violators, n_zero_free, out=np.full_like(n_violators, np.nan),
        where=n_zero_free > 0.0)

    # ------------------------------------------------------------------
    # Exact KKT-promotion outcomes.
    # ------------------------------------------------------------------
    n_promoted = _series(merged, "promotion_n_promoted")
    n_dropped = _series(merged, "promotion_n_zero_dropped")
    n_active_after = _series(merged, "promotion_n_active_after")

    retained = n_promoted-n_dropped
    retention_fraction = np.divide(
        retained, n_promoted, out=np.full_like(retained, np.nan),
        where=n_promoted > 0.0)

    # ------------------------------------------------------------------
    # Constraint and amplitude state.
    # ------------------------------------------------------------------
    alpha = _series(merged, "alpha", "promotion_alpha")
    constraint_dot_lambda = _series(merged, "constraint_dot_lambda")
    ridge = _series(merged, "ridge")

    # ------------------------------------------------------------------
    # Runtime.
    #
    # t_sec is emitted on the KKT record, so its difference measures the
    # complete previous promotion/re-solve/ATAz cycle rather than merely
    # plotting overhead.
    # ------------------------------------------------------------------
    elapsed = _series(merged, "t_sec")
    iter_time = np.full_like(elapsed, np.nan)

    if elapsed.size > 1:
        iter_time[1:] = np.diff(elapsed)

    rolling_time = np.full_like(iter_time, np.nan)
    window = min(5, iter_time.size)

    for ii in range(window-1, iter_time.size):
        values = iter_time[ii-window+1:ii+1]
        values = values[np.isfinite(values)]

        if values.size:
            rolling_time[ii] = float(np.median(values))

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(2, 4, figure=fig)

    axes = {
        "kkt": fig.add_subplot(grid[0, 0]),
        "gradients": fig.add_subplot(grid[0, 1]),
        "violators": fig.add_subplot(grid[0, 2]),
        "support": fig.add_subplot(grid[0, 3]),
        "promotion": fig.add_subplot(grid[1, 0]),
        "constraint": fig.add_subplot(grid[1, 1]),
        "runtime": fig.add_subplot(grid[1, 2]),
        "status": fig.add_subplot(grid[1, 3]),
    }

    # ------------------------------------------------------------------
    # Panel 1: the actual convergence certificate.
    # ------------------------------------------------------------------
    axis = axes["kkt"]

    _plot_finite(axis, iterations, kkt_ratio, "Overall KKT / tol",
        positive_log=True, lw=1.7)
    _plot_finite(axis, iterations, active_ratio, "Active / tol",
        positive_log=True, lw=1.2)
    _plot_finite(axis, iterations, dual_ratio, "Inactive dual / tol",
        positive_log=True, lw=1.2)

    axis.axhline(1.0, color="black", linestyle="--", lw=1.0,
        label="Convergence boundary")
    axis.set_title("Full-space KKT convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Violation / tolerance")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 2: what currently dominates the gradient scale?
    # ------------------------------------------------------------------
    axis = axes["gradients"]

    _plot_finite(axis, iterations, grad_data, r"$|g_{\rm data}|_{\max}$",
        positive_log=True, lw=1.4)
    _plot_finite(axis, iterations, grad_regularisation,
        r"$|g_{\rm smooth}|_{\max}$", positive_log=True, lw=1.4)
    _plot_finite(axis, iterations, grad_constraint,
        r"$|g_{\rm constraint}|_{\max}$", positive_log=True, lw=1.4)
    _plot_finite(axis, iterations, grad_ridge, r"$|g_{\rm ridge}|_{\max}$",
        positive_log=True, lw=1.0, linestyle=":")

    axis.set_title("Gradient decomposition")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Maximum magnitude")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 3: unresolved full-space dual violations.
    # ------------------------------------------------------------------
    axis = axes["violators"]

    _plot_finite(axis, iterations, n_violators, "Violating columns", lw=1.7)
    axis.set_title("Unresolved dual violations")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)

    axis_fraction = axis.twinx()
    _plot_finite(axis_fraction, iterations, 100.0*violator_fraction,
        "Violating fraction", lw=1.2, linestyle="--")
    axis_fraction.set_ylabel("Zero/free columns [%]")
    axis_fraction.set_ylim(bottom=0.0)
    _homogenise_ticks(axis_fraction)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_fraction.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 4: support evolution.
    # ------------------------------------------------------------------
    axis = axes["support"]

    _plot_finite(axis, iterations, n_active, "Working support", lw=1.6)
    _plot_finite(axis, iterations, n_positive, "Positive coefficients",
        lw=1.4)

    setup_C = _finite_scalar(setup, "C")

    if np.isfinite(setup_C) and setup_C > 0.0:
        axis.axhline(setup_C, color="black", linestyle=":", lw=1.0,
            label=rf"$C={int(setup_C)}$ floor")

    axis.set_title("Active-set size")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 5: do promoted KKT violators survive the exact re-solve?
    # ------------------------------------------------------------------
    axis = axes["promotion"]

    _plot_finite(axis, iterations, n_promoted, "Promoted", lw=1.4)
    _plot_finite(axis, iterations, retained, "Retained after solve", lw=1.6)
    _plot_finite(axis, iterations, n_dropped, "Returned to zero", lw=1.2)

    axis.set_title("KKT-promotion outcome")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)

    axis_retention = axis.twinx()
    _plot_finite(axis_retention, iterations, 100.0*retention_fraction,
        "Retention", lw=1.1, linestyle="--")
    axis_retention.set_ylim(0.0, 105.0)
    axis_retention.set_ylabel("Retention [%]")
    _homogenise_ticks(axis_retention)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_retention.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 6: hard-constraint health and fitted common mass.
    # ------------------------------------------------------------------
    axis = axes["constraint"]

    _plot_finite(axis, iterations, alpha, r"Common mass $\alpha$", lw=1.6)
    axis.set_title("Hard-constraint health")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$\alpha$")
    _homogenise_ticks(axis)

    axis_constraint = axis.twinx()
    _plot_finite(axis_constraint, iterations,
        np.abs(constraint_dot_lambda),
        r"$|\mathbf{1}^{T}\lambda|$", positive_log=True, lw=1.3)

    axis_constraint.set_ylabel("Amplitude-KKT residual")
    _homogenise_ticks(axis_constraint)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_constraint.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 7: computational health.
    # ------------------------------------------------------------------
    axis = axes["runtime"]

    _plot_finite(axis, iterations, iter_time, "Iteration time", lw=1.0,
        alpha=0.55)
    _plot_finite(axis, iterations, rolling_time, "5-iteration median",
        lw=1.7)

    axis.set_title("Runtime health")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Seconds / iteration")
    _homogenise_ticks(axis)

    axis_elapsed = axis.twinx()
    _plot_finite(axis_elapsed, iterations, elapsed/3600.0, "Elapsed",
        lw=1.0, linestyle=":")
    axis_elapsed.set_ylabel("Elapsed [h]")
    _homogenise_ticks(axis_elapsed)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_elapsed.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 8: current solver state.
    # ------------------------------------------------------------------
    axis = axes["status"]
    axis.set_axis_off()

    latest_kkt = _latest_finite(kkt_ratio)
    latest_active = _latest_finite(active_ratio)
    latest_dual = _latest_finite(dual_ratio)
    latest_violators = _latest_finite(n_violators)
    latest_support = _latest_finite(n_active)
    latest_positive = _latest_finite(n_positive)
    latest_alpha = _latest_finite(alpha)
    latest_ridge = _latest_finite(ridge)
    latest_time = _latest_finite(elapsed)
    latest_retention = _latest_finite(retention_fraction)

    latest_record = merged[-1]
    best_col = _finite_scalar(latest_record, "best_dual_col")
    best_orbit = _finite_scalar(latest_record, "best_dual_orbit")
    best_population = _finite_scalar(latest_record, "best_dual_population")
    best_value = _finite_scalar(latest_record, "best_dual_value")

    converged = np.isfinite(latest_kkt) and latest_kkt <= 1.0

    if final_record:
        stop_converged = bool(final_record.get("stop_converged", False))
        stop_reason = str(final_record.get("stop_reason", "unknown"))

        if stop_converged:
            phase = "CONVERGED"
        else:
            phase = f"STOPPED: {stop_reason.replace('_', ' ')}"
    elif converged:
        phase = "KKT SATISFIED"
    else:
        phase = "OPTIMIZING"

    lines = [
        rf"$\bf{{{phase}}}$",
        "",
        f"Iteration: {int(iterations[-1])}",
        f"KKT / tol: {latest_kkt:.3g}"
        if np.isfinite(latest_kkt) else "KKT / tol: n/a",
        f"Active / tol: {latest_active:.3g}"
        if np.isfinite(latest_active) else "Active / tol: n/a",
        f"Dual / tol: {latest_dual:.3g}"
        if np.isfinite(latest_dual) else "Dual / tol: n/a",
        f"Dual violators: {int(latest_violators)}"
        if np.isfinite(latest_violators) else "Dual violators: n/a",
        f"Support: {int(latest_support)} working / "
        f"{int(latest_positive)} positive"
        if np.isfinite(latest_support) and np.isfinite(latest_positive)
        else "Support: n/a",
    ]

    if np.isfinite(setup_C) and setup_C > 0.0 and np.isfinite(latest_positive):
        lines.append(
            f"Positive / component: {latest_positive/setup_C:.2f}")

    if np.isfinite(latest_retention):
        lines.append(f"Last promotion retention: "
            f"{100.0*latest_retention:.1f}%")

    if np.isfinite(best_col) and best_col >= 0.0:
        lines.extend([
            "",
            f"Worst inactive: column {int(best_col)}",
            f"orbit={int(best_orbit)}, population={int(best_population)}",
            f"reduced gradient={best_value:.3e}",
        ])

    if np.isfinite(latest_alpha):
        lines.append(rf"$\alpha={latest_alpha:.6g}$")

    if np.isfinite(latest_ridge):
        lines.append(f"ridge={latest_ridge:.3e}")

    if np.isfinite(latest_time):
        lines.append(f"elapsed={latest_time/3600.0:.2f} h")

    final_data = _finite_scalar(final_record, "data_objective")
    final_reg = _finite_scalar(
        final_record, "scientific_regularisation_objective")
    final_scientific = _finite_scalar(final_record, "scientific_objective")
    final_ridge_obj = _finite_scalar(final_record,
        "numerical_ridge_objective")
    final_total = _finite_scalar(final_record, "total_objective")
    final_orbit_linf = _finite_scalar(final_record, "orbit_resid_linf")

    if final_record:
        lines.extend(["", "Final objective"])

        if np.isfinite(final_data):
            lines.append(f"data: {final_data:.4e}")
        if np.isfinite(final_reg):
            lines.append(f"smoothness: {final_reg:.4e}")
        if np.isfinite(final_scientific):
            lines.append(f"scientific: {final_scientific:.4e}")
        if np.isfinite(final_ridge_obj):
            lines.append(f"ridge: {final_ridge_obj:.4e}")
        if np.isfinite(final_total):
            lines.append(f"total: {final_total:.4e}")
        if np.isfinite(final_orbit_linf):
            lines.append(f"orbit Linf: {final_orbit_linf:.3e}")

    axis.text(0.04, 0.97, "\n".join(lines), transform=axis.transAxes,
        ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.90))

    axis.set_title("Current solver state")

    # ------------------------------------------------------------------
    # Figure-level summary.
    # ------------------------------------------------------------------
    setup_P = _finite_scalar(setup, "P")
    setup_reg = _finite_scalar(setup, "regularisation_scale")
    setup_smooth = _finite_scalar(setup, "effective_smoothness_scale")
    setup_batch = _finite_scalar(setup, "kkt_promote_batch")
    setup_ridge = _finite_scalar(setup, "numerical_ridge")

    title_parts = [
        "CubeFit exact constrained active-set diagnostics",
        f"iterations={iterations[0]}-{iterations[-1]}",
    ]

    if np.isfinite(setup_C) and np.isfinite(setup_P):
        title_parts.append(f"C={int(setup_C)}, P={int(setup_P)}")

    if np.isfinite(setup_reg):
        title_parts.append(
            rf"$\eta={_fmt_sci(setup_reg)}$")

    if np.isfinite(setup_batch):
        title_parts.append(f"KKT batch={int(setup_batch)}")

    if np.isfinite(latest_kkt):
        title_parts.append(
            rf"$\mathrm{{KKT}}/\mathrm{{tol}}={_fmt_sci(latest_kkt)}$")

    if np.isfinite(latest_violators):
        title_parts.append(f"violators={int(latest_violators)}")

    if np.isfinite(latest_positive):
        title_parts.append(f"positive={int(latest_positive)}")

    if np.isfinite(latest_time):
        title_parts.append(f"elapsed={latest_time/3600.0:.2f} h")

    if final_record:
        stop_reason = str(final_record.get("stop_reason", "unknown"))
        title_parts.append(f"stop={stop_reason}")

    # Keep x-axis formatting consistent across every time-history panel.
    for name, axis in axes.items():
        if name == "status":
            continue

        axis.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, integer=True))
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(_fmt_compact))
        axis.tick_params(axis="x", which="major", labelsize=9)
        axis.xaxis.labelpad = 2
        axis.yaxis.labelpad = 2

    # Also format the secondary y axes created by twinx().
    for axis in fig.axes:
        if axis not in axes.values():
            axis.tick_params(axis="y", which="major", labelsize=8)
            axis.yaxis.labelpad = 2

    fig.suptitle(" | ".join(title_parts), fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.075, top=0.925,
        hspace=0.30, wspace=0.38)

    if save_path:
        fig.savefig(save_path, dpi=175, bbox_inches="tight",
            pad_inches=0.04)

    if show:
        plt.show()

    plt.close(fig)

    return raw_records

# ------------------------------------------------------------------------------
