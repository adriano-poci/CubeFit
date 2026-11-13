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
v2.2:   Moved `plot_chemodynamical_atlas` here;
        Show SFH densities instead of discrete bins in
            `plot_chemodynamical_atlas`;
        Added `plot_population_radial_profiles` to produce intrinsic weighted
            radial profiles of the SFH. 3 November 2026

"""

from __future__ import annotations

import json
import pathlib as plp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import colormaps, gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker

from CubeFit.logger import get_logger
from dynamics.IFU.Constants import UnitStr
from dynamics.IFU.Functions import Plot

UTS = UnitStr()
POT = Plot()
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
    show: bool = False, figsize: tuple[float, float] = (22.0, 14.0),
) -> list[dict]:
    """
    Plot diagnostics for the exact constrained KKT active-set solver.

    The dashboard combines ``kkt`` and ``kkt_promotion`` records by outer
    iteration. It is designed for the exact convex-QP solver in which
    full-space KKT violations are deterministically promoted into the working
    set and the complete constrained reduced problem is then re-solved.

    The panels show:

    1. Full-space KKT convergence relative to tolerance.
    2. KKT convergence rate and short-term progress.
    3. Data, regularisation, constraint, and ridge gradient magnitudes.
    4. Number and severity of unresolved inactive dual violations.
    5. Working and strictly positive support sizes.
    6. Fraction of the working set carrying positive coefficients.
    7. KKT-promotion, zero-drop, and net support-change behaviour.
    8. Equality-constraint feasibility when emitted by the solver.
    9. Common-amplitude and constraint-stationarity stability.
    10. Objective evolution when emitted per iteration.
    11. Per-iteration and cumulative runtime.
    12. Current solver state and final objective summary.

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
            axis.plot(x_values[valid], y_values[valid], label=label, **kwargs)

    def _rolling_median(values: np.ndarray, window: int = 5) -> np.ndarray:
        """Return a finite-value rolling median."""
        values = np.asarray(values, dtype=np.float64)
        output = np.full_like(values, np.nan)

        if values.size == 0:
            return output

        window = max(1, min(int(window), values.size))

        for ii in range(window-1, values.size):
            local = values[ii-window+1:ii+1]
            local = local[np.isfinite(local)]

            if local.size:
                output[ii] = float(np.median(local))

        return output

    def _unavailable(axis, text: str) -> None:
        """Mark a diagnostic panel as unavailable."""
        axis.text(0.5, 0.5, text, transform=axis.transAxes,
            ha="center", va="center", fontsize=9, color="0.35")
        axis.set_xticks([])
        axis.set_yticks([])

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
    # support transition committed from that state. final_solution.x is
    # deliberately excluded because it can be very large.
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

    kkt_ratio = np.divide(kkt_violation, tol,
        out=np.full_like(tol, np.nan), where=tol > 0.0)
    active_ratio = np.divide(np.abs(grad_active), tol,
        out=np.full_like(tol, np.nan), where=tol > 0.0)
    dual_ratio = np.divide(np.maximum(grad_dual, 0.0), tol,
        out=np.full_like(tol, np.nan), where=tol > 0.0)

    # ------------------------------------------------------------------
    # KKT progress.
    #
    # Positive values mean that KKT/tol decreased during the iteration.
    # Zero means no progress and negative values mean regression.
    # ------------------------------------------------------------------
    kkt_progress = np.full_like(kkt_ratio, np.nan)

    if kkt_ratio.size > 1:
        valid = (np.isfinite(kkt_ratio[:-1]) & np.isfinite(kkt_ratio[1:])
            & (kkt_ratio[:-1] > 0.0) & (kkt_ratio[1:] > 0.0))
        inds = np.flatnonzero(valid)+1
        kkt_progress[inds] = np.log10(
            kkt_ratio[:-1][valid]/kkt_ratio[1:][valid])

    rolling_kkt_progress = _rolling_median(kkt_progress, 5)

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

    support_efficiency = np.divide(n_positive, n_active,
        out=np.full_like(n_positive, np.nan), where=n_active > 0.0)

    # ------------------------------------------------------------------
    # Exact KKT-promotion outcomes.
    #
    # Do not interpret n_zero_dropped as failed promotions unless the solver
    # explicitly guarantees that semantic. It is therefore shown directly as
    # zero-dropped support. Net support change is obtained independently from
    # n_active_after - n_active.
    # ------------------------------------------------------------------
    n_promoted = _series(merged, "promotion_n_promoted")
    n_dropped = _series(merged, "promotion_n_zero_dropped")
    n_active_after = _series(merged, "promotion_n_active_after")

    support_change = np.subtract(n_active_after, n_active,
        out=np.full_like(n_active_after, np.nan),
        where=np.isfinite(n_active_after) & np.isfinite(n_active))

    promotion_survival = np.divide(support_change, n_promoted,
        out=np.full_like(support_change, np.nan), where=n_promoted > 0.0)

    # ------------------------------------------------------------------
    # Constraint and amplitude state.
    # ------------------------------------------------------------------
    alpha = _series(merged, "alpha", "promotion_alpha")
    constraint_dot_lambda = _series(merged, "constraint_dot_lambda")
    ridge = _series(merged, "ridge")

    alpha_reference = _latest_finite(alpha)
    alpha_relative = np.divide(alpha-alpha_reference, abs(alpha_reference),
        out=np.full_like(alpha, np.nan),
        where=np.isfinite(alpha_reference) and alpha_reference != 0.0)

    constraint_ratio = np.divide(np.abs(constraint_dot_lambda), tol,
        out=np.full_like(constraint_dot_lambda, np.nan), where=tol > 0.0)

    # ------------------------------------------------------------------
    # Primal equality-constraint feasibility.
    #
    # These are optional until the solver emits them on each KKT record.
    # Several aliases are accepted so this plotter remains tolerant of naming
    # changes.
    # ------------------------------------------------------------------
    orbit_resid_linf = _series(merged, "orbit_resid_linf",
        "constraint_resid_linf", "orbit_Linf")
    orbit_resid_rel_linf = _series(merged, "orbit_resid_rel_linf")
    orbit_resid_l2 = _series(merged, "orbit_resid_l2",
        "constraint_resid_l2", "orbit_L2")

    constraint_scale = np.abs(alpha)

    orbit_resid_linf_relative = orbit_resid_rel_linf.copy()
    orbit_resid_l2_relative = np.divide(
        orbit_resid_l2, constraint_scale,
        out=np.full_like(orbit_resid_l2, np.nan),
        where=np.isfinite(constraint_scale) & (constraint_scale > 0.0))

    # ------------------------------------------------------------------
    # Objective history.
    #
    # These arrays are populated automatically once the corresponding values
    # are emitted on each KKT record. Current files containing only the final
    # objective_summary remain fully supported.
    # ------------------------------------------------------------------
    data_objective = _series(merged, "data_objective")
    regularisation_objective = _series(merged,
        "scientific_regularisation_objective", "regularisation_objective")
    scientific_objective = _series(merged, "scientific_objective")
    ridge_objective = _series(merged, "numerical_ridge_objective")
    total_objective = _series(merged, "total_objective")

    objective_progress = np.full_like(total_objective, np.nan)

    if total_objective.size > 1:
        previous = total_objective[:-1]
        current = total_objective[1:]
        valid = (np.isfinite(previous) & np.isfinite(current)
            & (np.abs(previous) > 0.0))
        inds = np.flatnonzero(valid)+1
        objective_progress[inds] = (
            (previous[valid]-current[valid])/np.abs(previous[valid]))

    rolling_objective_progress = _rolling_median(objective_progress, 5)

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

    rolling_time = _rolling_median(iter_time, 5)

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(3, 4, figure=fig)

    axes = {
        "kkt": fig.add_subplot(grid[0, 0]),
        "kkt_progress": fig.add_subplot(grid[0, 1]),
        "gradients": fig.add_subplot(grid[0, 2]),
        "violators": fig.add_subplot(grid[0, 3]),
        "support": fig.add_subplot(grid[1, 0]),
        "efficiency": fig.add_subplot(grid[1, 1]),
        "promotion": fig.add_subplot(grid[1, 2]),
        "feasibility": fig.add_subplot(grid[1, 3]),
        "constraint": fig.add_subplot(grid[2, 0]),
        "objective": fig.add_subplot(grid[2, 1]),
        "runtime": fig.add_subplot(grid[2, 2]),
        "status": fig.add_subplot(grid[2, 3]),
    }

    # ------------------------------------------------------------------
    # Panel 1: full-space convergence certificate.
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
    # Panel 2: KKT convergence rate.
    # ------------------------------------------------------------------
    axis = axes["kkt_progress"]

    axis.axhline(0.0, color="black", linestyle=":", lw=0.8)
    _plot_finite(axis, iterations, kkt_progress, "Per iteration",
        lw=0.9, alpha=0.45)
    _plot_finite(axis, iterations, rolling_kkt_progress,
        "5-iteration median", lw=1.7)

    axis.set_title("KKT progress rate")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$\log_{10}(K_{k-1}/K_k)$")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 3: gradient decomposition.
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
    # Panel 4: unresolved dual violations and their severity.
    # ------------------------------------------------------------------
    axis = axes["violators"]

    _plot_finite(axis, iterations, n_violators, "Violating columns", lw=1.7)
    axis.set_title("Inactive-set violations")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Violating columns")
    axis.set_ylim(bottom=0.0)
    _homogenise_ticks(axis)

    axis_severity = axis.twinx()
    _plot_finite(axis_severity, iterations, dual_ratio,
        "Worst violation / tol", positive_log=True, lw=1.3, linestyle="--")
    axis_severity.axhline(1.0, color="black", linestyle=":", lw=0.9)
    axis_severity.set_ylabel("Worst dual / tolerance")
    _homogenise_ticks(axis_severity)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_severity.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 5: support evolution.
    # ------------------------------------------------------------------
    axis = axes["support"]

    _plot_finite(axis, iterations, n_active, "Working support", lw=1.6)
    _plot_finite(axis, iterations, n_positive, "Positive coefficients", lw=1.4)

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
    # Panel 6: global working-set efficiency.
    # ------------------------------------------------------------------
    axis = axes["efficiency"]

    _plot_finite(axis, iterations, 100.0*support_efficiency,
        "Positive / working", lw=1.7)

    axis.set_ylim(0.0, 105.0)
    axis.set_title("Working-set efficiency")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Positive support [%]")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 7: support transition after each promotion/re-solve.
    # ------------------------------------------------------------------
    axis = axes["promotion"]

    _plot_finite(axis, iterations, n_promoted, "Promoted", lw=1.4)
    _plot_finite(axis, iterations, n_dropped, "Zero-dropped", lw=1.3)
    _plot_finite(axis, iterations, support_change, "Net support change", lw=1.6)

    axis.axhline(0.0, color="black", linestyle=":", lw=0.8)
    axis.set_title("Active-set turnover")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)

    axis_survival = axis.twinx()
    _plot_finite(axis_survival, iterations, 100.0*promotion_survival,
        "Net gain / promoted", lw=1.1, linestyle="--")
    axis_survival.set_ylabel("Net gain / promoted [%]")
    _homogenise_ticks(axis_survival)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_survival.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 8: primal equality-constraint feasibility.
    # ------------------------------------------------------------------
    axis = axes["feasibility"]
    has_constraint_history = (
        np.any(np.isfinite(orbit_resid_linf_relative))
        or np.any(np.isfinite(orbit_resid_l2_relative))
    )

    if has_constraint_history:
        _plot_finite(axis, iterations, orbit_resid_linf_relative,
            r"$\|Cx-d\|_\infty/|\alpha|$", positive_log=True, lw=1.6)
        _plot_finite(axis, iterations, orbit_resid_l2_relative,
            r"$\|Cx-d\|_2/|\alpha|$", positive_log=True, lw=1.3)

        axis.set_xlabel("Iteration")
        axis.set_ylabel("Relative residual")
        _homogenise_ticks(axis)
        axis.legend(fontsize=8, loc="best")
    else:
        _unavailable(axis,
            "Per-iteration primal constraint\nresidual not emitted")

    axis.set_title("Hard-constraint feasibility")

    # ------------------------------------------------------------------
    # Panel 9: common-amplitude and constraint-stationarity stability.
    # ------------------------------------------------------------------
    axis = axes["constraint"]

    _plot_finite(axis, iterations, 1e6*alpha_relative,
        r"$10^6(\alpha-\alpha_{\rm final})/\alpha_{\rm final}$", lw=1.5)

    axis.axhline(0.0, color="black", linestyle=":", lw=0.8)
    axis.set_title("Amplitude / stationarity")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$\Delta\alpha/\alpha$ [ppm]")
    _homogenise_ticks(axis)

    axis_constraint = axis.twinx()
    _plot_finite(axis_constraint, iterations, constraint_ratio,
        "Amplitude KKT / tol", positive_log=True, lw=1.3, linestyle="--")
    axis_constraint.axhline(1.0, color="black", linestyle=":", lw=0.8)
    axis_constraint.set_ylabel("Stationarity / tolerance")
    _homogenise_ticks(axis_constraint)

    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis_constraint.get_legend_handles_labels()
    axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 10: objective evolution.
    # ------------------------------------------------------------------
    axis = axes["objective"]
    has_objective_history = (
        np.any(np.isfinite(total_objective))
        or np.any(np.isfinite(scientific_objective))
        or np.any(np.isfinite(data_objective))
    )

    if has_objective_history:
        reference = _latest_finite(total_objective)

        if np.isfinite(reference):
            relative_objective = np.divide(
                total_objective-reference, max(abs(reference), eps),
                out=np.full_like(total_objective, np.nan),
                where=np.isfinite(total_objective))
            _plot_finite(axis, iterations, relative_objective,
                r"$(\Phi-\Phi_{\rm final})/|\Phi_{\rm final}|$", lw=1.6)

        axis.axhline(0.0, color="black", linestyle=":", lw=0.8)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Relative objective")
        _homogenise_ticks(axis)

        axis_progress = axis.twinx()
        _plot_finite(axis_progress, iterations,
            rolling_objective_progress, "5-iteration improvement",
            lw=1.2, linestyle="--")
        axis_progress.axhline(0.0, color="black", linestyle=":", lw=0.7)
        axis_progress.set_ylabel("Fractional improvement")
        _homogenise_ticks(axis_progress)

        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = axis_progress.get_legend_handles_labels()
        axis.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="best")
    else:
        _unavailable(axis,
            "Per-iteration objective\nhistory not emitted")

    axis.set_title("Objective evolution")

    # ------------------------------------------------------------------
    # Panel 11: computational health.
    # ------------------------------------------------------------------
    axis = axes["runtime"]

    _plot_finite(axis, iterations, iter_time, "Iteration time", lw=1.0,
        alpha=0.55)
    _plot_finite(axis, iterations, rolling_time, "5-iteration median", lw=1.7)

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
    # Panel 12: current solver state.
    # ------------------------------------------------------------------
    axis = axes["status"]
    axis.set_axis_off()

    latest_kkt = _latest_finite(kkt_ratio)
    latest_active = _latest_finite(active_ratio)
    latest_dual = _latest_finite(dual_ratio)
    latest_violators = _latest_finite(n_violators)
    latest_support = _latest_finite(n_active)
    latest_positive = _latest_finite(n_positive)
    latest_efficiency = _latest_finite(support_efficiency)
    latest_alpha = _latest_finite(alpha)
    latest_ridge = _latest_finite(ridge)
    latest_time = _latest_finite(elapsed)
    latest_support_change = _latest_finite(support_change)
    latest_constraint_linf = _latest_finite(orbit_resid_linf_relative)

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

    if np.isfinite(latest_efficiency):
        lines.append(f"Working-set efficiency: "
            f"{100.0*latest_efficiency:.1f}%")

    if np.isfinite(setup_C) and setup_C > 0.0 and np.isfinite(latest_positive):
        lines.append(
            f"Positive / component: {latest_positive/setup_C:.2f}")

    if np.isfinite(latest_support_change):
        lines.append(f"Last net support change: "
            f"{latest_support_change:+.0f}")

    if np.isfinite(latest_constraint_linf):
        lines.append(f"Constraint Linf / alpha: "
            f"{latest_constraint_linf:.3e}")

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
    final_ridge_obj = _finite_scalar(
        final_record, "numerical_ridge_objective")
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
        ha="left", va="top", fontsize=9.0,
        bbox=dict(boxstyle="round", fc="white", alpha=0.90))

    axis.set_title("Current solver state")

    # ------------------------------------------------------------------
    # Figure-level summary.
    # ------------------------------------------------------------------
    setup_P = _finite_scalar(setup, "P")
    setup_reg = _finite_scalar(setup, "regularisation_scale")
    setup_batch = _finite_scalar(setup, "kkt_promote_batch")

    title_parts = [
        "CubeFit exact constrained active-set diagnostics",
        f"iterations={iterations[0]}-{iterations[-1]}",
    ]

    if np.isfinite(setup_C) and np.isfinite(setup_P):
        title_parts.append(f"C={int(setup_C)}, P={int(setup_P)}")

    if np.isfinite(setup_reg):
        title_parts.append(rf"$\eta={_fmt_sci(setup_reg)}$")

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

        if axis.get_xticks().size:
            axis.xaxis.set_major_locator(
                mticker.MaxNLocator(nbins=5, integer=True))
            axis.xaxis.set_major_formatter(
                mticker.FuncFormatter(_fmt_compact))
            axis.tick_params(axis="x", which="major", labelsize=9)
            axis.xaxis.labelpad = 2
            axis.yaxis.labelpad = 2

    # Also format secondary y axes created by twinx().
    for axis in fig.axes:
        if axis not in axes.values():
            axis.tick_params(axis="y", which="major", labelsize=8)
            axis.yaxis.labelpad = 2

    fig.suptitle(" | ".join(title_parts), fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.055, top=0.94,
        hspace=0.38, wspace=0.42)

    if save_path:
        fig.savefig(save_path, dpi=175, bbox_inches="tight",
            pad_inches=0.04)

    if show:
        plt.show()

    plt.close(fig)

    return raw_records

# ------------------------------------------------------------------------------

def plot_chemodynamical_atlas(arSOL, umetals, uages, ualphas, nzComp, pred,
    oDict, otypes, rM, lZMSign, lXMSign, lLMSign, save_path=None,
    cmap="inferno", bin_threshold=20, n_circ_bins=15, render_n_age=180,
    render_n_metal=120, smoothing_age=0.65, smoothing_metal=0.65,
    density_floor=1e-6):
    """
    Plot the complete fitted component-by-population CubeFit solution.

    Dynamical components are arranged vertically in three intrinsic orbital
    families. Within each family, components are ordered by their physically
    appropriate circularity, with dynamical-weighted radius used as the
    secondary ordering coordinate.

    Each dynamical row and alpha enhancement defines an independent miniature
    age-metallicity plane. The fitted discrete SSP populations in that plane
    are rendered as a smooth mass-preserving density field. Smoothing is
    confined to each component-alpha plane and therefore never mixes
    dynamical components or alpha enhancements.

    Colour gives the logarithmic fraction of the complete fitted population.
    Regions with zero or negligible population density are transparent.
    Marker size in the circularity panel represents relative Schwarzschild
    dynamical weight.

    Parameters
    ----------
    arSOL : ndarray
        Fitted SSP coefficients with shape
        ``(C, nMetals, nAges, nAlphas)``.
    umetals : ndarray
        SSP metallicities with shape ``(nMetals,)``.
    uages : ndarray
        SSP ages with shape ``(nAges,)``.
    ualphas : ndarray
        SSP alpha enhancements with shape ``(nAlphas,)``.
    nzComp : ndarray
        Original decomposition component numbers represented by ``arSOL``.
    pred : str
        Integer formatting specification used for decomposition keys.
    oDict : dict
        Dynamical decomposition dictionary containing ``wheres`` and
        ``weights``.
    otypes : ndarray
        Intrinsic component types: 0 for z-tube, 1 for x-tube, and 2 for
        boxes.
    rM : ndarray
        Intrinsic radius of every Schwarzschild orbit.
    lZMSign : ndarray
        Signed z-axis circularity of every Schwarzschild orbit.
    lXMSign : ndarray
        Signed x-axis circularity of every Schwarzschild orbit.
    lLMSign : ndarray
        Total-angular-momentum circularity of every Schwarzschild orbit.
    save_path : pathlib.Path or str, optional
        Output figure path.
    cmap : str, optional
        Matplotlib colormap. Default is ``"inferno"``.
    bin_threshold : int, optional
        Maximum number of fitted dynamical components shown individually.
        Above this threshold, components are rebinned onto fixed circularity
        grids within each orbital family. Default is 20.
    n_circ_bins : int, optional
        Number of circularity bins per orbital family when component binning
        is used. Default is 15.
    render_n_age : int, optional
        Horizontal resolution of each age-metallicity plane.
    render_n_metal : int, optional
        Vertical resolution of each age-metallicity plane.
    smoothing_age : float, optional
        Gaussian standard deviation along the rendered age axis, in pixels.
        Default is 2.5.
    smoothing_metal : float, optional
        Gaussian standard deviation along the rendered metallicity axis, in
        pixels. Default is 2.0.
    density_floor : float, optional
        Minimum smoothed support retained in the rendered image. Pixels below
        this value are transparent. Default is 1e-3.

    Returns
    -------
    out : dict
        Figure and all component ordering, dynamical-coordinate, mass, and
        displayed population quantities.

    Raises
    ------
    ValueError
        If supplied arrays have inconsistent dimensions or rendering
        parameters are invalid.
    RuntimeError
        If component types are invalid, a component has no valid constituent
        orbital weight, or the fitted solution has no positive mass.

    Examples
    --------
    >>> out = plot_chemodynamical_atlas(
    ...     arSOL, umetals, uages, ualphas, nzComp, pred, oDict, otypes,
    ...     rM, lZMSign, lXMSign, lLMSign,
    ...     save_path="chemodynamical_atlas.png")
    """
    arSOL = np.asarray(arSOL, dtype=np.float64)
    umetals = np.asarray(umetals, dtype=np.float64).ravel(order="C")
    uages = np.asarray(uages, dtype=np.float64).ravel(order="C")
    ualphas = np.asarray(ualphas, dtype=np.float64).ravel(order="C")
    nzComp = np.asarray(nzComp, dtype=np.int64).ravel(order="C")
    otypes = np.asarray(otypes, dtype=np.int64).ravel(order="C")
    rM = np.asarray(rM, dtype=np.float64).ravel(order="C")
    lZMSign = np.asarray(lZMSign, dtype=np.float64).ravel(order="C")
    lXMSign = np.asarray(lXMSign, dtype=np.float64).ravel(order="C")
    lLMSign = np.asarray(lLMSign, dtype=np.float64).ravel(order="C")

    if arSOL.ndim != 4:
        raise ValueError(
            "arSOL must have shape (C, nMetals, nAges, nAlphas).")

    C, nMetals, nAges, nAlphas = map(int, arSOL.shape)
    NOrbs = int(rM.size)
    bin_threshold = int(bin_threshold)
    n_circ_bins = int(n_circ_bins)
    render_n_age = int(render_n_age)
    render_n_metal = int(render_n_metal)
    density_floor = float(density_floor)

    if bin_threshold < 1:
        raise ValueError("bin_threshold must be positive.")
    if n_circ_bins < 1:
        raise ValueError("n_circ_bins must be positive.")
    if render_n_age < 16 or render_n_metal < 16:
        raise ValueError("Rendering dimensions must be at least 16.")
    if density_floor < 0.0 or density_floor >= 1.0:
        raise ValueError("density_floor must lie in [0, 1).")

    binned = C > bin_threshold

    if nzComp.size != C or otypes.size != C:
        raise ValueError(
            "arSOL, nzComp, and otypes have inconsistent component counts.")
    if umetals.size != nMetals:
        raise ValueError("umetals is inconsistent with arSOL.")
    if uages.size != nAges:
        raise ValueError("uages is inconsistent with arSOL.")
    if ualphas.size != nAlphas:
        raise ValueError("ualphas is inconsistent with arSOL.")

    for name, values in (("lZMSign", lZMSign), ("lXMSign", lXMSign),
        ("lLMSign", lLMSign)):
        if values.size != NOrbs:
            raise ValueError(f"{name} has {values.size} entries; expected "
                f"{NOrbs}.")

    if not np.all(np.isin(otypes, (0, 1, 2))):
        raise RuntimeError(
            f"Invalid intrinsic component types: {np.unique(otypes)}.")

    families = ((0, r"$z$-tube", lZMSign, r"$\lambda_z$"),
        (1, r"$x$-tube", lXMSign, r"$\lambda_x$"),
        (2, "box", lLMSign, r"$\lambda_L$"))

    component_circularity = np.full(C, np.nan, dtype=np.float64)
    component_radius = np.full(C, np.nan, dtype=np.float64)
    component_orbit_weight = np.zeros(C, dtype=np.float64)
    component_orbit_fraction = np.zeros((C, NOrbs), dtype=np.float64)

    for type_id, _, circularity, _ in families:
        for cc, component in enumerate(nzComp):
            if otypes[cc] != type_id:
                continue

            key = f"{component:{pred}d}"
            mask = np.asarray(oDict["wheres"][key], dtype=bool)
            weight = np.asarray(oDict["weights"][key], dtype=np.float64)

            if mask.size != NOrbs or weight.size != NOrbs:
                raise ValueError(f"Component {component} orbit arrays have "
                    "incorrect dimensions.")

            valid = mask & np.isfinite(weight) & (weight > 0.0)
            valid &= np.isfinite(rM) & np.isfinite(circularity)

            if not np.any(valid):
                raise RuntimeError(
                    f"Component {component} has no valid orbital weight.")

            total_weight = float(np.sum(weight[valid]))
            component_orbit_weight[cc] = total_weight
            component_orbit_fraction[cc, valid] = weight[valid]/total_weight
            component_circularity[cc] = float(
                np.sum(weight[valid]*circularity[valid])/total_weight)
            component_radius[cc] = float(
                np.sum(weight[valid]*rM[valid])/total_weight)

    family_orders = []
    family_bounds = []
    row_start = 0

    for type_id, _, _, _ in families:
        family = np.flatnonzero(otypes == type_id)

        if family.size:
            local = np.lexsort((component_radius[family],
                -component_circularity[family]))
            family = family[local]

        family_orders.append(family)
        family_bounds.append((row_start, row_start+family.size))
        row_start += family.size

    order = np.concatenate(family_orders)

    if order.size != C or np.unique(order).size != C:
        raise RuntimeError("Internal component ordering does not contain "
            "every component exactly once.")

    component_mass = np.sum(arSOL, axis=(1, 2, 3))
    total_mass = float(np.sum(component_mass))

    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("The fitted solution has no positive total mass.")

    circ_edges = None
    circ_centers = None

    if not binned:
        display_solution = arSOL[order]
        display_mass = component_mass[order]
        display_orbit_weight = component_orbit_weight[order]
        display_circularity = component_circularity[order]
        display_radius = component_radius[order]
        display_types = otypes[order]
        display_components = nzComp[order]
        display_family_bounds = family_bounds
        display_C = C

    else:
        circ_edges = np.linspace(-1.0, 1.0, n_circ_bins+1,
            dtype=np.float64)
        circ_centers = 0.5*(circ_edges[:-1]+circ_edges[1:])
        display_C = len(families)*n_circ_bins

        display_solution = np.zeros(
            (display_C, nMetals, nAges, nAlphas), dtype=np.float64)
        display_mass = np.zeros(display_C, dtype=np.float64)
        display_orbit_weight = np.zeros(display_C, dtype=np.float64)
        display_circularity = np.zeros(display_C, dtype=np.float64)
        display_radius = np.full(display_C, np.nan, dtype=np.float64)
        display_types = np.empty(display_C, dtype=np.int64)
        display_components = np.full(display_C, -1, dtype=np.int64)

        display_family_bounds = []
        row_start = 0

        for type_id, _, circularity, _ in families:
            family_rows = slice(row_start, row_start+n_circ_bins)
            display_circularity[family_rows] = circ_centers[::-1]
            display_types[family_rows] = type_id

            radius_num = np.zeros(n_circ_bins, dtype=np.float64)
            radius_den = np.zeros(n_circ_bins, dtype=np.float64)

            for cc in np.flatnonzero(otypes == type_id):
                orbit_fraction = component_orbit_fraction[cc]
                valid = ((orbit_fraction > 0.0) & np.isfinite(circularity)
                    & np.isfinite(rM))

                if not np.any(valid):
                    continue

                bin_fraction = np.histogram(circularity[valid],
                    bins=circ_edges, weights=orbit_fraction[valid])[0]
                fraction_sum = float(np.sum(bin_fraction))

                if not np.isclose(fraction_sum, 1.0, rtol=1e-12,
                    atol=1e-12):
                    raise RuntimeError(f"Component {nzComp[cc]} circularity "
                        f"binning conserves {fraction_sum:.16e}, expected 1.")

                for bb in np.flatnonzero(bin_fraction > 0.0):
                    rr = row_start+(n_circ_bins-1-bb)
                    frac = float(bin_fraction[bb])
                    display_solution[rr] += frac*arSOL[cc]
                    display_orbit_weight[rr] += (
                        frac*component_orbit_weight[cc])

                orbit_bin = np.searchsorted(circ_edges, circularity[valid],
                    side="right")-1
                orbit_bin = np.clip(orbit_bin, 0, n_circ_bins-1)
                frac_weight = orbit_fraction[valid]

                np.add.at(radius_num, orbit_bin, frac_weight*rM[valid])
                np.add.at(radius_den, orbit_bin, frac_weight)

            radius_mean = np.divide(radius_num, radius_den,
                out=np.full(n_circ_bins, np.nan, dtype=np.float64),
                where=radius_den > 0.0)

            display_radius[family_rows] = radius_mean[::-1]
            display_family_bounds.append(
                (row_start, row_start+n_circ_bins))
            row_start += n_circ_bins

        display_mass = np.sum(display_solution, axis=(1, 2, 3))
        binned_total = float(np.sum(display_solution))

        if not np.isclose(binned_total, total_mass, rtol=1e-12,
            atol=1e-12*max(1.0, total_mass)):
            raise RuntimeError("Circularity binning does not conserve fitted "
                f"population mass: {binned_total:.16e} != "
                f"{total_mass:.16e}.")

        print(f"SFH atlas: rebinned {C} components onto "
            f"{n_circ_bins} circularity bins per family ({display_C} rows).")

    population_fraction = display_solution/total_mass
    family_bounds = display_family_bounds

    # ------------------------------------------------------------------
    # Mass-conserving KDE on the true SSP sampling.
    #
    # Each fitted SSP coefficient is deposited as a Gaussian centred on its
    # actual (age, metallicity) coordinate. Every discretized kernel is
    # normalized to unit mass before multiplication by the SSP fraction, so
    # smoothing changes morphology but cannot change an SSP's total weight.
    # ------------------------------------------------------------------
    smoothing_age = float(smoothing_age)
    smoothing_metal = float(smoothing_metal)

    if smoothing_age <= 0.0 or smoothing_metal <= 0.0:
        raise ValueError("SSP smoothing scales must be positive.")

    def _cell_edges(values):
        values = np.asarray(values, dtype=np.float64)

        if values.size == 1:
            return np.array([values[0]-0.5, values[0]+0.5],
                dtype=np.float64)

        edges = np.empty(values.size+1, dtype=np.float64)
        edges[1:-1] = 0.5*(values[:-1]+values[1:])
        edges[0] = values[0]-0.5*(values[1]-values[0])
        edges[-1] = values[-1]+0.5*(values[-1]-values[-2])
        return edges

    age_edges = _cell_edges(uages)
    metal_edges = _cell_edges(umetals)

    age_min, age_max = float(age_edges[0]), float(age_edges[-1])
    metal_min, metal_max = float(metal_edges[0]), float(metal_edges[-1])
    age_width = age_max-age_min

    age_render = np.linspace(age_min, age_max, render_n_age)
    metal_render = np.linspace(metal_min, metal_max, render_n_metal)

    age_cell_width = np.diff(age_edges)
    metal_cell_width = np.diff(metal_edges)

    sigma_age = smoothing_age*age_cell_width
    sigma_metal = smoothing_metal*metal_cell_width

    # Precompute separable Gaussian factors. Do not normalize these
    # individually: the full 2-D kernel is normalized below after sampling
    # onto the finite rendering domain.
    age_kernel = np.exp(-0.5*(
        (age_render[:, None]-uages[None, :])/sigma_age[None, :])**2)
    metal_kernel = np.exp(-0.5*(
        (metal_render[:, None]-umetals[None, :])
        / sigma_metal[None, :])**2)

    rendered_linear = np.zeros(
        (display_C, nAlphas, render_n_metal, render_n_age),
        dtype=np.float64)

    for cc in range(display_C):
        for aa in range(nAlphas):
            weights = population_fraction[cc, :, :, aa]
            density = rendered_linear[cc, aa]

            for mm in range(nMetals):
                for tt in range(nAges):
                    weight = float(weights[mm, tt])

                    if weight <= 0.0:
                        continue
                    kernel = np.outer(
                        metal_kernel[:, mm], age_kernel[:, tt])

                    kernel_max = float(np.max(kernel))

                    if kernel_max > 0.0:
                        density += weight*(kernel/kernel_max)

    max_fraction = float(np.max(population_fraction))
    rendered_linear = np.minimum(rendered_linear, max_fraction)

    # Verify that smoothing has not changed the fitted mass represented by
    # any component-alpha plane.
    for cc in range(display_C):
        for aa in range(nAlphas):
            input_mass = float(np.sum(
                population_fraction[cc, :, :, aa]))
            output_mass = float(np.sum(rendered_linear[cc, aa]))

    positive = population_fraction[population_fraction > 0.0]

    if positive.size == 0:
        raise RuntimeError("No positive fitted population available to plot.")

    # Keep visibility determined by the smoothed KDE, but colour normalization
    # tied to the actual fitted SSP population fractions.
    rendered = np.full_like(rendered_linear, np.nan)
    visible = rendered_linear >= density_floor*np.max(rendered_linear)
    rendered[visible] = np.log10(rendered_linear[visible])

    sfh_min = float(np.log10(np.min(positive)))
    wmin = sfh_min
    wmax = float(np.log10(np.max(positive)))

    print(f"SFH atlas KDE conservation: "
        f"{np.sum(rendered_linear):.16f}")
    print(f"SFH atlas limits: {wmin:.2f} to {wmax:.2f}")

    young_lim = 3.0  # Gyr
    young = uages <= young_lim

    young_mass = np.sum(arSOL[:, :, young, :], axis=(1, 2, 3))
    total_mass = np.sum(arSOL, axis=(1, 2, 3))
    young_frac = np.divide(young_mass, total_mass,
        out=np.zeros_like(young_mass), where=total_mass > 0.0)

    print("\nYoung-component check")
    print(" comp   type       <R>     young frac")
    for cc in np.flatnonzero(young_frac > 1e-3):
        print(f"{nzComp[cc]:5d}  {otypes[cc]:5d}  "
            f"{component_radius[cc]:8.3f}  {young_frac[cc]:10.3e}")

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=np.asarray((12.0, 15.0))*0.8)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=(0.65, 8.0),
        wspace=0.035)

    ax_dyn = fig.add_subplot(gs[0, 0])
    ax_pop = fig.add_subplot(gs[0, 1])

    cmap_obj = colormaps[cmap].copy()
    cmap_obj.set_bad((0.0, 0.0, 0.0, 0.0))
    norm = Normalize(vmin=wmin, vmax=wmax)

    # One logical vertical unit per dynamical row. Within that unit the
    # complete metallicity range is represented continuously.
    for cc in range(display_C):
        y0 = float(cc)
        y1 = float(cc+1)

        for aa in range(nAlphas):
            image = rendered[cc, aa]

            if not np.any(np.isfinite(image)):
                continue

            x0 = aa*age_width
            x1 = (aa+1)*age_width

            ax_pop.imshow(image, cmap=cmap_obj, norm=norm, origin="lower",
                interpolation="bilinear", aspect="auto",
                extent=(x0, x1, y1, y0), rasterized=True)

    ax_pop.set_facecolor("black")
    ax_pop.set_xlim(0.0, nAlphas*age_width)
    ax_pop.set_ylim(display_C, 0.0)

    for aa, alpha in enumerate(ualphas):
        xpos = (aa+0.5)/nAlphas
        lT = ax_pop.text(xpos, 1.012, rf"${alpha:.2f}$", ha="center",
            va="bottom", color=POT.pgreen, transform=ax_pop.transAxes,
            clip_on=False)
        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
            foreground="k")])

    lT = ax_pop.text(-0.025, 1.012, r"$[\alpha/{\rm Fe}]=$", ha="left",
        va="bottom", color=POT.pgreen, transform=ax_pop.transAxes,
        clip_on=False)
    lT.set_path_effects([
        PathEffects.withStroke(linewidth=1.5, foreground="k")])

    for aa in range(1, nAlphas):
        ax_pop.axvline(aa*age_width, color="black", lw=1.0)

    age_locator = mticker.MaxNLocator(nbins=4)
    base_ticks = age_locator.tick_values(
        float(np.min(uages)), float(np.max(uages)))
    base_ticks = base_ticks[(base_ticks >= np.min(uages))
        & (base_ticks <= np.max(uages))]

    age_ticks = []
    age_labels = []

    for aa in range(nAlphas):
        age_ticks.extend(aa*age_width+(base_ticks-age_min))
        age_labels.extend([f"{age:g}" for age in base_ticks])

    ax_pop.set_xticks(age_ticks)
    ax_pop.set_xticklabels(age_labels, fontsize=8)
    ax_pop.set_xlabel(r"$t\ [{\rm Gyr}]$")
    ax_pop.set_yticks([])

    # Very subtle component boundaries; strong boundaries only between
    # intrinsic orbital families.
    for cc in range(1, display_C):
        ax_pop.axhline(cc, color="0.5", lw=0.15, alpha=0.75)

    family_colors = ("tab:blue", "tab:orange", "tab:green")

    for start, stop in family_bounds:
        if stop <= start or start == 0:
            continue

        for axis in (ax_dyn, ax_pop):
            axis.axhline(float(start), color="black", lw=0.9, zorder=10)

    row_centers = np.arange(display_C, dtype=np.float64)+0.5

    if binned:
        marker_weight = display_mass
    else:
        marker_weight = display_orbit_weight

    marker_total = float(np.sum(marker_weight))

    if marker_total > 0.0:
        dynamical_mass_fraction = marker_weight/marker_total
    else:
        dynamical_mass_fraction = np.zeros(display_C, dtype=np.float64)

    max_marker_weight = float(np.max(marker_weight))

    if max_marker_weight > 0.0:
        mass_scale = marker_weight/max_marker_weight
    else:
        mass_scale = np.zeros(display_C, dtype=np.float64)

    marker_sizes = 5.0+75.0*mass_scale

    for ff, (type_id, _, _, _) in enumerate(families):
        family_rows = np.flatnonzero(display_types == type_id)
        populated = display_mass[family_rows] > 0.0
        family_rows = family_rows[populated]

        ax_dyn.scatter(display_circularity[family_rows],
            row_centers[family_rows], s=marker_sizes[family_rows],
            color=family_colors[ff], edgecolor="none", zorder=3)

    ax_dyn.set_xlim(-1.05, 1.05)
    ax_dyn.set_ylim(display_C, 0.0)
    ax_dyn.set_xlabel(r"$\lambda$")
    ax_dyn.set_ylabel("")
    ax_dyn.set_yticks([])

    for ff, ((start, stop), (_, label, _, _)) in enumerate(
        zip(family_bounds, families)):

        if stop <= start:
            continue

        # Upper-left of each family is empty by circularity ordering.
        yc = start+0.15

        ax_dyn.text(0.08, yc, label,
            color=family_colors[ff], fontsize=10, fontweight="bold",
            rotation=90, rotation_mode="anchor",
            ha="left", va="top",
            transform=ax_dyn.get_yaxis_transform(),
            clip_on=True, zorder=5)

    fig.subplots_adjust(left=0.055, right=0.96, bottom=0.07, top=0.92)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])

    miText = POT.prec(1, wmin)
    maText = POT.prec(1, wmax)
    cax = POT.attachAxis(ax_pop, "right", 0.025)
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical")

    cax.text(0.5, 0.5, r"$\log_{10}$ Population Fraction", va="center",
        ha="center", rotation=270.0, color=POT.pgreen,
        transform=cax.transAxes, path_effects=[
            PathEffects.withStroke(linewidth=1.5, foreground="k")])
    cax.text(0.5, 5e-3, miText, va="bottom", ha="center", color="k",
        transform=cax.transAxes, rotation=270.0)
    cax.text(0.5, 1.0-5e-3, maText, va="top", ha="center", color="w",
        transform=cax.transAxes, rotation=270.0)
    cb.set_ticks([])

    if save_path is not None:
        save_path = plp.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.03)

    return {
        "figure": fig,
        "binned": binned,
        "bin_threshold": bin_threshold,
        "n_circularity_bins": n_circ_bins if binned else None,
        "circularity_edges": circ_edges,
        "circularity_centers": circ_centers,
        "order": order,
        "family_orders": family_orders,
        "family_bounds": family_bounds,
        "component_numbers": nzComp,
        "component_types": otypes,
        "component_circularity": component_circularity,
        "component_radius": component_radius,
        "component_coefficient_mass": component_mass,
        "component_dynamical_weight": component_orbit_weight,
        "display_solution": display_solution,
        "display_components": display_components,
        "display_types": display_types,
        "display_circularity": display_circularity,
        "display_radius": display_radius,
        "display_coefficient_mass": display_mass,
        "display_dynamical_weight": display_orbit_weight,
        "display_mass_fraction": dynamical_mass_fraction,
        "population_fraction": population_fraction,
        "rendered_population_fraction": rendered,
        "age_render": age_render,
        "metal_render": metal_render,
        "smoothing_age": smoothing_age,
        "smoothing_metal": smoothing_metal,
        "density_floor": density_floor,
        "wmin": wmin,
        "wmax": wmax,
    }

# ------------------------------------------------------------------------------

def plot_population_circularity_profiles(arSOL, umetals, uages, ualphas,
    nzComp, pred, oDict, otypes, rM, lZMSign, lXMSign, lLMSign,
    save_path=None, min_mass_fraction=0.0):
    """
    Plot population-weighted age, metallicity, and alpha enhancement against
    orbital circularity for z-tube, x-tube, and box components.

    Each fitted dynamical component contributes one point. Circularity is the
    orbit-weighted mean appropriate to that orbital family: lambda_z for
    z-tubes, lambda_x for x-tubes, and lambda_L for boxes. Population
    properties are weighted directly by the fitted SSP coefficients.

    Parameters
    ----------
    arSOL : ndarray
        Fitted SSP coefficients with shape
        ``(C, nMetals, nAges, nAlphas)``.
    umetals : ndarray
        SSP metallicities with shape ``(nMetals,)``.
    uages : ndarray
        SSP ages with shape ``(nAges,)``.
    ualphas : ndarray
        SSP alpha enhancements with shape ``(nAlphas,)``.
    nzComp : ndarray
        Original decomposition component numbers represented by ``arSOL``.
    pred : str
        Integer formatting specification used for decomposition keys.
    oDict : dict
        Dynamical decomposition dictionary containing ``wheres`` and
        ``weights``.
    otypes : ndarray
        Intrinsic component types: 0 for z-tube, 1 for x-tube, and 2 for
        boxes.
    rM : ndarray
        Intrinsic radius of every Schwarzschild orbit.
    lZMSign : ndarray
        Signed z-axis circularity of every Schwarzschild orbit.
    lXMSign : ndarray
        Signed x-axis circularity of every Schwarzschild orbit.
    lLMSign : ndarray
        Total-angular-momentum circularity of every Schwarzschild orbit.
    save_path : pathlib.Path or str, optional
        Output figure path.
    min_mass_fraction : float, optional
        Minimum fraction of the total fitted population mass required for a
        component to be plotted. Default is zero.

    Returns
    -------
    out : dict
        Computed component circularities, radii, masses, population-weighted
        properties, masks, and figure.

    Raises
    ------
    ValueError
        If supplied arrays have inconsistent dimensions or
        ``min_mass_fraction`` is invalid.
    RuntimeError
        If orbital types are invalid or no valid fitted components remain.

    Examples
    --------
    >>> out = plot_population_circularity_profiles(
    ...     arSOL, umetals, uages, ualphas, nzComp, pred, oDict, otypes,
    ...     rM, lZMSign, lXMSign, lLMSign,
    ...     save_path="population_circularity_profiles.png")
    """
    arSOL = np.asarray(arSOL, dtype=np.float64)
    umetals = np.asarray(umetals, dtype=np.float64).ravel(order="C")
    uages = np.asarray(uages, dtype=np.float64).ravel(order="C")
    ualphas = np.asarray(ualphas, dtype=np.float64).ravel(order="C")
    nzComp = np.asarray(nzComp, dtype=np.int64).ravel(order="C")
    otypes = np.asarray(otypes, dtype=np.int64).ravel(order="C")
    rM = np.asarray(rM, dtype=np.float64).ravel(order="C")
    lZMSign = np.asarray(lZMSign, dtype=np.float64).ravel(order="C")
    lXMSign = np.asarray(lXMSign, dtype=np.float64).ravel(order="C")
    lLMSign = np.asarray(lLMSign, dtype=np.float64).ravel(order="C")
    min_mass_fraction = float(min_mass_fraction)

    if arSOL.ndim != 4:
        raise ValueError(
            "arSOL must have shape (C, nMetals, nAges, nAlphas).")

    C, nMetals, nAges, nAlphas = map(int, arSOL.shape)
    NOrbs = int(rM.size)

    if nzComp.size != C or otypes.size != C:
        raise ValueError(
            "arSOL, nzComp, and otypes have inconsistent component counts.")
    if umetals.size != nMetals:
        raise ValueError("umetals is inconsistent with arSOL.")
    if uages.size != nAges:
        raise ValueError("uages is inconsistent with arSOL.")
    if ualphas.size != nAlphas:
        raise ValueError("ualphas is inconsistent with arSOL.")
    if min_mass_fraction < 0.0 or not np.isfinite(min_mass_fraction):
        raise ValueError("min_mass_fraction must be finite and nonnegative.")

    for name, values in (("lZMSign", lZMSign), ("lXMSign", lXMSign),
        ("lLMSign", lLMSign)):
        if values.size != NOrbs:
            raise ValueError(f"{name} has {values.size} entries; expected "
                f"{NOrbs}.")

    if not np.all(np.isin(otypes, (0, 1, 2))):
        raise RuntimeError(
            f"Invalid intrinsic component types: {np.unique(otypes)}.")

    families = ((0, r"$z$-tube", lZMSign, r"$\lambda_z$"),
        (1, r"$x$-tube", lXMSign, r"$\lambda_x$"),
        (2, "box", lLMSign, r"$\lambda_L$"))

    component_circularity = np.full(C, np.nan, dtype=np.float64)
    component_radius = np.full(C, np.nan, dtype=np.float64)
    component_orbit_weight = np.zeros(C, dtype=np.float64)

    for type_id, _, circularity, _ in families:
        for cc, component in enumerate(nzComp):
            if otypes[cc] != type_id:
                continue

            key = f"{component:{pred}d}"
            mask = np.asarray(oDict["wheres"][key], dtype=bool)
            weight = np.asarray(oDict["weights"][key], dtype=np.float64)

            if mask.size != NOrbs or weight.size != NOrbs:
                raise ValueError(f"Component {component} orbit arrays have "
                    "incorrect dimensions.")

            valid = mask & np.isfinite(weight) & (weight > 0.0)
            valid &= np.isfinite(rM) & np.isfinite(circularity)

            if not np.any(valid):
                continue

            total_weight = float(np.sum(weight[valid]))
            component_orbit_weight[cc] = total_weight
            component_circularity[cc] = float(
                np.sum(weight[valid]*circularity[valid])/total_weight)
            component_radius[cc] = float(
                np.sum(weight[valid]*rM[valid])/total_weight)

    component_mass = np.sum(arSOL, axis=(1, 2, 3))
    total_mass = float(np.sum(component_mass))

    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("The fitted solution has no positive total mass.")

    mass_fraction = component_mass/total_mass

    zz, tt, aa = np.meshgrid(
        umetals, uages, ualphas, indexing="ij")

    mean_metal = np.divide(
        np.sum(arSOL*zz[None, :, :, :], axis=(1, 2, 3)),
        component_mass, out=np.full(C, np.nan, dtype=np.float64),
        where=component_mass > 0.0)

    mean_age = np.divide(
        np.sum(arSOL*tt[None, :, :, :], axis=(1, 2, 3)),
        component_mass, out=np.full(C, np.nan, dtype=np.float64),
        where=component_mass > 0.0)

    mean_alpha = np.divide(
        np.sum(arSOL*aa[None, :, :, :], axis=(1, 2, 3)),
        component_mass, out=np.full(C, np.nan, dtype=np.float64),
        where=component_mass > 0.0)

    valid_component = (np.isfinite(component_circularity)
        & np.isfinite(mean_age) & np.isfinite(mean_metal)
        & np.isfinite(mean_alpha) & (component_mass > 0.0)
        & (mass_fraction >= min_mass_fraction))

    if not np.any(valid_component):
        raise RuntimeError("No valid fitted components remain to plot.")

    max_mass = float(np.max(component_mass[valid_component]))
    marker_scale = np.divide(component_mass, max_mass,
        out=np.zeros(C, dtype=np.float64), where=max_mass > 0.0)
    marker_size = 10.0+90.0*np.sqrt(marker_scale)

    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0), sharex="col")
    family_colors = ("tab:blue", "tab:orange", "tab:green")

    properties = (
        (mean_age, r"$\langle t\rangle\ [{\rm Gyr}]$"),
        (mean_metal, r"$\langle[Z/H]\rangle$"),
        (mean_alpha, r"$\langle[\alpha/{\rm Fe}]\rangle$"),
    )

    for ff, (type_id, label, _, lambda_label) in enumerate(families):
        family = valid_component & (otypes == type_id)
        inds = np.flatnonzero(family)

        if inds.size:
            order = np.argsort(component_circularity[inds])
            inds = inds[order]

        for rr, (values, ylabel) in enumerate(properties):
            ax = axes[rr, ff]

            if inds.size:
                ax.scatter(component_circularity[inds], values[inds],
                    s=marker_size[inds], color=family_colors[ff],
                    edgecolor="none", alpha=0.8, zorder=3)

                if inds.size > 1:
                    ax.plot(component_circularity[inds], values[inds],
                        color=family_colors[ff], lw=0.7, alpha=0.35,
                        zorder=2)

            ax.axvline(0.0, color="0.5", lw=0.6, linestyle=":")
            ax.set_xlim(-1.05, 1.05)

            if ff == 0:
                ax.set_ylabel(ylabel)

            if rr == 0:
                ax.set_title(label, color=family_colors[ff],
                    fontweight="bold")

            if rr == 2:
                ax.set_xlabel(lambda_label)

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.075, top=0.95,
        hspace=0.12, wspace=0.18)

    if save_path is not None:
        save_path = plp.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
            pad_inches=0.03)

    return {
        "figure": fig,
        "component_numbers": nzComp,
        "component_types": otypes,
        "component_circularity": component_circularity,
        "component_radius": component_radius,
        "component_orbit_weight": component_orbit_weight,
        "component_mass": component_mass,
        "component_mass_fraction": mass_fraction,
        "mean_age": mean_age,
        "mean_metallicity": mean_metal,
        "mean_alpha": mean_alpha,
        "valid_component": valid_component,
    }

# ------------------------------------------------------------------------------

def plot_population_radial_profiles(arSOL, umetals, uages, ualphas, intData,
    cRadius, otypes, dataMax, save_path=None):
    """
    Plot intrinsic cylindrical radial profiles of the fitted stellar
    populations for the three orbital families.

    The complete fitted SSP distribution of each dynamical component is
    projected into intrinsic cylindrical-radius bins according to that
    component's intrinsic fitted-mass distribution. Components are then
    combined separately for z-tubes, x-tubes, and box orbits before computing
    population-weighted age, metallicity, and alpha enhancement.

    Parameters
    ----------
    arSOL : ndarray
        Fitted SSP coefficients with shape
        ``(C, nMetals, nAges, nAlphas)``.
    umetals : ndarray
        SSP metallicities with shape ``(nMetals,)``.
    uages : ndarray
        SSP ages with shape ``(nAges,)``.
    ualphas : ndarray
        SSP alpha enhancements with shape ``(nAlphas,)``.
    intData : ndarray
        Intrinsic component data with shape ``(18, C, nIntrinsic)``.
        Index 1 is fitted mass, index 3 is intrinsic X, and index 4 is
        intrinsic Y.
    cRadius : ndarray
        Cylindrical radial-bin coordinates constructed by ``loadCubeFit``.
        These are treated in the same way as the radial coordinates used by
        the existing intrinsic-profile machinery.
    otypes : ndarray
        Intrinsic component types: 0 for z-tubes, 1 for x-tubes, and 2 for
        box orbits.
    save_path : pathlib.Path or str, optional
        Output figure path.

    Returns
    -------
    out : dict
        Figure, radial coordinates, family-resolved SSP distributions,
        population masses, and radial population properties.

    Raises
    ------
    ValueError
        If supplied arrays have inconsistent dimensions.
    RuntimeError
        If no positive fitted population or intrinsic radial mass is
        available.

    Examples
    --------
    >>> out = plot_population_radial_profiles(
    ...     arSOL, umetals, uages, ualphas, intData, cRadius, otypes,
    ...     save_path="population_radial_profiles.png")
    """
    arSOL = np.asarray(arSOL, dtype=np.float64)
    umetals = np.asarray(umetals, dtype=np.float64).ravel(order="C")
    uages = np.asarray(uages, dtype=np.float64).ravel(order="C")
    ualphas = np.asarray(ualphas, dtype=np.float64).ravel(order="C")
    cRadius = np.asarray(cRadius, dtype=np.float64).ravel(order="C")
    otypes = np.asarray(otypes, dtype=np.int64).ravel(order="C")
    intData = np.ma.asarray(intData, dtype=np.float64)
    dataMax = float(np.nanmax(np.asarray(dataMax, dtype=np.float64)))

    if arSOL.ndim != 4:
        raise ValueError(
            "arSOL must have shape (C, nMetals, nAges, nAlphas).")

    C, nMetals, nAges, nAlphas = map(int, arSOL.shape)

    if umetals.size != nMetals:
        raise ValueError("umetals is inconsistent with arSOL.")
    if uages.size != nAges:
        raise ValueError("uages is inconsistent with arSOL.")
    if ualphas.size != nAlphas:
        raise ValueError("ualphas is inconsistent with arSOL.")
    if otypes.size != C:
        raise ValueError("otypes is inconsistent with arSOL.")
    if intData.ndim != 3 or intData.shape[0] < 5:
        raise ValueError("intData must have shape (>=5, C, nIntrinsic).")
    if intData.shape[1] != C:
        raise ValueError("intData component dimension is inconsistent "
            "with arSOL.")
    if cRadius.size < 1:
        raise ValueError("cRadius is empty.")
    if not np.all(np.isfinite(cRadius)):
        raise ValueError("cRadius contains non-finite values.")
    if np.any(np.diff(cRadius) <= 0.0):
        raise ValueError("cRadius must be strictly increasing.")
    if not np.all(np.isin(otypes, (0, 1, 2))):
        raise ValueError(
            f"Invalid intrinsic orbital types: {np.unique(otypes)}.")

    component_mass = np.sum(arSOL, axis=(1, 2, 3))
    total_population_mass = float(np.sum(component_mass))

    if not np.isfinite(total_population_mass) or total_population_mass <= 0.0:
        raise RuntimeError("The fitted solution has no positive population "
            "mass.")

    # Match the existing broadBetaCompsCyl convention: cRadius gives the
    # lower radial coordinates and one open outer edge catches all remaining
    # intrinsic cells.
    radial_edges = np.insert(cRadius, 0, 0.0)
    nRad = int(cRadius.size)
    radial_coordinates = np.sqrt(radial_edges[:-1]*radial_edges[1:])
    radial_coordinates[0] = cRadius[0]

    XX = np.ma.asarray(intData[3], dtype=np.float64)
    YY = np.ma.asarray(intData[4], dtype=np.float64)
    fitMass = np.ma.asarray(intData[1], dtype=np.float64)
    intrinsic_radius = np.ma.sqrt(XX*XX+YY*YY)

    component_radial_mass = np.zeros((C, nRad), dtype=np.float64)

    for cc in range(C):
        radius = np.ma.asarray(intrinsic_radius[cc]).filled(np.nan)
        mass = np.ma.asarray(fitMass[cc]).filled(0.0)

        valid = np.isfinite(radius) & np.isfinite(mass) & (mass > 0.0)

        if not np.any(valid):
            continue

        radius_valid = radius[valid]
        mass_valid = mass[valid]

        for rr in range(nRad):
            if rr == 0:
                wss = ((radius_valid >= radial_edges[rr])
                    & (radius_valid <= radial_edges[rr+1]))
            else:
                wss = ((radius_valid > radial_edges[rr])
                    & (radius_valid <= radial_edges[rr+1]))

            if np.any(wss):
                component_radial_mass[cc, rr] = np.sum(mass_valid[wss])

        radial_total = float(np.sum(component_radial_mass[cc]))

    # ------------------------------------------------------------------
    # Component stellar-population properties.
    #
    # arSOL determines the mean population carried by each fitted dynamical
    # component. The absolute coefficient normalization is deliberately
    # removed here; intrinsic radial weighting is supplied by fitMass below,
    # exactly analogous to the aperMass weighting used by the 2-D maps.
    # ------------------------------------------------------------------
    component_mass = np.sum(arSOL, axis=(1, 2, 3))

    zz, tt, aa = np.meshgrid(
        umetals, uages, ualphas, indexing="ij")

    component_age = np.divide(
        np.sum(arSOL*tt[None, :, :, :], axis=(1, 2, 3)),
        component_mass,
        out=np.full(C, np.nan, dtype=np.float64),
        where=component_mass > 0.0)

    component_metal = np.divide(
        np.sum(arSOL*zz[None, :, :, :], axis=(1, 2, 3)),
        component_mass,
        out=np.full(C, np.nan, dtype=np.float64),
        where=component_mass > 0.0)

    component_alpha = np.divide(
        np.sum(arSOL*aa[None, :, :, :], axis=(1, 2, 3)),
        component_mass,
        out=np.full(C, np.nan, dtype=np.float64),
        where=component_mass > 0.0)

    valid_component = ((component_mass > 0.0)
        & np.isfinite(component_age)
        & np.isfinite(component_metal)
        & np.isfinite(component_alpha)
        & (np.sum(component_radial_mass, axis=1) > 0.0))

    if not np.any(valid_component):
        raise RuntimeError("No fitted component has positive intrinsic radial "
            "mass.")

    families = (
        (0, r"$z$-tube", "tab:blue"),
        (1, r"$x$-tube", "tab:orange"),
        (2, "Box orbits", "tab:green"),
    )

    nFamilies = len(families)

    family_radial_mass = np.zeros((nFamilies, nRad), dtype=np.float64)

    mean_age = np.full(
        (nFamilies, nRad), np.nan, dtype=np.float64)
    mean_metal = np.full(
        (nFamilies, nRad), np.nan, dtype=np.float64)
    mean_alpha = np.full(
        (nFamilies, nRad), np.nan, dtype=np.float64)

    sigma_age = np.full(
        (nFamilies, nRad), np.nan, dtype=np.float64)
    sigma_metal = np.full(
        (nFamilies, nRad), np.nan, dtype=np.float64)
    sigma_alpha = np.full(
        (nFamilies, nRad), np.nan, dtype=np.float64)

    component_properties = (component_age, component_metal, component_alpha)
    family_means = (mean_age, mean_metal, mean_alpha)
    family_sigmas = (sigma_age, sigma_metal, sigma_alpha)

    # ------------------------------------------------------------------
    # Intrinsic dynamical-mass weighting.
    #
    # At each radius the population property of an orbital family is the
    # fitMass-weighted mean of the fitted population properties of its
    # constituent dynamical components.
    # ------------------------------------------------------------------
    for ff, (type_id, _, _) in enumerate(families):
        family = valid_component & (otypes == type_id)

        if not np.any(family):
            continue

        weights = component_radial_mass[family]
        family_radial_mass[ff] = np.sum(weights, axis=0)

        for prop, mean, sigma in zip(
            component_properties, family_means, family_sigmas):

            numerator = np.sum(weights*prop[family, None], axis=0)

            mean[ff] = np.divide(numerator, family_radial_mass[ff],
                out=np.full(nRad, np.nan, dtype=np.float64),
                where=family_radial_mass[ff] > 0.0)

            variance = np.sum(weights*(prop[family, None]-mean[ff, None, :])**2,
                axis=0)

            sigma[ff] = np.sqrt(np.divide(variance, family_radial_mass[ff],
                out=np.full(nRad, np.nan, dtype=np.float64),
                where=family_radial_mass[ff] > 0.0))

    total_intrinsic_mass = float(np.sum(component_radial_mass))
    family_intrinsic_mass = float(np.sum(family_radial_mass))

    if not np.isclose(family_intrinsic_mass, total_intrinsic_mass,
        rtol=1e-12, atol=1e-12*max(1.0, total_intrinsic_mass)):
        raise RuntimeError("Orbital-family radial binning does not conserve "
            "intrinsic dynamical mass.")

    family_mass_fraction = np.divide(family_radial_mass,
        np.sum(family_radial_mass, axis=0)[None, :],
        out=np.zeros_like(family_radial_mass),
        where=np.sum(family_radial_mass, axis=0)[None, :] > 0.0)

    print("Population radial-profile dynamical-mass conservation: "
        f"{family_intrinsic_mass/total_intrinsic_mass:.16f}")

    # Plot at the radial-bin coordinates already constructed by loadCubeFit.
    # The same x-coordinate is used for all three properties.
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)

    properties = (
        (mean_age, sigma_age, r"$\langle t\rangle\ [{\rm Gyr}]$"),
        (mean_metal, sigma_metal, r"$\langle[Z/H]\rangle$"),
        (mean_alpha, sigma_alpha, r"$\langle[\alpha/{\rm Fe}]\rangle$"),
    )

    for pp, (values, sigma, ylabel) in enumerate(properties):
        ax = axes[pp]

        for ff, (_, label, color) in enumerate(families):
            valid = (np.isfinite(values[ff]) & np.isfinite(sigma[ff])
                & (family_radial_mass[ff] > 0.0))

            if not np.any(valid):
                continue

            radius = np.log10(radial_coordinates[valid])
            mean = values[ff, valid]
            std = sigma[ff, valid]

            ax.fill_between(radius, mean-std, mean+std, color=color,
                alpha=0.20, linewidth=0.0)

            ax.plot(radius, mean, color=color, lw=1.5, marker="o", ms=3.5,
                label=label)

        ax.axvline(np.log10(dataMax), color="k", ls="--", lw=1.0)
        ax.set_ylabel(ylabel)

        if pp < 2:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    axes[-1].set_xlabel(r"$\log_{10}(R_{\rm cyl}\ [{\rm arcsec}])$")
    axes[0].legend(frameon=False, loc="best")

    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.09, top=0.98,
        hspace=0.0)

    if save_path is not None:
        save_path = plp.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
            pad_inches=0.03)

    return {
        "figure": fig,
        "radial_coordinates": radial_coordinates,
        "radial_edges": radial_edges,
        "component_radial_mass": component_radial_mass,
        "family_radial_mass": family_radial_mass,
        "family_mass_fraction": family_mass_fraction,
        "component_mass": component_mass,
        "component_age": component_age,
        "component_metallicity": component_metal,
        "component_alpha": component_alpha,
        "mean_age": mean_age,
        "mean_metallicity": mean_metal,
        "mean_alpha": mean_alpha,
        "sigma_age": sigma_age,
        "sigma_metallicity": sigma_metal,
        "sigma_alpha": sigma_alpha,
        "valid_component": valid_component,
        "intrinsic_mass": total_intrinsic_mass,
    }

# ------------------------------------------------------------------------------
