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
    show: bool = False, figsize: tuple[float, float] = (24.0, 13.0),
) -> list[dict]:
    """
    Plot diagnostics for the flexible-amplitude hard-prior solver.

    The dashboard combines ``iter_pre``, ``iter_post``, setup, and final
    records emitted by the constrained streaming active-set solver. Records
    sharing an iteration number are merged so that pre-solve gradients and
    post-solve objective, support, and constraint diagnostics appear on the
    same iteration axis.

    The panels show:

    1. Accepted and trial data-objective evolution.
    2. Exploration objective gains and acceptance threshold.
    3. Current-support KKT convergence.
    4. Inactive-column constrained-gradient state.
    5. Committed and trial support sizes.
    6. Exploration rounds and consecutive failures.
    7. Proposed, retained, and discarded trial columns.
    8. Remaining candidate search space.
    9. Fitted hard-orbit amplitude and numerical ridge.
    10. Orbit-amplitude stationarity.
    11. Cumulative and per-iteration runtime.
    12. Current-support stationarity versus exploration acceptance.

    Parameters
    ----------
    jsonl_path : str
        Path to the solver diagnostics JSONL file.
    max_points : int or None, optional
        Maximum number of merged iteration records to retain. ``None`` keeps
        all records.
    save_path : str or None, optional
        Output image path. If ``None``, the figure is not written.
    show : bool, optional
        If True, display the figure with ``plt.show()``.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated dashboard figure.
    axes : dict of str to matplotlib.axes.Axes
        Named subplot axes.
    records : list of dict
        Raw JSONL records successfully parsed from the file.

    Raises
    ------
    FileNotFoundError
        If ``jsonl_path`` does not exist.
    ValueError
        If ``max_points`` is not positive or ``None``.
    OSError
        If the input file cannot be read or the output cannot be written.

    Examples
    --------
    >>> fig, axes, records = plot_diagnostic_jsonl_dashboard(
    ...     "diagnostics.jsonl",
    ...     save_path="diagnostics_dashboard.png",
    ... )
    """
    eps = 1e-300

    if max_points is not None and int(max_points) <= 0:
        raise ValueError("max_points must be positive or None.")

    def _load_records() -> list[dict]:
        paths = (
            [jsonl_paths] if isinstance(jsonl_paths, str)
            else list(jsonl_paths)
        )
        parsed = []

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
                        parsed.append(record)

        return parsed

    def _finite_scalar(record: dict, *keys: str, default: float = np.nan
        ) -> float:
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

    def _vector(record: dict, *keys: str) -> np.ndarray | None:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            try:
                array = np.asarray(value, dtype=np.float64).ravel(order="C")
            except (TypeError, ValueError):
                continue
            if array.size > 0:
                return array

        return None

    def _merge_iteration_records(
        raw_records: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Merge exploration-solver diagnostics by outer iteration.

        Parameters
        ----------
        raw_records : list of dict
            Raw JSONL diagnostic records.

        Returns
        -------
        merged : list of dict
            One merged diagnostic record per outer iteration.
        non_iteration : dict
            Setup and final records without an iteration number.

        Raises
        ------
        None

        Examples
        --------
        >>> merged, non_iteration = _merge_iteration_records(records)
        """
        merged_by_iter: dict[int, dict] = {}
        non_iteration: dict[str, dict] = {}
        setup_records = []

        for record in raw_records:
            kind = str(record.get("kind", ""))

            try:
                iteration = int(record["iter"])
            except (KeyError, TypeError, ValueError):
                if kind:
                    non_iteration[kind] = dict(record)

                    if kind == "mono_setup":
                        setup_records.append(dict(record))

                continue

            current = merged_by_iter.setdefault(iteration,
                {"iter": iteration})

            if kind == "exploration_trial":
                current.setdefault("_exploration_trials", []).append(dict(record))

                # Retain the best successful trial as the scalar representative of this
                # iteration. The complete trial set remains available above.
                old_obj = _finite_scalar(current,
                    "exploration_trial_trial_total_obj",
                    "exploration_trial_trial_data_obj")
                new_obj = _finite_scalar(record,
                    "trial_total_obj",
                    "trial_data_obj")

                if not np.isfinite(old_obj) or (
                    np.isfinite(new_obj) and new_obj < old_obj):
                    for key, value in record.items():
                        if key not in {"kind", "iter"}:
                            current[f"exploration_trial_{key}"] = value

                current["_has_exploration_trial"] = True
                continue

            if kind == "kkt":
                for key, value in record.items():
                    if key not in {"kind", "iter"}:
                        current[key] = value

                current["_has_kkt"] = True
                continue

            prefix = f"{kind}_"

            for key, value in record.items():
                if key not in {"kind", "iter"}:
                    current[f"{prefix}{key}"] = value

            if kind:
                current[f"_has_{kind}"] = True

        merged = [
            merged_by_iter[key]
            for key in sorted(merged_by_iter)
        ]

        if max_points is not None:
            merged = merged[-int(max_points):]

        non_iteration["_setup_records"] = setup_records

        return merged, non_iteration

    def _series(
        merged: list[dict],
        *keys: str,
        default: float = np.nan,
    ) -> np.ndarray:
        return np.asarray([_finite_scalar(record, *keys, default=default)
            for record in merged], dtype=np.float64)

    def _plot_finite(axis, x_values: np.ndarray, y_values: np.ndarray,
        label: str, *, absolute: bool = False, positive_log: bool = False,
        color: str | None = None, linestyle: str | None = None, **kwargs
        ) -> None:
        x_array = np.asarray(x_values, dtype=np.float64)
        y_array = np.asarray(y_values, dtype=np.float64)

        mask = np.isfinite(x_array) & np.isfinite(y_array)
        if not np.any(mask):
            return

        x_plot = x_array[mask]
        y_plot = y_array[mask]

        if absolute:
            y_plot = np.abs(y_plot)

        if positive_log:
            positive = y_plot > 0.0
            if not np.any(positive):
                return

            axis.semilogy(x_plot[positive], np.maximum(y_plot[positive], eps),
                label=label, color=color, linestyle=linestyle, **kwargs)
        else:
            axis.plot(x_plot, y_plot, label=label, color=color,
                linestyle=linestyle, **kwargs)

    def _fmt_sci(value):
        """Format a scalar as compact scientific-notation mathtext."""
        if not np.isfinite(value):
            return r"\mathrm{nan}"
        if value == 0.0:
            return "0"

        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10.0 ** exponent
        return rf"{mantissa:.2f}\times 10^{{{exponent}}}"

    def _latest_vector(merged: list[dict], *keys: str) -> np.ndarray | None:
        for record in reversed(merged):
            value = _vector(record, *keys)

            if value is not None:
                return value

        return None

    def _trial_records(merged: list[dict]) -> list[dict]:
        """
        Flatten all exploration trials while retaining their iteration.

        Parameters
        ----------
        merged : list of dict
            Merged outer-iteration diagnostic records.

        Returns
        -------
        trials : list of dict
            Individual exploration-trial records.

        Raises
        ------
        None

        Examples
        --------
        >>> trials = _trial_records(merged)
        """
        trials = []

        for record in merged:
            iteration = int(record["iter"])

            for trial in record.get("_exploration_trials", []):
                item = dict(trial)
                item["_plot_iter"] = iteration
                trials.append(item)

        return trials

    def _trial_series(trials: list[dict], *keys: str, default: float = np.nan,
    ) -> np.ndarray:
        """
        Extract a scalar series from individual exploration trials.

        Parameters
        ----------
        trials : list of dict
            Exploration-trial records.
        *keys : str
            Keys searched in priority order.
        default : float, optional
            Value used when no finite scalar is available.

        Returns
        -------
        values : ndarray
            Extracted float64 values.

        Raises
        ------
        None

        Examples
        --------
        >>> values = _trial_series(trials, "trial_total_obj")
        """
        return np.asarray([_finite_scalar(record, *keys, default=default)
            for record in trials], dtype=np.float64)

    raw_records = _load_records()
    merged, non_iteration = _merge_iteration_records(raw_records)
    trials = _trial_records(merged)

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(3, 4, figure=fig)

    axes = {
        "objective": fig.add_subplot(grid[0, 0]),
        "improvement": fig.add_subplot(grid[0, 1]),
        "kkt": fig.add_subplot(grid[0, 2]),
        "dual": fig.add_subplot(grid[0, 3]),
        "support": fig.add_subplot(grid[1, 0]),
        "exploration": fig.add_subplot(grid[1, 1]),
        "support_changes": fig.add_subplot(grid[1, 2]),
        "candidate_space": fig.add_subplot(grid[1, 3]),
        "amplitude": fig.add_subplot(grid[2, 0]),
        "stationarity": fig.add_subplot(grid[2, 1]),
        "runtime": fig.add_subplot(grid[2, 2]),
        "search_state": fig.add_subplot(grid[2, 3]),
    }

    if not merged:
        for axis in axes.values():
            axis.set_axis_off()

        axes["objective"].set_axis_on()
        axes["objective"].text(0.5, 0.5,
            "No iteration diagnostics found", ha="center", va="center",
            transform=axes["objective"].transAxes)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return raw_records

    iterations = np.asarray([int(record["iter"]) for record in merged],
        dtype=np.int64)

    # ------------------------------------------------------------------
    # Extract exploration-solver histories
    # ------------------------------------------------------------------
    # Committed state at the start of each exploration iteration.
    data_objective = _series(merged,
        "exploration_trial_current_data_obj",
        "exploration_accept_data_objective_before")

    regularisation_objective = _series(merged,
        "exploration_trial_current_regularisation_obj")

    total_objective = _series(merged,
        "exploration_trial_current_total_obj")

    # Every joint-support trial is retained separately. This is essential now
    # that several candidate batches may be tested within one outer iteration.
    trial_iterations = np.asarray([int(record["_plot_iter"]) for record in trials
        ], dtype=np.int64)

    trial_numbers = _trial_series(trials, "trial")
    trial_data_objective = _trial_series(trials, "trial_data_obj")
    trial_regularisation_objective = _trial_series(
        trials, "trial_regularisation_obj")
    trial_total_objective = _trial_series(trials, "trial_total_obj")

    trial_data_improvement = _trial_series(trials, "data_improvement")
    trial_regularisation_change = _trial_series(
        trials, "regularisation_change")
    trial_total_improvement = _trial_series(
        trials, "total_improvement", "improvement")
    trial_best_improvement = _trial_series(
        trials, "best_total_improvement")

    trial_rel_data_improvement = _trial_series(
        trials, "relative_data_improvement")
    trial_rel_total_improvement = _trial_series(
        trials, "relative_total_improvement", "relative_improvement")

    trial_improvement_tol = _trial_series(trials, "improvement_tol")

    trial_accepted_all = _trial_series(
        trials, "accepted", default=0.0)
    trial_solve_accepted = _trial_series(
        trials, "solve_accepted", default=np.nan)

    trial_support_all = _trial_series(trials, "trial_n_active")
    proposal_n_all = _trial_series(trials, "proposal_n")

    alpha = _series(merged, "alpha")
    ridge = _series(merged, "ridge")
    grad_active = _series(merged, "max_grad_active")
    grad_inactive = _series(merged, "max_grad_inactive")
    grad_dual = _series(merged, "max_grad_dual")

    kkt_violation = _series(merged, "kkt_violation")
    kkt_tol = _series(merged, "kkt_tol", "tol")

    best_dual_value = _series(merged, "best_dual_value")
    n_active = _series(merged, "n_working_active",
        "exploration_accept_n_active_after", "exploration_n_active")
    n_positive = _series(merged, "n_positive")
    n_zero_free = _series(merged, "n_zero_free",)
    n_candidates = _series(merged, "exploration_n_candidates",)

    trial_support = _series(merged, "exploration_trial_trial_n_active",)
    proposal_n = _series(merged, "exploration_trial_proposal_n",)
    n_added = np.asarray([float(len(record.get(
        "exploration_accept_added_columns", [])))
        if "_has_exploration_accept" in record else np.nan
        for record in merged], dtype=np.float64)
    n_zero_dropped = _series(merged,"exploration_accept_n_zero_dropped",)

    explore_round = _series(merged, "explore_round",
        "exploration_explore_round", "exploration_trial_explore_round")

    explore_fail_count = _series(merged, "exploration_explore_fail_count",
        "explore_fail_count", default=0.0)

    support_stationary = _series(merged, "support_stationary",
        default=np.nan)

    trial_accepted = _series(merged, "exploration_trial_accepted",
        default=np.nan)

    shape_dot_lambda = _series(merged, "shape_dot_lambda",)

    elapsed_time = _series(merged, "t_sec",)

    constraint_l1 = _series(merged, "orbit_constraint_l1",
        "orbit_resid_l1")
    constraint_l2 = _series(merged, "orbit_constraint_l2",
        "orbit_resid_l2")
    constraint_linf = _series(merged, "orbit_constraint_linf",
        "orbit_resid_linf")
    alpha_stationarity = _series(merged, "alpha_stationarity")

    def _trial_gradient_max(record: dict, scalar_key: str, vector_key: str,
        ) -> float:
        value = _finite_scalar(record, scalar_key)

        if np.isfinite(value):
            return value

        vector = _vector(record, vector_key)

        if vector is None:
            return np.nan

        return float(np.max(np.abs(vector)))

    trial_grad_data_max = np.asarray([_trial_gradient_max(record,
            "proposal_grad_data_max", "proposal_grad_data")
        for record in trials], dtype=np.float64)

    trial_grad_constraint_max = np.asarray([_trial_gradient_max(record,
            "proposal_grad_constraint_max", "proposal_grad_constraint")
        for record in trials], dtype=np.float64)

    trial_grad_total_max = np.asarray([_trial_gradient_max(record,
            "proposal_grad_total_max","proposal_grad_total")
        for record in trials], dtype=np.float64)


    any_trial_accepted = np.asarray([float(any(bool(trial.get("accepted", False))
            for trial in record.get("_exploration_trials", [])))
        if record.get("_exploration_trials") else np.nan for record in merged
        ], dtype=np.float64)


    # ------------------------------------------------------------------
    # Panel 1: scientific-objective evolution
    # ------------------------------------------------------------------
    axis = axes["objective"]

    _plot_finite(axis, iterations, data_objective,
        "Committed data objective", lw=1.5, color="tab:blue")

    _plot_finite(axis, iterations, total_objective,
        "Committed total objective", lw=1.6, color="tab:green")

    # Show every tested joint-support solution rather than only the best
    # representative trial retained by the iteration merger.
    finite_trial = (np.isfinite(trial_iterations)
        & np.isfinite(trial_total_objective))

    if np.any(finite_trial):
        axis.scatter(trial_iterations[finite_trial],
            trial_total_objective[finite_trial], s=14, alpha=0.45,
            color="tab:orange", label="Trial total objective")

    accepted_trial = (finite_trial & np.isfinite(trial_accepted_all)
        & (trial_accepted_all > 0.5))

    if np.any(accepted_trial):
        axis.scatter(trial_iterations[accepted_trial],
            trial_total_objective[accepted_trial], s=28, marker="o",
            facecolors="none", edgecolors="tab:green", label="Accepted trial",
            zorder=4)

    axis.set_title("Scientific objective evolution")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$J_\mathrm{data} + J_\mathrm{reg}$")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 2: joint-support objective-gain decomposition
    # ------------------------------------------------------------------
    axis = axes["improvement"]

    _plot_finite(axis, trial_iterations, trial_data_improvement,
        "Data-fit gain", lw=1.0, color="tab:blue", alpha=0.75)

    _plot_finite(axis, trial_iterations, trial_regularisation_change,
        "Regularisation gain", lw=1.0, color="tab:orange", alpha=0.75)

    _plot_finite(axis, trial_iterations, trial_total_improvement,
        "Total scientific gain", lw=1.5, color="tab:green")

    axis.axhline(0.0, lw=0.9, color="black", linestyle=":",
        label="No improvement")

    # The tolerance is tiny compared with genuine improvements, but plotting
    # it makes numerical non-improvements such as ~1e-2 objective changes
    # immediately identifiable.
    finite_tol = (np.isfinite(trial_iterations)
        & np.isfinite(trial_improvement_tol))

    if np.any(finite_tol):
        axis.plot(trial_iterations[finite_tol],
            trial_improvement_tol[finite_tol], lw=1.0, color="tab:red",
            linestyle="--", label="Acceptance tolerance")

    axis.set_title("Joint re-solve gain decomposition")
    axis.set_ylabel(r"$J_\mathrm{current} - J_\mathrm{trial}$")
    axis.set_xlabel("Iteration")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 3: constrained KKT convergence
    # ------------------------------------------------------------------
    axis = axes["kkt"]

    kkt_ratio = np.divide(
        kkt_violation,
        kkt_tol,
        out=np.full_like(kkt_violation, np.nan),
        where=kkt_tol > 0.0,
    )

    active_ratio = np.divide(
        np.abs(grad_active),
        kkt_tol,
        out=np.full_like(grad_active, np.nan),
        where=kkt_tol > 0.0,
    )

    dual_ratio = np.divide(
        np.maximum(grad_dual, 0.0),
        kkt_tol,
        out=np.full_like(grad_dual, np.nan),
        where=kkt_tol > 0.0,
    )

    _plot_finite(
        axis,
        iterations,
        kkt_ratio,
        "Overall KKT / tolerance",
        positive_log=True,
        lw=1.6,
        color="tab:red",
    )

    _plot_finite(
        axis,
        iterations,
        active_ratio,
        "Positive stationarity / tolerance",
        positive_log=True,
        lw=1.2,
        color="tab:blue",
    )

    _plot_finite(
        axis,
        iterations,
        dual_ratio,
        "Zero-variable dual / tolerance",
        positive_log=True,
        lw=1.2,
        color="tab:orange",
    )

    axis.axhline(
        1.0,
        lw=1.0,
        color="black",
        linestyle="--",
        label="KKT boundary",
    )

    axis.set_title("Current-support KKT")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Violation / tolerance")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 4: candidate-screening decomposition
    # ------------------------------------------------------------------
    axis = axes["dual"]

    _plot_finite(axis, trial_iterations, trial_grad_data_max,
        "Data term", positive_log=True, lw=1.2, color="tab:blue", alpha=0.8)

    _plot_finite(axis, trial_iterations, trial_grad_constraint_max,
        "Orbit-constraint term", positive_log=True,lw=1.2, color="tab:orange",
        alpha=0.8)

    _plot_finite(axis, trial_iterations, trial_grad_total_max,
        "Combined screening gradient", positive_log=True, lw=1.5,
        color="tab:green")

    axis.set_title("Candidate screening")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"Maximum $|g|$ in proposal")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 5: active and effective support
    # ------------------------------------------------------------------
    axis = axes["support"]

    _plot_finite(
        axis,
        iterations,
        n_active,
        "Working support",
        lw=1.6,
        color="tab:blue",
    )

    _plot_finite(
        axis,
        iterations,
        n_positive,
        "Positive coefficients",
        lw=1.2,
        color="tab:green",
    )

    _plot_finite(
        axis,
        iterations,
        trial_support,
        "Trial support",
        lw=1.0,
        color="tab:orange",
        alpha=0.8,
    )

    axis.set_title("Support evolution")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 6: active-set dynamics
    # ------------------------------------------------------------------
    axis = axes["exploration"]

    _plot_finite(
        axis,
        iterations,
        explore_round,
        "Exploration round",
        lw=1.4,
        color="tab:blue",
    )

    _plot_finite(
        axis,
        iterations,
        explore_fail_count,
        "Consecutive failures",
        lw=1.4,
        color="tab:red",
    )

    setup_record = non_iteration.get("mono_setup", {})
    explore_patience = _finite_scalar(
        setup_record,
        "explore_fail_patience",
    )

    if np.isfinite(explore_patience):
        axis.axhline(
            explore_patience,
            lw=1.0,
            color="black",
            linestyle="--",
            label="Failure patience",
        )

    axis.set_title("Exploration progress")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Round / count")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 7: proposal and support survival
    # ------------------------------------------------------------------
    axis = axes["support_changes"]

    _plot_finite(axis, trial_iterations, proposal_n_all,
        "Columns tested", lw=1.0, color="tab:blue", alpha=0.65)

    _plot_finite(axis, iterations, n_added,
        "Columns retained", lw=1.4, color="tab:green")

    _plot_finite(axis, iterations, n_zero_dropped,
        "Trial columns returned to zero", lw=1.2, color="tab:orange")

    axis.set_title("Joint-support survival")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 8: hard-prior feasibility
    # ------------------------------------------------------------------
    axis = axes["candidate_space"]

    _plot_finite(axis, iterations, n_candidates, "Remaining candidates",
        lw=1.4, color="tab:blue")

    coverage_axis = axis.twinx()
    tested_fraction = np.divide(proposal_n, n_candidates,
        out=np.full_like(proposal_n, np.nan), where=n_candidates > 0.0,)
    _plot_finite(coverage_axis, iterations, 100.0 * tested_fraction,
        "Batch fraction", lw=1.2, color="tab:orange")

    axis.set_title("Candidate search effort")
    axis.set_ylabel("Remaining columns")
    coverage_axis.set_ylabel("Batch [% of candidates]", color="tab:orange")
    axis.set_xlabel("Iteration")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 9: hard-orbit amplitude and numerical stabilisation
    # ------------------------------------------------------------------
    axis = axes["amplitude"]

    _plot_finite(
        axis,
        iterations,
        alpha,
        r"Fitted $\alpha$",
        lw=1.5,
        color="tab:blue",
    )

    axis.set_title("Orbit amplitude & numerical stability")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$\alpha$")
    _homogenise_ticks(axis)

    ridge_axis = axis.twinx()

    _plot_finite(
        ridge_axis,
        iterations,
        ridge,
        "Numerical stabilisation",
        positive_log=True,
        lw=1.1,
        color="tab:orange",
    )

    ridge_axis.set_ylabel("Numerical ridge", color="tab:orange")
    ridge_axis.tick_params(axis="y", colors="tab:orange")
    _homogenise_ticks(ridge_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = ridge_axis.get_legend_handles_labels()

    axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        fontsize=8,
        loc="best",
    )

    # ------------------------------------------------------------------
    # Panel 10: orbit-level population structure
    # ------------------------------------------------------------------
    axis = axes["stationarity"]

    _plot_finite(
        axis,
        iterations,
        np.abs(shape_dot_lambda),
        r"$|w^T\lambda|$",
        positive_log=True,
        lw=1.4,
        color="tab:blue",
    )

    axis.set_title("Orbit-amplitude stationarity")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$|w^T\lambda|$")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 11: solver progress and stall state
    # ------------------------------------------------------------------
    axis = axes["runtime"]

    _plot_finite(
        axis,
        iterations,
        elapsed_time,
        "Elapsed runtime",
        lw=1.5,
        color="tab:blue",
    )

    iter_time = np.full_like(
        elapsed_time,
        np.nan,
    )

    if elapsed_time.size > 1:
        iter_time[1:] = np.diff(
            elapsed_time
        )
    time_axis = axis.twinx()

    _plot_finite(
        time_axis,
        iterations,
        iter_time,
        "Iteration time",
        lw=1.1,
        color="tab:orange",
    )

    time_axis.set_ylabel(
        "Seconds / iteration",
        color="tab:orange",
    )
    time_axis.tick_params(
        axis="y",
        colors="tab:orange",
    )
    _homogenise_ticks(time_axis)

    axis.set_title("Runtime")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Elapsed seconds")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 12: runtime and numerical conditioning
    # ------------------------------------------------------------------
    axis = axes["search_state"]

    _plot_finite(
        axis,
        iterations,
        support_stationary,
        "Support KKT stationary",
        lw=1.4,
        color="tab:blue",
    )

    _plot_finite(
        axis,
        iterations,
        any_trial_accepted,
        "Joint search improved",
        lw=1.2,
        color="tab:green",
    )

    axis.set_ylim(-0.05, 1.05)
    axis.set_yticks([0.0, 1.0])
    axis.set_yticklabels(["No", "Yes"])

    axis.set_title("Local stationarity vs joint search")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("State")
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Figure-level summary
    # ------------------------------------------------------------------
    final_record = non_iteration.get("objective_summary", {})

    final_data_objective = _finite_scalar(final_record, "data_objective")
    final_total_objective = _finite_scalar(final_record, "total_objective")

    final_scientific_obj = _finite_scalar(final_record,
        "scientific_regularisation_objective")
    final_numerical_obj = _finite_scalar(final_record,
        "numerical_ridge_objective")
    final_scientific_scale = _finite_scalar(final_record,
        "scientific_regularisation", "regularisation_scale")
    final_numerical_ridge = _finite_scalar(final_record, "numerical_ridge",
        "ridge")

    final_alpha_summary = _finite_scalar(final_record, "alpha")
    if not np.isfinite(final_alpha_summary):
        finite_alpha = alpha[np.isfinite(alpha)]
        final_alpha_summary = (float(finite_alpha[-1]) if finite_alpha.size
            else np.nan)

    title_parts = ["CubeFit constrained streaming diagnostics",
        f"iterations={iterations[0]}-{iterations[-1]}"]
    setup_record = non_iteration.get("mono_setup", {})
    setup_C = _finite_scalar(setup_record, "C")
    setup_P = _finite_scalar(setup_record, "P")
    setup_reg = _finite_scalar(setup_record, "regularisation_scale")
    if np.isfinite(setup_C) and np.isfinite(setup_P):
        title_parts.append(f"C={int(setup_C)}, P={int(setup_P)}")
    if np.isfinite(setup_reg):
        title_parts.append(
            rf"$\mathrm{{reg}}={_fmt_sci(setup_reg)}$")
    if np.isfinite(final_scientific_obj):
        title_parts.append(
            rf"$J_\mathrm{{reg}}={_fmt_sci(final_scientific_obj)}$"
        )

    finite_dual = grad_dual[np.isfinite(grad_dual)]
    finite_kkt = kkt_violation[
        np.isfinite(kkt_violation)
    ]
    finite_tol = kkt_tol[
        np.isfinite(kkt_tol)
    ]

    if (
        finite_kkt.size
        and finite_tol.size
        and finite_tol[-1] > 0.0
    ):
        final_kkt_ratio = (
            finite_kkt[-1]
            / finite_tol[-1]
        )

        title_parts.append(
            rf"$\mathrm{{KKT}}/\mathrm{{tol}}="
            rf"{_fmt_sci(final_kkt_ratio)}$"
        )
    finite_fail = explore_fail_count[
        np.isfinite(explore_fail_count)
    ]

    finite_round = explore_round[
        np.isfinite(explore_round)
    ]

    if finite_round.size:
        title_parts.append(
            f"explore={int(finite_round[-1])}"
        )

    if finite_fail.size:
        if np.isfinite(explore_patience):
            title_parts.append(
                f"fail={int(finite_fail[-1])}/"
                f"{int(explore_patience)}"
            )
        else:
            title_parts.append(
                f"fail={int(finite_fail[-1])}"
            )

    if np.isfinite(final_alpha_summary):
        title_parts.append(rf"$\alpha={_fmt_sci(final_alpha_summary)}$")

    if np.isfinite(final_data_objective):
        title_parts.append(rf"data objective=${_fmt_sci(final_data_objective)}$")
    elif np.isfinite(final_total_objective):
        title_parts.append(rf"objective=${_fmt_sci(final_total_objective)}$")

    for axis in axes.values():
        if axis.axison:
            axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5,
                integer=True))
            axis.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_compact))
            axis.tick_params(axis="x", which="major", labelsize=9)

    fig.suptitle(" | ".join(title_parts), fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.055, top=0.955,
        hspace=0.28, wspace=0.32)
    for axis in fig.axes:
        axis.xaxis.labelpad = 2
        axis.yaxis.labelpad = 2

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.04)
    if show:
        plt.show()

    plt.close('all')
    return raw_records

# ------------------------------------------------------------------------------
