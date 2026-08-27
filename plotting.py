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

def plot_diagnostic_jsonl_dashboard(jsonl_path: str, *,
    max_points: int | None = 5000, save_path: str | None = None,
    show: bool = False, figsize: tuple[float, float] = (18.0, 14.0),
) -> list[dict]:
    """
    Plot diagnostics for the flexible-amplitude hard-prior solver.

    The dashboard combines ``iter_pre``, ``iter_post``, setup, and final
    records emitted by the constrained streaming active-set solver. Records
    sharing an iteration number are merged so that pre-solve gradients and
    post-solve objective, support, and constraint diagnostics appear on the
    same iteration axis.

    The panels show:

    1. Global data-objective evolution and relative improvement.
    2. Physical solution-vector norm and relative solution change.
    3. Active and inactive constrained KKT convergence.
    4. Active-set size, physical nnz(x), effective support, and concentration.
    5. Promotions, failures, survival, exploration, and near-noop events.
    6. Hard-prior residuals and alpha stationarity.
    7. Promotion eligibility, cooldown state, and eligible gradients.
    8. Final orbit masses, targets, and relative residuals.
    9. Iteration runtime, ridge, and reduced-system conditioning.

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
        parsed = []

        with open(jsonl_path, "r", encoding="utf-8") as handle:
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
        Merge solver diagnostics by iteration without losing event semantics.
        """
        merged_by_iter: dict[int, dict] = {}
        non_iteration: dict[str, dict] = {}

        for record in raw_records:
            kind = str(record.get("kind", ""))

            try:
                iteration = int(record["iter"])
            except (KeyError, TypeError, ValueError):
                if kind:
                    non_iteration[kind] = dict(record)
                continue

            current = merged_by_iter.setdefault(iteration,
                {"iter": iteration})

            if kind in {"iter_pre", "iter_post"}:
                current.update(record)
                current[f"_has_{kind}"] = True
                continue

            prefix = f"{kind}_" if kind else "event_"
            for key, value in record.items():
                if key not in {"kind", "iter"}:
                    current[f"{prefix}{key}"] = value

            if kind:
                current[f"_has_{kind}"] = True

        merged = [merged_by_iter[key] for key in sorted(merged_by_iter)]

        if max_points is not None:
            merged = merged[-int(max_points):]

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

    raw_records = _load_records()
    merged, non_iteration = _merge_iteration_records(raw_records)

    fig = plt.figure(figsize=figsize,)
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.35)

    axes = {
        "objective": fig.add_subplot(grid[0, 0]),
        "amplitude": fig.add_subplot(grid[0, 1]),
        "gradients": fig.add_subplot(grid[0, 2]),
        "support": fig.add_subplot(grid[1, 0]),
        "promotions": fig.add_subplot(grid[1, 1]),
        "constraints": fig.add_subplot(grid[1, 2]),
        "eligibility": fig.add_subplot(grid[2, 0]),
        "orbit_final": fig.add_subplot(grid[2, 1]),
        "numerics": fig.add_subplot(grid[2, 2]),
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
    # Extract primary scalar histories
    # ------------------------------------------------------------------
    data_objective = _series(merged, "data_objective")
    obj_old = _series(merged, "obj_old")
    obj_new = _series(merged, "obj_new")
    obj_gain = _series(merged, "obj_gain")
    rel_gain = _series(merged, "promotion_outcome_rel_gain")

    alpha = _series(merged, "alpha")
    norm_old = _series(merged, "norm_old")
    norm_new = _series(merged, "norm_new")
    z_step_rel = _series(merged, "promotion_outcome_z_step_rel")

    grad_total = _series(merged, "max_grad_total")
    grad_active = _series(merged, "max_grad_active")
    grad_inactive = _series(merged, "max_grad_inactive")
    grad_promo = _series(merged, "max_grad_promo")
    tol_here = _series(merged, "promotion_attempt_tol_here")

    n_active = _series(merged, "n_active", "k_active", "active")
    n_promoted = _series(merged, "n_promoted", default=0.0)
    n_failed = _series(merged, "n_failed", default=0.0)
    n_dropped = _series(merged, "n_dropped", default=0.0)
    n_survived = _series(merged, "promotion_outcome_n_survived",
        default=0.0)

    n_candidates = _series(merged, "promotion_attempt_n_candidates")
    n_eligible = _series(merged, "promotion_attempt_n_eligible")
    n_cooldown = _series(merged, "promotion_attempt_n_cooldown")
    n_col_cooldown = _series(merged,
        "promotion_attempt_n_col_cooldown")
    n_orbit_cooldown = _series(merged,
        "promotion_attempt_n_orbit_cooldown")

    grad_eligible = _series(merged,
        "promotion_attempt_max_grad_eligible")
    score_eligible = _series(merged,
        "promotion_attempt_max_score_eligible")

    constraint_l1 = _series(merged, "orbit_constraint_l1",
        "orbit_resid_l1")
    constraint_l2 = _series(merged, "orbit_constraint_l2",
        "orbit_resid_l2")
    constraint_linf = _series(merged, "orbit_constraint_linf",
        "orbit_resid_linf")
    alpha_stationarity = _series(merged, "alpha_stationarity")

    iteration_time = _series(merged, "t_iter_sec")
    ridge = _series(merged, "ridge")
    eig_min = _series(merged, "emin")
    eig_max = _series(merged, "emax")

    did_explore = np.asarray([
        bool(record.get("promotion_attempt_did_explore",
            record.get("did_explore", False))) for record in merged],
        dtype=bool)

    force_explore = np.asarray([
        bool(record.get("promotion_attempt_force_explore", False))
        for record in merged], dtype=bool)

    near_noop = np.asarray([
        bool(record.get("promotion_outcome_near_noop", False))
        for record in merged], dtype=bool)

    rejected = np.asarray([
        bool(record.get("reject", False)) for record in merged],
        dtype=bool)

    mean_orbit_nz = np.full(iterations.shape, np.nan, dtype=np.float64)
    mean_eff_support = np.full(iterations.shape, np.nan, dtype=np.float64)
    max_top_share = np.full(iterations.shape, np.nan, dtype=np.float64)

    # ------------------------------------------------------------------
    # Physical solution-vector histories
    # ------------------------------------------------------------------
    n_iter = iterations.size

    x_norm = np.full(n_iter, np.nan, dtype=np.float64)
    x_rel_step = np.full(n_iter, np.nan, dtype=np.float64)
    x_nnz = np.full(n_iter, np.nan, dtype=np.float64)
    x_eff_support = np.full(n_iter, np.nan, dtype=np.float64)
    x_top_share = np.full(n_iter, np.nan, dtype=np.float64)
    x_min = np.full(n_iter, np.nan, dtype=np.float64)

    previous_x = None

    for index, record in enumerate(merged):
        x_vec = _vector(record, "x", "x_current")

        if x_vec is None:
            continue

        finite = np.isfinite(x_vec)

        if not np.all(finite):
            continue

        norm = float(np.linalg.norm(x_vec))
        mass = float(np.sum(x_vec))
        sq_mass = float(np.dot(x_vec, x_vec))

        x_norm[index] = norm
        x_nnz[index] = float(
            np.count_nonzero(x_vec > 0.0)
        )
        x_min[index] = float(np.min(x_vec))

        if sq_mass > 0.0:
            x_eff_support[index] = mass * mass / sq_mass

        if mass > 0.0:
            x_top_share[index] = float(np.max(x_vec)) / mass

        if (previous_x is not None
            and previous_x.size == x_vec.size
        ):
            dx = x_vec - previous_x

            x_rel_step[index] = float(np.linalg.norm(dx))/ max(norm, eps)

        previous_x = x_vec.copy()

    for index, record in enumerate(merged):
        orbit_nz = _vector(record, "orbit_nz")
        orbit_eff = _vector(record, "orbit_eff_support")
        orbit_top = _vector(record, "orbit_top_share")

        if orbit_nz is not None:
            mean_orbit_nz[index] = float(np.nanmean(orbit_nz))

        if orbit_eff is not None:
            mean_eff_support[index] = float(np.nanmean(orbit_eff))

        if orbit_top is not None:
            max_top_share[index] = float(np.nanmax(orbit_top))

        mass = _vector(record, "orbit_mass")
        target = _vector(record, "orbit_target")

    # ------------------------------------------------------------------
    # Panel 1: objective convergence
    # ------------------------------------------------------------------
    axis = axes["objective"]

    _plot_finite(axis, iterations, data_objective, "Global data objective",
        lw=1.6, color="tab:blue")
    _plot_finite(axis, iterations, obj_new, "Reduced objective",
        lw=1.0, color="tab:orange", alpha=0.75)

    axis.set_title("Objective convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective")
    _homogenise_ticks(axis)

    gain_axis = axis.twinx()
    _plot_finite(gain_axis, iterations, np.abs(rel_gain),
        "Reduced relative gain", positive_log=True, lw=1.1,
        color="tab:green")

    gain_axis.set_ylabel("Relative objective gain", color="tab:green")
    gain_axis.tick_params(axis="y", colors="tab:green")
    gain_axis.spines["right"].set_color("tab:green")
    _homogenise_ticks(gain_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = gain_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right,
        fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 2: solution-vector convergence
    # ------------------------------------------------------------------
    axis = axes["amplitude"]

    _plot_finite(axis, iterations, x_norm, r"$||x||_2$",
        positive_log=True, lw=1.5, color="tab:blue")

    axis.set_title("Solution-vector convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(r"$||x||_2$", color="tab:blue")
    axis.tick_params(axis="y", which="both", colors="tab:blue")
    axis.spines["left"].set_color("tab:blue")
    _homogenise_ticks(axis)

    step_axis = axis.twinx()

    _plot_finite(step_axis, iterations, x_rel_step,
        r"$||\Delta x||_2/||x||_2$", positive_log=True, lw=1.3,
        color="tab:orange")
    _plot_finite(step_axis, iterations, z_step_rel,
        r"$||\Delta z||_2/||z||_2$", positive_log=True, lw=1.1,
        color="tab:green", alpha=0.8)

    step_axis.set_ylabel("Relative change", color="tab:orange")
    step_axis.tick_params(axis="y", which="both", colors="tab:orange")
    step_axis.spines["right"].set_color("tab:orange")
    _homogenise_ticks(step_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = step_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right,
        fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 3: constrained KKT convergence
    # ------------------------------------------------------------------
    axis = axes["gradients"]

    _plot_finite(axis, iterations, grad_active,
        "Active stationarity", absolute=True, positive_log=True,
        lw=1.4, color="tab:blue")
    _plot_finite(axis, iterations, np.maximum(grad_inactive, 0.0),
        "Inactive violation", positive_log=True, lw=1.4,
        color="tab:orange")
    _plot_finite(axis, iterations, np.maximum(grad_promo, 0.0),
        "Promotable violation", positive_log=True, lw=1.0,
        color="tab:green", alpha=0.75)
    _plot_finite(axis, iterations, tol_here, "KKT tolerance",
        positive_log=True, lw=1.0, color="black", linestyle="--")

    axis.set_title("Constrained KKT convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("KKT violation")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 4: active and effective support
    # ------------------------------------------------------------------
    axis = axes["support"]

    _plot_finite(axis, iterations, n_active,
        "Active columns", lw=1.6, color="tab:blue")
    _plot_finite(axis, iterations, mean_orbit_nz,
        "Mean nonzero/orbit", lw=1.2, color="tab:orange")
    _plot_finite(axis, iterations, mean_eff_support,
        "Mean effective support", lw=1.2, color="tab:green")
    _plot_finite(axis, iterations, x_nnz,
        "nnz(x)", lw=1.2, color="tab:purple", alpha=0.85)

    axis.set_title("Support evolution")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Count")
    _homogenise_ticks(axis)

    share_axis = axis.twinx()

    _plot_finite(share_axis, iterations, max_top_share,
        "Largest orbit top-share", lw=1.0, color="tab:red", alpha=0.70)
    _plot_finite(share_axis, iterations, x_top_share,
        "max(x) / sum(x)", lw=1.2, color="tab:brown", alpha=0.85)

    share_axis.set_ylim(0.0, 1.05)
    share_axis.tick_params(axis="y", which="both", colors="tab:red")
    share_axis.spines["right"].set_color("tab:red")
    share_axis.set_ylabel("Maximum top-share", color="tab:red")
    _homogenise_ticks(share_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = share_axis.get_legend_handles_labels()

    if handles_left or handles_right:
        axis.legend(handles_left + handles_right,
            labels_left + labels_right,
            fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 5: active-set dynamics
    # ------------------------------------------------------------------
    axis = axes["promotions"]

    _plot_finite(axis, iterations, n_promoted, "Promoted",
        lw=1.3, color="tab:blue")
    _plot_finite(axis, iterations, n_failed, "Failed",
        lw=1.3, color="tab:orange")
    _plot_finite(axis, iterations, n_dropped, "Dropped",
        lw=1.3, color="tab:green")

    explore_mask = did_explore | force_explore
    if np.any(explore_mask):
        axis.scatter(iterations[explore_mask], n_promoted[explore_mask],
            marker="x", s=28, color="tab:red", label="Exploration", zorder=5)

    if np.any(near_noop):
        axis.scatter(iterations[near_noop], n_promoted[near_noop], marker="o",
            s=38, facecolors="none", edgecolors="black", label="Near-noop",
            zorder=6)

    if np.any(rejected):
        axis.scatter(iterations[rejected], n_promoted[rejected], marker="s",
            s=30, facecolors="none", edgecolors="tab:red", label="Rejected",
            zorder=6)

    axis.set_title("Active-set dynamics")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Columns")
    _homogenise_ticks(axis)

    survival_axis = axis.twinx()
    promotion_survival = np.divide(n_survived, n_promoted,
        out=np.full_like(n_promoted, np.nan), where=n_promoted > 0.0)

    _plot_finite(survival_axis, iterations, promotion_survival,
        "Promotion survival", lw=1.2, color="tab:purple")

    survival_axis.set_ylim(-0.05, 1.05)
    survival_axis.set_ylabel("Surviving fraction", color="tab:purple")
    survival_axis.tick_params(axis="y", which="both", colors="tab:purple")
    survival_axis.spines["right"].set_color("tab:purple")
    _homogenise_ticks(survival_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = survival_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right,
        fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 6: hard-prior feasibility
    # ------------------------------------------------------------------
    axis = axes["constraints"]

    _plot_finite(axis, iterations, constraint_l1, "Orbit residual L1",
        absolute=True, positive_log=True, lw=1.3, color="tab:blue")
    _plot_finite(axis, iterations, constraint_l2, "Orbit residual L2",
        absolute=True, positive_log=True, lw=1.1, color="tab:purple")
    _plot_finite(axis, iterations, constraint_linf, "Orbit residual Linf",
        absolute=True, positive_log=True, lw=1.3, color="tab:orange")
    _plot_finite(axis, iterations, alpha_stationarity,
        r"$|w^T\lambda|$", absolute=True, positive_log=True, lw=1.2,
        color="tab:green")

    axis.set_title("Hard-prior feasibility")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Constraint residual")
    _homogenise_ticks(axis)
    axis.legend(fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 7: promotion eligibility and cooldown
    # ------------------------------------------------------------------
    axis = axes["eligibility"]

    _plot_finite(axis, iterations, n_candidates, "Inactive candidates",
        lw=1.3, color="tab:blue")
    _plot_finite(axis, iterations, n_eligible, "Eligible candidates",
        lw=1.3, color="tab:green")
    _plot_finite(axis, iterations, n_cooldown, "Combined cooldown",
        lw=1.2, color="tab:red")
    _plot_finite(axis, iterations, n_col_cooldown, "Column cooldown",
        lw=1.0, color="tab:orange", alpha=0.75)
    _plot_finite(axis, iterations, n_orbit_cooldown, "Orbit cooldown",
        lw=1.0, color="tab:purple", alpha=0.75)

    axis.set_title("Promotion eligibility and cooldown")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Candidate count")
    _homogenise_ticks(axis)

    gradient_axis = axis.twinx()

    _plot_finite(gradient_axis, iterations, grad_eligible,
        "Best eligible gradient", lw=1.3, color="tab:brown")
    _plot_finite(gradient_axis, iterations, score_eligible,
        "Best eligible score", lw=1.0, color="tab:pink", alpha=0.8)

    gradient_axis.axhline(0.0, lw=0.8, color="black", alpha=0.5)
    gradient_axis.set_ylabel("Eligible gradient / score")
    _homogenise_ticks(gradient_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = gradient_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right,
        fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Panel 8: final orbit comparison
    # ------------------------------------------------------------------
    axis = axes["orbit_final"]

    final_record = non_iteration.get("objective_summary", {})

    final_mass = _vector(final_record, "orbit_mass")
    final_target = _vector(final_record, "orbit_target")
    final_resid = _vector(final_record, "orbit_resid")

    if final_mass is None:
        final_mass = _latest_vector(merged, "orbit_mass")
    if final_target is None:
        final_target = _latest_vector(merged, "orbit_target")
    if final_resid is None:
        final_resid = _latest_vector(merged, "orbit_resid")

    if final_mass is not None:
        orbit_indices = np.arange(final_mass.size, dtype=np.int64)
        width = 0.38

        axis.bar(orbit_indices - width / 2.0, final_mass, width=width,
            label="Fitted mass")

        if (final_target is not None and final_target.size == final_mass.size):
            axis.bar(orbit_indices + width / 2.0, final_target, width=width,
                label=r"Target $\alpha w$")

        axis.set_xlabel("Orbit index")
        axis.set_ylabel("Physical mass")

        if (final_resid is None and final_target is not None and
            final_target.size == final_mass.size):
            final_resid = final_mass - final_target

        relative_residual = None

        if (final_resid is not None and final_resid.size == final_mass.size
            and final_target is not None
            and final_target.size == final_mass.size):
            relative_residual = np.divide(final_resid, final_target,
                out=np.full_like(final_resid, np.nan),
                where=np.abs(final_target) > 0.0)

            residual_axis = axis.twinx()
            residual_axis.plot(orbit_indices, relative_residual, marker=".",
                lw=1.0, color="tab:red", label="Relative residual")
            residual_axis.axhline(0.0, lw=0.8, c='k', alpha=0.65)
            residual_axis.tick_params(axis="y", which="both", colors="tab:red")
            residual_axis.spines["right"].set_color("tab:red")
            residual_axis.set_ylabel("Relative residual", color="tab:red")
            _homogenise_ticks(residual_axis)

        absolute_l1 = (float(np.sum(np.abs(final_resid))) if final_resid is not
            None else np.nan)
        absolute_linf = (float(np.max(np.abs(final_resid))) if final_resid is not
            None else np.nan)

        axis.set_title(rf"$L_1={_fmt_sci(absolute_l1)},\quad "
            rf"L_\infty={_fmt_sci(absolute_linf)}$")
        print(f"Absolute L1: {absolute_l1} {_fmt_sci(absolute_l1)}, Absolute L-infinity: {absolute_linf} {_fmt_sci(absolute_linf)}")
        axis.legend(fontsize=8, loc="best")
    else:
        axis.set_axis_off()
        axis.text(0.5, 0.5, "No final orbit masses", ha="center", va="center",
            transform=axis.transAxes)

    # ------------------------------------------------------------------
    # Panel 9: runtime and numerical conditioning
    # ------------------------------------------------------------------
    axis = axes["numerics"]

    _plot_finite(axis, iterations, iteration_time, "Iteration time",
        positive_log=True, lw=1.4, color="tab:blue")
    _plot_finite(axis, iterations, ridge, "Ridge", positive_log=True,
        lw=1.1, color="tab:orange")

    axis.set_title("Runtime and numerical conditioning")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Seconds / ridge")
    _homogenise_ticks(axis)

    condition_estimate = np.divide(np.abs(eig_max),
        np.maximum(np.abs(eig_min), eps),
        out=np.full_like(eig_max, np.nan),
        where=np.isfinite(eig_max) & np.isfinite(eig_min))

    condition_axis = axis.twinx()

    _plot_finite(condition_axis, iterations, condition_estimate,
        "Condition estimate", positive_log=True, lw=1.2, color="tab:green")
    _plot_finite(condition_axis, iterations, np.abs(eig_min),
        r"$|\lambda_{\min}|$", positive_log=True, lw=1.0,
        color="tab:red", alpha=0.8)

    condition_axis.set_ylabel(
        r"Condition estimate / $|\lambda_{\min}|$")
    _homogenise_ticks(condition_axis)

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = condition_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right,
        fontsize=8, loc="best")

    # ------------------------------------------------------------------
    # Figure-level summary
    # ------------------------------------------------------------------
    final_record = non_iteration.get("objective_summary", {})

    final_data_objective = _finite_scalar(final_record, "data_objective")
    final_total_objective = _finite_scalar(final_record, "total_objective")

    final_alpha_summary = _finite_scalar(final_record, "alpha")
    if not np.isfinite(final_alpha_summary):
        finite_alpha = alpha[np.isfinite(alpha)]
        final_alpha_summary = (float(finite_alpha[-1]) if finite_alpha.size
            else np.nan)

    title_parts = ["CubeFit constrained streaming diagnostics",
        f"iterations={iterations[0]}-{iterations[-1]}"]

    if np.isfinite(final_alpha_summary):
        title_parts.append(f"alpha={final_alpha_summary:.6e}")

    if np.isfinite(final_data_objective):
        title_parts.append(f"data objective={final_data_objective:.6e}")
    elif np.isfinite(final_total_objective):
        title_parts.append(f"objective={final_total_objective:.6e}")

    for axis in axes.values():
        if axis.axison:
            axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5,
                integer=True))
            axis.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_compact))
            axis.tick_params(axis="x", which="major", labelsize=9)
    fig.suptitle(" | ".join(title_parts), fontsize=12, y=0.95)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close('all')
    return raw_records

# ------------------------------------------------------------------------------
