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

def plot_diagnostic_jsonl_dashboard(
    jsonl_path: str,
    *,
    max_points: int | None = 5000,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (18.0, 14.0),
) -> tuple[
    plt.Figure,
    dict[str, plt.Axes],
    list[dict],
]:
    """
    Plot diagnostics for the flexible-amplitude hard-prior solver.

    The dashboard combines ``iter_pre``, ``iter_post``, setup, and final
    records emitted by the constrained streaming active-set solver. Records
    sharing an iteration number are merged so that pre-solve gradients and
    post-solve objective, support, and constraint diagnostics appear on the
    same iteration axis.

    The panels show:

    1. Global data-objective evolution and relative improvement.
    2. Fitted global amplitude and coefficient norms.
    3. Raw, constraint, and constrained reduced gradients.
    4. Active-set size, effective support, and orbit occupation.
    5. Promotions, failures, drops, and promotion survival.
    6. Hard-constraint residuals and alpha stationarity.
    7. Orbit-mass ratio history.
    8. Final orbit masses, targets, and relative residuals.
    9. Iteration runtime, ridge, and reduced-Hessian condition estimate.

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
        raise ValueError(
            "max_points must be positive or None."
        )

    def _load_records() -> list[dict]:
        parsed = []

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as handle:
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

    def _finite_scalar(
        record: dict,
        *keys: str,
        default: float = np.nan,
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

    def _vector(
        record: dict,
        *keys: str,
    ) -> np.ndarray | None:
        for key in keys:
            value = record.get(key)

            if value is None:
                continue

            try:
                array = np.asarray(
                    value,
                    dtype=np.float64,
                ).ravel(order="C")
            except (TypeError, ValueError):
                continue

            if array.size > 0:
                return array

        return None

    def _merge_iteration_records(
        raw_records: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Merge pre- and post-solve records for each iteration.

        Later records overwrite earlier scalar fields, while vector fields
        remain available from whichever phase emitted them most recently.
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

            current = merged_by_iter.setdefault(
                iteration,
                {
                    "iter": iteration,
                },
            )

            current.update(record)

            if kind == "iter_pre":
                current["_has_pre"] = True
            elif kind == "iter_post":
                current["_has_post"] = True

        merged = [
            merged_by_iter[key]
            for key in sorted(merged_by_iter)
        ]

        if max_points is not None:
            merged = merged[-int(max_points):]

        return merged, non_iteration

    def _series(
        merged: list[dict],
        *keys: str,
        default: float = np.nan,
    ) -> np.ndarray:
        return np.asarray(
            [
                _finite_scalar(
                    record,
                    *keys,
                    default=default,
                )
                for record in merged
            ],
            dtype=np.float64,
        )

    def _plot_finite(
        axis,
        x_values: np.ndarray,
        y_values: np.ndarray,
        label: str,
        *,
        absolute: bool = False,
        positive_log: bool = False,
        color: str | None = None,
        linestyle: str | None = None,
        **kwargs,
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

            axis.semilogy(
                x_plot[positive],
                np.maximum(y_plot[positive], eps),
                label=label,
                color=color,
                linestyle=linestyle,
                **kwargs,
            )
        else:
            axis.plot(
                x_plot,
                y_plot,
                label=label,
                color=color,
                linestyle=linestyle,
                **kwargs,
            )

    def _latest_vector(
        merged: list[dict],
        *keys: str,
    ) -> np.ndarray | None:
        for record in reversed(merged):
            value = _vector(
                record,
                *keys,
            )

            if value is not None:
                return value

        return None

    raw_records = _load_records()
    merged, non_iteration = _merge_iteration_records(
        raw_records
    )

    fig = plt.figure(
        figsize=figsize,
    )
    grid = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.36,
        wspace=0.30,
    )

    axes = {
        "objective": fig.add_subplot(grid[0, 0]),
        "amplitude": fig.add_subplot(grid[0, 1]),
        "gradients": fig.add_subplot(grid[0, 2]),
        "support": fig.add_subplot(grid[1, 0]),
        "promotions": fig.add_subplot(grid[1, 1]),
        "constraints": fig.add_subplot(grid[1, 2]),
        "orbit_history": fig.add_subplot(grid[2, 0]),
        "orbit_final": fig.add_subplot(grid[2, 1]),
        "numerics": fig.add_subplot(grid[2, 2]),
    }

    if not merged:
        for axis in axes.values():
            axis.set_axis_off()

        axes["objective"].set_axis_on()
        axes["objective"].text(
            0.5,
            0.5,
            "No iteration diagnostics found",
            ha="center",
            va="center",
            transform=axes["objective"].transAxes,
        )

        if save_path:
            fig.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        return fig, axes, raw_records

    iterations = np.asarray(
        [
            int(record["iter"])
            for record in merged
        ],
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Extract primary scalar histories
    # ------------------------------------------------------------------
    data_objective = _series(
        merged,
        "data_objective",
    )
    obj_old = _series(
        merged,
        "obj_old",
    )
    obj_new = _series(
        merged,
        "obj_new",
    )
    obj_gain = _series(
        merged,
        "obj_gain",
    )

    alpha = _series(
        merged,
        "alpha",
    )
    norm_old = _series(
        merged,
        "norm_old",
    )
    norm_new = _series(
        merged,
        "norm_new",
    )

    grad_total = _series(
        merged,
        "max_grad_total",
    )
    grad_data = _series(
        merged,
        "max_grad_data",
    )
    grad_orbit = _series(
        merged,
        "max_grad_orbit",
    )
    grad_promo = _series(
        merged,
        "max_grad_promotable",
        "max_grad_promo",
    )

    n_active = _series(
        merged,
        "n_active",
        "k_active",
        "active",
    )
    n_promoted = _series(
        merged,
        "n_promoted",
        default=0.0,
    )
    n_failed = _series(
        merged,
        "n_failed",
        default=0.0,
    )
    n_dropped = _series(
        merged,
        "n_dropped",
        default=0.0,
    )

    constraint_l1 = _series(
        merged,
        "orbit_constraint_l1",
        "orbit_resid_l1",
    )
    constraint_linf = _series(
        merged,
        "orbit_constraint_linf",
        "orbit_resid_linf",
    )
    alpha_stationarity = _series(
        merged,
        "alpha_stationarity",
    )

    iteration_time = _series(
        merged,
        "t_iter_sec",
    )
    ridge = _series(
        merged,
        "ridge",
    )
    eig_min = _series(
        merged,
        "emin",
    )
    eig_max = _series(
        merged,
        "emax",
    )

    mean_orbit_nz = np.full(
        iterations.shape,
        np.nan,
        dtype=np.float64,
    )
    mean_eff_support = np.full(
        iterations.shape,
        np.nan,
        dtype=np.float64,
    )
    max_top_share = np.full(
        iterations.shape,
        np.nan,
        dtype=np.float64,
    )

    orbit_ratio_rows = []
    orbit_ratio_iters = []
    orbit_count = None

    for index, record in enumerate(merged):
        orbit_nz = _vector(
            record,
            "orbit_nz",
        )
        orbit_eff = _vector(
            record,
            "orbit_eff_support",
        )
        orbit_top = _vector(
            record,
            "orbit_top_share",
        )

        if orbit_nz is not None:
            mean_orbit_nz[index] = float(
                np.nanmean(orbit_nz)
            )

        if orbit_eff is not None:
            mean_eff_support[index] = float(
                np.nanmean(orbit_eff)
            )

        if orbit_top is not None:
            max_top_share[index] = float(
                np.nanmax(orbit_top)
            )

        ratio = _vector(
            record,
            "orbit_ratio",
        )
        mass = _vector(
            record,
            "orbit_mass",
        )
        target = _vector(
            record,
            "orbit_target",
        )

        if (
            ratio is None
            and mass is not None
            and target is not None
            and mass.size == target.size
        ):
            ratio = np.divide(
                mass,
                target,
                out=np.full_like(
                    mass,
                    np.nan,
                ),
                where=np.abs(target) > 0.0,
            )

        if ratio is None:
            continue

        if orbit_count is None:
            orbit_count = int(ratio.size)

        if ratio.size != orbit_count:
            continue

        orbit_ratio_rows.append(
            ratio.copy()
        )
        orbit_ratio_iters.append(
            int(record["iter"])
        )

    # ------------------------------------------------------------------
    # Panel 1: data objective
    # ------------------------------------------------------------------
    axis = axes["objective"]

    _plot_finite(
        axis,
        iterations,
        data_objective,
        "Global data objective",
        lw=1.6, color="tab:blue"
    )
    _plot_finite(
        axis,
        iterations,
        obj_new,
        "Reduced objective",
        lw=1.0, color="tab:orange",
        alpha=0.75,
    )

    axis.set_title(
        "Data-fit objective"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Objective"
    )

    objective_for_gain = data_objective.copy()

    if not np.any(np.isfinite(objective_for_gain)):
        objective_for_gain = obj_new.copy()

    relative_gain = np.full_like(
        objective_for_gain,
        np.nan,
    )

    for index in range(
        1,
        objective_for_gain.size,
    ):
        previous = objective_for_gain[index - 1]
        current = objective_for_gain[index]

        if (
            np.isfinite(previous)
            and np.isfinite(current)
        ):
            relative_gain[index] = (
                previous - current
            ) / max(
                1.0,
                abs(previous),
            )

    objective_axis_right = axis.twinx()

    _plot_finite(
        objective_axis_right,
        iterations,
        relative_gain,
        "Relative gain",
        absolute=True,
        positive_log=True,
        lw=1.0, color="tab:green",
        alpha=0.65,
    )

    objective_axis_right.yaxis.set_major_locator(
        mticker.LogLocator(base=10, numticks=4)
    )
    objective_axis_right.yaxis.set_major_formatter(
        mticker.LogFormatterExponent(base=10)
    )
    objective_axis_right.tick_params(
        axis="y",
        colors="tab:green",
        labelsize=8,
        pad=2,
    )
    objective_axis_right.spines["right"].set_color("tab:green")
    objective_axis_right.set_ylabel(
        r"$\log_{10}|\text{Relative improvement}|$", color="tab:green"
    )

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = objective_axis_right.get_legend_handles_labels()

    if handles_left or handles_right:
        axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            fontsize=8,
            loc="best",
        )

    # ------------------------------------------------------------------
    # Panel 2: alpha and coefficient norms
    # ------------------------------------------------------------------
    axis = axes["amplitude"]

    _plot_finite(
        axis,
        iterations,
        norm_old,
        "Norm old",
        positive_log=True,
        lw=1.0, color="tab:blue",
        alpha=0.70,
    )
    _plot_finite(
        axis,
        iterations,
        norm_new,
        "Norm new",
        positive_log=True,
        lw=1.2, color="tab:orange",
    )

    axis.set_title(
        "Global amplitude and coefficient norm"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Coefficient norm"
    )

    amp_axis = axis.twinx()
    _plot_finite(
        amp_axis,
        iterations,
        alpha,
        "Fitted alpha",
        lw=1.7, color="tab:green"
    )

    amp_axis.tick_params(axis="y", colors="tab:green")
    amp_axis.spines["right"].set_color("tab:green")
    amp_axis.set_ylabel("Fitted alpha", color="tab:green")

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = amp_axis.get_legend_handles_labels()

    if handles_left or handles_right:
        axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            fontsize=8,
            loc="best",
        )

    # ------------------------------------------------------------------
    # Panel 3: reduced-gradient optimality
    # ------------------------------------------------------------------
    axis = axes["gradients"]

    _plot_finite(
        axis,
        iterations,
        grad_data,
        "Raw data gradient",
        absolute=True,
        positive_log=True,
        lw=1.2, color="tab:blue"
    )
    _plot_finite(
        axis,
        iterations,
        grad_orbit,
        "Constraint correction",
        absolute=True,
        positive_log=True,
        lw=1.2, color="tab:orange"
    )
    _plot_finite(
        axis,
        iterations,
        grad_total,
        "Constrained reduced gradient",
        absolute=True,
        positive_log=True,
        lw=1.6, color="tab:green"
    )
    _plot_finite(
        axis,
        iterations,
        grad_promo,
        "Promotable reduced gradient",
        absolute=True,
        positive_log=True,
        lw=1.1, color="tab:red",
        alpha=0.85,
    )

    axis.set_title(
        "Constrained optimality"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Absolute gradient"
    )
    axis.legend(
        fontsize=8,
        loc="best",
    )

    # ------------------------------------------------------------------
    # Panel 4: active and effective support
    # ------------------------------------------------------------------
    axis = axes["support"]

    _plot_finite(
        axis,
        iterations,
        n_active,
        "Active columns",
        lw=1.6, color="tab:blue"
    )
    _plot_finite(
        axis,
        iterations,
        mean_orbit_nz,
        "Mean nonzero/orbit",
        lw=1.2, color="tab:orange"
    )
    _plot_finite(
        axis,
        iterations,
        mean_eff_support,
        "Mean effective support",
        lw=1.2, color="tab:green"
    )

    axis.set_title(
        "Support size and diversity"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Count"
    )

    share_axis = axis.twinx()

    _plot_finite(
        share_axis,
        iterations,
        max_top_share,
        "Largest orbit top-share",
        lw=1.0, color="tab:red",
        alpha=0.70,
    )

    share_axis.set_ylim(
        0.0,
        1.05,
    )
    share_axis.tick_params(axis="y", colors="tab:red")
    share_axis.spines["right"].set_color("tab:red")
    share_axis.set_ylabel(
        "Maximum top-share", color="tab:red"
    )

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = share_axis.get_legend_handles_labels()

    if handles_left or handles_right:
        axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            fontsize=8,
            loc="best",
        )

    # ------------------------------------------------------------------
    # Panel 5: active-set churn
    # ------------------------------------------------------------------
    axis = axes["promotions"]

    _plot_finite(
        axis,
        iterations,
        n_promoted,
        "Promoted",
        lw=1.3, color="tab:blue"
    )
    _plot_finite(
        axis,
        iterations,
        n_failed,
        "Failed",
        lw=1.3, color="tab:orange"
    )
    _plot_finite(
        axis,
        iterations,
        n_dropped,
        "Dropped",
        lw=1.3, color="tab:green"
    )

    promotion_survival = np.divide(
        n_promoted - n_failed,
        n_promoted,
        out=np.full_like(
            n_promoted,
            np.nan,
        ),
        where=n_promoted > 0.0,
    )

    survival_axis = axis.twinx()

    _plot_finite(
        survival_axis,
        iterations,
        promotion_survival,
        "Promotion survival",
        lw=1.2, color="tab:red"
    )

    survival_axis.set_ylim(
        -0.05,
        1.05,
    )
    survival_axis.tick_params(axis="y", colors="tab:red")
    survival_axis.spines["right"].set_color("tab:red")
    survival_axis.set_ylabel(
        "Surviving fraction", color="tab:red"
    )

    axis.set_title(
        "Active-set churn"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Columns"
    )

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = (
        survival_axis.get_legend_handles_labels()
    )

    axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        fontsize=8,
        loc="best",
    )

    # ------------------------------------------------------------------
    # Panel 6: exact hard-constraint diagnostics
    # ------------------------------------------------------------------
    axis = axes["constraints"]

    _plot_finite(
        axis,
        iterations,
        constraint_l1,
        "Orbit residual L1",
        absolute=True,
        positive_log=True,
        lw=1.3, color="tab:blue"
    )
    _plot_finite(
        axis,
        iterations,
        constraint_linf,
        "Orbit residual Linf",
        absolute=True,
        positive_log=True,
        lw=1.3, color="tab:orange"
    )
    _plot_finite(
        axis,
        iterations,
        alpha_stationarity,
        "|shape dot lambda|",
        absolute=True,
        positive_log=True,
        lw=1.3, color="tab:green"
    )
    axis.yaxis.set_major_locator(
        mticker.LogLocator(base=10, numticks=3)
    )
    axis.yaxis.set_major_formatter(
        mticker.LogFormatterExponent(base=10)
    )
    axis.set_title(
        "Hard-prior feasibility"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Absolute residual"
    )
    axis.legend(
        fontsize=8,
        loc="best",
    )

    # ------------------------------------------------------------------
    # Panel 7: orbit-ratio history
    # ------------------------------------------------------------------
    axis = axes["orbit_history"]

    if orbit_ratio_rows:
        ratio_history = np.asarray(
            orbit_ratio_rows,
            dtype=np.float64,
        )
        ratio_iterations = np.asarray(
            orbit_ratio_iters,
            dtype=np.float64,
        )

        log_ratio = np.log10(
            np.maximum(
                ratio_history,
                eps,
            )
        )

        finite_log = log_ratio[
            np.isfinite(log_ratio)
        ]

        if finite_log.size > 0:
            limit = float(
                np.nanpercentile(
                    np.abs(finite_log),
                    99.0,
                )
            )
            limit = max(
                limit,
                1e-12,
            )
        else:
            limit = 1.0

        image = axis.imshow(
            log_ratio.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                float(ratio_iterations[0]),
                float(ratio_iterations[-1]),
                -0.5,
                float(log_ratio.shape[1]) - 0.5,
            ],
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(
            r"Orbit prior history: "
            r"$\log_{10}(M_c/(\alpha w_c))$"
        )
        axis.set_xlabel(
            "Iteration"
        )
        axis.set_ylabel(
            "Orbit index"
        )

        colorbar = fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(
            r"$\log_{10}(M_c/(\alpha w_c))$"
        )
    else:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No orbit-ratio history",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    # ------------------------------------------------------------------
    # Panel 8: final orbit comparison
    # ------------------------------------------------------------------
    axis = axes["orbit_final"]

    final_mass = _latest_vector(
        merged,
        "orbit_mass",
    )
    final_target = _latest_vector(
        merged,
        "orbit_target",
    )
    final_resid = _latest_vector(
        merged,
        "orbit_resid",
    )
    final_shape = _latest_vector(
        merged,
        "orbit_shape",
    )

    if final_mass is not None:
        orbit_indices = np.arange(
            final_mass.size,
            dtype=np.int64,
        )
        width = 0.38

        axis.bar(
            orbit_indices - width / 2.0,
            final_mass,
            width=width,
            label="Fitted mass",
        )

        if (
            final_target is not None
            and final_target.size == final_mass.size
        ):
            axis.bar(
                orbit_indices + width / 2.0,
                final_target,
                width=width,
                label=r"Target $\alpha w$",
            )

        axis.set_xlabel(
            "Orbit index"
        )
        axis.set_ylabel(
            "Physical mass"
        )

        if (
            final_resid is None
            and final_target is not None
            and final_target.size == final_mass.size
        ):
            final_resid = (
                final_mass - final_target
            )

        relative_residual = None

        if (
            final_resid is not None
            and final_resid.size == final_mass.size
            and final_target is not None
            and final_target.size == final_mass.size
        ):
            relative_residual = np.divide(
                final_resid,
                final_target,
                out=np.full_like(
                    final_resid,
                    np.nan,
                ),
                where=np.abs(final_target) > 0.0,
            )

            residual_axis = axis.twinx()
            residual_axis.plot(
                orbit_indices,
                relative_residual,
                marker=".",
                lw=1.0, color="tab:red",
                label="Relative residual",
            )
            residual_axis.axhline(
                0.0,
                lw=0.8, c='k',
                alpha=0.65,
            )
            residual_axis.tick_params(axis="y", colors="tab:red")
            residual_axis.spines["right"].set_color("tab:red")
            residual_axis.set_ylabel(
                "Relative residual", color="tab:red"
            )

        absolute_l1 = (
            float(np.sum(np.abs(final_resid)))
            if final_resid is not None
            else np.nan
        )
        absolute_linf = (
            float(np.max(np.abs(final_resid)))
            if final_resid is not None
            else np.nan
        )

        final_alpha = alpha[
            np.flatnonzero(np.isfinite(alpha))[-1]
        ] if np.any(np.isfinite(alpha)) else np.nan

        axis.set_title(
            "Final hard-prior match\n"
            f"alpha={final_alpha:.4e}, "
            f"L1={absolute_l1:.3e}, "
            f"Linf={absolute_linf:.3e}"
        )
        axis.legend(
            fontsize=8,
            loc="best",
        )
    else:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No final orbit masses",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    # ------------------------------------------------------------------
    # Panel 9: runtime and numerical conditioning
    # ------------------------------------------------------------------
    axis = axes["numerics"]

    _plot_finite(
        axis,
        iterations,
        iteration_time,
        "Iteration time",
        positive_log=True,
        lw=1.4, color="tab:blue"
    )
    _plot_finite(
        axis,
        iterations,
        ridge,
        "Ridge",
        positive_log=True,
        lw=1.1, color="tab:orange"
    )

    condition_estimate = np.divide(
        np.abs(eig_max),
        np.maximum(
            np.abs(eig_min),
            eps,
        ),
        out=np.full_like(
            eig_max,
            np.nan,
        ),
        where=(
            np.isfinite(eig_max)
            & np.isfinite(eig_min)
        ),
    )

    condition_axis = axis.twinx()

    _plot_finite(
        condition_axis,
        iterations,
        condition_estimate,
        "Condition estimate",
        positive_log=True,
        lw=1.2, color="tab:green"
    )

    axis.set_title(
        "Cost and reduced-system conditioning"
    )
    axis.set_xlabel(
        "Iteration"
    )
    axis.set_ylabel(
        "Seconds / ridge"
    )
    condition_axis.tick_params(axis="y", colors="tab:green")
    condition_axis.spines["right"].set_color("tab:green")
    condition_axis.set_ylabel(
        "Condition estimate", color="tab:green"
    )

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = (
        condition_axis.get_legend_handles_labels()
    )

    if handles_left or handles_right:
        axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            fontsize=8,
            loc="best",
        )

    # ------------------------------------------------------------------
    # Figure-level summary
    # ------------------------------------------------------------------
    final_record = non_iteration.get(
        "objective_summary",
        {},
    )

    final_data_objective = _finite_scalar(
        final_record,
        "data_objective",
    )
    final_total_objective = _finite_scalar(
        final_record,
        "total_objective",
    )

    final_alpha_summary = _finite_scalar(
        final_record,
        "alpha",
    )

    if not np.isfinite(final_alpha_summary):
        finite_alpha = alpha[
            np.isfinite(alpha)
        ]
        final_alpha_summary = (
            float(finite_alpha[-1])
            if finite_alpha.size
            else np.nan
        )

    title_parts = [
        "CubeFit constrained streaming diagnostics",
        f"iterations={iterations[0]}-{iterations[-1]}",
    ]

    if np.isfinite(final_alpha_summary):
        title_parts.append(
            f"alpha={final_alpha_summary:.6e}"
        )

    if np.isfinite(final_data_objective):
        title_parts.append(
            f"data objective={final_data_objective:.6e}"
        )
    elif np.isfinite(final_total_objective):
        title_parts.append(
            f"objective={final_total_objective:.6e}"
        )

    fig.suptitle(
        " | ".join(title_parts),
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.975),
    )

    if save_path:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes, raw_records

# ------------------------------------------------------------------------------
