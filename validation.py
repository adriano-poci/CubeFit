#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
    validation.py
    Adriano Poci
    University of Oxford
    2026

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Create a minimal CubeFit validation input containing synthetic spectra
    generated from the /ModelCube of a completed CubeFit fit.

    The original file is never modified.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   29 September 2026
v1.1:   Generalised to include all validation tests;
        Produce a minimal mock dataset, which does not need `/HyperCube` and
            other model products in `makeValidationCube`. 3 October 2026
v1.2:   Updated `makeValidationCube` to allow for noise in the mock data, either
            drawn from the data STAT cube, or from a previous fit's residuals. 4
            October 2026
"""
# need to set up the logger before any other imports
import pathlib as plp
from CubeFit.logger import get_logger
print("[CubeFit] Initializing CubeFit logger...")
curdir = plp.Path(__file__).parent
lfn = curdir/'kz_run.log'
logger = get_logger(lfn, mode='w')

import argparse, os, shutil
import h5py
import numpy as np
import pathlib as plp
import matplotlib.pyplot as plt

from CubeFit.hdf5_manager import open_h5

# ------------------------------------------------------------------------------

def makeValidationCube(source_path, output_path, noise="none",
    noise_scale=1.0, stat_cube=None, seed=42):
    """
    Create a minimal CubeFit validation input from an existing model cube.

    Synthetic spectra are generated from ``/ModelCube`` with optional noise.
    The source file is never modified. Noise may be omitted, drawn from a
    supplied observational 1-sigma error cube, or generated from the empirical
    residuals of the original CubeFit solution.

    Parameters
    ----------
    source_path : str
        Completed CubeFit HDF5 file containing /ModelCube and /ObsPix.
    output_path : str
        Output validation HDF5 path.
    noise : {'none', 'stat', 'residual'}, optional
        Noise model. ``'none'`` gives exact model closure. ``'stat'`` draws
        independent Gaussian noise using ``stat_cube`` as the per-pixel
        1-sigma uncertainty. ``'residual'`` randomly permutes the original
        DataCube-ModelCube residuals independently within each spectrum.
        Default is ``'none'``.
    noise_scale : float, optional
        Multiplicative scale applied to the generated noise. Default is 1.0.
    stat_cube : ndarray, optional
        Per-pixel 1-sigma observational uncertainties with shape (S, L).
        Required when ``noise='stat'``.
    seed : int, optional
        Random seed used to generate reproducible noise. Default is 42.

    Returns
    -------
    str
        Output filename.

    Raises
    ------
    FileNotFoundError
        If the source HDF5 file does not exist.
    RuntimeError
        If required source datasets are missing or invalid.
    ValueError
        If the paths, noise mode, noise scale, or supplied error cube are
        invalid.

    Examples
    --------
    >>> makeValidationCube(
    ...     "hypercube_3_01.h5", "hypercube_3_01_mock-data.h5")
    'hypercube_3_01_mock-data.h5'
    >>> makeValidationCube(
    ...     "hypercube_3_01.h5", "hypercube_3_01_mock-stat.h5",
    ...     noise="stat", stat_cube=statGrid.T, seed=42)
    'hypercube_3_01_mock-stat.h5'
    """
    source_path = os.path.abspath(str(source_path))
    output_path = os.path.abspath(str(output_path))

    if source_path == output_path:
        raise ValueError("source_path and output_path must be different.")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)

    with h5py.File(source_path, "r") as f:
        required = ("/ModelCube", "/ObsPix")
        missing = [name for name in required if name not in f]
        if missing:
            raise RuntimeError(f"Missing required datasets: {missing}")

        model = np.asarray(f["/ModelCube"][...], dtype=np.float64)
        obs_pix = np.asarray(f["/ObsPix"][...], dtype=np.float64)

        if model.ndim != 2:
            raise RuntimeError(
                f"Expected ModelCube (S, L), got {model.shape}.")
        if obs_pix.ndim != 1 or obs_pix.size != model.shape[1]:
            raise RuntimeError(
                "ObsPix is inconsistent with ModelCube.")
        if not np.all(np.isfinite(model)):
            bad = int(np.count_nonzero(~np.isfinite(model)))
            raise RuntimeError(
                f"ModelCube contains {bad} non-finite values.")
        if not np.all(np.isfinite(obs_pix)):
            raise RuntimeError("ObsPix contains non-finite values.")

    noise = str(noise).lower()
    noise_scale = float(noise_scale)
    rng = np.random.default_rng(seed)

    if noise_scale < 0.0 or not np.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite and non-negative.")

    if noise == "none":
        noise_cube = np.zeros_like(model)
        validation_kind = "exact_model_closure"

    elif noise == "stat":
        if stat_cube is None:
            raise ValueError("stat_cube is required for noise='stat'.")

        stat = np.asarray(stat_cube, dtype=np.float64)
        if stat.shape != model.shape:
            raise ValueError(
                f"stat_cube shape {stat.shape} != model shape {model.shape}.")

        if np.any(~np.isfinite(stat)) or np.any(stat < 0.0):
            raise ValueError("stat_cube contains invalid 1-sigma errors.")

        noise_cube = rng.normal(0.0, stat)
        validation_kind = "gaussian_observational_noise"

    elif noise == "residual":
        with h5py.File(source_path, "r") as f:
            if "/DataCube" not in f:
                raise RuntimeError(
                    "noise='residual' requires source /DataCube.")

            data = np.asarray(f["/DataCube"][...], dtype=np.float64)

        if data.shape != model.shape:
            raise RuntimeError(
                f"DataCube shape {data.shape} != ModelCube {model.shape}.")

        residual = data-model
        noise_cube = np.empty_like(residual)

        # Preserve each spaxel's empirical residual distribution while
        # destroying wavelength-coherent correspondence with the original fit.
        for ss in range(model.shape[0]):
            noise_cube[ss] = rng.permutation(residual[ss])

        validation_kind = "permuted_empirical_residual_noise"

    else:
        raise ValueError("noise must be one of 'none', 'stat', or 'residual'.")

    mock = model + noise_scale*noise_cube

    with h5py.File(output_path, "w") as f:
        ds = f.create_dataset("/DataCube", data=mock, dtype=np.float64,
            compression="gzip", shuffle=True)
        f.create_dataset("/ObsPix", data=obs_pix, dtype=np.float64)

        ds.attrs["validation_source"] = "/ModelCube"
        ds.attrs["validation_source_file"] = source_path
        ds.attrs["validation_kind"] = validation_kind
        ds.attrs["noise_mode"] = noise
        ds.attrs["noise_scale"] = noise_scale
        ds.attrs["noise_seed"] = int(seed)

    print()
    print("Validation input created.")
    print(f"DataCube shape: {model.shape}")
    print(f"DataCube range: {np.min(model):.6e} .. {np.max(model):.6e}")
    print(f"Output: {output_path}")

    return output_path

# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=(
        "Convert a CubeFit /ModelCube into a synthetic "
        "CubeFit validation input."))

    parser.add_argument("source", help="Completed CubeFit HDF5 file.")

    parser.add_argument("output", help="Output validation HDF5 file.")

    parser.add_argument("--rebuild-hypercube",
        action="store_true",
        help=("Remove /HyperCube from the validation file so that "
            "it must be regenerated."))

    args = parser.parse_args()

    makeValidationCube(args.source, args.output)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------

def loadValidationCube(validationPath, spLL, nSpat, *, dataset="/DataCube"):
    """
    Load a CubeFit model cube for use as synthetic validation data.

    The returned array follows the orientation expected internally by
    ``genCubeFit``: ``(nLSpec, nSpat)``. CubeFit HDF5 datasets use the
    opposite orientation, ``(nSpat, nLSpec)``, so the selected dataset
    is transposed before returning.

    The wavelength grid stored in the validation file is required to
    agree with the wavelength grid produced by ``_oneTimeSpec``. This
    prevents a model cube generated on one spectral grid from being
    silently fitted on another.

    Parameters
    ----------
    validationPath : str or pathlib.Path
        CubeFit HDF5 file containing the synthetic validation cube.
    spLL : ndarray
        Current observed log-wavelength grid from ``_oneTimeSpec``, with
        shape ``(nLSpec,)``.
    nSpat : int
        Number of spatial bins expected by the current CubeFit run.
    dataset : str, optional
        HDF5 dataset containing the validation spectra. Default is
        ``"/DataCube"``.

    Returns
    -------
    validationCube : ndarray
        Synthetic spectral cube with shape ``(nLSpec, nSpat)`` and
        dtype ``float64``.

    Raises
    ------
    FileNotFoundError
        If ``validationPath`` does not exist.
    RuntimeError
        If the requested dataset or wavelength grid is missing, or if
        the validation cube contains non-finite values.
    ValueError
        If the validation cube dimensions or wavelength grid do not
        match the current CubeFit configuration.

    Examples
    --------
    >>> laGrid = loadValidationCube(
    ...     "NGC4365/hypercube_003_00.h5",
    ...     spLL,
    ...     nSpat,
    ... )
    """
    validationPath = plp.Path(validationPath)

    if not validationPath.is_file():
        raise FileNotFoundError(
            f"Validation HDF5 file not found: {validationPath}")

    spLL = np.asarray(spLL, dtype=np.float64).ravel(order="C")

    with open_h5(str(validationPath), role="reader") as f:
        if dataset not in f:
            raise RuntimeError(f"{validationPath} does not contain {dataset}.")
        if "/ObsPix" not in f:
            raise RuntimeError(f"{validationPath} does not contain /ObsPix.")

        cube = np.asarray(f[dataset][...], dtype=np.float64, order="C")
        obs_pix = np.asarray(f["/ObsPix"][...], dtype=np.float64).ravel(
            order="C")

    if cube.ndim != 2:
        raise ValueError(f"{dataset} must be two-dimensional; "
            f"got shape {cube.shape}.")

    expected_shape = (int(nSpat), int(spLL.size),)

    if cube.shape != expected_shape:
        raise ValueError("Validation cube shape mismatch: "
            f"{dataset} has {cube.shape}, but the current run "
            f"expects {expected_shape} = (nSpat, nLSpec).")
    if obs_pix.shape != spLL.shape:
        raise ValueError("Validation wavelength-grid length mismatch: "
            f"{obs_pix.size} != {spLL.size}.")
    if not np.allclose(obs_pix, spLL, rtol=1e-12, atol=1e-12,):
        max_diff = float(np.max(np.abs(obs_pix - spLL)))
        raise ValueError("Validation /ObsPix does not match the current "
            "_oneTimeSpec wavelength grid. "
            f"Maximum absolute difference = {max_diff:.6e}.")
    if not np.all(np.isfinite(cube)):
        n_bad = int(np.count_nonzero(~np.isfinite(cube)))
        raise RuntimeError(f"{dataset} contains {n_bad} non-finite values.")

    # HDF5 CubeFit convention: (nSpat, nLSpec)
    # kz_fitSpec convention:    (nLSpec, nSpat)
    validationCube = np.ascontiguousarray(cube.T, dtype=np.float64)

    logger.log("[CubeFit][validation] Loaded synthetic data from "
        f"{validationPath}:{dataset}; shape={validationCube.shape}, "
        f"min={np.min(validationCube):.6e}, max={np.max(validationCube):.6e}")

    return validationCube

# ------------------------------------------------------------------------------

def compare_validation_solution(input_path, validation_path, save_path=None):
    """
    Compare a validation solution directly with its generating solution.

    The input and validation paths are the corresponding HyperCube HDF5
    files. Their sibling ``x_*.h5`` files are located automatically and
    ``/X_global`` is compared coefficient by coefficient.

    Parameters
    ----------
    input_path : str or pathlib.Path
        HyperCube whose ``/ModelCube`` supplied the validation data.
    validation_path : str or pathlib.Path
        HyperCube produced by fitting those validation data.
    save_path : str or pathlib.Path, optional
        Output figure path. If None, the figure is not written.

    Returns
    -------
    out : dict
        Input and recovered coefficients, residuals, and summary metrics.

    Raises
    ------
    FileNotFoundError
        If either HyperCube or corresponding solution file is absent.
    RuntimeError
        If ``/X_global`` is absent from either solution file.
    ValueError
        If the input and recovered coefficient shapes differ.

    Examples
    --------
    >>> out = compare_validation_solution(
    ...     "CubeFit/NGC4365/hypercube_20_01.h5",
    ...     "CubeFit/NGC4365/hypercube_20_01_validation.h5")
    """
    input_path = plp.Path(input_path)
    validation_path = plp.Path(validation_path)

    def _x_path(path):
        name = path.name.replace("hypercube", "x", 1)
        return path.with_name(name)

    input_x_path = _x_path(input_path)
    validation_x_path = _x_path(validation_path)

    for path in (input_path, validation_path, input_x_path,
        validation_x_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required HDF5 file not found: {path}")

    with h5py.File(input_x_path, "r") as f:
        if "/X_global" not in f:
            raise RuntimeError(f"{input_x_path} has no /X_global.")
        x_input = np.asarray(f["/X_global"][...], dtype=np.float64)
    with h5py.File(validation_x_path, "r") as f:
        if "/X_global" not in f:
            raise RuntimeError(f"{validation_x_path} has no /X_global.")
        x_fit = np.asarray(f["/X_global"][...], dtype=np.float64)
    if x_input.shape != x_fit.shape:
        raise ValueError(
            f"X_global shapes differ: {x_input.shape} != {x_fit.shape}.")
    if x_input.ndim != 2:
        raise ValueError(
            f"X_global must have shape (C, P); got {x_input.shape}.")

    C, P = x_input.shape
    resid = x_fit - x_input

    input_flat = x_input.ravel(order="C")
    fit_flat = x_fit.ravel(order="C")
    resid_flat = resid.ravel(order="C")

    l1 = float(np.sum(np.abs(resid_flat)))
    l2 = float(np.linalg.norm(resid_flat))
    linf = float(np.max(np.abs(resid_flat)))
    input_l1 = float(np.sum(np.abs(input_flat)))
    input_l2 = float(np.linalg.norm(input_flat))
    input_linf = float(np.max(np.abs(input_flat)))

    rel_l1 = l1 / input_l1 if input_l1 > 0.0 else np.nan
    rel_l2 = l2 / input_l2 if input_l2 > 0.0 else np.nan
    rel_linf = linf / input_linf if input_linf > 0.0 else np.nan

    denom = float(np.linalg.norm(input_flat) * np.linalg.norm(fit_flat))
    cosine = float(np.dot(input_flat, fit_flat) / denom
        ) if denom > 0.0 else np.nan

    if np.std(input_flat) > 0.0 and np.std(fit_flat) > 0.0:
        corr = float(np.corrcoef(input_flat, fit_flat)[0, 1])
    else:
        corr = np.nan

    input_pop = np.sum(x_input, axis=0)
    fit_pop = np.sum(x_fit, axis=0)
    input_comp = np.sum(x_input, axis=1)
    fit_comp = np.sum(x_fit, axis=1)

    vmax = float(max(np.max(input_flat), np.max(fit_flat), 0.0))
    pad = 0.025 * vmax if vmax > 0.0 else 1.0
    lim = (-pad, vmax + pad)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(input_flat, fit_flat, s=8, alpha=0.5)
    ax.plot(lim, lim, "k--", lw=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"input $x_{c,p}$")
    ax.set_ylabel(r"recovered $x_{c,p}$")
    ax.set_title("Coefficient recovery")

    text = (rf"$C={C},\ P={P}$" "\n"
        rf"$||\Delta x||_1/||x||_1={rel_l1:.3e}$" "\n"
        rf"$||\Delta x||_2/||x||_2={rel_l2:.3e}$" "\n"
        rf"$||\Delta x||_\infty/||x||_\infty={rel_linf:.3e}$" "\n"
        rf"$\rho={corr:.5f}$" "\n"
        rf"$\cos\theta={cosine:.5f}$")
    ax.text(0.03, 0.97, text, va="top", ha="left", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax = axes[1]
    pop_max = float(max(np.max(input_pop), np.max(fit_pop), 0.0))
    pop_pad = 0.025 * pop_max if pop_max > 0.0 else 1.0
    pop_lim = (-pop_pad, pop_max + pop_pad)
    ax.scatter(input_pop, fit_pop, s=12, alpha=0.6)
    ax.plot(pop_lim, pop_lim, "k--", lw=1.0)
    ax.set_xlim(pop_lim)
    ax.set_ylim(pop_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"input $\sum_c x_{c,p}$")
    ax.set_ylabel(r"recovered $\sum_c x_{c,p}$")
    ax.set_title("Population recovery")

    ax = axes[2]
    comp_max = float(max(np.max(input_comp), np.max(fit_comp), 0.0))
    comp_pad = 0.025 * comp_max if comp_max > 0.0 else 1.0
    comp_lim = (-comp_pad, comp_max + comp_pad)
    ax.scatter(input_comp, fit_comp, s=20)
    ax.plot(comp_lim, comp_lim, "k--", lw=1.0)
    ax.set_xlim(comp_lim)
    ax.set_ylim(comp_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"input $\sum_p x_{c,p}$")
    ax.set_ylabel(r"recovered $\sum_p x_{c,p}$")
    ax.set_title("Component coefficient mass")

    fig.tight_layout()

    if save_path is not None:
        save_path = plp.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    out = {
        "x_input": x_input,
        "x_fit": x_fit,
        "residual": resid,
        "input_population": input_pop,
        "fit_population": fit_pop,
        "input_component": input_comp,
        "fit_component": fit_comp,
        "relative_l1": rel_l1,
        "relative_l2": rel_l2,
        "relative_linf": rel_linf,
        "correlation": corr,
        "cosine_similarity": cosine,
        "figure": fig,
    }

    print("[validation] coefficient recovery")
    print(f"  shape                 = {x_input.shape}")
    print(f"  relative L1 error     = {rel_l1:.6e}")
    print(f"  relative L2 error     = {rel_l2:.6e}")
    print(f"  relative Linf error   = {rel_linf:.6e}")
    print(f"  correlation           = {corr:.8f}")
    print(f"  cosine similarity     = {cosine:.8f}")

    return out

# ------------------------------------------------------------------------------