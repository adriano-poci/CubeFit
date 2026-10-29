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
v1.3:   Confirm if validation run used the same SSP library, and produce
            valid comparisons regardless, in `compare_validation_solution`. 5
            October 2026
"""
# need to set up the logger before any other imports
import pathlib as plp
from CubeFit.logger import get_logger
print("[CubeFit] Initialising CubeFit logger...")
curdir = plp.Path(__file__).parent
lfn = curdir/'kz_run.log'
logger = get_logger(lfn, mode='w')

import argparse, os, shutil
import h5py
import numpy as np
import pathlib as plp
import matplotlib.pyplot as plt

from CubeFit.hdf5_manager import open_h5
from CubeFit import cube_utils as cu
from dynamics.IFU.Constants import Constants, Units, UnitStr

mDir = curdir.parent/'muse'
dDir = cu._ddir()
UTS = UnitStr()

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

def _best_two_box_locs(*curves):
    """Choose distinct low-occupancy corners for statistics and legend."""
    points = []

    for x, y in curves:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        points.append((x[good], y[good]))

    if not any(x.size for x, _ in points):
        return "upper left", "upper right"

    xmin = min(np.min(x) for x, _ in points if x.size)
    xmax = max(np.max(x) for x, _ in points if x.size)
    ymin = min(np.min(y) for _, y in points if y.size)
    ymax = max(np.max(y) for _, y in points if y.size)

    dx = max(xmax - xmin, np.finfo(np.float64).eps)
    dy = max(ymax - ymin, np.finfo(np.float64).eps)

    points = [((x - xmin)/dx, (y - ymin)/dy) for x, y in points]

    xfrac = 0.42
    yfrac = 0.28

    scores = {}

    for loc in ("upper left", "upper right", "lower left", "lower right"):
        score = 0.0
        left = "left" in loc
        upper = "upper" in loc

        for x, y in points:
            xmask = x <= xfrac if left else x >= 1.0 - xfrac
            ymask = y >= 1.0 - yfrac if upper else y <= yfrac

            score += 10.0 * np.count_nonzero(xmask & ymask)

            xc = 0.0 if left else 1.0
            yc = 1.0 if upper else 0.0
            dist2 = ((x - xc)/xfrac)**2 + ((y - yc)/yfrac)**2
            score += np.sum(np.exp(-2.0 * dist2))

        scores[loc] = score

    ordered = sorted(scores, key=scores.get)
    return ordered[0], ordered[1]

def _best_stats_loc(ax, legend, box, *curves):
    """Choose the least obstructed corner for an OffsetBox."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    legend_bbox = legend.get_window_extent(renderer=renderer)
    box_bbox = box.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent(renderer=renderer)

    bw = box_bbox.width
    bh = box_bbox.height

    pad = 6.0
    corners = {
        "upper left": (
            ax_bbox.x0 + pad,
            ax_bbox.y1 - pad - bh,
        ),
        "upper right": (
            ax_bbox.x1 - pad - bw,
            ax_bbox.y1 - pad - bh,
        ),
        "lower left": (
            ax_bbox.x0 + pad,
            ax_bbox.y0 + pad,
        ),
        "lower right": (
            ax_bbox.x1 - pad - bw,
            ax_bbox.y0 + pad,
        ),
    }

    scores = {}

    for loc, (x0, y0) in corners.items():
        x1 = x0 + bw
        y1 = y0 + bh

        # Actual prospective statistics-box rectangle.
        overlap_x = max(
            0.0, min(x1, legend_bbox.x1) - max(x0, legend_bbox.x0))
        overlap_y = max(
            0.0, min(y1, legend_bbox.y1) - max(y0, legend_bbox.y0))

        # Make overlap with the legend effectively forbidden.
        score = 1.0e6 * overlap_x * overlap_y

        for x, y in curves:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            good = np.isfinite(x) & np.isfinite(y)

            if not np.any(good):
                continue

            xy = ax.transData.transform(
                np.column_stack((x[good], y[good])))

            inside = (
                (xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
                (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            )
            score += 1.0e4 * np.count_nonzero(inside)

            # Also discourage putting the box very close to the curves.
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            sx = max(bw, 1.0)
            sy = max(bh, 1.0)

            dist2 = ((xy[:, 0] - cx) / sx) ** 2 + \
                ((xy[:, 1] - cy) / sy) ** 2
            score += np.sum(np.exp(-2.0 * dist2))

        scores[loc] = score

    return min(scores, key=scores.get)

# ------------------------------------------------------------------------------

def compare_validation_solution(input_path, validation_path, mPath,
    input_nCuts, validation_nCuts, save_path=None, dataLabels=['Mock', 'Fit']):
    """
    Compare a validation fit with its generating CubeFit solution.

    The global stellar-population comparison is independent of the dynamical
    component basis. When the two solutions use different dynamical
    decompositions, both solutions are additionally expanded onto the common
    Schwarzschild orbit library and rebinned into z-tube, x-tube, and box
    orbital families.

    Parameters
    ----------
    input_path : str or pathlib.Path
        HyperCube whose ``/ModelCube`` supplied the validation data.
    validation_path : str or pathlib.Path
        HyperCube produced by fitting the validation data.
    mPath : str or pathlib.Path
        Schwarzschild model directory beneath ``mDir/'tri_models'``.
    input_nCuts : int
        Number of dynamical components in the generating decomposition.
    validation_nCuts : int
        Number of dynamical components in the validation decomposition.
    save_path : str or pathlib.Path, optional
        Output path for the primary stellar-population figure. Additional
        figures are written beside it with descriptive suffixes.
    dataLabels : list of str, optional
        Labels for the two datasets in the comparison plots. Default is
        ``['Mock', 'Fit']``.

    Returns
    -------
    out : dict
        SSP identity tests, fitted coefficients, global physical summaries,
        orbit-space solutions, orbital-family summaries, direct coefficient
        metrics when applicable, and generated figures.

    Raises
    ------
    FileNotFoundError
        If a required HyperCube, solution, decomposition, or orbit file is
        absent.
    RuntimeError
        If required solution, SSP, or dynamical metadata are absent.
    ValueError
        If coefficient arrays, SSP grids, or dynamical decompositions are
        inconsistent.

    Examples
    --------
    >>> out = compare_validation_solution(
    ...     "CubeFit/NGC4365/hypercube_3_00.h5",
    ...     "CubeFit/NGC4365/hypercube_125_00_validation.h5",
    ...     "hdhdc4365", 3, 125)
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker,\
        VPacker
    from matplotlib.patches import Rectangle

    input_path = plp.Path(input_path)
    validation_path = plp.Path(validation_path)
    bDir = mDir/'tri_models'/mPath

    iLabel, vLabel = dataLabels

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _x_path(path):
        """Return the sibling solution-file path."""
        name = path.name.replace("hypercube", "x", 1)
        return path.with_name(name)

    def _load_ssp(path):
        """Load templates, physical SSP axes, and library provenance."""
        with h5py.File(path, "r") as f:
            if "/Templates" not in f or "/TemPix" not in f:
                raise RuntimeError(f"{path} has incomplete SSP information.")

            templates = np.asarray(
                f["/Templates"][...], dtype=np.float64)
            tem_pix = np.asarray(
                f["/TemPix"][...], dtype=np.float64)

            required = ("/SSPMetals", "/SSPAges", "/SSPAlphas")
            if all(name in f for name in required):
                axes = tuple(
                    np.asarray(f[name][...], dtype=np.float64)
                    for name in required)
            else:
                axes = None

            library = f["/Templates"].attrs.get("ssp_library", None)
            if isinstance(library, bytes):
                library = library.decode("utf-8")
            if library is not None:
                library = str(library)

        return templates, tem_pix, axes, library

    def _physical_summary(x, axes):
        """Calculate marginalized SSP distributions and mean properties."""
        metals, ages, alphas = axes
        C, P = x.shape
        expected = metals.size*ages.size*alphas.size

        if P != expected:
            raise ValueError(
                f"X has P={P}, while SSP axes imply P={expected}.")

        sfh = x.reshape(C, metals.size, ages.size, alphas.size)
        mass = sfh.sum(axis=(1, 2, 3))

        z_grid = metals[:, None, None]
        age_grid = ages[None, :, None]
        alpha_grid = alphas[None, None, :]

        def _mean(grid):
            """Return component-wise coefficient-weighted mean."""
            num = np.sum(sfh*grid[None, ...], axis=(1, 2, 3))
            return np.divide(num, mass, out=np.full(C, np.nan),
                where=mass > 0.0)

        return {
            "sfh": sfh,
            "mass": mass,
            "age_distribution": sfh.sum(axis=(0, 1, 3)),
            "metal_distribution": sfh.sum(axis=(0, 2, 3)),
            "alpha_distribution": sfh.sum(axis=(0, 1, 2)),
            "mean_age": _mean(age_grid),
            "mean_metal": _mean(z_grid),
            "mean_alpha": _mean(alpha_grid),
        }

    def _global_properties(summary):
        """Return coefficient-weighted global stellar-population means."""
        mass = summary["mass"]
        total = float(np.sum(mass))

        if total <= 0.0:
            return {"age": np.nan, "metal": np.nan, "alpha": np.nan}

        return {
            "age": float(np.sum(mass*summary["mean_age"])/total),
            "metal": float(np.sum(mass*summary["mean_metal"])/total),
            "alpha": float(np.sum(mass*summary["mean_alpha"])/total),
        }

    def _normalise(values):
        """Return a unit-sum copy of a marginalized distribution."""
        values = np.asarray(values, dtype=np.float64)
        total = float(np.sum(values))
        if total <= 0.0:
            return np.zeros_like(values)
        return values/total

    def _load_decomposition(nCuts):
        """Load one decomposition from the model's current decDir."""
        nCuts = int(nCuts)

        dec_file = bDir/'decomp.dir'
        if not dec_file.is_file():
            raise FileNotFoundError(f"Missing decomposition file: {dec_file}")

        with open(dec_file, 'r+') as dd:
            decDir = dd.readline().strip()

        if not decDir:
            raise RuntimeError(f"{dec_file} contains an empty decDir.")

        direc = bDir/decDir/f"decomp_{nCuts:d}"
        decomp_file = direc/f"decomp_{nCuts:d}.plt"

        if not decomp_file.is_file():
            raise FileNotFoundError(
                f"Missing decomposition file: {decomp_file}")

        oDict = cu.Load.lzma(decomp_file)
        nzComp = np.asarray(oDict["nzComp"], dtype=int)
        pred = f"0{len(repr(nCuts)):d}"

        return {
            "decDir": decDir,
            "direc": direc,
            "nCuts": nCuts,
            "oDict": oDict,
            "nzComp": nzComp,
            "pred": pred,
        }

    def _orbit_library_path(decomp):
        """Reconstruct nn_orb.out path in the same way as loadCubeFit."""
        nnOrb = plp.Path(*decomp["oDict"]["nnOrb"])
        return plp.Path(
            bDir, decomp["decDir"], nnOrb.parent.name, nnOrb.name)

    def _component_orbit_matrix(decomp, n_orbits, n_components):
        """Return normalized orbit fractions of each dynamical component."""
        oDict = decomp["oDict"]
        nzComp = decomp["nzComp"]
        pred = decomp["pred"]

        if nzComp.size != n_components:
            raise ValueError(
                f"decomp_{decomp['nCuts']} contains {nzComp.size} active "
                f"components, but X_global has C={n_components}.")

        matrix = np.zeros((n_orbits, n_components), dtype=np.float64)

        for cc, comp in enumerate(nzComp):
            key = f"{comp:{pred}d}"

            if key not in oDict["weights"] or key not in oDict["wheres"]:
                raise RuntimeError(
                    f"Component {comp} is absent from decomposition metadata.")

            weights = np.asarray(oDict["weights"][key], dtype=np.float64)
            mask = np.asarray(oDict["wheres"][key], dtype=bool)

            if weights.shape != mask.shape:
                raise ValueError(
                    f"Component {comp}: weight/mask shape mismatch "
                    f"{weights.shape} != {mask.shape}.")

            if weights.size != n_orbits:
                raise ValueError(f"Component {comp}: decomposition contains "
                    f"{weights.size} orbits, expected {n_orbits}.")

            weights = np.where(mask, weights, 0.0)
            total = float(np.sum(weights))

            if total > 0.0:
                matrix[:, cc] = weights/total

        return matrix

    def _family_summary(orbit_x, orbit_types, axes):
        """Collapse an orbit-space solution onto intrinsic orbit families."""
        families = (("z_tube", "z-tubes", 3), ("x_tube", "x-tubes", 1),
            ("box", "boxes", 4))

        result = {}

        for key, label, orbit_type in families:
            mask = orbit_types == orbit_type
            family_x = np.sum(orbit_x[mask], axis=0)

            entry = {
                "label": label,
                "orbit_type": orbit_type,
                "mask": mask,
                "n_orbits": int(np.count_nonzero(mask)),
                "x": family_x,
                "coefficient_mass": float(np.sum(family_x)),
            }

            if axes is not None:
                physical = _physical_summary(family_x[None, :], axes)
                entry["physical"] = physical
                entry["global"] = _global_properties(physical)
            else:
                entry["physical"] = None
                entry["global"] = None

            result[key] = entry

        return result

    # ------------------------------------------------------------------
    # Load the two coefficient solutions.
    # ------------------------------------------------------------------

    input_x_path = _x_path(input_path)
    validation_x_path = _x_path(validation_path)

    for path in (input_path, validation_path, input_x_path, validation_x_path):
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

    if x_input.ndim != 2 or x_fit.ndim != 2:
        raise ValueError("X_global must have shape (C, P).")

    C_input = int(x_input.shape[0])
    C_fit = int(x_fit.shape[0])
    same_components = C_input == C_fit

    # ------------------------------------------------------------------
    # SSP metadata.
    # ------------------------------------------------------------------

    t_input, l_input, axes_input, lib_input = _load_ssp(input_path)
    t_fit, l_fit, axes_fit, lib_fit = _load_ssp(validation_path)

    physical_available = axes_input is not None and axes_fit is not None

    if physical_available:
        phys_input = _physical_summary(x_input, axes_input)
        phys_fit = _physical_summary(x_fit, axes_fit)
        global_input = _global_properties(phys_input)
        global_fit = _global_properties(phys_fit)
    else:
        phys_input = None
        phys_fit = None
        global_input = None
        global_fit = None

    same_library = (lib_input is not None and lib_fit is not None
        and lib_input == lib_fit)

    same_axes = (physical_available and all(a.shape == b.shape
        and np.allclose(a, b, rtol=1e-12, atol=1e-14)
        for a, b in zip(axes_input, axes_fit)))

    same_templates = (t_input.shape == t_fit.shape
        and l_input.shape == l_fit.shape and np.allclose(
            l_input, l_fit, rtol=1e-12, atol=1e-14)
        and np.allclose(t_input, t_fit, rtol=1e-10, atol=1e-12))

    same_ssp = same_library and same_axes and same_templates
    coefficient_comparable = same_ssp and same_components

    print("[validation] SSP comparison")
    print(f"  generating library     = {lib_input}")
    print(f"  fitted library         = {lib_fit}")
    print(f"  same library           = {same_library}")
    print(f"  same physical grid     = {same_axes}")
    print(f"  same template spectra  = {same_templates}")
    print(f"  generating C           = {C_input}")
    print(f"  fitted C               = {C_fit}")
    print(f"  same components        = {same_components}")
    print(f"  coefficient comparable = {coefficient_comparable}")

    # ------------------------------------------------------------------
    # Dynamical decomposition and common Schwarzschild orbit library.
    # ------------------------------------------------------------------

    input_decomp = _load_decomposition(input_nCuts)
    fit_decomp = _load_decomposition(validation_nCuts)

    input_nnOrb = _orbit_library_path(input_decomp)
    fit_nnOrb = _orbit_library_path(fit_decomp)

    if not input_nnOrb.is_file():
        raise FileNotFoundError(f"Input orbit library not found: {input_nnOrb}")
    if not fit_nnOrb.is_file():
        raise FileNotFoundError(f"Fit orbit library not found: {fit_nnOrb}")

    if input_nnOrb.resolve() != fit_nnOrb.resolve():
        raise ValueError('Generating and validation decompositions do not '
            'reference the same Schwarzschild orbit library:\n'
            f"  {iLabel}: {input_nnOrb}\n"
            f"  {vLabel}:  {fit_nnOrb}")

    NOrbs, inds, energs, I2s, I3s, regs, orbit_types, \
        schwarzschild_weights, lcuts = cu.Read.orbits(input_nnOrb)

    orbit_types = np.asarray(orbit_types, dtype=np.int64).ravel()
    schwarzschild_weights = np.asarray(schwarzschild_weights,
        dtype=np.float64).ravel()

    n_orbits = int(orbit_types.size)

    if schwarzschild_weights.size != n_orbits:
        raise RuntimeError("Orbit-type and Schwarzschild-weight arrays have "
            "different lengths.")

    if int(NOrbs) != n_orbits:
        raise RuntimeError(f"Orbit library reports NOrbs={NOrbs}, but contains "
            f"{n_orbits} orbit classifications.")

    # The only supported intrinsic families in the current decomposition.
    supported_types = np.array([1, 3, 4], dtype=np.int64)
    populated_types = np.unique(orbit_types[schwarzschild_weights > 0.0])

    unsupported = populated_types[~np.isin(populated_types, supported_types)]

    if unsupported.size:
        raise RuntimeError('Positive-weight Schwarzschild orbits contain '
            f"unsupported types {unsupported.tolist()}.")

    M_input = _component_orbit_matrix(input_decomp, n_orbits, C_input)
    M_fit = _component_orbit_matrix(fit_decomp, n_orbits, C_fit)

    # Expand both arbitrary dynamical component bases onto the common
    # Schwarzschild orbit basis.
    orbit_x_input = M_input @ x_input
    orbit_x_fit = M_fit @ x_fit

    input_population_mass = np.sum(x_input, axis=0)
    fit_population_mass = np.sum(x_fit, axis=0)

    input_orbit_population_mass = np.sum(orbit_x_input, axis=0)
    fit_orbit_population_mass = np.sum(orbit_x_fit, axis=0)

    if not np.allclose(input_orbit_population_mass, input_population_mass,
        rtol=1e-10, atol=1e-12):
        err = float(np.max(np.abs(input_orbit_population_mass
        - input_population_mass)))
        raise RuntimeError(f"{iLabel} component-to-orbit expansion does not "
            f"conserve SSP coefficient mass; max error={err:.6e}.")

    if not np.allclose(fit_orbit_population_mass, fit_population_mass,
        rtol=1e-10, atol=1e-12):
        err = float(np.max(np.abs(fit_orbit_population_mass-
            fit_population_mass)))
        raise RuntimeError(f"{vLabel} component-to-orbit expansion does not "
            f"conserve SSP coefficient mass; max error={err:.6e}.")

    orbital_input = _family_summary(orbit_x_input, orbit_types, axes_input)
    orbital_fit = _family_summary(orbit_x_fit, orbit_types, axes_fit)

    print("[validation] common orbit-space reconstruction")
    print(f"  orbit library           = {input_nnOrb}")
    print(f"  number of orbits        = {n_orbits}")
    print(f"  {iLabel} decomposition      = {input_nCuts}")
    print(f"  {vLabel} decomposition       = {validation_nCuts}")

    for key in ("z_tube", "x_tube", "box"):
        mock_mass = orbital_input[key]["coefficient_mass"]
        fit_mass = orbital_fit[key]["coefficient_mass"]
        print(f"  {orbital_input[key]['label']:8s}: "
            f"{iLabel}={mock_mass:.6e} {vLabel}={fit_mass:.6e}")

    # ------------------------------------------------------------------
    # Metrics and figures.
    # ------------------------------------------------------------------

    resid = None
    rel_l1 = rel_l2 = rel_linf = np.nan
    corr = cosine = np.nan

    spectral_resid = None
    relative_l2_spectral = None
    fractional_rms = None

    physical_figure = None
    coefficient_figure = None
    spectral_figure = None

    # ------------------------------------------------------------------
    # Global stellar-population comparison.
    #
    # Keep this figure unchanged: it is deliberately independent of the
    # dynamical component basis.
    # ------------------------------------------------------------------

    if physical_available:
        physical_figure, axes = plt.subplots(2, 3,
            figsize=np.asarray((15, 8.0))*0.9)

        physical_axes = axes[0]
        orbital_axes = axes[1]

        distributions = (("age_distribution", 1, "age",
                rf"Age $[{{{UTS.gyr}}}]$", "Age distribution"),
            ("metal_distribution", 0, "metal",
                r"[Z/H] $[{\rm dex}]$", "Metallicity distribution"),
            ("alpha_distribution", 2, "alpha",
                r"[$\alpha$/Fe] $[{\rm dex}]$", "Alpha distribution"))

        if same_library:
            input_label = iLabel
            fit_label = vLabel
        else:
            input_label = (f"{iLabel}: {lib_input}"
                if lib_input is not None else iLabel)
            fit_label = (f"{vLabel}: {lib_fit}"
                if lib_fit is not None else vLabel)

        for ax, (key, idx, prop, xlabel, title) in zip(
            physical_axes, distributions):
            grid_input = axes_input[idx]
            grid_fit = axes_fit[idx]
            yi = _normalise(phys_input[key])
            yf = _normalise(phys_fit[key])

            ax.plot(grid_input, yi, "o-", label=input_label)
            ax.plot(grid_fit, yf, "s--", label=fit_label)

            stats_loc, legend_loc = _best_two_box_locs(
                (grid_input, yi), (grid_fit, yf))

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Fractional Weight")
            ax.set_title(title)

            mock_mean = global_input[prop]
            fit_mean = global_fit[prop]

            if prop == "age":
                labels = (rf"$\langle{{\rm Age}}\rangle_{{\rm {iLabel}}}$",
                    rf"$\langle{{\rm Age}}\rangle_{{\rm {vLabel}}}$")
                values = (rf"$= {mock_mean:.2f}\ [{UTS.gyr}]$",
                    rf"$= {fit_mean:.2f}\ [{UTS.gyr}]$",)
            elif prop == "metal":
                labels = (rf"$\langle[Z/H]\rangle_{{\rm {iLabel}}}$",
                    rf"$\langle[Z/H]\rangle_{{\rm {vLabel}}}$")
                values = (rf"$= {mock_mean:.3f}\ [{{\rm dex}}]$",
                    rf"$= {fit_mean:.3f}\ [{{\rm dex}}]$")
            else:
                labels = (
                    rf"$\langle[\alpha/{{\rm Fe}}]\rangle_{{\rm {iLabel}}}$",
                    rf"$\langle[\alpha/{{\rm Fe}}]\rangle_{{\rm {vLabel}}}$")
                values = (rf"$= {mock_mean:.3f}\ [{{\rm dex}}]$",
                    rf"$= {fit_mean:.3f}\ [{{\rm dex}}]$")

            rows = []
            for label, value in zip(labels, values):
                rows.append(
                    HPacker(children=[TextArea(label,
                        textprops=dict(ha="right")),
                        TextArea(value, textprops=dict(ha="left"))],
                        align="center", pad=0, sep=4))
            box = VPacker(children=rows, align="right", pad=0, sep=2)
            anchored = AnchoredOffsetbox(loc=stats_loc, child=box, frameon=True,
                borderpad=0.4, pad=0.3)
            ax.add_artist(anchored)

            physical_figure.canvas.draw()
            renderer = physical_figure.canvas.get_renderer()
            bbox_disp = anchored.get_window_extent(renderer=renderer)
            bbox_data = bbox_disp.transformed(ax.transData.inverted())

            padx = 0.02 * (ax.get_xlim()[1]-ax.get_xlim()[0])
            pady = 0.02 * (ax.get_ylim()[1]-ax.get_ylim()[0])

            legend_blocker = Rectangle((bbox_data.x0-padx, bbox_data.y0-pady),
                bbox_data.width+2.0*padx, bbox_data.height+2.0*pady,
                facecolor="none", edgecolor="none", alpha=0.0)

            ax.add_patch(legend_blocker)
            ax.legend(loc="best")

        print("[validation] physical SSP recovery")
        print(f"  input P                = {x_input.shape[1]}")
        print(f"  recovered P            = {x_fit.shape[1]}")

        if same_components:
            for key, label, unit in (("mean_age", "age", UTS.gyr),
                ("mean_metal", "[Z/H]", "dex"),
                ("mean_alpha", "[alpha/Fe]", "dex")):
                delta = phys_fit[key]-phys_input[key]
                print(f"  max component delta {label:10s} = "
                    f"{np.nanmax(np.abs(delta)):.6e} [{unit}]")
        else:
            print("  component-wise comparison skipped "
                f"(C={C_input} vs C={C_fit})")

        print("[validation] global stellar populations")

        for key, label, unit in (("age", "age", UTS.gyr),
            ("metal", "[Z/H]", "dex"),
            ("alpha", "[alpha/Fe]", "dex")):
            delta = global_fit[key]-global_input[key]
            print(f"  {label:10s}: {global_input[key]:.6g} -> "
                f"{global_fit[key]:.6g}  delta={delta:+.6g} [{unit}]")

    # ------------------------------------------------------------------
    # Common orbital-family comparison.
    #
    # This is valid even when C differs because both decompositions have
    # first been expanded onto the same underlying Schwarzschild orbits.
    # ------------------------------------------------------------------
    if physical_available:
        family_order = ("z_tube", "x_tube", "box")
        family_labels = ("z-tubes", "x-tubes", "boxes")
        xx = np.arange(len(family_order), dtype=np.float64)

        properties = (
            ("age", rf"$\langle{{\rm Age}}\rangle\, [{{{UTS.gyr}}}]$",
                1, 0.2),
            ("metal", r"$\langle[Z/H]\rangle\, [{\rm dex}]$",
                0, 0.1),
            ("alpha", r"$\langle[\alpha/{\rm Fe}]\rangle\, [{\rm dex}]$",
                2, 0.1),
        )

        for ax, (prop, ylabel, axis_idx, min_span) in zip(orbital_axes,
            properties):
            mock_values = np.asarray([orbital_input[key]["global"][prop]
                for key in family_order], dtype=np.float64)

            fit_values = np.asarray([orbital_fit[key]["global"][prop]
                for key in family_order], dtype=np.float64)

            ax.plot(xx, mock_values, "o-", label=input_label)
            ax.plot(xx, fit_values, "s--", label=fit_label)

            values = np.concatenate((mock_values, fit_values))
            values = values[np.isfinite(values)]

            if values.size:
                data_min = float(np.min(values))
                data_max = float(np.max(values))

                lib_min = float(min(np.min(axes_input[axis_idx]),
                    np.min(axes_fit[axis_idx])))
                lib_max = float(max(np.max(axes_input[axis_idx]),
                    np.max(axes_fit[axis_idx])))

                # Establish the unpadded plotting interval. Use the actual data
                # extent unless it is smaller than the desired minimum span.
                data_span = data_max-data_min

                if data_span < min_span:
                    span = min(min_span, lib_max-lib_min)
                    centre = 0.5*(data_min+data_max)
                    ymin = centre-0.5*span
                    ymax = centre+0.5*span

                    # Keep the artificial minimum-span expansion within the SSP
                    # library, shifting rather than shrinking where possible.
                    if ymin < lib_min:
                        ymax += lib_min-ymin
                        ymin = lib_min

                    if ymax > lib_max:
                        ymin -= ymax-lib_max
                        ymax = lib_max

                    ymin = max(ymin, lib_min)
                    ymax = min(ymax, lib_max)
                else:
                    ymin = data_min
                    ymax = data_max

                # Apply Matplotlib's normal fractional margin to the final
                # interval, including any artificially imposed minimum span.
                ymargin = ax.margins()[1]
                padding = ymargin*(ymax-ymin)

                ax.set_ylim(
                    ymin-padding,
                    ymax+padding,
                )

            ax.set_xticks(xx)
            ax.set_xticklabels(family_labels)
            ax.set_ylabel(ylabel)
            ax.legend(loc="best")

        physical_figure.tight_layout()

        print("[validation] orbital-family stellar populations")

        for key in family_order:
            label = orbital_input[key]["label"]
            mi = orbital_input[key]["global"]
            mf = orbital_fit[key]["global"]

            print(f"  {label}")
            print(f"    age       : {mi['age']:.6g} -> "
                f"{mf['age']:.6g} [{UTS.gyr}]")
            print(f"    [Z/H]     : {mi['metal']:.6g} -> "
                f"{mf['metal']:.6g} [dex]")
            print(f"    [alpha/Fe]: {mi['alpha']:.6g} -> "
                f"{mf['alpha']:.6g} [dex]")

        print("[validation] orbital-family stellar populations")

        for key in family_order:
            label = orbital_input[key]["label"]
            mi = orbital_input[key]["global"]
            mf = orbital_fit[key]["global"]

            print(f"  {label}")
            print(f"    age       : {mi['age']:.6g} -> "
                f"{mf['age']:.6g} [{UTS.gyr}]")
            print(f"    [Z/H]     : {mi['metal']:.6g} -> "
                f"{mf['metal']:.6g} [dex]")
            print(f"    [alpha/Fe]: {mi['alpha']:.6g} -> "
                f"{mf['alpha']:.6g} [dex]")

    # ------------------------------------------------------------------
    # Direct coefficient comparison.
    # ------------------------------------------------------------------

    if coefficient_comparable:
        resid = x_fit-x_input
        input_flat = x_input.ravel(order="C")
        fit_flat = x_fit.ravel(order="C")
        resid_flat = resid.ravel(order="C")

        l1 = float(np.sum(np.abs(resid_flat)))
        l2 = float(np.linalg.norm(resid_flat))
        linf = float(np.max(np.abs(resid_flat)))

        input_l1 = float(np.sum(np.abs(input_flat)))
        input_l2 = float(np.linalg.norm(input_flat))
        input_linf = float(np.max(np.abs(input_flat)))

        rel_l1 = (l1/input_l1 if input_l1 > 0.0 else np.nan)
        rel_l2 = (l2/input_l2 if input_l2 > 0.0 else np.nan)
        rel_linf = (linf/input_linf if input_linf > 0.0 else np.nan)

        denom = float(np.linalg.norm(input_flat)
            * np.linalg.norm(fit_flat))

        if denom > 0.0:
            cosine = float(np.dot(input_flat, fit_flat)/denom)

        if np.std(input_flat) > 0.0 and np.std(fit_flat) > 0.0:
            corr = float(np.corrcoef(input_flat, fit_flat)[0, 1])

        input_pop = np.sum(x_input, axis=0)
        fit_pop = np.sum(x_fit, axis=0)
        input_comp = np.sum(x_input, axis=1)
        fit_comp = np.sum(x_fit, axis=1)

        coefficient_figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        ax = axes[0]
        vmax = float(max(np.max(input_flat), np.max(fit_flat), 0.0))
        pad = 0.025*vmax if vmax > 0.0 else 1.0
        lim = (-pad, vmax+pad)

        ax.scatter(input_flat, fit_flat, s=8, alpha=0.5)
        ax.plot(lim, lim, "k--", lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"input $x_{c,p}$")
        ax.set_ylabel(r"recovered $x_{c,p}$")
        ax.set_title("Coefficient recovery")

        text = (rf"$C={C_input},\ P={x_input.shape[1]}$" "\n"
            rf"$||\Delta x||_1/||x||_1={rel_l1:.3e}$" "\n"
            rf"$||\Delta x||_2/||x||_2={rel_l2:.3e}$" "\n"
            rf"$||\Delta x||_\infty/||x||_\infty="
            rf"{rel_linf:.3e}$" "\n"
            rf"$\rho={corr:.5f}$" "\n"
            rf"$\cos\theta={cosine:.5f}$")

        ax.text(0.03, 0.97, text, va="top", ha="left", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        ax = axes[1]
        vmax = float(max(np.max(input_pop), np.max(fit_pop), 0.0))
        pad = 0.025*vmax if vmax > 0.0 else 1.0
        lim = (-pad, vmax+pad)

        ax.scatter(input_pop, fit_pop, s=12, alpha=0.6)
        ax.plot(lim, lim, "k--", lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"input $\sum_c x_{c,p}$")
        ax.set_ylabel(r"recovered $\sum_c x_{c,p}$")
        ax.set_title("Population recovery")

        ax = axes[2]
        vmax = float(max(np.max(input_comp), np.max(fit_comp), 0.0))
        pad = 0.025*vmax if vmax > 0.0 else 1.0
        lim = (-pad, vmax+pad)

        ax.scatter(input_comp, fit_comp, s=20)
        ax.plot(lim, lim, "k--", lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"input $\sum_p x_{c,p}$")
        ax.set_ylabel(r"recovered $\sum_p x_{c,p}$")
        ax.set_title("Component coefficient mass")

        coefficient_figure.tight_layout()

        print("[validation] coefficient recovery")
        print(f"  shape                 = {x_input.shape}")
        print(f"  relative L1 error     = {rel_l1:.6e}")
        print(f"  relative L2 error     = {rel_l2:.6e}")
        print(f"  relative Linf error   = {rel_linf:.6e}")
        print(f"  correlation           = {corr:.8f}")
        print(f"  cosine similarity     = {cosine:.8f}")

    # ------------------------------------------------------------------
    # Historic fallback when physical SSP coordinates are unavailable.
    # ------------------------------------------------------------------

    if not physical_available:
        with h5py.File(input_path, "r") as f:
            if "/ModelCube" not in f:
                raise RuntimeError(f"{input_path} has no /ModelCube.")
            model_input = np.asarray(f["/ModelCube"][...], dtype=np.float64)
            wave_input = np.asarray(f["/ObsPix"][...], dtype=np.float64)

        with h5py.File(validation_path, "r") as f:
            if "/ModelCube" not in f:
                raise RuntimeError(f"{validation_path} has no /ModelCube.")
            model_fit = np.asarray(f["/ModelCube"][...], dtype=np.float64)
            wave_fit = np.asarray(f["/ObsPix"][...], dtype=np.float64)

        if model_input.shape != model_fit.shape:
            raise ValueError(f"ModelCube shapes differ: "
                f"{model_input.shape} != {model_fit.shape}.")

        if not np.allclose(wave_input, wave_fit, rtol=1e-12, atol=1e-14):
            raise ValueError("ModelCube wavelength grids differ.")

        spectral_resid = model_fit-model_input
        model_norm = float(np.linalg.norm(model_input))

        relative_l2_spectral = (float(np.linalg.norm(spectral_resid))/model_norm
            if model_norm > 0.0 else np.nan)

        scale = np.sqrt(np.mean(model_input*model_input, axis=1))
        rms = np.sqrt(np.mean(spectral_resid*spectral_resid, axis=1))

        fractional_rms = np.divide(rms, scale, out=np.full_like(rms, np.nan),
            where=scale > 0.0)

        spectral_figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.plot(wave_input, np.mean(model_input, axis=0), label=iLabel)
        ax.plot(wave_fit, np.mean(model_fit, axis=0), "--", label=vLabel)
        ax.set_xlabel(r"$\ln\lambda$")
        ax.set_ylabel("Mean model flux")
        ax.set_title("Mean reconstructed spectrum")
        ax.legend()

        ax = axes[1]
        valid = fractional_rms[np.isfinite(fractional_rms)]
        ax.hist(valid, bins=40)
        ax.set_xlabel("Fractional spectral RMS")
        ax.set_ylabel("Spatial bins")
        ax.set_title(rf"Spectral recovery: "
            rf"$\Delta_2={relative_l2_spectral:.3e}$")

        spectral_figure.tight_layout()

        print("[validation] historic SSP provenance")
        print("  assuming different SSP models")
        print("  coefficient comparison is not applicable")
        print("  physical SSP comparison is not available")
        print(f"  spectral relative L2 = "
            f"{relative_l2_spectral:.6e}")

    # ------------------------------------------------------------------
    # Save figures.
    # ------------------------------------------------------------------

    if save_path is not None:
        save_path = plp.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        primary_figure = (physical_figure if physical_figure is not None
            else spectral_figure)

        if primary_figure is not None:
            primary_figure.savefig(save_path, dpi=200, bbox_inches="tight")

        if coefficient_figure is not None:
            coeff_path = save_path.with_name(f"{save_path.stem}_coefficients"
                f"{save_path.suffix}")
            coefficient_figure.savefig(coeff_path, dpi=200, bbox_inches="tight")

    primary_figure = (physical_figure if physical_figure is not None
        else spectral_figure)

    # ------------------------------------------------------------------
    # Return all comparison products.
    # ------------------------------------------------------------------

    out = {
        "x_input": x_input,
        "x_fit": x_fit,
        "same_ssp": same_ssp,
        "C_input": C_input,
        "C_fit": C_fit,
        "same_components": same_components,
        "coefficient_comparable": coefficient_comparable,
        "same_library": same_library,
        "same_axes": same_axes,
        "same_templates": same_templates,
        "physical_comparison_available": physical_available,
        "input_ssp_library": lib_input,
        "fit_ssp_library": lib_fit,
        "input_ssp_axes": axes_input,
        "fit_ssp_axes": axes_fit,
        "input_physical": phys_input,
        "fit_physical": phys_fit,
        "global_physical_input": global_input,
        "global_physical_fit": global_fit,

        "orbit_library": input_nnOrb,
        "orbit_types": orbit_types,
        "schwarzschild_weights": schwarzschild_weights,
        "input_decomposition": input_decomp,
        "fit_decomposition": fit_decomp,
        "input_component_orbit_matrix": M_input,
        "fit_component_orbit_matrix": M_fit,
        "input_orbit_x": orbit_x_input,
        "fit_orbit_x": orbit_x_fit,
        "input_orbital_families": orbital_input,
        "fit_orbital_families": orbital_fit,

        "residual": resid,
        "relative_l1": rel_l1,
        "relative_l2": rel_l2,
        "relative_linf": rel_linf,
        "correlation": corr,
        "cosine_similarity": cosine,
        "spectral_resid": spectral_resid,
        "relative_l2_spectral": relative_l2_spectral,
        "fractional_rms": fractional_rms,

        "figure": primary_figure,
        "physical_figure": physical_figure,
        "coefficient_figure": coefficient_figure,
        "spectral_figure": spectral_figure,
    }

    return out

# ------------------------------------------------------------------------------