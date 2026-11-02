# -*- coding: utf-8 -*-
r"""
    cube_debug.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    A collection of helpful functionality to assist in debugging aspects of the CubeFit pipeline.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Added `nnls_seed_diagnostics` to investigate how the NNLS seed behaves
            as a full solution. 13 December 2025
"""
import time, os
import numpy as np
import pathlib as plp
from copy import copy
from tqdm.auto import tqdm
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import read_global_column_energy
from CubeFit import cube_utils as cu
from CubeFit import kz_init as kzi
from CubeFit.live_fit_dashboard import (
    render_aperture_fits_with_x,
    render_sfh_from_x,
)
from dynamics.IFU.Functions import Plot, Geometric

POT = Plot()
GEO = Geometric()

curdir = plp.Path(__file__).parent
dDir = cu._ddir()
mDir = curdir.parent/'muse'

# ------------------------------------------------------------------------------

def _read_C_P(f) -> tuple[int, int]:
    M = f["/HyperCube/models"]
    _, C, P, _ = map(int, M.shape)
    return C, P

def _row_or_vec_to_CP(arr: np.ndarray, C: int, P: int) -> np.ndarray:
    """Map various X layouts to a (C,P) array."""
    X = np.asarray(arr, np.float64)
    if X.ndim == 2 and X.shape == (C, P):
        return X.copy()
    if X.ndim == 2 and X.shape == (P, C):
        return X.T.copy()
    v = X.ravel(order="C")
    if v.size != C * P:
        raise ValueError(
            f"Cannot reshape X of size {v.size} to (C,P)=({C},{P})."
        )
    return v.reshape(C, P, order="C").copy()

def _read_orbit_weights(f) -> np.ndarray:
    # Same preference order you use elsewhere
    if "/Fit/orbit_weights" in f:
        w = np.asarray(f["/Fit/orbit_weights"][...], np.float64)
    elif "/CompWeights" in f:
        w = np.asarray(f["/CompWeights"][...], np.float64)
    else:
        raise RuntimeError("No orbital weights found (/Fit/orbit_weights or /CompWeights).")
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

def _read_X(h5_path: str,
            x_dset: str | None,
            C: int,
            P: int) -> np.ndarray:
    """
    Read a real solution / seed vector and return it as (C,P).

    x_dset=None → use your usual preference order.
    """
    with open_h5(h5_path, role="reader", swmr=True) as f:
        if x_dset is not None:
            if x_dset not in f:
                raise RuntimeError(f"Requested x_dset '{x_dset}' not found.")
            X_raw = np.asarray(f[x_dset][...], np.float64)
        else:
            # same search order as compare_usage_to_orbit_weights, but main-file only
            for name in ("/X_global",
                         "/Fit/x_latest",
                         "/Seeds/x0_nnls_patch",
                         "/Fit/x_best",
                         "/Fit/x_last",
                         "/Fit/x_epoch_last"):
                if name in f:
                    X_raw = np.asarray(f[name][...], np.float64)
                    break
            else:
                raise RuntimeError("No X dataset found in main HDF5.")
    return _row_or_vec_to_CP(X_raw, C, P)

def _usage(X: np.ndarray,
           E: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (raw_usage, normalized_usage) per component.
    If E is provided, usage is energy-weighted: s_c = sum_p X[c,p]*E[c,p].
    """
    X64 = np.asarray(X, np.float64)
    if E is None:
        s = X64.sum(axis=1)
    else:
        E64 = np.asarray(E, np.float64)
        s = (X64 * E64).sum(axis=1)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.maximum(s, 0.0)
    S = float(s.sum() or 1.0)
    return s, s / S

def debug_test_projectors_on_h5(
    h5_path: str,
    x_dset: str | None = None,
    *,
    use_energy_metric: bool = True,
) -> None:
    """
    Test project_to_component_weights and project_to_component_weights_strict
    on *real* data from an HDF5, as close to runtime as possible.
    """
    with open_h5(h5_path, role="reader", swmr=True) as f:
        C, P = _read_C_P(f)
        w_raw = _read_orbit_weights(f)

    # Reduce orbit_weights to (C,) if needed (C*P -> sum over P)
    w_vec = np.asarray(w_raw, np.float64).ravel(order="C")
    if w_vec.size == C:
        w_c = w_vec.copy()
    elif w_vec.size == C * P:
        w_c = w_vec.reshape(C, P, order="C").sum(axis=1)
    else:
        raise ValueError(
            f"orbit_weights length {w_vec.size} incompatible with C={C}, C*P={C*P}."
        )
    w_c = np.maximum(np.nan_to_num(w_c, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    Wsum = float(w_c.sum() or 1.0)
    w_fracs = w_c / Wsum

    # Real X_cp from file
    X0 = _read_X(h5_path, x_dset=x_dset, C=C, P=P)
    # Real global column energy (same as softbox/usage)
    E_cp = read_global_column_energy(h5_path)

    print(f"[debug] h5={h5_path}")
    print(f"[debug] shapes: X0={X0.shape}, E_cp={E_cp.shape}, w_c={w_c.shape}")

    # Baseline usage
    s_plain0, u_plain0 = _usage(X0, E=None)
    s_energy0, u_energy0 = _usage(X0, E_cp if use_energy_metric else None)

    def _report(label: str, u: np.ndarray) -> None:
        diff = u - w_fracs
        l1 = float(np.sum(np.abs(diff)))
        linf = float(np.max(np.abs(diff)))
        print(f"[{label}] L1={l1:.3e}  L∞={linf:.3e}")

    print("\n[baseline] before any projection:")
    _report("plain-usage", u_plain0)
    _report("energy-usage", u_energy0)
    print("  mass_plain =", float(s_plain0.sum()))
    print("  mass_energy =", float(s_energy0.sum()))

    # ---------- inter-epoch projector (gentle) ----------
    X1 = X0.copy()
    t0 = time.perf_counter()
    cu.project_to_component_weights(
        X1,
        t_vec=w_c,                # same target as runtime
        E_cp=(E_cp if use_energy_metric else None),
        minw=1e-12,
    )
    dt1 = time.perf_counter() - t0

    s_plain1, u_plain1 = _usage(X1, E=None)
    s_energy1, u_energy1 = _usage(X1, E_cp if use_energy_metric else None)

    print("\n[proj] after project_to_component_weights:")
    print(f"  runtime = {dt1:.4e} s")
    print("  finite X1? ", np.isfinite(X1).all(),
          "  min(X1) =", float(np.nanmin(X1)))
    print("  mass_plain  (before/after) =",
          float(s_plain0.sum()), "→", float(s_plain1.sum()))
    print("  mass_energy (before/after) =",
          float(s_energy0.sum()), "→", float(s_energy1.sum()))
    _report("plain-usage", u_plain1)
    _report("energy-usage", u_energy1)

    # # ---------- strict projector (hard constraint, post-epoch) ----------
    # X2 = X0.copy()
    # t0 = time.perf_counter()
    # cu.project_to_component_weights_strict(
    #     X2,
    #     orbit_weights=w_c,        # or full (C*P,) if you want; fn should handle both
    #     E_cp=(E_cp if use_energy_metric else None),
    #     min_target=1e-12,
    # )
    # dt2 = time.perf_counter() - t0

    # s_plain2, u_plain2 = _usage(X2, E=None)
    # s_energy2, u_energy2 = _usage(X2, E_cp if use_energy_metric else None)

    # print("\n[strict] after project_to_component_weights_strict:")
    # print(f"  runtime = {dt2:.4e} s")
    # print("  finite X2? ", np.isfinite(X2).all(),
    #       "  min(X2) =", float(np.nanmin(X2)))
    # print("  mass_plain  (before/after) =",
    #       float(s_plain0.sum()), "→", float(s_plain2.sum()))
    # print("  mass_energy (before/after) =",
    #       float(s_energy0.sum()), "→", float(s_energy2.sum()))
    # _report("plain-usage", u_plain2)
    # _report("energy-usage", u_energy2)

    print("\n[debug] done.\n")

# ------------------------------------------------------------------------------

def nnls_seed_diagnostics(
    h5_path: str,
    *,
    seed_path: Optional[str] = None,
    n_apertures: int = 6,
    apertures: Optional[Sequence[int]] = None,
    show_residual: bool = True,
) -> dict:
    """
    Treat the NNLS seed as the final solution and render standard diagnostics.

    This is a convenience wrapper around `render_aperture_fits_with_x` and
    `render_sfh_from_x` that:
      - reads the seed vector X from the main HDF5 file,
      - flattens it to (C*P,) in the runtime layout,
      - renders a small set of representative aperture fits, and
      - renders the orbital SFH panel.

    Parameters
    ----------
    h5_path : str
        Path to the main CubeFit HDF5 file.
    seed_path : str, optional
        Dataset to use for the seed X. Defaults to the environment
        variable CUBEFIT_SEED_PATH if set, otherwise "/Seeds/x0_nnls_patch".
    n_apertures : int, optional
        Number of apertures to sample uniformly across the field if
        `apertures` is not provided. Default is 6.
    apertures : sequence of int, optional
        Explicit list of aperture indices to plot. If provided, this
        overrides `n_apertures`.
    show_residual : bool, optional
        Whether to overlay residuals in the aperture-fit panel.

    Returns
    -------
    info : dict
        Dictionary with keys:
          - "seed_path" : str
          - "apertures" : list[int]
          - "fits_png" : str
          - "sfh_png" : str
    Execution
    ---------
    >>> info = cdbg.nnls_seed_diagnostics("/data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/NGC4365_207_12.h5", n_apertures=1)
    """

    if seed_path is None:
        seed_path = os.environ.get(
            "CUBEFIT_SEED_PATH", "/Seeds/x0_nnls_patch"
        )

    # Read geometry and seed vector as (C,P)
    with open_h5(h5_path, role="reader", swmr=True) as f:
        C, P = _read_C_P(f)
        S = int(f["/DataCube"].shape[0])

    X_cp = _read_X(h5_path, x_dset=seed_path, C=C, P=P)
    x_flat = np.asarray(X_cp, dtype=np.float64).ravel(order="C")
    x_flat = np.nan_to_num(
        x_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Choose apertures to plot
    if apertures is not None:
        ap_idx = [int(i) for i in apertures if 0 <= int(i) < S]
    else:
        n_plot = int(min(max(1, n_apertures), S))
        ap_idx = np.linspace(0, S - 1, n_plot, dtype=int).tolist()

    base = Path(h5_path).parent / "figures"
    base.mkdir(parents=True, exist_ok=True)

    fits_png = base / "nnls_seed_fits.png"
    sfh_png = base / "nnls_seed_sfh.png"

    render_sfh_from_x(
        h5_path,
        x_flat,
        sfh_png,
    )
    
    if ap_idx:
        render_aperture_fits_with_x(
            h5_path,
            x_flat,
            fits_png,
            apertures=ap_idx,
            show_residual=show_residual,
            title="NNLS seed treated as final solution",
        )


    return {
        "seed_path": str(seed_path),
        "apertures": ap_idx,
        "fits_png": str(fits_png),
        "sfh_png": str(sfh_png),
    }

# ------------------------------------------------------------------------------

def compare_sidecar_solution_to_hypercube(
    x_h5: str,
    hypercube_h5: str,
    *,
    x_key_candidates: tuple[str, ...] = (
        "/X_global",
    ),
    save: str | None = None,
    title: str | None = None,
) -> dict:
    """
    Compare the coefficient solution stored in a dedicated sidecar HDF5 file
    against the solution stored inside a hypercube HDF5 file.

    The function reads the full coefficient matrix from each file, compares the
    coefficient vectors directly, and also compares the per-orbit masses
    obtained by summing over the population axis.

    Parameters
    ----------
    x_h5 : str
        Path to the dedicated solution HDF5 file, e.g. ``x_154_00.h5``.
    hypercube_h5 : str
        Path to the hypercube HDF5 file containing ``/HyperCube/models`` and
        the embedded solution.
    x_key_candidates : tuple of str, optional
        Candidate dataset paths to probe, in order, for the solution vector.
    save : str, optional
        If provided, save the comparison figure to this path.
    title : str, optional
        Optional figure title.

    Returns
    -------
    dict
        Summary dictionary containing coefficient norms, per-orbit residuals,
        and the loaded coefficient arrays.

    Raises
    ------
    RuntimeError
        If a required dataset is missing.
    ValueError
        If the coefficient shapes are incompatible.

    Notes
    -----
    The hypercube solution is taken from the embedded coefficient store, not
    from ``/ModelCube``. This is the cleanest way to detect whether the saved
    solver state itself is stale.
    """
    import os

    def _load_first_solution(
        f,
        *,
        C: int,
        P: int,
        label: str,
    ) -> np.ndarray:
        for key in x_key_candidates:
            if key in f:
                arr = np.asarray(f[key][...], dtype=np.float64)
                break
        else:
            raise RuntimeError(
                f"No solution dataset found in {label}; tried: "
                + ", ".join(x_key_candidates)
            )

        if arr.ndim == 1:
            if arr.size != C * P:
                raise ValueError(
                    f"{label} solution length {arr.size} != C*P={C*P}."
                )
            arr = arr.reshape(C, P)
        elif arr.ndim == 2:
            if arr.shape != (C, P):
                raise ValueError(
                    f"{label} solution shape {arr.shape} != (C,P)=({C},{P})."
                )
        else:
            raise ValueError(
                f"{label} solution must be 1-D or 2-D, got ndim={arr.ndim}."
            )

        return np.asarray(arr, dtype=np.float64, order="C")

    with open_h5(hypercube_h5, role="reader") as fh:
        if "/HyperCube/models" not in fh:
            raise RuntimeError(
                f"{hypercube_h5} does not contain /HyperCube/models."
            )
        _, C, P, _ = map(int, fh["/HyperCube/models"].shape)
        x_hyper = _load_first_solution(
            fh,
            C=C,
            P=P,
            label="hypercube",
        )

    with open_h5(x_h5, role="reader") as fx:
        x_side = _load_first_solution(
            fx,
            C=C,
            P=P,
            label="sidecar",
        )

    if x_side.shape != x_hyper.shape:
        raise ValueError(
            f"Shape mismatch: sidecar {x_side.shape} vs hypercube {x_hyper.shape}."
        )

    dx = x_hyper - x_side
    side_mass = np.sum(x_side, axis=1)
    hyper_mass = np.sum(x_hyper, axis=1)
    dmass = hyper_mass - side_mass

    x_l1 = float(np.sum(np.abs(dx)))
    x_l2 = float(np.linalg.norm(dx))
    x_linf = float(np.max(np.abs(dx))) if dx.size else 0.0

    mass_l1 = float(np.sum(np.abs(dmass)))
    mass_l2 = float(np.linalg.norm(dmass))
    mass_linf = float(np.max(np.abs(dmass))) if dmass.size else 0.0

    rel_mass_l2 = float(mass_l2 / (np.linalg.norm(side_mass) + 1e-30))
    rel_mass_linf = float(
        mass_linf / (np.max(np.abs(side_mass)) + 1e-30)
    )

    eps = 1e-30
    log_side = np.log10(np.maximum(side_mass, eps))
    log_hyper = np.log10(np.maximum(hyper_mass, eps))

    fig = plt.figure(figsize=(10.0, 4.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32)

    # Left: per-orbit mass comparison.
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(log_side, log_hyper, "o", alpha=0.8, ms=5)
    lo = float(np.min([log_side.min(), log_hyper.min()])) if C else -1.0
    hi = float(np.max([log_side.max(), log_hyper.max()])) if C else 1.0
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax0.set_xlabel(r"$\log_{10}(M_{\rm sidecar})$")
    ax0.set_ylabel(r"$\log_{10}(M_{\rm hypercube})$")
    ax0.set_title("Solution comparison")

    txt = (
        rf"$||x_h-x_s||_1={x_l1:.3e}$" "\n"
        rf"$||x_h-x_s||_2={x_l2:.3e}$" "\n"
        rf"$||x_h-x_s||_\infty={x_linf:.3e}$" "\n"
        rf"$||M_h-M_s||_2={mass_l2:.3e}$" "\n"
        rf"rel$_2$={rel_mass_l2:.3e}, rel$_\infty$={rel_mass_linf:.3e}"
    )
    ax0.text(
        0.03,
        0.97,
        txt,
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Right: per-orbit residuals.
    ax1 = fig.add_subplot(gs[0, 1])
    idx = np.arange(C, dtype=int)
    ax1.axhline(0.0, color="k", lw=1.0)
    ax1.bar(idx, dmass, width=0.8)
    ax1.set_xlabel("orbit index")
    ax1.set_ylabel(r"$M_{\rm hypercube}-M_{\rm sidecar}$")
    ax1.set_title("Per-orbit residual")

    if C <= 24:
        ax1.set_xticks(idx)

    if title is not None:
        fig.suptitle(title, y=0.98)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "x_l1": x_l1,
        "x_l2": x_l2,
        "x_linf": x_linf,
        "mass_l1": mass_l1,
        "mass_l2": mass_l2,
        "mass_linf": mass_linf,
        "rel_mass_l2": rel_mass_l2,
        "rel_mass_linf": rel_mass_linf,
        "sidecar_mass": side_mass,
        "hypercube_mass": hyper_mass,
        "mass_resid": dmass,
        "x_sidecar": x_side,
        "x_hypercube": x_hyper,
    }

# ------------------------------------------------------------------------------

def tryC():
    PD = kzi.props('NGC4365')
    galaxy = PD['galaxy']
    mPath = PD['mPath']
    decDir = None
    nCuts = PD['nCuts']
    full = PD['full']
    SN = PD['SN']
    method = 'fsf'
    weighting = 'luminosity'
    slope = 1.30
    proj = 'i'
    lOrder = PD['lOrder']
    IMF = 'KB'
    iso = 'BaSTI'
    # Directories
    bDir = mDir/'tri_models'/mPath
    pDir = curdir.parent/'pxf'
    figDir = curdir/galaxy/'figures'
    MKDIRS = [bDir, pDir, figDir]
    [plp.Path(DIR).mkdir(parents=True, exist_ok=True) for DIR in MKDIRS]
    if isinstance(decDir, type(None)):
        with open(bDir/'decomp.dir', 'r+') as dd:
            decDir = dd.readline().strip()
    if isinstance(nCuts, type(None)):
        direc = list(filter(lambda xd: xd.is_dir(),
            (bDir/decDir).glob('decomp_*')))[0]
    else:
        direc = bDir/decDir/f"decomp_{nCuts:d}"
    if 'fif' in method:
        IMF = 'FIF'
        iso = 'fif'
    if not full:
        tEnd = 'trunc'
    else:
        tEnd = 'full'
    w8Str = f"{weighting[0].upper()}W"
    tag = f"_SN{int(SN):02d}_{iso}_{IMF}{slope:.2f}_{w8Str}"
    # Filenames
    kin = pDir/galaxy/f"kinematics_SN{SN:02d}.xz"
    pfs = pDir/galaxy/f"pixels_SN{SN:02d}.xz"
    sfs =  pDir/galaxy/f"selection_SN{SN:02d}_{tEnd}.xz"
    vbSpec = pDir/galaxy/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    mlfn = pDir/galaxy/f"ML{tag}.xz"
    infn = bDir/'infil.xz'
    gfn = curdir/'obsData'/f"{galaxy}.xz"

    INF = cu.Load.lzma(infn)
    PA = INF['angle'][0]

    xpix, ypix, sele, pixs = cu.Load.lzma(pfs)
    # saur,goods = cu.Load.lzma(sfs)
    # del saur
    xbix, ybix = GEO.rotate2D(xpix, ypix, PA)
    pfn = dDir.parent/'muse'/'obsData'/f"{galaxy}-poly-rot.xz"
    polyProps = dict(ec=POT.brown, linestyle='--', fill=False, zorder=100,
        lw=0.75, salpha=0.5)
    if pfn.is_file():
        aShape = cu.Load.lzma(pfn)
        aShape, pPatch = POT.polyPatch(POLYGON=aShape, Xpo=xbix, Ypo=ybix,
            **polyProps)
    else:
        aShape, pPatch = POT.polyPatch(Xpo=xbix, Ypo=ybix, **polyProps)
        cu.Write.lzma(pfn, aShape)
    xmin, xmax = np.amin(xbix), np.amax(xbix)
    ymin, ymax = np.amin(ybix), np.amax(ybix)
    xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels

    saur, goods = cu.Load.lzma(pDir/galaxy/f"selection_SN{SN:02d}_{tEnd}.xz")
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    xbix = np.compress(goods, xbix)
    ybix = np.compress(goods, ybix)
    decDir, cDirs, cKeys, nComp, teLL, lnGrid, histBinSize, dataVelScale,\
        RZ, spLL, laGrid, lmin, lmax, umetals, uages, ualphas, pixOff = \
        cu._oneTimeSpec(**PD)
    nLSpec, nSpat = laGrid.shape
    nTSpec, nMetals, nAges, nAlphas = lnGrid.shape
    nSSP = int(np.prod((nMetals, nAges, nAlphas), dtype=int))
    pred = f"0{len(repr(nComp)):d}"
    nComp = int(nComp)

    oDict = cu.Load.lzma(direc/f"decomp_{nCuts:d}.plt")
    binFN = oDict['binFN']
    apFN = oDict['apFN']
    dnPix, dgrid = cu.Read.bins(bDir/'infil'/binFN)
    dnbins = int(np.max(dgrid))
    dgrid -= 1
    dss = np.where(dgrid >= 0)[0]
    dx0, dx1, dnx, dy0, dy1, dny, dtheta = cu.Read.aperture(
        bDir/'infil'/apFN)
    ddx = np.abs((dx1-dx0)/dnx)
    ddy = np.abs((dy1-dy0)/dny)
    dpixs = np.min([ddx, ddy])
    dxr = np.arange(dnx)*dpixs + dx0 + 0.5*dpixs
    dyr = np.arange(dny)*dpixs + dy0 + 0.5*dpixs
    dxtss = np.einsum('i,k->ki', dxr, np.full_like(dyr, 1)).ravel()[dss]
    dytss = np.einsum('i,k->ki', np.full_like(dxr, 1), dyr).ravel()[dss]
    dtestX, dtestY = GEO.rotate2D(dxtss, dytss, dtheta)
    duPix, dpInverse, dpCounts = np.unique(dgrid[dss], return_inverse=True,
        return_counts=True)
    dpCount = dpCounts[dpInverse]

    biI = INF['bins'][0]
    bCount = biI['pCountsBin']
    # grid = np.array(biI['grid'], dtype=int).ravel()-1
    grid = np.array(biI['grid'], dtype=int).T.ravel()-1
    nbins = np.max(grid).astype(int)+1
    ss = np.where(grid >= 0)[0]

    if np.max(dpCount) > 1: # at least one bin contains more than one pixel
        # a quick way to check if the oberved scheme was used
        dgrid = grid
        dss = ss
        dnbins = nbins
        dpCount = bCount

    nzComp = np.array(oDict['nzComp'], dtype=int)
    nnOrb = plp.Path(*oDict['nnOrb'])
    oClass = plp.Path(*oDict['oClass'])
    obClass = plp.Path(*oDict['obClass'])
    bLKey = cu.keySep.join([nnOrb.parent.parent.name, nnOrb.parent.name])
    bLKey = cu.rReplace(bLKey, cu.keySep, os.sep, 1)
    nnOrb = plp.Path(bDir, decDir, nnOrb.parent.name, nnOrb.name)
    oClass = plp.Path(bDir, decDir, oClass.parent.name, oClass.name)
    obClass = plp.Path(bDir, decDir, obClass.parent.name, obClass.name)
    fpd = cu._deetExtr(bLKey)
    NOrbs, inds, energs, I2s, I3s, regs, types, weights, lcuts =\
        cu.Read.orbits(nnOrb)
    cWeights = np.array([
        np.ma.sum(oDict['weights'][f"{comp:{pred}d}"]) for comp in nzComp])

    if 'componentTypes' in oDict:
        componentTypes = np.asarray(
            oDict['componentTypes'], dtype=np.int64)
        if componentTypes.size < np.max(nzComp):
            raise RuntimeError(
                "Stored componentTypes is inconsistent with nzComp.")

        otypes = componentTypes[nzComp - 1]
        if not np.all(np.isin(otypes, [0, 1, 2])):
            raise RuntimeError(
                f"Invalid intrinsic orbital types: {np.unique(otypes)}.")

        allCuts = np.asarray(
            [oDict['cuts'][f"{component:{pred}d}"] for component in nzComp],
            dtype=np.float64)

        diskComps = np.flatnonzero(
            (otypes == 0) & (allCuts[:, 2] > 0.5))
        bulgeComps = np.setdiff1d(np.arange(nComp), diskComps)
    elif oDict['cuts'] and len(oDict['cuts']) > 0:
        componentTypes = np.full(nzComp.size, -1, dtype=np.int64)

        for ii, component in enumerate(nzComp):
            key = f"{component:{pred}d}"
            mask = np.asarray(oDict['wheres'][key], dtype=bool)
            orbitTypes = np.unique(types[mask])

            if orbitTypes.size != 1:
                raise RuntimeError(
                    f"Component {component} contains orbital types "
                    f"{orbitTypes.tolist()}.")

            if orbitTypes[0] == 3:
                componentTypes[ii] = 0
            elif orbitTypes[0] == 1:
                componentTypes[ii] = 1
            elif orbitTypes[0] == 4:
                componentTypes[ii] = 2
            else:
                raise RuntimeError(
                    f"Component {component} contains unsupported orbital type "
                    f"{orbitTypes[0]}.")

        otypes = componentTypes
        if not np.all(np.isin(otypes, [0, 1, 2])):
            raise RuntimeError(
                f"Invalid intrinsic orbital types: {np.unique(otypes)}.")

        allCuts = np.asarray(
            [oDict['cuts'][f"{component:{pred}d}"] for component in nzComp],
            dtype=np.float64)

        diskComps = np.flatnonzero(
            (otypes == 0) & (allCuts[:, 2] > 0.5))
        bulgeComps = np.setdiff1d(np.arange(nComp), diskComps)
    else:
        otypes = copy(nzComp) - 1
        diskComps = bulgeComps = None
    
    c3Path = '/data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/hypercube_3_00.h5'
    h5Path = '/data/phys-gal-dynamics/phys2603/CubeFit/NGC4365/hypercube_125_00.h5'

    # # orbital decomposition check
    # with open_h5(c3Path, role="reader") as f:
    #     w3 = np.asarray(f["/CompWeights"][...], dtype=np.float64)

    # with open_h5(h5Path, role="reader") as f:
    #     w125 = np.asarray(f["/CompWeights"][...], dtype=np.float64)

    # w3 /= np.sum(w3)
    # w125 /= np.sum(w125)

    # for tt in range(3):
    #     print(
    #         tt,
    #         w3[tt],
    #         np.sum(w125[otypes == tt]),
    #         np.sum(w125[otypes == tt]) - w3[tt],
    #     )
    
    x0 = cu.constrainUpscaledSolution(str(h5Path).replace('hypercube', 'x').replace(str(nComp), str(3)), h5Path, otypes,
        orbit_weights=cWeights)
    breakpoint()

# ------------------------------------------------------------------------------

def check_c3_cfine_additivity(
    c3_h5: str,
    cfine_h5: str,
    component_types: np.ndarray,
    *,
    n_spax: int = 8,
    n_ssp: int = 8,
) -> dict:
    """
    Quickly test C=3 versus fine-component HyperCube additivity.

    Parameters
    ----------
    c3_h5 : str
        C=3 HyperCube HDF5 file.
    cfine_h5 : str
        Fine-component HyperCube HDF5 file.
    component_types : ndarray
        Fine-component intrinsic types, with values 0, 1, and 2.
    n_spax : int, optional
        Number of spatial pixels to sample.
    n_ssp : int, optional
        Number of SSPs to sample.

    Returns
    -------
    results : dict
        Relative and maximum absolute errors for each sampled family and SSP.

    Raises
    ------
    ValueError
        If HyperCube dimensions or component types are inconsistent.

    Examples
    --------
    >>> out = check_c3_cfine_additivity(
    ...     c3_h5, cfine_h5, component_types)
    """
    component_types = np.asarray(
        component_types, dtype=np.int64).ravel()

    with open_h5(c3_h5, role="reader") as f3, \
            open_h5(cfine_h5, role="reader") as ff:
        M3 = f3["/HyperCube/models"]
        Mf = ff["/HyperCube/models"]

        S3, C3, P3, L3 = map(int, M3.shape)
        Sf, Cf, Pf, Lf = map(int, Mf.shape)

        if C3 != 3 or (S3, P3, L3) != (Sf, Pf, Lf):
            raise ValueError(
                f"Incompatible shapes: C=3 {M3.shape}, fine {Mf.shape}.")

        if component_types.size != Cf:
            raise ValueError(
                f"component_types has {component_types.size} entries; "
                f"expected {Cf}.")

        s_idx = np.unique(np.linspace(
            0, S3 - 1, min(n_spax, S3), dtype=np.int64))
        p_idx = np.unique(np.linspace(
            0, P3 - 1, min(n_ssp, P3), dtype=np.int64))

        results = {}

        print("type  SSP      rel_L2       max_rel      scale")

        for tt in range(3):
            c_idx = np.flatnonzero(component_types == tt)

            if c_idx.size == 0:
                raise ValueError(
                    f"No fine components found for type {tt}.")

            for pp in p_idx:
                A3 = np.asarray(
                    M3[s_idx, tt, int(pp), :], dtype=np.float64)

                Af = np.asarray(
                    Mf[s_idx, :, int(pp), :], dtype=np.float64)

                Af = np.sum(Af[:, c_idx, :], axis=1)

                diff = Af - A3

                norm3 = float(np.linalg.norm(A3))
                rel_l2 = float(
                    np.linalg.norm(diff) / max(norm3, 1e-300))

                scale = float(np.max(np.abs(A3)))
                max_rel = float(
                    np.max(np.abs(diff)) / max(scale, 1e-300))

                results[(tt, int(pp))] = {
                    "rel_l2": rel_l2,
                    "max_rel": max_rel,
                    "scale": scale,
                }

                print(
                    f"{tt:4d} {pp:4d} "
                    f"{rel_l2:11.3e} {max_rel:11.3e} "
                    f"{scale:11.3e}"
                )

    worst_l2 = max(
        value["rel_l2"] for value in results.values())
    worst_max = max(
        value["max_rel"] for value in results.values())

    print(
        f"[ADDITIVITY] worst rel_L2={worst_l2:.3e} "
        f"worst max_rel={worst_max:.3e}"
    )

    return results

# ------------------------------------------------------------------------------

def test_c3_copy_in_cfine(
    c3_solution_path: str,
    cfine_h5_path: str,
    component_types: np.ndarray,
    *,
    s0: int = 1000,
    n_spax: int = 2,
) -> tuple[np.ndarray, float]:
    """
    Test a direct C=3 population copy in a fine-component HyperCube.

    Each fine component receives the complete population vector of its parent
    C=3 intrinsic orbital type,

        x_fine[c, :] = x3[component_types[c], :].

    The model is evaluated over one small contiguous spatial block. Fine
    components are read individually as ``(Sblk, P, L)`` slabs to avoid
    loading an entire ``(Sblk, C, P, L)`` HyperCube block into memory.

    Parameters
    ----------
    c3_solution_path : str
        HDF5 solution file containing the C=3 solution in ``/X_global``.
    cfine_h5_path : str
        Fine-component HyperCube HDF5 file.
    component_types : ndarray
        Zero-based intrinsic orbital type of every fine component. Values
        must be 0, 1, or 2.
    s0 : int, optional
        First spatial pixel in the contiguous test block.
    n_spax : int, optional
        Number of contiguous spatial pixels to evaluate.

    Returns
    -------
    x0 : ndarray, shape (C, P)
        Direct C=3-to-fine coefficient mapping.
    rmse : float
        RMSE over the selected spatial block and fitted wavelengths.

    Raises
    ------
    ValueError
        If dimensions, component types, or spatial bounds are inconsistent.
    RuntimeError
        If the C=3 solution or calculated residuals are invalid.

    Examples
    --------
    >>> x0, rmse = test_c3_copy_in_cfine(
    ...     c3_solution_path, cfine_h5_path, component_types,
    ...     s0=1000, n_spax=4)
    """
    if int(n_spax) <= 0:
        raise ValueError(
            "n_spax must be positive.")

    component_types = np.asarray(
        component_types, dtype=np.int64).ravel(order="C")

    with open_h5(c3_solution_path, role="reader") as f:
        if "/X_global" not in f:
            raise RuntimeError(
                f"{c3_solution_path} does not contain /X_global.")

        x3 = np.asarray(
            f["/X_global"][...], dtype=np.float64)

    if x3.ndim == 1:
        if x3.size % 3 != 0:
            raise ValueError(
                f"C=3 solution has invalid size {x3.size}.")

        x3 = x3.reshape(3, x3.size // 3)

    if x3.ndim != 2 or x3.shape[0] != 3:
        raise ValueError(
            f"C=3 solution has shape {x3.shape}; expected (3, P).")

    if not np.all(np.isfinite(x3)):
        raise RuntimeError(
            "C=3 solution contains non-finite coefficients.")

    if np.any(x3 < 0.0):
        raise RuntimeError(
            "C=3 solution contains negative coefficients.")

    with open_h5(cfine_h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        S, C, P, L = map(int, M.shape)

        if DC.shape != (S, L):
            raise ValueError(
                "DataCube and HyperCube dimensions are inconsistent.")

        if x3.shape[1] != P:
            raise ValueError(
                f"C=3 solution has P={x3.shape[1]}; "
                f"fine HyperCube has P={P}.")

        if component_types.size != C:
            raise ValueError(
                f"component_types has size {component_types.size}; "
                f"expected C={C}.")

        if not np.all(np.isin(component_types, [0, 1, 2])):
            raise ValueError(
                "component_types must contain only 0, 1, and 2.")

        s0 = int(s0)
        s1 = min(S, s0 + int(n_spax))

        if s0 < 0 or s0 >= S or s1 <= s0:
            raise ValueError(
                f"Invalid spatial range [{s0}, {s1}) for S={S}.")

        mask = cu._get_mask(f)
        keep_idx = (
            np.flatnonzero(mask)
            if mask is not None else None
        )

        x0 = x3[component_types, :].copy()

        Y = np.asarray(
            DC[s0:s1, :],
            dtype=np.float64,
            order="C",
        )

        if keep_idx is not None:
            Y = Y[:, keep_idx]

        yhat = np.zeros_like(Y)

        for cc in tqdm(
            range(C),
            desc="[C3->C] direct-copy test",
            leave=False,
        ):
            # Read only one component at a time. This keeps peak memory small
            # and follows the same HDF5 access pattern as the streaming
            # full-gradient worker.
            M_cc = M[s0:s1, cc, :, :][...]

            if keep_idx is not None:
                M_cc = M_cc[:, :, keep_idx]

            M_cc = np.asarray(
                M_cc,
                dtype=np.float64,
                order="C",
            )

            yhat += np.tensordot(
                x0[cc],
                M_cc,
                axes=(0, 1),
            )

            del M_cc

    resid = Y - yhat

    if not np.all(np.isfinite(resid)):
        raise RuntimeError(
            "Non-finite residual encountered in direct-copy test.")

    rmse = float(
        np.sqrt(
            np.mean(resid * resid)
        )
    )

    print(
        f"[C3->C direct] C={C} P={P} "
        f"spax={s0}:{s1} "
        f"nnz={np.count_nonzero(x0)}/{x0.size} "
        f"rmse={rmse:.6e}",
        flush=True,
    )

    return x0, rmse

# ------------------------------------------------------------------------------

def test_solution_rmse(
    solution_path: str,
    h5_path: str,
    *,
    s0: int = 1000,
    n_spax: int = 2,
) -> float:
    """
    Compute the RMSE of an existing CubeFit solution over a small spatial block.

    Parameters
    ----------
    solution_path : str
        HDF5 solution file containing ``/X_global``.
    h5_path : str
        Matching HyperCube HDF5 file containing ``/DataCube`` and
        ``/HyperCube/models``.
    s0 : int, optional
        First spatial pixel in the contiguous test block.
    n_spax : int, optional
        Number of contiguous spatial pixels to evaluate.

    Returns
    -------
    rmse : float
        RMSE over the selected spatial pixels and fitted wavelengths.

    Raises
    ------
    ValueError
        If the solution and HyperCube dimensions are inconsistent.
    RuntimeError
        If the solution or calculated residuals contain non-finite values.

    Examples
    --------
    >>> rmse = test_solution_rmse(
    ...     c3_solution_path, c3_h5_path, s0=1000, n_spax=2)
    """
    with open_h5(solution_path, role="reader") as f:
        if "/X_global" not in f:
            raise RuntimeError(
                f"{solution_path} does not contain /X_global.")

        x = np.asarray(
            f["/X_global"][...], dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(
            f"X_global has shape {x.shape}; expected (C, P).")

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            "X_global contains non-finite coefficients.")

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]
        M = f["/HyperCube/models"]

        S, C, P, L = map(int, M.shape)

        if x.shape != (C, P):
            raise ValueError(
                f"X_global has shape {x.shape}; expected {(C, P)}.")

        s0 = int(s0)
        s1 = min(S, s0 + int(n_spax))

        if s0 < 0 or s0 >= S or s1 <= s0:
            raise ValueError(
                f"Invalid spatial range [{s0}, {s1}) for S={S}.")

        mask = cu._get_mask(f)
        keep_idx = (
            np.flatnonzero(mask)
            if mask is not None else None
        )

        Y = np.asarray(
            DC[s0:s1, :], dtype=np.float64, order="C")

        if keep_idx is not None:
            Y = Y[:, keep_idx]

        yhat = np.zeros_like(Y)

        # Read one component at a time rather than the complete C axis.
        for cc in range(C):
            M_cc = M[s0:s1, cc, :, :][...]

            if keep_idx is not None:
                M_cc = M_cc[:, :, keep_idx]

            M_cc = np.asarray(
                M_cc, dtype=np.float64, order="C")

            yhat += np.tensordot(
                x[cc], M_cc, axes=(0, 1))

            del M_cc

    resid = Y - yhat

    if not np.all(np.isfinite(resid)):
        raise RuntimeError(
            "Non-finite residual encountered.")

    rmse = float(np.sqrt(np.mean(resid * resid)))

    print(
        f"[RMSE] C={C} P={P} spax={s0}:{s1} "
        f"rmse={rmse:.6e}",
        flush=True,
    )

    return rmse

# ------------------------------------------------------------------------------

def diagnose_orbit_weight_encoding(
    hypercube_path: str,
    solution_path: str,
) -> dict:
    """
    Compare HyperCube component amplitudes, orbit weights, and a free solution.

    Parameters
    ----------
    hypercube_path : str
        HyperCube HDF5 containing the models and orbit weights.
    solution_path : str
        HDF5 containing the unconstrained solution.

    Returns
    -------
    result : dict
        Per-component amplitudes, weights, coefficient masses, and correlations.

    Raises
    ------
    RuntimeError
        If required HyperCube normalization data are unavailable.
    ValueError
        If the stored dimensions are inconsistent.

    Examples
    --------
    >>> out = diagnose_orbit_weight_encoding(hypercube_path, solution_path)
    """
    with open_h5(hypercube_path, role="reader") as f:
        C, P = _read_C_P(f)
        w_raw = _read_orbit_weights(f)

        if "/HyperCube/norm/losvd_amp" not in f:
            raise RuntimeError(
                "Missing /HyperCube/norm/losvd_amp.")

        amp_sc = np.asarray(
            f["/HyperCube/norm/losvd_amp"][...],
            dtype=np.float64,
        )
        norm_mode = str(
            f["/HyperCube"].attrs.get("norm.mode", "unknown")
        ).lower()

    w = np.asarray(w_raw, dtype=np.float64).ravel(order="C")
    if w.size == C * P:
        w = w.reshape(C, P, order="C").sum(axis=1)
    elif w.size != C:
        raise ValueError(
            f"orbit_weights has size {w.size}; expected {C} or {C * P}.")

    if amp_sc.ndim != 2 or amp_sc.shape[1] != C:
        raise ValueError(
            f"losvd_amp has shape {amp_sc.shape}; expected (S, {C}).")

    x = _read_X(solution_path, x_dset=None, C=C, P=P)

    if not np.all(np.isfinite(x)):
        raise ValueError("Solution contains non-finite coefficients.")
    if np.any(x < 0.0):
        raise ValueError("Solution contains negative coefficients.")

    amp = np.sum(amp_sc, axis=0, dtype=np.float64)
    x_mass = np.sum(x, axis=1, dtype=np.float64)

    def _normalize(v):
        v = np.asarray(v, dtype=np.float64)
        total = float(np.sum(v, dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Cannot normalize a non-positive vector.")
        return v / total

    def _logcorr(a, b):
        good = (
            np.isfinite(a) & np.isfinite(b)
            & (a > 0.0) & (b > 0.0)
        )
        if np.count_nonzero(good) < 2:
            return np.nan
        return float(np.corrcoef(
            np.log10(a[good]),
            np.log10(b[good]),
        )[0, 1])

    w_norm = _normalize(w)
    amp_norm = _normalize(amp)
    x_mass_norm = _normalize(x_mass)
    effective_norm = _normalize(amp * x_mass)

    ratio = np.full(C, np.nan, dtype=np.float64)
    good = w_norm > 0.0
    ratio[good] = amp_norm[good] / w_norm[good]

    inv_amp = np.zeros(C, dtype=np.float64)
    good_amp = amp > 0.0
    inv_amp[good_amp] = 1.0 / amp[good_amp]

    corr_amp_weight = _logcorr(amp, w)
    corr_x_weight = _logcorr(x_mass, w)
    corr_x_amp = _logcorr(x_mass, amp)
    corr_x_inv_amp = _logcorr(x_mass, inv_amp)

    print(f"[orbit encoding] norm.mode={norm_mode}")
    print(
        " c       weight        LOSVD          x-sum"
        "        A*x          A/w"
    )
    for cc in range(C):
        print(
            f"{cc:2d}  {w_norm[cc]:12.5e}  {amp_norm[cc]:12.5e}  "
            f"{x_mass_norm[cc]:12.5e}  "
            f"{effective_norm[cc]:12.5e}  {ratio[cc]:9.4f}"
        )

    print()
    print(f"corr(log A, log weight) = {corr_amp_weight:+.6f}")
    print(f"corr(log x, log weight) = {corr_x_weight:+.6f}")
    print(f"corr(log x, log A)      = {corr_x_amp:+.6f}")
    print(f"corr(log x, log 1/A)    = {corr_x_inv_amp:+.6f}")

    return {
        "norm_mode": norm_mode,
        "orbit_weights": w,
        "orbit_weights_norm": w_norm,
        "losvd_amplitude": amp,
        "losvd_amplitude_norm": amp_norm,
        "x_orbit_mass": x_mass,
        "x_orbit_mass_norm": x_mass_norm,
        "effective_amplitude_norm": effective_norm,
        "losvd_over_weight": ratio,
        "corr_log_losvd_weight": corr_amp_weight,
        "corr_log_x_weight": corr_x_weight,
        "corr_log_x_losvd": corr_x_amp,
        "corr_log_x_inverse_losvd": corr_x_inv_amp,
    }

# ------------------------------------------------------------------------------

def test_hypercube_boundary_convex_hull(h5_path, mask=None, component=None,
    max_iter=2000, ftol=1e-12):
    """
    Test whether interior HyperCube population columns lie in the convex hull
    of SSP-grid boundary columns.

    For each dynamical component, every interior population column is fitted
    using a nonnegative unit-sum mixture of boundary population columns from
    the same component:

        min_w ||B w - a||_2^2

    subject to

        w >= 0
        sum(w) = 1.

    This tests the actual spectral-model geometry seen by CubeFit, including
    the dynamical component, LOSVD convolution, spectral sampling, and any
    other transformations already encoded in the HyperCube.

    Parameters
    ----------
    h5_path : str
        CubeFit HyperCube HDF5 file.
    mask : ndarray, optional
        Boolean wavelength mask. True retains a wavelength pixel. If None,
        ``/Mask`` is used when available, otherwise all wavelengths are used.
    component : int, optional
        Test only this dynamical-component index. If None, test every
        component.
    max_iter : int, optional
        Maximum SLSQP iterations for each convex-hull projection.
    ftol : float, optional
        SLSQP objective tolerance.

    Returns
    -------
    out : dict
        SSP coordinates, boundary mask, tested components, relative
        reconstruction errors, and convex-hull weights.

    Raises
    ------
    ValueError
        If dimensions, mask, or component index are invalid.
    RuntimeError
        If required HyperCube datasets are absent or an optimization fails.

    Examples
    --------
    >>> out = test_hypercube_boundary_convex_hull(
    ...     "hypercube_109_00.h5", component=0)
    """
    h5_path = str(h5_path)

    with open_h5(h5_path, role="reader") as f:
        required = (
            "/HyperCube/models",
            "/SSPMetals",
            "/SSPAges",
            "/SSPAlphas",
        )

        missing = [name for name in required if name not in f]

        if missing:
            raise RuntimeError(
                f"HyperCube is missing required datasets: {missing}")

        M = f["/HyperCube/models"]

        if M.ndim != 4:
            raise RuntimeError(
                "/HyperCube/models must have shape (S, C, P, L).")

        S, C, P, L = map(int, M.shape)

        metals = np.asarray(f["/SSPMetals"][...],
            dtype=np.float64).ravel(order="C")
        ages = np.asarray(f["/SSPAges"][...],
            dtype=np.float64).ravel(order="C")
        alphas = np.asarray(f["/SSPAlphas"][...],
            dtype=np.float64).ravel(order="C")

        nMetals = int(metals.size)
        nAges = int(ages.size)
        nAlphas = int(alphas.size)

        if nMetals*nAges*nAlphas != P:
            raise RuntimeError(
                "SSP dimensions are inconsistent with HyperCube P: "
                f"{nMetals}*{nAges}*{nAlphas} != {P}.")

        if mask is None and "/Mask" in f:
            candidate = np.asarray(
                f["/Mask"][...], dtype=bool).ravel(order="C")

            if candidate.size == L:
                mask = candidate

        if mask is None:
            keep_idx = None
            L_keep = L
        else:
            mask = np.asarray(mask, dtype=bool).ravel(order="C")

            if mask.size != L:
                raise ValueError(
                    f"mask has size {mask.size}; expected L={L}.")

            keep_idx = np.flatnonzero(mask).astype(np.int64)
            L_keep = int(keep_idx.size)

    # ------------------------------------------------------------
    # Population-grid boundary.
    # ------------------------------------------------------------
    boundary = np.zeros(
        (nMetals, nAges, nAlphas), dtype=bool)

    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    boundary_flat = boundary.ravel(order="C")
    boundary_idx = np.flatnonzero(boundary_flat).astype(np.int64)
    interior_idx = np.flatnonzero(~boundary_flat).astype(np.int64)

    if component is None:
        components = np.arange(C, dtype=np.int64)
    else:
        component = int(component)

        if component < 0 or component >= C:
            raise ValueError(
                f"component must lie in [0, {C - 1}].")

        components = np.array([component], dtype=np.int64)

    print(
        f"HyperCube: S={S}, C={C}, P={P}, L={L}, "
        f"L_keep={L_keep}")
    print(
        f"SSP grid: {nMetals} x {nAges} x {nAlphas}")
    print(
        f"Boundary populations: {boundary_idx.size}; "
        f"interior populations: {interior_idx.size}")
    print(
        f"Testing {components.size} dynamical component(s).")

    # One result per C,P location. Boundary populations are left NaN because
    # the primary experiment is specifically interior -> boundary.
    residual_relative = np.full(
        (C, P), np.nan, dtype=np.float64)

    residual_rms = np.full(
        (C, P), np.nan, dtype=np.float64)

    weights = {}

    def _solve_convex_hull(target, basis):
        """Project one model column onto the convex hull of basis columns."""
        n_basis = int(basis.shape[1])

        x0 = np.full(
            n_basis, 1.0/n_basis, dtype=np.float64)

        def objective(w):
            resid = basis @ w-target
            return 0.5*float(np.dot(resid, resid))

        def gradient(w):
            return basis.T @ (basis @ w-target)

        result = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, None)]*n_basis,
            constraints={
                "type": "eq",
                "fun": lambda w: np.sum(w)-1.0,
                "jac": lambda w: np.ones_like(w),
            },
            options={
                "maxiter": int(max_iter),
                "ftol": float(ftol),
                "disp": False,
            },
        )

        if not result.success:
            raise RuntimeError(
                "Convex-hull projection failed: "
                f"{result.message}")

        w = np.maximum(
            np.asarray(result.x, dtype=np.float64), 0.0)

        total = float(np.sum(w))

        if total <= 0.0:
            raise RuntimeError(
                "Convex-hull projection returned zero total weight.")

        w /= total

        return w

    # ------------------------------------------------------------
    # Read and test one component at a time.
    #
    # This deliberately avoids loading the complete S x C x P x L
    # HyperCube into memory.
    # ------------------------------------------------------------
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]

        for nc, cc in enumerate(components, start=1):
            print(
                f"\nComponent {cc} "
                f"({nc}/{components.size})")

            # Shape after transpose/reshape:
            #
            #     (P, S * L_keep)
            #
            # Each row is therefore the complete model column for one
            # population of this dynamical component.
            if keep_idx is None:
                block = np.asarray(
                    M[:, cc, :, :], dtype=np.float64)
            else:
                block = np.asarray(
                    M[:, cc, :, :], dtype=np.float64)
                block = block[:, :, keep_idx]

            spectra = block.transpose(1, 0, 2).reshape(P, -1)

            del block

            # Normalize each complete HyperCube column to unit Euclidean norm.
            # This tests spectral/model SHAPE rather than arbitrary column
            # amplitude. It also mirrors the solver's column normalization
            # conceptually.
            norms = np.linalg.norm(spectra, axis=1)

            valid_norm = np.isfinite(norms) & (norms > 0.0)

            if not np.all(valid_norm[boundary_idx]):
                bad = boundary_idx[~valid_norm[boundary_idx]]

                raise RuntimeError(
                    f"Component {cc} has zero/nonfinite boundary "
                    f"columns: {bad.tolist()}")

            spectra = np.divide(
                spectra,
                norms[:, None],
                out=np.zeros_like(spectra),
                where=norms[:, None] > 0.0)

            # Basis columns are all boundary populations for this same
            # dynamical component.
            basis = spectra[boundary_idx, :].T

            for ni, pp in enumerate(interior_idx, start=1):
                if not valid_norm[pp]:
                    continue

                target = spectra[pp, :]

                w = _solve_convex_hull(target, basis)
                model = basis @ w
                resid = target-model

                residual_rms[cc, pp] = float(
                    np.sqrt(np.mean(resid*resid)))

                residual_relative[cc, pp] = float(
                    np.linalg.norm(resid)/
                    max(np.linalg.norm(target), 1e-30))

                weights[(int(cc), int(pp))] = {
                    "source_indices": boundary_idx.copy(),
                    "weights": w.copy(),
                }

                print(
                    f"\r  interior {ni:3d}/{interior_idx.size}: "
                    f"p={pp:3d} "
                    f"rel={residual_relative[cc, pp]:.3e}",
                    end="",
                    flush=True,
                )

            print()

            del spectra
            del basis

    # ------------------------------------------------------------
    # Global summary.
    # ------------------------------------------------------------
    tested = residual_relative[np.isfinite(residual_relative)]

    print()
    print("Interior -> boundary convex-hull reconstruction")
    print("------------------------------------------------")

    if tested.size:
        print(
            f"N       = {tested.size}")
        print(
            f"min     = {np.min(tested):.6e}")
        print(
            f"p16     = {np.percentile(tested, 16):.6e}")
        print(
            f"median  = {np.median(tested):.6e}")
        print(
            f"p84     = {np.percentile(tested, 84):.6e}")
        print(
            f"max     = {np.max(tested):.6e}")

    return {
        "h5_path": h5_path,
        "shape": (S, C, P, L),
        "metals": metals,
        "ages": ages,
        "alphas": alphas,
        "boundary_mask": boundary,
        "boundary_indices": boundary_idx,
        "interior_indices": interior_idx,
        "components": components,
        "residual_rms": residual_rms,
        "residual_relative": residual_relative,
        "weights": weights,
    }

# ------------------------------------------------------------------------------

def diagnose_dynamical_population_mirroring(hypercube_path, x_path,
    population_floor=1e-8, dynamical_cosine_cut=0.2,
    population_cosine_cut=0.9):
    """
    Test whether dynamically distinct components receive similar SSP mixtures.

    Parameters
    ----------
    hypercube_path : str or pathlib.Path
        CubeFit HyperCube HDF5 file.
    x_path : str or pathlib.Path
        CubeFit solution HDF5 file containing ``/X_global``.
    population_floor : float, optional
        Relative coefficient floor applied separately to each component
        before calculating population similarity. Default is 1e-8.
    dynamical_cosine_cut : float, optional
        Component pairs below this spatial-amplitude cosine are classified
        as dynamically dissimilar. Default is 0.2.
    population_cosine_cut : float, optional
        Population cosine above this value is classified as strongly
        mirrored. Default is 0.9.

    Returns
    -------
    out : dict
        Component and pairwise diagnostics.

    Raises
    ------
    ValueError
        If stored dimensions are inconsistent.
    RuntimeError
        If required datasets are unavailable or invalid.

    Examples
    --------
    >>> out = diagnose_dynamical_population_mirroring(
    ...     hdf5Path, xPath)
    """
    from CubeFit.hdf5_manager import open_h5

    hypercube_path = str(hypercube_path)
    x_path = str(x_path)

    with open_h5(hypercube_path, role="reader") as f:
        required = (
            "/HyperCube/norm/losvd_amp",
            "/SSPMetals",
            "/SSPAges",
            "/SSPAlphas",
        )

        missing = [key for key in required if key not in f]

        if missing:
            raise RuntimeError(
                "Missing HyperCube datasets: " + ", ".join(missing))

        amp = np.asarray(
            f["/HyperCube/norm/losvd_amp"][...],
            dtype=np.float64, order="C")

        metals = np.asarray(
            f["/SSPMetals"][...],
            dtype=np.float64).ravel(order="C")

        ages = np.asarray(
            f["/SSPAges"][...],
            dtype=np.float64).ravel(order="C")

        alphas = np.asarray(
            f["/SSPAlphas"][...],
            dtype=np.float64).ravel(order="C")

    with open_h5(x_path, role="reader") as f:
        if "/X_global" not in f:
            raise RuntimeError("Solution file has no /X_global.")

        X = np.asarray(
            f["/X_global"][...],
            dtype=np.float64, order="C")

    if X.ndim != 2:
        raise ValueError(
            f"/X_global must have shape (C,P); got {X.shape}.")

    C, P = map(int, X.shape)
    S = int(amp.shape[0])

    n_metals = int(metals.size)
    n_ages = int(ages.size)
    n_alphas = int(alphas.size)

    if amp.shape != (S, C):
        raise ValueError(
            f"losvd_amp has shape {amp.shape}; expected (S,{C}).")

    if n_metals*n_ages*n_alphas != P:
        raise ValueError(
            "SSP dimensions are inconsistent with /X_global.")

    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(amp)):
        raise RuntimeError("Stored solution/amplitudes contain non-finite values.")

    scale = max(float(np.max(np.abs(X))), 1.0)

    if np.min(X) < -1e-10*scale:
        raise RuntimeError(
            "/X_global contains materially negative coefficients.")

    X = np.maximum(X, 0.0)

    # ------------------------------------------------------------------
    # Normalize each component population independently.
    # ------------------------------------------------------------------
    component_mass = np.sum(X, axis=1)

    if np.any(component_mass <= 0.0):
        raise RuntimeError(
            "At least one fitted component has zero coefficient mass.")

    Q = X/component_mass[:, None]

    # Remove numerical tails independently for each component, then
    # renormalize. This prevents tiny coefficients from masquerading as
    # genuine population overlap.
    for cc in range(C):
        cut = population_floor*float(np.max(Q[cc]))
        Q[cc, Q[cc] < cut] = 0.0

        total = float(np.sum(Q[cc]))

        if total > 0.0:
            Q[cc] /= total

    # ------------------------------------------------------------------
    # Population cosine similarity.
    # ------------------------------------------------------------------
    qnorm = np.linalg.norm(Q, axis=1)

    Qn = np.divide(
        Q, qnorm[:, None],
        out=np.zeros_like(Q),
        where=qnorm[:, None] > 0.0)

    population_cosine = Qn @ Qn.T
    np.clip(population_cosine, 0.0, 1.0, out=population_cosine)

    # Total-variation distance is useful alongside cosine:
    #
    #     0 -> identical mixtures
    #     1 -> disjoint mixtures.
    population_tv = np.zeros((C, C), dtype=np.float64)

    for cc in range(C):
        population_tv[cc] = 0.5*np.sum(
            np.abs(Q[cc][None, :]-Q), axis=1)

    # ------------------------------------------------------------------
    # Dynamical spatial-map cosine similarity.
    # ------------------------------------------------------------------
    amp_norm = np.linalg.norm(amp, axis=0)

    An = np.divide(
        amp,
        amp_norm[None, :],
        out=np.zeros_like(amp),
        where=amp_norm[None, :] > 0.0)

    dynamical_cosine = An.T @ An
    np.clip(dynamical_cosine, 0.0, 1.0, out=dynamical_cosine)

    # ------------------------------------------------------------------
    # SSP coordinates and boundary masks.
    # ------------------------------------------------------------------
    indices = np.asarray([
        np.unravel_index(
            pp, (n_metals, n_ages, n_alphas), order="C")
        for pp in range(P)
    ], dtype=np.int64)

    iz = indices[:, 0]
    ia = indices[:, 1]
    ialpha = indices[:, 2]

    metal_boundary = (iz == 0) | (iz == n_metals-1)
    age_boundary = (ia == 0) | (ia == n_ages-1)
    alpha_boundary = (ialpha == 0) | (ialpha == n_alphas-1)

    any_boundary = (
        metal_boundary | age_boundary | alpha_boundary)

    # ------------------------------------------------------------------
    # Per-component population summaries.
    # ------------------------------------------------------------------
    dominant_population = np.argmax(Q, axis=1)

    dominant_fraction = Q[
        np.arange(C), dominant_population]

    boundary_fraction = np.sum(
        Q[:, any_boundary], axis=1)

    age_boundary_fraction = np.sum(
        Q[:, age_boundary], axis=1)

    metal_boundary_fraction = np.sum(
        Q[:, metal_boundary], axis=1)

    alpha_boundary_fraction = np.sum(
        Q[:, alpha_boundary], axis=1)

    metal_grid = metals[iz]
    age_grid = ages[ia]
    alpha_grid = alphas[ialpha]

    mean_metal = Q @ metal_grid
    mean_age = Q @ age_grid
    mean_alpha = Q @ alpha_grid

    projected_amp = np.sum(amp, axis=0)
    relative_amp = projected_amp/np.max(projected_amp)

    # ------------------------------------------------------------------
    # Pair statistics.
    # ------------------------------------------------------------------
    iu = np.triu_indices(C, k=1)

    dyn_pair = dynamical_cosine[iu]
    pop_pair = population_cosine[iu]
    tv_pair = population_tv[iu]

    dissimilar = dyn_pair < dynamical_cosine_cut
    mirrored = pop_pair > population_cosine_cut

    dissimilar_mirrored = dissimilar & mirrored

    # Exact same dominant SSP is an even simpler mirroring statistic.
    same_dominant = (
        dominant_population[iu[0]]
        == dominant_population[iu[1]])

    dissimilar_same_dominant = (
        dissimilar & same_dominant)

    # Store explicit component-pair records.
    pair_records = []

    for kk in range(dyn_pair.size):
        cc = int(iu[0][kk])
        dd = int(iu[1][kk])

        pair_records.append((
            cc,
            dd,
            float(dyn_pair[kk]),
            float(pop_pair[kk]),
            float(tv_pair[kk]),
            bool(same_dominant[kk]),
        ))

    # The most interesting pairs are dynamically most dissimilar while
    # retaining high population similarity.
    suspicious = [
        item for item in pair_records
        if item[2] < dynamical_cosine_cut
    ]

    suspicious.sort(
        key=lambda item: (-item[3], item[2]))

    # ------------------------------------------------------------------
    # Dominant-SSP occupancy.
    # ------------------------------------------------------------------
    occupancy = np.bincount(
        dominant_population, minlength=P)

    occupied = np.flatnonzero(occupancy > 0)

    occupancy_order = occupied[
        np.argsort(occupancy[occupied])[::-1]]

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print()
    print("="*78)
    print("DYNAMICAL / POPULATION MIRRORING DIAGNOSTIC")
    print("="*78)
    print(f"C={C}  P={P}  S={S}")
    print(f"Distinct dominant SSPs      : {occupied.size}/{C}")
    print(
        f"Median boundary mass        : "
        f"{np.median(boundary_fraction):.6f}")
    print(
        f"Median age-boundary mass    : "
        f"{np.median(age_boundary_fraction):.6f}")
    print(
        f"Median metal-boundary mass  : "
        f"{np.median(metal_boundary_fraction):.6f}")
    print(
        f"Median alpha-boundary mass  : "
        f"{np.median(alpha_boundary_fraction):.6f}")

    print()
    print("All component pairs:")
    print(
        f"  dynamical cosine min/median/max = "
        f"{np.min(dyn_pair):.6f} / "
        f"{np.median(dyn_pair):.6f} / "
        f"{np.max(dyn_pair):.6f}")
    print(
        f"  population cosine min/median/max = "
        f"{np.min(pop_pair):.6f} / "
        f"{np.median(pop_pair):.6f} / "
        f"{np.max(pop_pair):.6f}")
    print(
        f"  population TV min/median/max = "
        f"{np.min(tv_pair):.6f} / "
        f"{np.median(tv_pair):.6f} / "
        f"{np.max(tv_pair):.6f}")

    print()
    print(
        f"Dynamically dissimilar pairs "
        f"(cos < {dynamical_cosine_cut:g}): "
        f"{np.count_nonzero(dissimilar)}/{dyn_pair.size}")

    if np.any(dissimilar):
        print(
            f"  population cosine min/median/max = "
            f"{np.min(pop_pair[dissimilar]):.6f} / "
            f"{np.median(pop_pair[dissimilar]):.6f} / "
            f"{np.max(pop_pair[dissimilar]):.6f}")
        print(
            f"  population TV min/median/max = "
            f"{np.min(tv_pair[dissimilar]):.6f} / "
            f"{np.median(tv_pair[dissimilar]):.6f} / "
            f"{np.max(tv_pair[dissimilar]):.6f}")
        print(
            f"  strongly mirrored "
            f"(pop cosine > {population_cosine_cut:g}) = "
            f"{np.count_nonzero(dissimilar_mirrored)}/"
            f"{np.count_nonzero(dissimilar)}")
        print(
            f"  same dominant SSP = "
            f"{np.count_nonzero(dissimilar_same_dominant)}/"
            f"{np.count_nonzero(dissimilar)}")

    print()
    print("Most commonly dominant SSPs:")
    print(
        "  p   count      [Z/H]      age    alpha   boundary")

    for pp in occupancy_order[:15]:
        print(
            f"{pp:3d}  {occupancy[pp]:5d}  "
            f"{metal_grid[pp]:9.4f}  "
            f"{age_grid[pp]:7.3f}  "
            f"{alpha_grid[pp]:7.3f}  "
            f"{str(bool(any_boundary[pp])):>8s}")

    print()
    print("Per component:")
    print(
        " comp    A/Amax  dom_p   dom_frac      <Z>      <age>   "
        "<alpha>  boundary")

    for cc in range(C):
        print(
            f"{cc:4d}  "
            f"{relative_amp[cc]:8.3e}  "
            f"{dominant_population[cc]:5d}  "
            f"{dominant_fraction[cc]:9.5f}  "
            f"{mean_metal[cc]:8.4f}  "
            f"{mean_age[cc]:8.3f}  "
            f"{mean_alpha[cc]:8.4f}  "
            f"{boundary_fraction[cc]:8.5f}")

    print()
    print("Most suspicious dynamically distinct pairs:")
    print(
        "  c1   c2   dyn_cos   pop_cos      TV    same_dom")

    for cc, dd, dyn, pop, tv, same in suspicious[:25]:
        print(
            f"{cc:4d} {dd:4d}  "
            f"{dyn:8.5f}  "
            f"{pop:8.5f}  "
            f"{tv:8.5f}  "
            f"{str(same):>8s}")

    print("="*78)
    print()

    return {
        "population": Q,
        "population_cosine": population_cosine,
        "population_tv": population_tv,
        "dynamical_cosine": dynamical_cosine,
        "component_mass": component_mass,
        "projected_amplitude": projected_amp,
        "relative_amplitude": relative_amp,
        "dominant_population": dominant_population,
        "dominant_fraction": dominant_fraction,
        "dominant_occupancy": occupancy,
        "boundary_fraction": boundary_fraction,
        "age_boundary_fraction": age_boundary_fraction,
        "metal_boundary_fraction": metal_boundary_fraction,
        "alpha_boundary_fraction": alpha_boundary_fraction,
        "mean_metal": mean_metal,
        "mean_age": mean_age,
        "mean_alpha": mean_alpha,
        "dynamically_dissimilar": dissimilar,
        "dissimilar_population_cosine": pop_pair[dissimilar],
        "dissimilar_population_tv": tv_pair[dissimilar],
        "dissimilar_mirrored": dissimilar_mirrored,
        "dissimilar_same_dominant": dissimilar_same_dominant,
        "suspicious_pairs": suspicious,
    }

# ------------------------------------------------------------------------------

def diagnose_extreme_ssp_reconstruction(h5_path,
    populations=(4, 129, 224, 310, 312, 314), coeff_floor=1e-4):
    """
    Inspect nonnegative reconstructions of selected SSP templates.

    Each selected SSP is reconstructed from every other SSP in the stored
    library. The function reports the reconstruction error and the physical
    SSP coordinates receiving significant reconstruction weight.

    Parameters
    ----------
    h5_path : str
        CubeFit HyperCube HDF5 path.
    populations : sequence of int, optional
        Flattened SSP indices to test.
    coeff_floor : float, optional
        Report reconstruction coefficients larger than this fraction of the
        total reconstruction coefficient sum. Default is 1e-4.

    Returns
    -------
    out : dict
        Reconstruction coefficients, errors, and physical summaries.

    Raises
    ------
    ValueError
        If inputs or stored SSP dimensions are invalid.
    RuntimeError
        If required HDF5 datasets are unavailable.

    Examples
    --------
    >>> out = diagnose_extreme_ssp_reconstruction(
    ...     hdf5Path, populations=(4, 129, 224, 310, 312, 314))
    """
    import numpy as np
    from scipy.optimize import nnls
    from CubeFit.hdf5_manager import open_h5

    populations = np.asarray(
        populations, dtype=np.int64).ravel(order="C")

    with open_h5(h5_path, role="reader") as f:
        required = (
            "/Templates",
            "/SSPMetals",
            "/SSPAges",
            "/SSPAlphas",
        )

        missing = [key for key in required if key not in f]

        if missing:
            raise RuntimeError(
                "Missing HDF5 datasets: " + ", ".join(missing))

        T = np.asarray(
            f["/Templates"][...],
            dtype=np.float64, order="C")

        metals = np.asarray(
            f["/SSPMetals"][...],
            dtype=np.float64).ravel(order="C")

        ages = np.asarray(
            f["/SSPAges"][...],
            dtype=np.float64).ravel(order="C")

        alphas = np.asarray(
            f["/SSPAlphas"][...],
            dtype=np.float64).ravel(order="C")

    n_metals = int(metals.size)
    n_ages = int(ages.size)
    n_alphas = int(alphas.size)
    P = n_metals*n_ages*n_alphas

    if T.ndim != 2 or T.shape[0] != P:
        raise ValueError(
            f"/Templates has shape {T.shape}; expected ({P}, T_c).")

    if np.any(populations < 0) or np.any(populations >= P):
        raise ValueError(
            f"populations must lie in [0, {P-1}].")

    if not np.isfinite(coeff_floor) or coeff_floor < 0.0:
        raise ValueError(
            "coeff_floor must be finite and nonnegative.")

    # Physical coordinates for every flattened population.
    indices = np.asarray([
        np.unravel_index(
            pp, (n_metals, n_ages, n_alphas), order="C")
        for pp in range(P)
    ], dtype=np.int64)

    iz = indices[:, 0]
    ia = indices[:, 1]
    ialpha = indices[:, 2]

    metal_grid = metals[iz]
    age_grid = ages[ia]
    alpha_grid = alphas[ialpha]

    # Normalize spectra exactly as in the previous redundancy test.
    template_norm = np.linalg.norm(T, axis=1)

    if np.any(~np.isfinite(template_norm)) or np.any(template_norm <= 0.0):
        raise RuntimeError(
            "/Templates contains invalid spectra.")

    Tn = T/template_norm[:, None]

    results = {}

    for pp in populations:
        pp = int(pp)

        use = np.ones(P, dtype=bool)
        use[pp] = False

        other = np.flatnonzero(use)

        coeff_other, residual = nnls(
            Tn[use].T,
            Tn[pp])

        coeff = np.zeros(P, dtype=np.float64)
        coeff[other] = coeff_other

        coeff_sum = float(np.sum(coeff))

        if coeff_sum <= 0.0:
            raise RuntimeError(
                f"SSP {pp} has an empty NNLS reconstruction.")

        # q is used only to interpret the reconstruction as a population
        # mixture. The actual spectral reconstruction remains coeff.
        q = coeff/coeff_sum

        significant = q >= coeff_floor
        significant_idx = np.flatnonzero(significant)

        order = significant_idx[
            np.argsort(q[significant_idx])[::-1]]

        mean_metal = float(np.sum(q*metal_grid))
        mean_age = float(np.sum(q*age_grid))
        mean_alpha = float(np.sum(q*alpha_grid))

        var_metal = float(np.sum(
            q*(metal_grid-mean_metal)**2))
        var_age = float(np.sum(
            q*(age_grid-mean_age)**2))
        var_alpha = float(np.sum(
            q*(alpha_grid-mean_alpha)**2))

        effective_n = float(np.exp(
            -np.sum(q[q > 0.0]*np.log(q[q > 0.0]))))

        target_coord = (
            float(metal_grid[pp]),
            float(age_grid[pp]),
            float(alpha_grid[pp]),
        )

        # Distances are reported axis by axis because the three SSP
        # coordinates have different physical units/scales.
        mean_abs_dmetal = float(np.sum(
            q*np.abs(metal_grid-metal_grid[pp])))
        mean_abs_dage = float(np.sum(
            q*np.abs(age_grid-age_grid[pp])))
        mean_abs_dalpha = float(np.sum(
            q*np.abs(alpha_grid-alpha_grid[pp])))

        results[pp] = {
            "coefficients": coeff,
            "population_fraction": q,
            "residual": float(residual),
            "coefficient_sum": coeff_sum,
            "significant_indices": order,
            "effective_n": effective_n,
            "mean_metal": mean_metal,
            "mean_age": mean_age,
            "mean_alpha": mean_alpha,
            "std_metal": np.sqrt(var_metal),
            "std_age": np.sqrt(var_age),
            "std_alpha": np.sqrt(var_alpha),
            "mean_abs_dmetal": mean_abs_dmetal,
            "mean_abs_dage": mean_abs_dage,
            "mean_abs_dalpha": mean_abs_dalpha,
        }

        print()
        print("="*76)
        print(
            f"SSP {pp}: "
            f"[Z/H]={target_coord[0]:.4f} "
            f"age={target_coord[1]:.3f} "
            f"alpha={target_coord[2]:.3f}")
        print("="*76)

        print(
            f"fractional spectral residual : {residual:.6e}")
        print(
            f"raw coefficient sum         : {coeff_sum:.6e}")
        print(
            f"effective reconstruction N  : {effective_n:.3f}")

        print(
            f"weighted mean               : "
            f"[Z/H]={mean_metal:.4f} "
            f"age={mean_age:.3f} "
            f"alpha={mean_alpha:.4f}")

        print(
            f"weighted std                : "
            f"[Z/H]={np.sqrt(var_metal):.4f} "
            f"age={np.sqrt(var_age):.3f} "
            f"alpha={np.sqrt(var_alpha):.4f}")

        print(
            f"mean |delta|                : "
            f"[Z/H]={mean_abs_dmetal:.4f} "
            f"age={mean_abs_dage:.3f} "
            f"alpha={mean_abs_dalpha:.4f}")

        print()
        print(
            f"Reconstruction populations "
            f"(fraction >= {coeff_floor:g}):")
        print(
            "  p      fraction      [Z/H]      age    alpha")

        for qq in order:
            print(
                f"{qq:3d}  "
                f"{q[qq]:12.6e}  "
                f"{metal_grid[qq]:9.4f}  "
                f"{age_grid[qq]:7.3f}  "
                f"{alpha_grid[qq]:7.3f}")

    print()
    print("="*76)
    print("SUMMARY")
    print("="*76)
    print(
        "  p       residual    Neff    "
        "<|dZ|>   <|dt|>  <|dalpha|>")

    for pp in populations:
        rr = results[int(pp)]

        print(
            f"{int(pp):3d}  "
            f"{rr['residual']:12.6e}  "
            f"{rr['effective_n']:6.2f}  "
            f"{rr['mean_abs_dmetal']:7.4f}  "
            f"{rr['mean_abs_dage']:7.3f}  "
            f"{rr['mean_abs_dalpha']:10.4f}")

    print("="*76)

    return results

# ------------------------------------------------------------------------------

def diagnose_ssp_neighbour_distances(h5_path):
    """
    Measure spectral distances between adjacent SSP grid points.

    Tests whether the physical SSP grid has strongly nonuniform spectral
    resolution, particularly at its boundaries.

    Parameters
    ----------
    h5_path : str
        CubeFit HyperCube HDF5 path.

    Returns
    -------
    out : dict
        Neighbour distances and summary statistics for each SSP axis.

    Raises
    ------
    ValueError
        If the stored SSP grid is inconsistent.
    RuntimeError
        If required datasets are unavailable or invalid.

    Examples
    --------
    >>> out = diagnose_ssp_neighbour_distances(hdf5Path)
    """
    import numpy as np
    from CubeFit.hdf5_manager import open_h5

    with open_h5(h5_path, role="reader") as f:
        required = (
            "/Templates",
            "/SSPMetals",
            "/SSPAges",
            "/SSPAlphas",
        )

        missing = [key for key in required if key not in f]

        if missing:
            raise RuntimeError(
                "Missing HDF5 datasets: " + ", ".join(missing))

        T = np.asarray(
            f["/Templates"][...], dtype=np.float64, order="C")

        metals = np.asarray(
            f["/SSPMetals"][...],
            dtype=np.float64).ravel(order="C")

        ages = np.asarray(
            f["/SSPAges"][...],
            dtype=np.float64).ravel(order="C")

        alphas = np.asarray(
            f["/SSPAlphas"][...],
            dtype=np.float64).ravel(order="C")

    n_metals = int(metals.size)
    n_ages = int(ages.size)
    n_alphas = int(alphas.size)
    shape = (n_metals, n_ages, n_alphas)
    P = int(np.prod(shape))

    if T.ndim != 2 or T.shape[0] != P:
        raise ValueError(
            f"/Templates has shape {T.shape}; expected ({P}, T_c).")

    norms = np.linalg.norm(T, axis=1)

    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("/Templates contains invalid spectra.")

    T = T/norms[:, None]

    def _p(iz, ia, ialpha):
        return np.ravel_multi_index(
            (iz, ia, ialpha), shape, order="C")

    records = {
        "metal": [],
        "age": [],
        "alpha": [],
    }

    # ------------------------------------------------------------------
    # Metallicity neighbours.
    # ------------------------------------------------------------------
    for iz in range(n_metals-1):
        for ia in range(n_ages):
            for ialpha in range(n_alphas):
                p0 = _p(iz, ia, ialpha)
                p1 = _p(iz+1, ia, ialpha)

                distance = float(np.linalg.norm(T[p1]-T[p0]))

                boundary = (
                    iz == 0
                    or iz+1 == n_metals-1
                )

                records["metal"].append((
                    p0, p1, distance, boundary,
                    metals[iz], metals[iz+1],
                    ages[ia], alphas[ialpha],
                ))

    # ------------------------------------------------------------------
    # Age neighbours.
    # ------------------------------------------------------------------
    for iz in range(n_metals):
        for ia in range(n_ages-1):
            for ialpha in range(n_alphas):
                p0 = _p(iz, ia, ialpha)
                p1 = _p(iz, ia+1, ialpha)

                distance = float(np.linalg.norm(T[p1]-T[p0]))

                boundary = (
                    ia == 0
                    or ia+1 == n_ages-1
                )

                records["age"].append((
                    p0, p1, distance, boundary,
                    ages[ia], ages[ia+1],
                    metals[iz], alphas[ialpha],
                ))

    # ------------------------------------------------------------------
    # Alpha neighbours.
    # ------------------------------------------------------------------
    for iz in range(n_metals):
        for ia in range(n_ages):
            for ialpha in range(n_alphas-1):
                p0 = _p(iz, ia, ialpha)
                p1 = _p(iz, ia, ialpha+1)

                distance = float(np.linalg.norm(T[p1]-T[p0]))

                boundary = (
                    ialpha == 0
                    or ialpha+1 == n_alphas-1
                )

                records["alpha"].append((
                    p0, p1, distance, boundary,
                    alphas[ialpha], alphas[ialpha+1],
                    metals[iz], ages[ia],
                ))

    out = {}

    print()
    print("="*78)
    print("SSP NEAREST-NEIGHBOUR SPECTRAL-DISTANCE DIAGNOSTIC")
    print("="*78)
    print(
        f"SSP grid: {n_metals} x {n_ages} x {n_alphas} = {P}")

    for axis in ("metal", "age", "alpha"):
        rec = records[axis]

        distance = np.asarray(
            [rr[2] for rr in rec], dtype=np.float64)

        boundary = np.asarray(
            [rr[3] for rr in rec], dtype=bool)

        interior = ~boundary

        boundary_values = distance[boundary]
        interior_values = distance[interior]

        med_all = float(np.median(distance))
        med_boundary = float(np.median(boundary_values))
        med_interior = float(np.median(interior_values))

        print()
        print(f"{axis.upper()} AXIS")
        print("-"*78)
        print(
            f"all edges      : N={distance.size:4d} "
            f"median={med_all:.6e} "
            f"min={np.min(distance):.6e} "
            f"max={np.max(distance):.6e}")
        print(
            f"boundary edges : N={boundary_values.size:4d} "
            f"median={med_boundary:.6e} "
            f"min={np.min(boundary_values):.6e} "
            f"max={np.max(boundary_values):.6e}")
        print(
            f"interior edges : N={interior_values.size:4d} "
            f"median={med_interior:.6e} "
            f"min={np.min(interior_values):.6e} "
            f"max={np.max(interior_values):.6e}")
        print(
            f"boundary/interior median distance = "
            f"{med_boundary/med_interior:.6f}")

        # Proposed raw spectral weight before clipping:
        #
        #     w = (median distance / local distance)^2
        #
        raw_weight = (med_all/distance)**2

        order_short = np.argsort(distance)
        order_long = order_short[::-1]

        print()
        print("10 spectrally closest adjacent pairs:")

        for kk in order_short[:10]:
            rr = rec[int(kk)]

            if axis == "metal":
                print(
                    f"  p={rr[0]:3d}->{rr[1]:3d} "
                    f"d={rr[2]:.6e} "
                    f"w={raw_weight[kk]:8.3f} "
                    f"Z={rr[4]:.3f}->{rr[5]:.3f} "
                    f"age={rr[6]:.1f} "
                    f"alpha={rr[7]:.1f} "
                    f"boundary={rr[3]}")

            elif axis == "age":
                print(
                    f"  p={rr[0]:3d}->{rr[1]:3d} "
                    f"d={rr[2]:.6e} "
                    f"w={raw_weight[kk]:8.3f} "
                    f"age={rr[4]:.1f}->{rr[5]:.1f} "
                    f"Z={rr[6]:.3f} "
                    f"alpha={rr[7]:.1f} "
                    f"boundary={rr[3]}")

            else:
                print(
                    f"  p={rr[0]:3d}->{rr[1]:3d} "
                    f"d={rr[2]:.6e} "
                    f"w={raw_weight[kk]:8.3f} "
                    f"alpha={rr[4]:.1f}->{rr[5]:.1f} "
                    f"Z={rr[6]:.3f} "
                    f"age={rr[7]:.1f} "
                    f"boundary={rr[3]}")

        print()
        print("10 spectrally most separated adjacent pairs:")

        for kk in order_long[:10]:
            rr = rec[int(kk)]

            if axis == "metal":
                print(
                    f"  p={rr[0]:3d}->{rr[1]:3d} "
                    f"d={rr[2]:.6e} "
                    f"w={raw_weight[kk]:8.3f} "
                    f"Z={rr[4]:.3f}->{rr[5]:.3f} "
                    f"age={rr[6]:.1f} "
                    f"alpha={rr[7]:.1f} "
                    f"boundary={rr[3]}")

            elif axis == "age":
                print(
                    f"  p={rr[0]:3d}->{rr[1]:3d} "
                    f"d={rr[2]:.6e} "
                    f"w={raw_weight[kk]:8.3f} "
                    f"age={rr[4]:.1f}->{rr[5]:.1f} "
                    f"Z={rr[6]:.3f} "
                    f"alpha={rr[7]:.1f} "
                    f"boundary={rr[3]}")

            else:
                print(
                    f"  p={rr[0]:3d}->{rr[1]:3d} "
                    f"d={rr[2]:.6e} "
                    f"w={raw_weight[kk]:8.3f} "
                    f"alpha={rr[4]:.1f}->{rr[5]:.1f} "
                    f"Z={rr[6]:.3f} "
                    f"age={rr[7]:.1f} "
                    f"boundary={rr[3]}")

        out[axis] = {
            "records": rec,
            "distance": distance,
            "boundary": boundary,
            "raw_weight": raw_weight,
            "median": med_all,
            "boundary_median": med_boundary,
            "interior_median": med_interior,
        }

    print()
    print("="*78)

    return out

# ------------------------------------------------------------------------------

def test_redundant_ssp_replacement(hypercube_path, x_path, error_cube,
    coeff_floor=1e-10, s_chunk=128):
    """
    Replace fitted SSP coefficients by their leave-one-out redundant
    representations and measure the observational effect.

    Only SSPs that actually carry fitted population are reconstructed.
    The modified solution preserves each dynamical component's total
    coefficient mass exactly.

    The test evaluates

        sqrt(mean((A @ (x_new - x_old) / sigma)**2))

    over fitted pixels with valid observational uncertainties.

    Parameters
    ----------
    hypercube_path : str or pathlib.Path
        CubeFit HyperCube HDF5 file.
    x_path : str or pathlib.Path
        CubeFit solution HDF5 file containing ``/X_global``.
    error_cube : ndarray
        Observational 1-sigma errors with shape ``(S, L)``.
    coeff_floor : float, optional
        Ignore fitted coefficients below this fraction of the largest
        coefficient in their component. Default is ``1e-10``.
    s_chunk : int, optional
        Spatial read chunk. Default is 128.

    Returns
    -------
    out : dict
        Original and redistributed solutions, reconstruction information,
        and model-change diagnostics.

    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    RuntimeError
        If required datasets are unavailable or invalid.

    Examples
    --------
    >>> out = test_redundant_ssp_replacement(
    ...     hdf5Path, xPath, errorCube)
    """
    from scipy.optimize import nnls
    from CubeFit.hdf5_manager import open_h5

    hypercube_path = str(hypercube_path)
    x_path = str(x_path)
    error_cube = np.asarray(error_cube, dtype=np.float64)
    s_chunk = int(s_chunk)

    # ------------------------------------------------------------------
    # Read the small quantities first.
    # ------------------------------------------------------------------
    with open_h5(hypercube_path, role="reader") as f:
        required = (
            "/HyperCube/models",
            "/Templates",
            "/Mask",
            "/SSPMetals",
            "/SSPAges",
            "/SSPAlphas",
        )

        missing = [key for key in required if key not in f]

        if missing:
            raise RuntimeError(
                "Missing datasets: " + ", ".join(missing))

        S, C, P, L = map(
            int, f["/HyperCube/models"].shape)

        T = np.asarray(
            f["/Templates"][...],
            dtype=np.float64, order="C")

        mask = np.asarray(
            f["/Mask"][...],
            dtype=bool).ravel(order="C")

        metals = np.asarray(
            f["/SSPMetals"][...],
            dtype=np.float64).ravel(order="C")

        ages = np.asarray(
            f["/SSPAges"][...],
            dtype=np.float64).ravel(order="C")

        alphas = np.asarray(
            f["/SSPAlphas"][...],
            dtype=np.float64).ravel(order="C")

    with open_h5(x_path, role="reader") as f:
        if "/X_global" not in f:
            raise RuntimeError("Solution file has no /X_global.")

        x_old = np.asarray(
            f["/X_global"][...],
            dtype=np.float64, order="C")

    if x_old.shape != (C, P):
        raise ValueError(
            f"/X_global has shape {x_old.shape}; expected {(C, P)}.")

    if error_cube.shape != (S, L):
        raise ValueError(
            f"error_cube has shape {error_cube.shape}; "
            f"expected {(S, L)}.")

    if T.shape[0] != P:
        raise ValueError(
            f"/Templates has shape {T.shape}; expected P={P}.")

    # ------------------------------------------------------------------
    # Identify only SSPs that actually carry fitted mass.
    # ------------------------------------------------------------------
    x_old = np.maximum(x_old, 0.0)

    used = np.zeros(P, dtype=bool)

    for cc in range(C):
        xmax = float(np.max(x_old[cc]))

        if xmax <= 0.0:
            continue

        used |= x_old[cc] > coeff_floor*xmax

    used_p = np.flatnonzero(used)

    print(
        f"[REDUNDANCY] fitted SSPs requiring reconstruction: "
        f"{used_p.size}/{P}",
        flush=True)

    # ------------------------------------------------------------------
    # Normalize the stored SSP spectra and calculate the leave-one-out
    # reconstruction only for SSPs actually used by this solution.
    # ------------------------------------------------------------------
    norm = np.linalg.norm(T, axis=1)

    if np.any(norm <= 0.0) or np.any(~np.isfinite(norm)):
        raise RuntimeError("/Templates contains invalid spectra.")

    Tn = T/norm[:, None]

    reconstruction = {}
    reconstruction_error = {}

    for pp in used_p:
        use = np.ones(P, dtype=bool)
        use[pp] = False

        other = np.flatnonzero(use)

        coeff, residual = nnls(
            Tn[use].T,
            Tn[pp])

        full = np.zeros(P, dtype=np.float64)
        full[other] = coeff

        # We need coefficient-mass conservation because the production
        # solution has an equal-component-mass constraint.
        total = float(np.sum(full))

        if total <= 0.0:
            raise RuntimeError(
                f"Empty reconstruction for SSP {pp}.")

        full /= total

        reconstruction[int(pp)] = full
        reconstruction_error[int(pp)] = float(residual)

    # ------------------------------------------------------------------
    # Replace every materially populated SSP by its redundant
    # representation.
    #
    # Each unit of x[:,pp] is replaced by a unit-sum nonnegative vector,
    # so component mass is conserved analytically.
    # ------------------------------------------------------------------
    x_new = np.zeros_like(x_old)

    for cc in range(C):
        xmax = float(np.max(x_old[cc]))

        if xmax <= 0.0:
            continue

        cut = coeff_floor*xmax

        for pp in range(P):
            value = float(x_old[cc, pp])

            if value <= 0.0:
                continue

            if value > cut and pp in reconstruction:
                x_new[cc] += value*reconstruction[pp]
            else:
                x_new[cc, pp] += value

    # Remove roundoff and enforce exactly the original component sums.
    x_new = np.maximum(x_new, 0.0)

    mass_old = np.sum(x_old, axis=1)
    mass_new = np.sum(x_new, axis=1)

    for cc in range(C):
        if mass_new[cc] > 0.0:
            x_new[cc] *= mass_old[cc]/mass_new[cc]

    mass_error = float(np.max(
        np.abs(np.sum(x_new, axis=1)-mass_old)))

    # ------------------------------------------------------------------
    # Determine which dynamical components actually changed.
    # ------------------------------------------------------------------
    delta = x_new-x_old

    delta_scale = max(
        float(np.max(np.abs(delta))), 1.0)

    changed_coeff = (
        np.abs(delta) > 1e-12*delta_scale)

    changed_components = np.flatnonzero(
        np.any(changed_coeff, axis=1))

    print(
        f"[REDUNDANCY] changed components: "
        f"{changed_components.size}/{C}",
        flush=True)

    print(
        f"[REDUNDANCY] component-mass error: "
        f"{mass_error:.6e}",
        flush=True)

    # ------------------------------------------------------------------
    # Valid observational pixels.
    # ------------------------------------------------------------------
    valid = (
        mask[None, :]
        & np.isfinite(error_cube)
        & (error_cube > 0.0)
    )

    n_valid = int(np.count_nonzero(valid))

    if n_valid == 0:
        raise RuntimeError(
            "No fitted pixels have valid positive errors.")

    # ------------------------------------------------------------------
    # ONE streaming HyperCube pass, reading only changed components.
    #
    # Directly calculate
    #
    #     delta_model = A @ (x_new - x_old)
    #
    # rather than constructing either complete model cube.
    # ------------------------------------------------------------------
    noise_ss = 0.0
    model_ss = 0.0
    model_n = 0
    max_noise_delta = 0.0

    with open_h5(hypercube_path, role="reader") as f:
        M = f["/HyperCube/models"]

        for s0 in range(0, S, s_chunk):
            s1 = min(s0+s_chunk, S)
            ns = s1-s0

            delta_model = np.zeros(
                (ns, L), dtype=np.float64)

            for cc in changed_components:
                block = np.asarray(
                    M[s0:s1, int(cc), :, :],
                    dtype=np.float64, order="C")

                delta_model += np.einsum(
                    "spl,p->sl",
                    block,
                    delta[cc],
                    optimize=True)

            local_valid = valid[s0:s1]
            local_sigma = error_cube[s0:s1]

            dv = delta_model[local_valid]
            sv = local_sigma[local_valid]

            weighted = dv/sv

            noise_ss += float(np.dot(weighted, weighted))

            model_ss += float(np.dot(dv, dv))
            model_n += int(dv.size)

            if weighted.size:
                max_noise_delta = max(max_noise_delta,
                    float(np.max(np.abs(weighted))))

    noise_rms = float(np.sqrt(noise_ss/n_valid))

    model_change_rms = float(np.sqrt(model_ss/max(model_n, 1)))

    # ------------------------------------------------------------------
    # Compact physical summaries before and after.
    # ------------------------------------------------------------------
    n_metals = int(metals.size)
    n_ages = int(ages.size)
    n_alphas = int(alphas.size)

    indices = np.asarray([np.unravel_index(
        pp, (n_metals, n_ages, n_alphas), order="C") for pp in range(P)
    ], dtype=np.int64)

    iz = indices[:, 0]
    ia = indices[:, 1]
    ialpha = indices[:, 2]

    boundary = ((iz == 0) | (iz == n_metals-1)
        | (ia == 0) | (ia == n_ages-1)
        | (ialpha == 0) | (ialpha == n_alphas-1))

    def _boundary_fraction(x):
        total = np.sum(x, axis=1)

        return np.divide(np.sum(x[:, boundary], axis=1), total,
            out=np.zeros(C, dtype=np.float64), where=total > 0.0)

    boundary_old = _boundary_fraction(x_old)
    boundary_new = _boundary_fraction(x_new)

    print()
    print("="*72)
    print("REDUNDANT SSP REPLACEMENT TEST")
    print("="*72)
    print(f"used SSPs                  : {used_p.size}/{P}")
    print(f"changed components         : "
        f"{changed_components.size}/{C}")
    print(f"component-mass error       : {mass_error:.6e}")
    print(f"model-change noise RMS     : {noise_rms:.6e} sigma")
    print(f"maximum pixel change       : {max_noise_delta:.6e} sigma")
    print(f"ordinary model-change RMS  : {model_change_rms:.6e}")
    print(f"median boundary mass before: {np.median(boundary_old):.6f}")
    print(f"median boundary mass after : {np.median(boundary_new):.6f}")

    print()
    print('Reconstructed fitted SSPs:')
    print('  p      error       dominant replacement')

    for pp in used_p:
        rr = reconstruction[int(pp)]
        qq = int(np.argmax(rr))

        print(f"{pp:3d}  {reconstruction_error[int(pp)]:.6e}  "
            f"{qq:3d} ({metals[iz[qq]]:.3f}, {ages[ia[qq]]:.1f}, "
            f"{alphas[ialpha[qq]]:.1f}) frac={rr[qq]:.5f}")

    print("="*72)

    return {
        "x_original": x_old,
        "x_redundant": x_new,
        "delta_x": delta,
        "used_populations": used_p,
        "changed_components": changed_components,
        "reconstruction": reconstruction,
        "reconstruction_error": reconstruction_error,
        "noise_rms": noise_rms,
        "noise_ss": noise_ss,
        "max_noise_delta": max_noise_delta,
        "model_change_rms": model_change_rms,
        "mass_error": mass_error,
        "boundary_fraction_original": boundary_old,
        "boundary_fraction_redundant": boundary_new,
    }

# ------------------------------------------------------------------------------
