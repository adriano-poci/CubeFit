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

def diagnose_young_continuum_observed(h5_path: str, max_order: int = 4,
    min_age_fraction: float = 1e-4, coeff_tol: float = 0.0,
    s_tile: int = 128, clip_sigma: float = 3.5,
    save: str | None = None, show: bool = True) -> dict:
    """
    Test young-population continuum substitution in observed HyperCube space.

    The fitted age distribution is split at the largest empty gap between
    occupied SSP ages. Only nonzero young fitted HyperCube columns are read to
    reconstruct the exact observed young contribution. The old model is then

        M_old = M_full - M_young.

    Multiplicative Legendre polynomials are fitted independently in every
    aperture to transform ``M_old`` into ``M_full``. This directly measures
    whether smooth continuum freedom can reproduce the spatially and
    kinematically convolved effect of the fitted young population.

    A second polynomial is fitted to ``Data/M_old`` to compare the continuum
    implied by the young population with that independently preferred by the
    observations.

    Parameters
    ----------
    h5_path : str
        Completed CubeFit HDF5 file.
    max_order : int, optional
        Highest Legendre order to test. Default is 4.
    min_age_fraction : float, optional
        Minimum global coefficient fraction for an age to count as occupied.
        Default is ``1e-4``.
    coeff_tol : float, optional
        Ignore young coefficients less than or equal to this value. Default is
        zero, giving an exact reconstruction of all positive young weights.
    s_tile : int, optional
        Spatial apertures read per HyperCube slab. Default is 128.
    clip_sigma : float, optional
        MAD clipping threshold for the data-continuum fit. Default is 3.5.
    save : str, optional
        Diagnostic figure output path. Default is None.
    show : bool, optional
        Display the diagnostic figure. Default is True.

    Returns
    -------
    result : dict
        Age split, young-column support, reconstructed young and old models,
        polynomial coefficients, shape recovery, data-fit recovery, and
        continuum-agreement diagnostics.

    Raises
    ------
    RuntimeError
        If required datasets are absent or separated age groups are not found.
    ValueError
        If SSP, model, or solution dimensions are inconsistent.

    Examples
    --------
    >>> out = diagnose_young_continuum_observed(
    ...     "hypercube_109_00.h5", max_order=4,
    ...     save="young_continuum_observed.png")
    """
    max_order = int(max_order)
    s_tile = max(1, int(s_tile))
    min_age_fraction = float(min_age_fraction)
    coeff_tol = float(coeff_tol)
    clip_sigma = float(clip_sigma)

    if max_order < 0:
        raise ValueError("max_order must be non-negative.")
    if min_age_fraction <= 0.0:
        raise ValueError("min_age_fraction must be positive.")
    if coeff_tol < 0.0:
        raise ValueError("coeff_tol must be non-negative.")
    if clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be positive.")

    required = ("/DataCube", "/ModelCube", "/HyperCube/models",
        "/X_global", "/SSPAges", "/SSPMetals", "/SSPAlphas",
        "/Templates")

    with open_h5(h5_path, role="reader", swmr=True) as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise RuntimeError(f"Missing required datasets: {missing}.")

        M = f["/HyperCube/models"]
        S, C, P, L = map(int, M.shape)

        data = np.asarray(f["/DataCube"][...], dtype=np.float64)
        model_full = np.asarray(f["/ModelCube"][...], dtype=np.float64)
        X = _row_or_vec_to_CP(
            np.asarray(f["/X_global"][...], dtype=np.float64), C, P)

        ages = np.asarray(f["/SSPAges"][...], dtype=np.float64).ravel()
        metals = np.asarray(f["/SSPMetals"][...], dtype=np.float64).ravel()
        alphas = np.asarray(f["/SSPAlphas"][...], dtype=np.float64).ravel()
        pop_shape = tuple(np.asarray(
            f["/Templates"].attrs["pop_shape"], dtype=np.int64))

        obs = np.asarray(f["/ObsPix"][...], dtype=np.float64).ravel() \
            if "/ObsPix" in f else np.arange(L, dtype=np.float64)

        if "/Mask" in f:
            mask = np.asarray(f["/Mask"][...], dtype=bool).ravel()
        else:
            mask = np.ones(L, dtype=bool)

        expected_shape = (metals.size, ages.size, alphas.size)
        if pop_shape != expected_shape:
            raise ValueError(
                f"pop_shape={pop_shape}; expected {expected_shape}.")
        if int(np.prod(pop_shape)) != P:
            raise ValueError(
                f"SSP shape {pop_shape} gives {np.prod(pop_shape)} "
                f"populations; expected {P}.")
        if data.shape != (S, L) or model_full.shape != (S, L):
            raise ValueError(
                f"Data/model shapes are {data.shape}/{model_full.shape}; "
                f"expected {(S, L)}.")
        if obs.size != L or mask.size != L:
            raise ValueError("Wavelength or mask dimensions are inconsistent.")

        # Exact age associated with every flattened SSP population.
        pop_ages = np.broadcast_to(
            ages[None, :, None], pop_shape).ravel(order="C")

        age_mass = np.array([
            np.sum(X[:, pop_ages == age]) for age in ages],
            dtype=np.float64)

        total_mass = float(np.sum(age_mass))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise RuntimeError("Fitted solution has zero population mass.")

        age_fraction = age_mass/total_mass
        occupied = age_fraction >= min_age_fraction
        occupied_idx = np.flatnonzero(occupied)

        if occupied_idx.size < 2:
            raise RuntimeError("Fewer than two fitted age bins are occupied.")

        gaps = []
        for lo, hi in zip(occupied_idx[:-1], occupied_idx[1:]):
            if hi-lo > 1:
                gaps.append((ages[hi]-ages[lo], int(lo), int(hi)))

        if not gaps:
            raise RuntimeError(
                "No empty SSP-age gap separates young and old populations.")

        _, young_hi, old_lo = max(gaps, key=lambda item: item[0])
        age_split = 0.5*(ages[young_hi]+ages[old_lo])
        young_pop = pop_ages < age_split

        # Read only physical (component, population) columns that actually
        # carry fitted young mass.
        young_support = (X > coeff_tol) & young_pop[None, :]
        support_c, support_p = np.nonzero(young_support)

        if support_c.size == 0:
            raise RuntimeError("No fitted young HyperCube columns are nonzero.")

        young_components = np.unique(support_c)
        young_populations = np.unique(support_p)
        young_mass = float(np.sum(X[young_support]))
        young_mass_fraction = young_mass/total_mass

        print("\nObserved-space young-population continuum test")
        print(f"occupied ages : {ages[occupied]}")
        print(f"empty gap     : {ages[young_hi]:.3f} -- "
            f"{ages[old_lo]:.3f} Gyr")
        print(f"age split     : {age_split:.3f} Gyr")
        print(f"young mass    : {100.0*young_mass_fraction:.3f}%")
        print(f"components    : {young_components.size}/{C}")
        print(f"populations   : {young_populations.size}/{P}")
        print(f"active columns: {support_c.size}/{C*P}")

        model_young = np.zeros((S, L), dtype=np.float64)

        # Group requests by physical component. For each component read the
        # smallest population range containing its nonzero young coefficients.
        by_component = {}
        for cc in young_components:
            pp = support_p[support_c == cc]
            by_component[int(cc)] = np.asarray(pp, dtype=np.int64)

        for s0 in tqdm(range(0, S, s_tile), desc="Young observed model"):
            s1 = min(s0+s_tile, S)

            for cc, pp in by_component.items():
                # Young age bins are contiguous in each metal/alpha block but
                # need not form one global contiguous population interval.
                # Group consecutive population indices to avoid useless reads.
                breaks = np.flatnonzero(np.diff(pp) > 1)+1
                groups = np.split(pp, breaks)

                for group in groups:
                    p0, p1 = int(group[0]), int(group[-1])+1
                    slab = np.asarray(M[s0:s1, cc, p0:p1, :],
                        dtype=np.float32, order="C")

                    weights = X[cc, p0:p1]
                    model_young[s0:s1] += np.tensordot(
                        slab, weights, axes=([1], [0]))
                    del slab

    model_old = model_full-model_young

    # Use only the wavelength support seen by the solver.
    fit_mask = mask & np.isfinite(obs)
    obs_fit = obs[fit_mask]

    if obs_fit.size < max(max_order+2, 8):
        raise RuntimeError("Too few fitted wavelengths for polynomial test.")

    span = float(np.ptp(obs_fit))
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("Fitted wavelength coordinate has zero extent.")

    leg_x = np.full(L, np.nan, dtype=np.float64)
    leg_x[fit_mask] = 2.0*(obs_fit-np.min(obs_fit))/span-1.0

    valid = fit_mask[None, :] & np.isfinite(data) \
        & np.isfinite(model_full) & np.isfinite(model_old) \
        & (model_old > 0.0)

    # Match the symmetric fractional residual used by CubeFit diagnostics.
    def _rss(mod):
        denom = 0.5*(data+mod)
        good = fit_mask[None, :] & np.isfinite(data) \
            & np.isfinite(mod) & np.isfinite(denom) & (denom > 0.0)

        positive = denom[good]
        floor = max(float(np.percentile(positive, 1.0)*1e-6), 1e-30) \
            if positive.size else 1e-30

        resid = np.zeros_like(data)
        resid[good] = (data[good]-mod[good]) \
            / np.maximum(denom[good], floor)

        return np.sum(resid*resid, axis=1)

    rss_full = _rss(model_full)
    rss_old = _rss(model_old)
    young_gain = rss_old-rss_full

    # Shape recovery is normalised by the exact observed young contribution.
    young_power = np.sum(
        np.where(fit_mask[None, :], model_young, 0.0)**2, axis=1)

    model_coeff = {}
    data_coeff = {}
    shape_recovery = {}
    data_recovery = {}
    rss_substitute = {}
    median_model_continuum = {}
    median_data_continuum = {}

    for order in range(max_order+1):
        c_model = np.full((S, order+1), np.nan, dtype=np.float64)
        c_data = np.full((S, order+1), np.nan, dtype=np.float64)
        substitute = model_old.copy()
        data_cont = np.full((S, L), np.nan, dtype=np.float64)
        model_cont = np.full((S, L), np.nan, dtype=np.float64)

        for s in range(S):
            good = valid[s]
            if np.count_nonzero(good) < max(order+2, 8):
                continue

            V = np.polynomial.legendre.legvander(leg_x[good], order)

            # Direct substitution: can a polynomial reproduce the actual
            # observed-space spectral effect supplied by the young SSPs?
            ratio_model = model_full[s, good]/model_old[s, good]
            cm, *_ = np.linalg.lstsq(V, ratio_model, rcond=None)
            pc_model = V@cm

            c_model[s] = cm
            model_cont[s, good] = pc_model
            substitute[s, good] = model_old[s, good]*pc_model

            # Independently estimate the smooth correction preferred by data.
            ratio_data = data[s, good]/model_old[s, good]
            keep = np.isfinite(ratio_data)

            for _ in range(5):
                if np.count_nonzero(keep) < max(order+2, 8):
                    break

                cd, *_ = np.linalg.lstsq(
                    V[keep], ratio_data[keep], rcond=None)
                delta = ratio_data-V@cd
                centre = np.median(delta[keep])
                mad = 1.4826*np.median(np.abs(delta[keep]-centre))

                if not np.isfinite(mad) or mad <= 0.0:
                    break

                new_keep = np.abs(delta-centre) <= clip_sigma*mad
                if np.array_equal(new_keep, keep):
                    break
                keep = new_keep

            if np.count_nonzero(keep) >= max(order+2, 8):
                cd, *_ = np.linalg.lstsq(
                    V[keep], ratio_data[keep], rcond=None)
                c_data[s] = cd
                data_cont[s, good] = V@cd

        model_coeff[order] = c_model
        data_coeff[order] = c_data
        rss_substitute[order] = _rss(substitute)

        shape_error = np.sum(np.where(fit_mask[None, :],
            model_full-substitute, 0.0)**2, axis=1)

        shape_recovery[order] = 1.0-np.divide(
            shape_error, young_power, out=np.full(S, np.nan),
            where=young_power > 0.0)

        substitute_loss = rss_substitute[order]-rss_full
        data_recovery[order] = 1.0-np.divide(
            substitute_loss, young_gain, out=np.full(S, np.nan),
            where=young_gain > 0.0)

        median_model_continuum[order] = np.nanmedian(model_cont, axis=0)
        median_data_continuum[order] = np.nanmedian(data_cont, axis=0)

    # Weight aperture summaries by observed young spectral power. Apertures
    # where the young solution has negligible effect should not dominate.
    power_sum = float(np.nansum(young_power))
    power_weight = young_power/power_sum if power_sum > 0.0 \
        else np.zeros_like(young_power)

    weighted_shape = np.full(max_order+1, np.nan, dtype=np.float64)
    weighted_data = np.full(max_order+1, np.nan, dtype=np.float64)

    print("\nPolynomial substitution")
    print("order  shape.rec  data.rec   med.shape   med.data")
    print("-"*54)

    for order in range(max_order+1):
        sr = shape_recovery[order]
        dr = data_recovery[order]

        gs = np.isfinite(sr) & (power_weight > 0.0)
        gd = np.isfinite(dr) & (power_weight > 0.0)

        if np.any(gs):
            w = power_weight[gs]
            weighted_shape[order] = np.sum(w*sr[gs])/np.sum(w)
            med_s = np.median(sr[gs])
        else:
            med_s = np.nan

        if np.any(gd):
            w = power_weight[gd]
            weighted_data[order] = np.sum(w*dr[gd])/np.sum(w)
            med_d = np.median(dr[gd])
        else:
            med_d = np.nan

        print(f"{order:5d} {weighted_shape[order]:10.3f} "
            f"{weighted_data[order]:9.3f} {med_s:11.3f} {med_d:10.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))

    ax = axes[0, 0]
    plot_fraction = np.where(age_fraction > 0.0, age_fraction, np.nan)
    ax.plot(ages, plot_fraction, "o-")
    ax.axvline(age_split, lw=0.8, color="0.4", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("SSP age [Gyr]")
    ax.set_ylabel("Fitted coefficient fraction")
    ax.set_title("Fitted age separation")

    ax = axes[0, 1]
    order = 1 if max_order >= 1 else 0
    mc = median_model_continuum[order]
    dc = median_data_continuum[order]
    ax.plot(obs[fit_mask], mc[fit_mask], lw=1.0,
        label="young SSP equivalent")
    ax.plot(obs[fit_mask], dc[fit_mask], lw=1.0,
        label="data preferred")
    ax.axhline(1.0, lw=0.7, color="0.4")
    ax.set_xlabel(r"$\log(\lambda)$")
    ax.set_ylabel("Multiplicative correction")
    ax.set_title(f"Order-{order} continuum shape")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    orders = np.arange(max_order+1)
    ax.plot(orders, weighted_shape, "o-", label="shape recovery")
    ax.plot(orders, weighted_data, "s-", label="data-fit recovery")
    ax.axhline(1.0, lw=0.7, color="0.4", linestyle="--")
    ax.set_xlabel("Legendre order")
    ax.set_ylabel("Young-power-weighted recovery")
    ax.set_title("Observed-space substitution")
    ax.set_xticks(orders)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    if max_order >= 1:
        sr = shape_recovery[1]
        dr = data_recovery[1]
        good = np.isfinite(sr) & np.isfinite(dr) & (young_power > 0.0)
        size = 10.0+40.0*young_power[good] \
            / max(float(np.max(young_power[good])), 1e-30)
        ax.scatter(sr[good], dr[good], s=size, alpha=0.6)
        ax.axvline(1.0, lw=0.7, color="0.4", linestyle="--")
        ax.axhline(1.0, lw=0.7, color="0.4", linestyle="--")
    ax.set_xlabel("Order-1 shape recovery")
    ax.set_ylabel("Order-1 data-fit recovery")
    ax.set_title("Aperture substitution")

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(ages=ages, age_mass=age_mass, age_fraction=age_fraction,
        occupied=occupied, age_split=age_split, young_pop=young_pop,
        young_support=young_support, young_components=young_components,
        young_populations=young_populations,
        young_mass_fraction=young_mass_fraction,
        model_young=model_young, model_old=model_old,
        rss_full=rss_full, rss_old=rss_old,
        rss_substitute=rss_substitute, model_coeff=model_coeff,
        data_coeff=data_coeff, shape_recovery=shape_recovery,
        data_recovery=data_recovery, weighted_shape=weighted_shape,
        weighted_data=weighted_data,
        median_model_continuum=median_model_continuum,
        median_data_continuum=median_data_continuum)

# ------------------------------------------------------------------------------

def diagnose_ssp_normalisation(h5_path: str, min_pop_fraction: float = 1e-6,
    save: str | None = None, show: bool = True) -> dict:
    """
    Diagnose normalization and shape degeneracy of occupied SSP templates.

    The function identifies fitted SSP populations from ``/X_global`` and
    compares their stored templates using raw normalization measures, optimal
    constant rescaling, and first-order multiplicative Legendre rescaling.

    It also compares the fitted young and old composite SSP spectra, using the
    largest empty gap between occupied ages to define the two age groups.

    No HyperCube data are read and no fit is performed.

    Parameters
    ----------
    h5_path : str
        Completed CubeFit HDF5 file.
    min_pop_fraction : float, optional
        Minimum fraction of total fitted coefficient mass required for an SSP
        population to count as occupied. Default is ``1e-6``.
    save : str, optional
        Diagnostic figure output path. Default is None.
    show : bool, optional
        Display the diagnostic figure. Default is True.

    Returns
    -------
    result : dict
        Population coordinates, normalization metrics, pairwise comparisons,
        and young/old composite-spectrum comparisons.

    Raises
    ------
    RuntimeError
        If required datasets are absent or no separated age groups exist.
    ValueError
        If SSP dimensions are inconsistent.

    Examples
    --------
    >>> out = diagnose_ssp_normalisation(
    ...     "hypercube_109_00.h5", save="ssp_normalisation.png")
    """
    min_pop_fraction = float(min_pop_fraction)
    if min_pop_fraction <= 0.0:
        raise ValueError("min_pop_fraction must be positive.")

    required = ("/Templates", "/X_global", "/SSPAges", "/SSPMetals",
        "/SSPAlphas", "/TemPix")

    with open_h5(h5_path, role="reader", swmr=True) as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise RuntimeError(f"Missing required datasets: {missing}.")

        templates = np.asarray(f["/Templates"][...], dtype=np.float64)
        X_raw = np.asarray(f["/X_global"][...], dtype=np.float64)
        ages = np.asarray(f["/SSPAges"][...], dtype=np.float64).ravel()
        metals = np.asarray(f["/SSPMetals"][...], dtype=np.float64).ravel()
        alphas = np.asarray(f["/SSPAlphas"][...], dtype=np.float64).ravel()
        tem_pix = np.asarray(f["/TemPix"][...], dtype=np.float64).ravel()
        pop_shape = tuple(np.asarray(
            f["/Templates"].attrs["pop_shape"], dtype=np.int64))

        if "/Mask" in f and "/ObsPix" in f:
            obs = np.asarray(f["/ObsPix"][...], dtype=np.float64).ravel()
            mask = np.asarray(f["/Mask"][...], dtype=bool).ravel()
        else:
            obs = mask = None

    P, T = templates.shape
    expected_shape = (metals.size, ages.size, alphas.size)

    if pop_shape != expected_shape:
        raise ValueError(
            f"pop_shape={pop_shape}; expected {expected_shape}.")
    if int(np.prod(pop_shape)) != P:
        raise ValueError(
            f"SSP shape {pop_shape} gives {np.prod(pop_shape)} "
            f"populations; expected P={P}.")
    if tem_pix.size != T:
        raise ValueError(f"TemPix has size {tem_pix.size}; expected {T}.")
    if X_raw.size % P != 0:
        raise ValueError(
            f"X_global size {X_raw.size} is not divisible by P={P}.")

    C = X_raw.size//P
    X = _row_or_vec_to_CP(X_raw, C, P)

    # Recover exact physical coordinates of every flattened SSP.
    iz, ia, ialpha = np.unravel_index(
        np.arange(P), pop_shape, order="C")
    pop_metal = metals[iz]
    pop_age = ages[ia]
    pop_alpha = alphas[ialpha]

    # Integrated fitted coefficient assigned to each SSP.
    pop_mass = np.sum(X, axis=0)
    total_mass = float(np.sum(pop_mass))

    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("Fitted solution has zero population mass.")

    pop_fraction = pop_mass/total_mass
    occupied = pop_fraction >= min_pop_fraction
    occupied_idx = np.flatnonzero(occupied)

    if occupied_idx.size < 2:
        raise RuntimeError("Fewer than two SSP populations are occupied.")

    # Infer young/old separation from occupied age bins.
    age_mass = np.array([
        np.sum(pop_mass[pop_age == age]) for age in ages],
        dtype=np.float64)
    age_fraction = age_mass/total_mass
    occupied_age = age_fraction >= min_pop_fraction
    occupied_age_idx = np.flatnonzero(occupied_age)

    gaps = []
    for lo, hi in zip(occupied_age_idx[:-1], occupied_age_idx[1:]):
        if hi-lo > 1:
            gaps.append((ages[hi]-ages[lo], int(lo), int(hi)))

    if not gaps:
        raise RuntimeError(
            "No empty SSP-age gap separates young and old populations.")

    _, young_hi, old_lo = max(gaps, key=lambda item: item[0])
    age_split = 0.5*(ages[young_hi]+ages[old_lo])
    young = pop_age < age_split
    old = ~young

    # Raw template normalization metrics.
    template_mean = np.mean(templates, axis=1)
    template_median = np.median(templates, axis=1)
    template_sum = np.sum(templates, axis=1)
    template_l2 = np.linalg.norm(templates, axis=1)

    # Composite spectra use exactly the fitted SSP coefficients.
    spec_full = pop_mass@templates
    spec_young = pop_mass[young]@templates[young]
    spec_old = pop_mass[old]@templates[old]

    closure = np.linalg.norm(spec_full-spec_old-spec_young)
    closure /= max(np.linalg.norm(spec_full), 1e-30)
    if closure > 1e-12:
        raise RuntimeError(
            f"Composite spectral closure failed: {closure:.3e}.")

    valid = np.isfinite(spec_full) & np.isfinite(spec_old) \
        & np.isfinite(spec_young) & (spec_old > 0.0)

    if np.count_nonzero(valid) < 8:
        raise RuntimeError("Insufficient valid template wavelengths.")

    # Legendre coordinate on the stored template wavelength grid.
    tmin = float(np.min(tem_pix[valid]))
    span = float(np.ptp(tem_pix[valid]))
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("Template wavelength grid has zero extent.")

    leg_x = np.full(T, np.nan, dtype=np.float64)
    leg_x[valid] = 2.0*(tem_pix[valid]-tmin)/span-1.0

    # Fit a multiplicative polynomial mapping source -> target.
    def _compare(source, target, order):
        good = np.isfinite(source) & np.isfinite(target) & (source > 0.0)
        V = np.polynomial.legendre.legvander(leg_x[good], order)
        ratio = target[good]/source[good]
        coeff, *_ = np.linalg.lstsq(V, ratio, rcond=None)

        prediction = source[good]*(V@coeff)
        delta = target[good]-prediction
        signal = target[good]-source[good]

        error = float(np.sum(delta*delta))
        signal_power = float(np.sum(signal*signal))
        target_power = float(np.sum(target[good]*target[good]))

        recovery = 1.0-error/signal_power if signal_power > 0.0 else np.nan
        rel_target_rms = np.sqrt(error/max(target_power, 1e-30))

        return coeff, recovery, rel_target_rms

    composite = {}
    for order in (0, 1):
        coeff, recovery, rel_rms = _compare(spec_old, spec_full, order)
        composite[order] = dict(coeff=coeff, recovery=recovery,
            relative_target_rms=rel_rms)

    # Compare every occupied young SSP against every occupied old SSP.
    young_idx = occupied_idx[young[occupied_idx]]
    old_idx = occupied_idx[old[occupied_idx]]
    pairwise = []

    for py in young_idx:
        for po in old_idx:
            c0, r0, e0 = _compare(templates[po], templates[py], 0)
            c1, r1, e1 = _compare(templates[po], templates[py], 1)

            pairwise.append(dict(young=int(py), old=int(po),
                young_age=float(pop_age[py]), old_age=float(pop_age[po]),
                young_metal=float(pop_metal[py]),
                old_metal=float(pop_metal[po]),
                young_alpha=float(pop_alpha[py]),
                old_alpha=float(pop_alpha[po]),
                young_fraction=float(pop_fraction[py]),
                old_fraction=float(pop_fraction[po]),
                scale=float(c0[0]), recovery0=float(r0),
                recovery1=float(r1), rel_rms0=float(e0),
                rel_rms1=float(e1), slope=float(c1[1])))

    print("\nOccupied SSP normalization diagnostic")
    print(f"age split : {age_split:.3f} Gyr")
    print(f"young SSPs: {young_idx.size}")
    print(f"old SSPs  : {old_idx.size}")
    print("")
    print(" idx    age       Z   alpha      frac       mean     median")
    print("-"*70)

    for pp in occupied_idx:
        print(f"{pp:4d} {pop_age[pp]:6.2f} {pop_metal[pp]:+7.3f} "
            f"{pop_alpha[pp]:+7.3f} {pop_fraction[pp]:9.3e} "
            f"{template_mean[pp]:9.3e} {template_median[pp]:9.3e}")

    print("\nComposite old -> full")
    print("order      scale       slope    recovery   rel.target.RMS")
    print("-"*60)

    c0 = composite[0]["coeff"]
    print(f"{0:5d} {c0[0]:10.6f} {'--':>11s} "
        f"{composite[0]['recovery']:11.6f} "
        f"{composite[0]['relative_target_rms']:16.6e}")

    c1 = composite[1]["coeff"]
    print(f"{1:5d} {c1[0]:10.6f} {c1[1]:+11.6f} "
        f"{composite[1]['recovery']:11.6f} "
        f"{composite[1]['relative_target_rms']:16.6e}")

    if pairwise:
        best0 = max(pairwise, key=lambda d: d["recovery0"])
        best1 = max(pairwise, key=lambda d: d["recovery1"])

        print("\nBest occupied SSP matches")
        print("         young old   y.age  o.age     scale   recovery")
        print("-"*61)
        print(f"order 0: {best0['young']:5d} {best0['old']:3d} "
            f"{best0['young_age']:6.2f} {best0['old_age']:6.2f} "
            f"{best0['scale']:9.5f} {best0['recovery0']:10.6f}")
        print(f"order 1: {best1['young']:5d} {best1['old']:3d} "
            f"{best1['young_age']:6.2f} {best1['old_age']:6.2f} "
            f"{best1['scale']:9.5f} {best1['recovery1']:10.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))

    ax = axes[0, 0]
    ax.scatter(pop_age[occupied_idx], template_mean[occupied_idx],
        s=30+300.0*pop_fraction[occupied_idx])
    ax.set_xlabel("SSP age [Gyr]")
    ax.set_ylabel("Template mean")
    ax.set_title("Stored template normalization")

    ax = axes[0, 1]
    ax.scatter(pop_age[occupied_idx], template_l2[occupied_idx],
        s=30+300.0*pop_fraction[occupied_idx])
    ax.set_xlabel("SSP age [Gyr]")
    ax.set_ylabel(r"$||T_p||_2$")
    ax.set_title("Template spectral norm")

    ax = axes[1, 0]
    norm = np.median(spec_full[valid])
    norm = norm if np.isfinite(norm) and norm != 0.0 else 1.0

    V0 = np.polynomial.legendre.legvander(leg_x[valid], 0)
    V1 = np.polynomial.legendre.legvander(leg_x[valid], 1)
    sub0 = spec_old[valid]*(V0@composite[0]["coeff"])
    sub1 = spec_old[valid]*(V1@composite[1]["coeff"])

    ax.plot(tem_pix[valid], spec_full[valid]/norm, lw=0.9, label="full")
    ax.plot(tem_pix[valid], spec_old[valid]/norm, lw=0.9, label="old")
    ax.plot(tem_pix[valid], sub0/norm, lw=0.9, label="old x constant")
    ax.plot(tem_pix[valid], sub1/norm, lw=0.9, label="old x linear")
    ax.set_xlabel(r"$\log(\lambda)$")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Composite population spectra")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ratio = spec_full[valid]/spec_old[valid]
    ax.plot(tem_pix[valid], ratio, lw=0.7, label="full / old")
    ax.plot(tem_pix[valid], V0@composite[0]["coeff"], lw=1.0,
        label="constant")
    ax.plot(tem_pix[valid], V1@composite[1]["coeff"], lw=1.0,
        label="linear")
    ax.axhline(1.0, lw=0.7, color="0.4")
    ax.set_xlabel(r"$\log(\lambda)$")
    ax.set_ylabel("Multiplicative ratio")
    ax.set_title("Continuum-equivalent young contribution")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(pop_age=pop_age, pop_metal=pop_metal,
        pop_alpha=pop_alpha, pop_mass=pop_mass,
        pop_fraction=pop_fraction, occupied=occupied,
        age_fraction=age_fraction, age_split=age_split,
        young_idx=young_idx, old_idx=old_idx,
        template_mean=template_mean, template_median=template_median,
        template_sum=template_sum, template_l2=template_l2,
        spec_full=spec_full, spec_old=spec_old,
        spec_young=spec_young, composite=composite,
        pairwise=pairwise)

# ------------------------------------------------------------------------------

def diagnose_young_line_signature(h5_path: str, continuum_order: int = 4,
    min_age_fraction: float = 1e-4, cache_path: str | None = None,
    s_tile: int = 128, save: str | None = None, show: bool = True) -> dict:
    """
    Test whether the observed data contain the fitted young SSP line signature.

    The function reconstructs only the nonzero young contribution from the
    existing HyperCube. This contribution is cached after the first run.
    Smooth Legendre continua are removed independently from the data and young
    model, leaving their line-scale spectral structure.

    The resulting high-pass spectra are compared aperture by aperture using
    normalized correlation and matched-filter amplitudes. No full model is
    reconstructed, no solver is run, and no HDF5 datasets are modified.

    This is a screening diagnostic rather than an exact likelihood test:
    old-population absorption features remain in the continuum-removed data.

    Parameters
    ----------
    h5_path : str
        CubeFit HDF5 file containing the data, HyperCube, and ``/X_global``.
    continuum_order : int, optional
        Legendre order removed independently from each spectrum. Default is 4.
    min_age_fraction : float, optional
        Minimum global coefficient fraction for an age to count as occupied.
        Default is ``1e-4``.
    cache_path : str, optional
        NPZ cache for the reconstructed young model. If None, a cache name is
        derived from ``h5_path``.
    s_tile : int, optional
        Number of apertures read per sparse HyperCube access. Default is 128.
    save : str, optional
        Diagnostic figure output path. Default is None.
    show : bool, optional
        Display the diagnostic figure. Default is True.

    Returns
    -------
    result : dict
        Age split, young support, reconstructed young model, high-pass spectra,
        correlations, matched-filter amplitudes, and wavelength diagnostics.

    Raises
    ------
    RuntimeError
        If required datasets are absent or no separated age groups are found.
    ValueError
        If stored dimensions or input parameters are inconsistent.

    Examples
    --------
    >>> out = diagnose_young_line_signature(
    ...     "hypercube_109_00.h5", continuum_order=4,
    ...     save="young_line_signature.png")
    """
    continuum_order = int(continuum_order)
    min_age_fraction = float(min_age_fraction)
    s_tile = max(1, int(s_tile))

    if continuum_order < 0:
        raise ValueError("continuum_order must be non-negative.")
    if min_age_fraction <= 0.0:
        raise ValueError("min_age_fraction must be positive.")

    required = ("/DataCube", "/HyperCube/models", "/X_global",
        "/SSPAges", "/SSPMetals", "/SSPAlphas", "/Templates")

    with open_h5(h5_path, role="reader", swmr=True) as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise RuntimeError(f"Missing required datasets: {missing}.")

        H = f["/HyperCube/models"]
        S, C, P, L = map(int, H.shape)

        data = np.asarray(f["/DataCube"][...], dtype=np.float64)
        X = _row_or_vec_to_CP(
            np.asarray(f["/X_global"][...], dtype=np.float64), C, P)

        ages = np.asarray(f["/SSPAges"][...], dtype=np.float64).ravel()
        metals = np.asarray(f["/SSPMetals"][...], dtype=np.float64).ravel()
        alphas = np.asarray(f["/SSPAlphas"][...], dtype=np.float64).ravel()
        pop_shape = tuple(np.asarray(
            f["/Templates"].attrs["pop_shape"], dtype=np.int64))

        obs = np.asarray(f["/ObsPix"][...], dtype=np.float64).ravel() \
            if "/ObsPix" in f else np.arange(L, dtype=np.float64)
        mask = np.asarray(f["/Mask"][...], dtype=bool).ravel() \
            if "/Mask" in f else np.ones(L, dtype=bool)

        expected_shape = (metals.size, ages.size, alphas.size)
        if pop_shape != expected_shape:
            raise ValueError(
                f"pop_shape={pop_shape}; expected {expected_shape}.")
        if int(np.prod(pop_shape)) != P:
            raise ValueError(
                f"SSP shape {pop_shape} gives {np.prod(pop_shape)} "
                f"populations; expected {P}.")
        if data.shape != (S, L):
            raise ValueError(
                f"/DataCube has shape {data.shape}; expected {(S, L)}.")
        if obs.size != L or mask.size != L:
            raise ValueError("Wavelength or mask dimensions are inconsistent.")

        pop_ages = np.broadcast_to(
            ages[None, :, None], pop_shape).ravel(order="C")
        age_mass = np.array([
            np.sum(X[:, pop_ages == age]) for age in ages],
            dtype=np.float64)

        total_mass = float(np.sum(age_mass))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise RuntimeError("Fitted solution has zero population mass.")

        age_fraction = age_mass/total_mass
        occupied = age_fraction >= min_age_fraction
        occupied_idx = np.flatnonzero(occupied)

        if occupied_idx.size < 2:
            raise RuntimeError("Fewer than two fitted age bins are occupied.")

        gaps = []
        for lo, hi in zip(occupied_idx[:-1], occupied_idx[1:]):
            if hi-lo > 1:
                gaps.append((ages[hi]-ages[lo], int(lo), int(hi)))

        if not gaps:
            raise RuntimeError(
                "No empty SSP-age gap separates young and old populations.")

        _, young_hi, old_lo = max(gaps, key=lambda item: item[0])
        age_split = 0.5*(ages[young_hi]+ages[old_lo])
        young_pop = pop_ages < age_split
        young_support = (X > 0.0) & young_pop[None, :]
        support_c, support_p = np.nonzero(young_support)

        if support_c.size == 0:
            raise RuntimeError("No positive fitted young coefficients exist.")

        if cache_path is None:
            cache_path = str(h5_path)+".young_model.npz"

        cache_ok = False
        try:
            cache = np.load(cache_path, allow_pickle=False)
            cached_split = float(cache["age_split"])
            cached_shape = tuple(cache["model_young"].shape)
            cached_support_c = np.asarray(cache["support_c"], dtype=np.int64)
            cached_support_p = np.asarray(cache["support_p"], dtype=np.int64)

            cache_ok = (
                cached_shape == (S, L)
                and np.isclose(cached_split, age_split, rtol=0.0, atol=0.0)
                and np.array_equal(cached_support_c, support_c)
                and np.array_equal(cached_support_p, support_p)
            )

            if cache_ok:
                model_young = np.asarray(
                    cache["model_young"], dtype=np.float64)
                print(f"Loaded young-model cache: {cache_path}")
        except (FileNotFoundError, KeyError, OSError, ValueError):
            cache_ok = False

        if not cache_ok:
            print("\nYoung-model reconstruction")
            print(f"age split     : {age_split:.3f} Gyr")
            print(f"components    : {np.unique(support_c).size}/{C}")
            print(f"populations   : {np.unique(support_p).size}/{P}")
            print(f"active columns: {support_c.size}/{C*P}")

            model_young = np.zeros((S, L), dtype=np.float64)

            by_component = {}
            for cc in np.unique(support_c):
                pp = np.sort(support_p[support_c == cc])
                by_component[int(cc)] = pp

            for s0 in tqdm(
                range(0, S, s_tile), desc="Young observed model"):
                s1 = min(s0+s_tile, S)

                for cc, pp in by_component.items():
                    breaks = np.flatnonzero(np.diff(pp) > 1)+1

                    for group in np.split(pp, breaks):
                        p0 = int(group[0])
                        p1 = int(group[-1])+1
                        weights = X[cc, p0:p1]

                        slab = np.asarray(
                            H[s0:s1, cc, p0:p1, :],
                            dtype=np.float32, order="C")
                        model_young[s0:s1] += np.tensordot(
                            slab, weights, axes=([1], [0]))
                        del slab

            np.savez_compressed(
                cache_path, model_young=model_young,
                age_split=np.asarray(age_split, dtype=np.float64),
                support_c=support_c, support_p=support_p)
            print(f"Saved young-model cache: {cache_path}")

    fit_mask = mask & np.isfinite(obs)
    obs_fit = obs[fit_mask]

    if obs_fit.size < max(continuum_order+2, 8):
        raise RuntimeError("Too few fitted wavelengths for continuum removal.")

    span = float(np.ptp(obs_fit))
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("Fitted wavelength coordinate has zero extent.")

    x_leg = 2.0*(obs_fit-np.min(obs_fit))/span-1.0
    V = np.polynomial.legendre.legvander(x_leg, continuum_order)

    data_hp = np.full_like(data, np.nan)
    young_hp = np.full_like(model_young, np.nan)
    correlation = np.full(S, np.nan, dtype=np.float64)
    amplitude = np.full(S, np.nan, dtype=np.float64)
    young_line_power = np.zeros(S, dtype=np.float64)

    # Divide by the fitted smooth continuum before subtracting unity. This
    # compares fractional line structure rather than absolute flux amplitude.
    for ss in range(S):
        good = fit_mask & np.isfinite(data[ss]) \
            & np.isfinite(model_young[ss]) & (data[ss] > 0.0) \
            & (model_young[ss] > 0.0)

        if np.count_nonzero(good) < max(continuum_order+2, 8):
            continue

        local_x = 2.0*(obs[good]-np.min(obs_fit))/span-1.0
        local_V = np.polynomial.legendre.legvander(
            local_x, continuum_order)

        cd, *_ = np.linalg.lstsq(local_V, data[ss, good], rcond=None)
        cy, *_ = np.linalg.lstsq(
            local_V, model_young[ss, good], rcond=None)

        cont_data = local_V@cd
        cont_young = local_V@cy

        valid_cont = np.isfinite(cont_data) & np.isfinite(cont_young) \
            & (cont_data > 0.0) & (cont_young > 0.0)

        if np.count_nonzero(valid_cont) < max(continuum_order+2, 8):
            continue

        inds = np.flatnonzero(good)[valid_cont]
        dline = data[ss, inds]/cont_data[valid_cont]-1.0
        yline = model_young[ss, inds]/cont_young[valid_cont]-1.0

        # Remove any remaining constant offset from the line vectors.
        dline -= np.mean(dline)
        yline -= np.mean(yline)

        data_hp[ss, inds] = dline
        young_hp[ss, inds] = yline

        dp = float(np.dot(dline, dline))
        yp = float(np.dot(yline, yline))
        cross = float(np.dot(dline, yline))

        young_line_power[ss] = yp

        if dp > 0.0 and yp > 0.0:
            correlation[ss] = cross/np.sqrt(dp*yp)
            amplitude[ss] = cross/yp

    # Weight summaries by the observed strength of the young line pattern.
    good = np.isfinite(correlation) & np.isfinite(amplitude) \
        & (young_line_power > 0.0)

    if not np.any(good):
        raise RuntimeError("No valid aperture line comparisons were obtained.")

    weights = young_line_power[good]
    weights /= np.sum(weights)

    weighted_corr = float(np.sum(weights*correlation[good]))
    weighted_amp = float(np.sum(weights*amplitude[good]))

    # Wavelength-resolved normalized coherence. The numerator retains sign.
    cross_lambda = np.nansum(data_hp*young_hp, axis=0)
    data_power_lambda = np.nansum(data_hp*data_hp, axis=0)
    young_power_lambda = np.nansum(young_hp*young_hp, axis=0)

    coherence_lambda = np.divide(
        cross_lambda,
        np.sqrt(data_power_lambda*young_power_lambda),
        out=np.full(L, np.nan, dtype=np.float64),
        where=(data_power_lambda > 0.0) & (young_power_lambda > 0.0))

    print("\nYoung line-signature diagnostic")
    print(f"occupied ages    : {ages[occupied]}")
    print(f"age split        : {age_split:.3f} Gyr")
    print(f"continuum order  : {continuum_order}")
    print(f"valid apertures  : {np.count_nonzero(good)}/{S}")
    print(f"weighted corr    : {weighted_corr:+.6f}")
    print(f"weighted amp     : {weighted_amp:+.6f}")
    print(f"median corr      : {np.median(correlation[good]):+.6f}")
    print(f"median amp       : {np.median(amplitude[good]):+.6f}")
    print(f"p16--p84 corr    : {np.percentile(correlation[good], 16):+.6f} -- "
        f"{np.percentile(correlation[good], 84):+.6f}")
    print(f"p16--p84 amp     : {np.percentile(amplitude[good], 16):+.6f} -- "
        f"{np.percentile(amplitude[good], 84):+.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))

    ax = axes[0, 0]
    ax.hist(correlation[good], bins=30, histtype="step")
    ax.axvline(0.0, color="0.4", lw=0.7)
    ax.axvline(weighted_corr, color="0.4", lw=0.8, linestyle="--")
    ax.set_xlabel("Data--young line correlation")
    ax.set_ylabel("Apertures")
    ax.set_title("Line-pattern correlation")

    ax = axes[0, 1]
    finite_amp = good & np.isfinite(amplitude)
    amp_plot = amplitude[finite_amp]
    lo, hi = np.percentile(amp_plot, [1.0, 99.0])
    amp_plot = amp_plot[(amp_plot >= lo) & (amp_plot <= hi)]
    ax.hist(amp_plot, bins=30, histtype="step")
    ax.axvline(0.0, color="0.4", lw=0.7)
    ax.axvline(1.0, color="0.4", lw=0.7, linestyle=":")
    ax.axvline(weighted_amp, color="0.4", lw=0.8, linestyle="--")
    ax.set_xlabel("Matched young-line amplitude")
    ax.set_ylabel("Apertures")
    ax.set_title("Young line strength in data")

    ax = axes[1, 0]
    ax.plot(obs[fit_mask], coherence_lambda[fit_mask], lw=0.8)
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xlabel(r"$\log(\lambda)$")
    ax.set_ylabel("Coherence")
    ax.set_title("Wavelength-resolved line agreement")

    ax = axes[1, 1]
    power = young_line_power.copy()
    positive = power > 0.0
    ax.scatter(
        power[positive], correlation[positive], s=8, alpha=0.5)
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Young line-pattern power")
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation versus young signal")

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(
        ages=ages, age_fraction=age_fraction, occupied=occupied,
        age_split=age_split, young_pop=young_pop,
        young_support=young_support, model_young=model_young,
        data_highpass=data_hp, young_highpass=young_hp,
        correlation=correlation, amplitude=amplitude,
        young_line_power=young_line_power,
        weighted_correlation=weighted_corr,
        weighted_amplitude=weighted_amp,
        coherence_lambda=coherence_lambda,
        cache_path=cache_path)

# ------------------------------------------------------------------------------

def diagnose_unique_young_features(h5_path: str, continuum_order: int = 4,
    feature_fraction: float = 0.10, min_pop_fraction: float = 1e-6,
    cache_path: str | None = None, save: str | None = None,
    show: bool = True) -> dict:
    """
    Test whether data preferentially support features unique to young SSPs.

    The fitted young SSP is continuum-normalized and projected onto the
    continuum-normalized subspace spanned by occupied old SSPs. Wavelengths
    where the residual from that projection is strongest define the spectral
    features that distinguish the young population from the fitted old
    population.

    Those wavelengths are mapped to the observed grid and used to compare the
    continuum-removed data with the cached observed-space young contribution.
    Correlation in young-discriminating wavelengths is compared with a matched
    control set drawn from the remaining fitted wavelengths.

    No HyperCube data are read if the young-model cache already exists.

    Parameters
    ----------
    h5_path : str
        CubeFit HDF5 file containing data, templates, and ``/X_global``.
    continuum_order : int, optional
        Legendre order removed from templates and observed spectra. Default is
        4.
    feature_fraction : float, optional
        Fraction of fitted wavelengths classified as most young-sensitive.
        Default is 0.10.
    min_pop_fraction : float, optional
        Minimum global coefficient fraction for an SSP to count as occupied.
        Default is ``1e-6``.
    cache_path : str, optional
        Young-model cache created by ``diagnose_young_line_signature``.
        Default derives the standard cache name from ``h5_path``.
    save : str, optional
        Diagnostic figure output path. Default is None.
    show : bool, optional
        Display the diagnostic figure. Default is True.

    Returns
    -------
    result : dict
        Young and old SSP indices, unique-young intrinsic signature, observed
        feature mask, feature/control correlations, and wavelength summaries.

    Raises
    ------
    RuntimeError
        If required data or the young-model cache are unavailable.
    ValueError
        If dimensions or input parameters are inconsistent.

    Examples
    --------
    >>> out = diagnose_unique_young_features(
    ...     "hypercube_109_00.h5", continuum_order=4,
    ...     save="unique_young_features.png")
    """
    continuum_order = int(continuum_order)
    feature_fraction = float(feature_fraction)
    min_pop_fraction = float(min_pop_fraction)

    if continuum_order < 0:
        raise ValueError("continuum_order must be non-negative.")
    if not 0.0 < feature_fraction < 0.5:
        raise ValueError("feature_fraction must lie between zero and 0.5.")
    if min_pop_fraction <= 0.0:
        raise ValueError("min_pop_fraction must be positive.")

    required = ("/DataCube", "/Templates", "/X_global", "/SSPAges",
        "/SSPMetals", "/SSPAlphas", "/TemPix")

    with open_h5(h5_path, role="reader", swmr=True) as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise RuntimeError(f"Missing required datasets: {missing}.")

        data = np.asarray(f["/DataCube"][...], dtype=np.float64)
        templates = np.asarray(f["/Templates"][...], dtype=np.float64)
        ages = np.asarray(f["/SSPAges"][...], dtype=np.float64).ravel()
        metals = np.asarray(f["/SSPMetals"][...], dtype=np.float64).ravel()
        alphas = np.asarray(f["/SSPAlphas"][...], dtype=np.float64).ravel()
        tem_pix = np.asarray(f["/TemPix"][...], dtype=np.float64).ravel()

        pop_shape = tuple(np.asarray(
            f["/Templates"].attrs["pop_shape"], dtype=np.int64))
        P = int(np.prod(pop_shape))

        X_raw = np.asarray(f["/X_global"][...], dtype=np.float64)
        C = int(X_raw.size//P)
        X = _row_or_vec_to_CP(X_raw, C, P)

        obs = np.asarray(f["/ObsPix"][...], dtype=np.float64).ravel() \
            if "/ObsPix" in f else np.arange(data.shape[1], dtype=np.float64)
        mask = np.asarray(f["/Mask"][...], dtype=bool).ravel() \
            if "/Mask" in f else np.ones(data.shape[1], dtype=bool)

    expected_shape = (metals.size, ages.size, alphas.size)
    if pop_shape != expected_shape:
        raise ValueError(
            f"pop_shape={pop_shape}; expected {expected_shape}.")
    if templates.shape[0] != P:
        raise ValueError(
            f"/Templates has P={templates.shape[0]}; expected {P}.")
    if templates.shape[1] != tem_pix.size:
        raise ValueError("Template wavelength dimensions are inconsistent.")
    if data.shape[1] != obs.size or mask.size != obs.size:
        raise ValueError("Observed wavelength dimensions are inconsistent.")

    iz, ia, ialpha = np.unravel_index(
        np.arange(P), pop_shape, order="C")
    pop_age = ages[ia]
    pop_metal = metals[iz]
    pop_alpha = alphas[ialpha]

    pop_mass = np.sum(X, axis=0)
    total_mass = float(np.sum(pop_mass))

    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("Fitted solution has zero population mass.")

    pop_fraction = pop_mass/total_mass
    occupied = pop_fraction >= min_pop_fraction
    occupied_idx = np.flatnonzero(occupied)

    age_mass = np.array([
        np.sum(pop_mass[pop_age == age]) for age in ages],
        dtype=np.float64)
    age_fraction = age_mass/total_mass
    occupied_age = age_fraction >= min_pop_fraction
    occupied_age_idx = np.flatnonzero(occupied_age)

    gaps = []
    for lo, hi in zip(occupied_age_idx[:-1], occupied_age_idx[1:]):
        if hi-lo > 1:
            gaps.append((ages[hi]-ages[lo], int(lo), int(hi)))

    if not gaps:
        raise RuntimeError(
            "No empty SSP-age gap separates young and old populations.")

    _, young_hi, old_lo = max(gaps, key=lambda item: item[0])
    age_split = 0.5*(ages[young_hi]+ages[old_lo])

    young_idx = occupied_idx[pop_age[occupied_idx] < age_split]
    old_idx = occupied_idx[pop_age[occupied_idx] >= age_split]

    if young_idx.size == 0 or old_idx.size == 0:
        raise RuntimeError("Young/old occupied SSP groups are empty.")

    # Use the fitted young mixture rather than assuming only one young SSP.
    young_weights = pop_mass[young_idx]
    young_weights /= np.sum(young_weights)
    young_template = young_weights@templates[young_idx]

    fit_t = np.isfinite(tem_pix) & np.all(
        np.isfinite(templates[np.r_[young_idx, old_idx]]), axis=0)

    if np.count_nonzero(fit_t) < max(continuum_order+2, 8):
        raise RuntimeError("Too few valid template wavelengths.")

    tmin = float(np.min(tem_pix[fit_t]))
    tspan = float(np.ptp(tem_pix[fit_t]))
    tx = 2.0*(tem_pix[fit_t]-tmin)/tspan-1.0
    TV = np.polynomial.legendre.legvander(tx, continuum_order)

    def _continuum_normalize(spec):
        coeff, *_ = np.linalg.lstsq(TV, spec[fit_t], rcond=None)
        continuum = TV@coeff

        if np.any(~np.isfinite(continuum)) or np.any(continuum <= 0.0):
            raise RuntimeError("Invalid fitted template continuum.")

        out = np.zeros(np.count_nonzero(fit_t), dtype=np.float64)
        out[:] = spec[fit_t]/continuum-1.0
        out -= np.mean(out)
        return out

    young_line = _continuum_normalize(young_template)
    old_lines = np.vstack([
        _continuum_normalize(templates[pp]) for pp in old_idx])

    # Weight old SSP basis vectors by fitted relevance before orthogonalizing.
    old_weights = np.sqrt(pop_mass[old_idx])
    old_basis = old_lines*old_weights[:, None]

    # SVD provides a stable orthonormal basis for the occupied old SSP space.
    _, singular, vh = np.linalg.svd(old_basis, full_matrices=False)
    sv_cut = max(old_basis.shape)*np.finfo(np.float64).eps*singular[0]
    rank = int(np.count_nonzero(singular > sv_cut))

    if rank == 0:
        raise RuntimeError("Occupied old SSP line subspace has zero rank.")

    basis = vh[:rank]
    projection = basis.T@(basis@young_line)
    unique_young = young_line-projection

    young_norm = float(np.linalg.norm(young_line))
    unique_norm = float(np.linalg.norm(unique_young))
    unique_fraction = unique_norm/max(young_norm, 1e-300)

    # Interpolate the magnitude of the unique intrinsic signature onto the
    # observed wavelength grid. TemPix and ObsPix are both log-lambda grids.
    score_t = np.abs(unique_young)
    score_obs = np.interp(
        obs, tem_pix[fit_t], score_t, left=np.nan, right=np.nan)

    valid_obs = mask & np.isfinite(obs) & np.isfinite(score_obs)
    n_valid = int(np.count_nonzero(valid_obs))

    if n_valid < 20:
        raise RuntimeError("Too few overlapping observed/template wavelengths.")

    n_feature = max(8, int(np.ceil(feature_fraction*n_valid)))
    valid_idx = np.flatnonzero(valid_obs)
    order = valid_idx[np.argsort(score_obs[valid_idx])[::-1]]

    feature_idx = np.sort(order[:n_feature])
    feature_mask = np.zeros(obs.size, dtype=bool)
    feature_mask[feature_idx] = True

    # Construct a control mask of the same size from wavelengths carrying the
    # weakest unique-young signature. This is intentionally deterministic.
    control_idx = np.sort(order[-n_feature:])
    control_mask = np.zeros(obs.size, dtype=bool)
    control_mask[control_idx] = True

    if cache_path is None:
        cache_path = str(h5_path)+".young_model.npz"

    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            model_young = np.asarray(
                cache["model_young"], dtype=np.float64)
            cached_split = float(cache["age_split"])
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise RuntimeError(
            "Young-model cache is unavailable. Run "
            "diagnose_young_line_signature() once first.") from exc

    if model_young.shape != data.shape:
        raise ValueError(
            f"Cached young model has shape {model_young.shape}; "
            f"expected {data.shape}.")
    if not np.isclose(cached_split, age_split, rtol=0.0, atol=0.0):
        raise RuntimeError(
            "Young-model cache was generated with a different age split.")

    # Continuum-remove observed data and observed-space young contribution.
    obs_fit = obs[mask & np.isfinite(obs)]
    omin = float(np.min(obs_fit))
    ospan = float(np.ptp(obs_fit))

    data_hp = np.full_like(data, np.nan)
    young_hp = np.full_like(model_young, np.nan)

    for ss in range(data.shape[0]):
        good = mask & np.isfinite(data[ss]) \
            & np.isfinite(model_young[ss]) & (data[ss] > 0.0) \
            & (model_young[ss] > 0.0)

        if np.count_nonzero(good) < max(continuum_order+2, 8):
            continue

        ox = 2.0*(obs[good]-omin)/ospan-1.0
        OV = np.polynomial.legendre.legvander(ox, continuum_order)

        cd, *_ = np.linalg.lstsq(OV, data[ss, good], rcond=None)
        cy, *_ = np.linalg.lstsq(
            OV, model_young[ss, good], rcond=None)

        cont_d = OV@cd
        cont_y = OV@cy
        okay = np.isfinite(cont_d) & np.isfinite(cont_y) \
            & (cont_d > 0.0) & (cont_y > 0.0)

        inds = np.flatnonzero(good)[okay]
        if inds.size < max(continuum_order+2, 8):
            continue

        dline = data[ss, inds]/cont_d[okay]-1.0
        yline = model_young[ss, inds]/cont_y[okay]-1.0
        dline -= np.mean(dline)
        yline -= np.mean(yline)

        data_hp[ss, inds] = dline
        young_hp[ss, inds] = yline

    def _masked_statistics(which):
        corr = np.full(data.shape[0], np.nan, dtype=np.float64)
        amp = np.full(data.shape[0], np.nan, dtype=np.float64)
        power = np.zeros(data.shape[0], dtype=np.float64)

        for ss in range(data.shape[0]):
            good = which & np.isfinite(data_hp[ss]) \
                & np.isfinite(young_hp[ss])

            if np.count_nonzero(good) < 4:
                continue

            d = data_hp[ss, good]
            y = young_hp[ss, good]
            dp = float(np.dot(d, d))
            yp = float(np.dot(y, y))
            cross = float(np.dot(d, y))
            power[ss] = yp

            if dp > 0.0 and yp > 0.0:
                corr[ss] = cross/np.sqrt(dp*yp)
                amp[ss] = cross/yp

        return corr, amp, power

    feature_corr, feature_amp, feature_power = \
        _masked_statistics(feature_mask)
    control_corr, control_amp, control_power = \
        _masked_statistics(control_mask)

    paired = np.isfinite(feature_corr) & np.isfinite(control_corr)
    paired_amp = np.isfinite(feature_amp) & np.isfinite(control_amp)

    if not np.any(paired):
        raise RuntimeError("No valid feature/control aperture comparisons.")

    delta_corr = feature_corr-control_corr
    delta_amp = feature_amp-control_amp

    print("\nUnique-young feature diagnostic")
    print(f"young SSPs          : {young_idx.tolist()}")
    print(f"old SSPs            : {old_idx.tolist()}")
    print(f"age split           : {age_split:.3f} Gyr")
    print(f"old subspace rank   : {rank}/{old_idx.size}")
    print(f"unique line fraction: {unique_fraction:.6f}")
    print(f"feature wavelengths : {n_feature}/{n_valid}")
    print("")
    print(f"feature corr median : "
        f"{np.nanmedian(feature_corr):+.6f}")
    print(f"control corr median : "
        f"{np.nanmedian(control_corr):+.6f}")
    print(f"delta corr median   : "
        f"{np.nanmedian(delta_corr[paired]):+.6f}")
    print(f"feature amp median  : "
        f"{np.nanmedian(feature_amp):+.6f}")
    print(f"control amp median  : "
        f"{np.nanmedian(control_amp):+.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))

    ax = axes[0, 0]
    ax.plot(tem_pix[fit_t], young_line, lw=0.7, label="young line spectrum")
    ax.plot(tem_pix[fit_t], unique_young, lw=0.8, label="unique young")
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xlabel(r"$\log(\lambda)$")
    ax.set_ylabel("Fractional line structure")
    ax.set_title("Young spectrum after old-subspace projection")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(obs[valid_obs], score_obs[valid_obs], lw=0.8)
    ax.scatter(obs[feature_mask], score_obs[feature_mask], s=8,
        label="young-discriminating")
    ax.set_xlabel(r"$\log(\lambda)$")
    ax.set_ylabel(r"$|Y_{\rm unique}|$")
    ax.set_title("Selected distinguishing wavelengths")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.hist(feature_corr[np.isfinite(feature_corr)], bins=30,
        histtype="step", label="features")
    ax.hist(control_corr[np.isfinite(control_corr)], bins=30,
        histtype="step", label="control")
    ax.axvline(0.0, color="0.4", lw=0.7)
    ax.set_xlabel("Data--young correlation")
    ax.set_ylabel("Apertures")
    ax.set_title("Discriminating versus control wavelengths")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.hist(delta_corr[paired], bins=30, histtype="step")
    ax.axvline(0.0, color="0.4", lw=0.7)
    ax.set_xlabel(r"$\rho_{\rm feature}-\rho_{\rm control}$")
    ax.set_ylabel("Apertures")
    ax.set_title("Preferential young-feature support")

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(young_idx=young_idx, old_idx=old_idx,
        age_split=age_split, old_subspace_rank=rank,
        young_line=young_line, old_line_basis=basis,
        unique_young=unique_young, unique_fraction=unique_fraction,
        feature_score=score_obs, feature_mask=feature_mask,
        control_mask=control_mask, feature_correlation=feature_corr,
        control_correlation=control_corr, delta_correlation=delta_corr,
        feature_amplitude=feature_amp, control_amplitude=control_amp,
        delta_amplitude=delta_amp, feature_power=feature_power,
        control_power=control_power, cache_path=cache_path)

# ------------------------------------------------------------------------------
