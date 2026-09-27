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