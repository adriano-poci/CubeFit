# -*- coding: utf-8 -*-
r"""
    kz_fitSpec.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Master script to prepare data products and execute the workflow of CubeFit.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   2025
v1.1:   Read `zarrDir` from `kwargs` instead of hardcoding it. 12 August 2025
v1.2:   Removed `cp_flux_ref` from `reconstruct_modelcube_fast`, since now all
            public `x` API is scaled to physical units. 5 December 2025
v1.3:   Read in and un-scale `x_global` by stored Hypercube scale in
            `loadCubeFit`. 8 January 2026
v1.4:   Universally removed all ad-hoc scalings;
        Converted to `cube_utils` functions instead of `muse`;
        Updated `compare_orbit_vs_solution` to new `orbit_weights` maths
            in the SPG solver. 25 January 2026
v1.5:   Get `warm_start` from `kwargs` in `genCubeFit`;
        Converted `compare_orbit_vs_solution` to be diagnostic in the context of
            the exact Lagrange multiplier orbit prior. 31 January 2026
v1.6:   Remove existing spectral fit plots before regenerating, to avoid
            confusion with stale files. 10 February 2026
v1.7:   Removed axis loops of hyper-cube in `reconstruct_modelcube_fast` for
            efficiency. 4 March 2026
v1.8:   Fixed bug in SFH limits in `loadCubeFit`. 6 March 2026
v1.9:   Defined `reconstruct_modelcube_fast_parallel` for multi-process
            reconstruction of the model-cube;
        Added global `CPU_PROCESSES` and `BLAS_THREADS` constants for default
            parallelism settings;
        Renamed HDF5 file to `hypercube_*.h5`. 13 March 2026
v1.10:  Get `orbit_beta` from kwargs. 15 March 2026
v1.11:  Removed lingering `orbit_beta`. 23 March 2026
v1.12:  Re-implemented `orbit_beta` support in `genCubeFit` and passed it to the
            solver. 30 March 2026
v1.13:  Fixed spectral fit plot unlinking glob in `loadCubeFit`. 31 March 2026
v1.14:  Do not return full slab in `_reconstruct_worker`, causing extreme memory
            requirements. 22 May 2026
v1.15:  Polished `orbitMaps` figure. 3 August 2026
v1.16:  Removed legacy keywords in solver calls in `genCubeFit`. 4 August 2026
v1.17:  Fixed bug in `loadCubeFit` with the orbit map data dictionary labels in
            the plotting loop. 5 August 2026
v1.18:  Updated hard orbit-prior diagnostics for the constrained solver's
            flexible global-amplitude formulation;
        Removed the obsolete ATy-median estimate of the absolute orbit target;
        Reconstructed the fitted target as the normalized a priori orbit shape
            multiplied by the total fitted coefficient mass;
        Corrected the absolute orbit-mass comparison plot and residual
            diagnostics to use the same target enforced by the solver. 6 August
            2026
v1.19:  Dynamically adjust parallelism in case of cpuset restrictions. 7 August
            2026
v1.20:  Made all orbital phase-space plots show the logarithmic mass fractions
            in `loadCubeFit`;
        Use `moncmap` for spatial maps in `loadCubeFit`. 10 August 2026
v1.21:  Added `plot_best_worst_spectrum_fits_stacked` to plot single
            representative spectral figure;
        Use new `cube_utils.resolve_parallelism` to access optimal CPU/BLAS
            configuration from all functions. 11 August 2026
v1.22:  Fixed bug in `loadCubeFit` where `cWeights` was being computed
            incorrectly from the orbit weights. 12 August 2026
v1.23:  Normalised all outputs to adhere to the same `nComp` schema. 19 August
            2026
v1.24:  Return `picks` in `plot_best_worst_spectrum_fits_stacked` for downstream
            use;
        Added `r_{s,\lambda}` residuals to each pair in
            `plot_best_worst_spectrum_fits_stacked`;
        Changed residuals in `parallel_spectrum_plots` to match new
            `r_{s,\lambda}`. 20 August 2026
v1.25:  Allow `cpu_processes` and `blas_threads` to be passed in `kwargs` to
            `genCubeFit`;
        Removed all legacy checkpointing and tracking;
        Added SFH corner plot for each phase-space pair of the SSP library in
            `loadCubeFit`. 26 August 2026
"""

# need to set up the logger before any other imports
import pathlib as plp
from CubeFit.logger import get_logger
print("[CubeFit] Initializing CubeFit logger...")
curdir = plp.Path(__file__).parent
lfn = curdir/'kz_run.log'
logger = get_logger(lfn, mode='w')
logger.log(f"[CubeFit] CubeFit logger initialised to {logger.logfile}",
    flush=True)

import os, pdb, math, ctypes, sys, builtins, traceback
import numpy as np
import hashlib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize
from matplotlib import colormaps, colors as mcolors
from copy import copy
import h5py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed,\
    ThreadPoolExecutor
from plotbin.display_pixels import display_pixels as dbi

from CubeFit.hdf5_manager import H5Manager, open_h5,\
    live_prefit_snapshot_from_models, invalidate_done
from CubeFit.hypercube_builder import build_hypercube, assert_preflight_ok,\
    estimate_global_velocity_bias_prebuild
from CubeFit.pipeline_runner   import PipelineRunner
from CubeFit import cube_utils as cu
from dynamics.IFU.Constants import Constants, Units, UnitStr
from dynamics.IFU.Functions import Plot, Geometric
from cythonModules import C_utils as Cu
from cythonModules import C_GHKinematics as Cgh

mDir = curdir.parent/'muse'
dDir = cu._ddir()

UTS = UnitStr()
UTT = Units()
CTS = Constants()
POT = Plot()
GEO = Geometric()

divcmap = 'GECKOSdr'
moncmap = 'inferno'
moncmapr = 'inferno_r'

os.environ["FITTRACKER_START"] = "fork"

CPU_PROCESSES = 12
BLAS_THREADS = 4

# ------------------------------------------------------------------------------

def _worker_compute_tile(h5_path, s0, s1, x_cp64):
    x_cp64 = np.asarray(x_cp64, dtype=np.float64, order="C")

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]                     # (S,C,P,L) f32
        _, C, P, L = map(int, M.shape)
        # infer P_chunk from dataset chunks; fallback to full P
        P_chunk = (M.chunks[2] if getattr(M, "chunks", None) else P)
        dS = s1 - s0
        Y  = np.zeros((dS, L), dtype=np.float64)

        for p0 in range(0, P, P_chunk):
            p1   = min(P, p0 + P_chunk)
            slab = M[s0:s1, :, p0:p1, :][...] # (dS,C,Pb,L) f32
            A64  = slab.astype(np.float64, copy=False) # cast once per P_chunk
            xblk = x_cp64[:, p0:p1] # (C,Pb) f64
            Y   += np.tensordot(A64, xblk, axes=([1, 2], [0, 1])) # (dS,L)
    return s0, Y

#------------------------------------------------------------------------------

def _MWProp(prop, aperMass):
    """
    This function computes a mass-weighted map of the given property
    Args
    ----
        prop (arr:float): the property to be mass-weighted, of shape `(nComp,)`
        aperMass (arr:float): an array of the aperture masses from the
            Schwarzschild model, of shape `(nSpat, nComp)`
    Returns
    -------
        mwP (arr:float): an array of the mass-weighting per component, as well
            as the total mass-weighted value, of shape `(nSpat, nComp+1)`
    """

    mwP = np.ma.sum((aperMass/np.ma.sum(aperMass, axis=1)[:, np.newaxis])*\
        prop[np.newaxis, :], axis=1)
    return mwP

# ------------------------------------------------------------------------------

def genCubeFit(galaxy, mPath, decDir=None, nCuts=None, proj='i', SN=90,
    full=False, slope=1.30, IMF='KB', iso='pad', weighting='luminosity',
    lOrder=4, rescale=False, specRange=None, lsf=False, band='r', smask=None,
    method='fsf', varIMF=False, source='ppxf', redraw=False, runSwitch='gen',
    **kwargs):
    """
    Overarching generative function controlling the hypercube build and
        streaming solver.

    Parameters
    ----------
    galaxy : str
        Name of the galaxy to process.
    mPath : str
        Path to the model directory.
    decDir : str, optional
        Decomposition directory name. Default None.
    nCuts : int, optional
        Number of cuts for decomposition. Default None.
    proj : str, optional
        Projection type. Default 'i'.
    SN : int, optional
        Signal-to-noise ratio. Default 90.
    full : bool, optional
        Whether to use full data or truncated. Default False.
    slope : float, optional
        Slope for the IMF. Default 1.30.
    IMF : str, optional
        Initial mass function type. Default 'KB'.
    iso : str, optional
        Isochrones type. Default 'pad'.
    weighting : str, optional
        Weighting scheme. Default 'luminosity'.
    lOrder : int, optional
        Polynomial order for the fit. Default 4.
    rescale : bool, optional
        Whether to rescale the data. Default False.
    specRange : tuple, optional
        Spectral range to consider. Default None.
    lsf : bool, optional
        Whether to apply LSF (Line Spread Function). Default False.
    band : str, optional
        Band to use for the fit. Default 'r'.
    smask : str, optional
        Mask for the spectra. Default None.
    method : str, optional
        Method to use for the fit. Default 'fsf'.
    varIMF : bool, optional
        Whether to use variable IMF. Default False.
    source : str, optional
        Source of the data. Default 'ppxf'.
    redraw : bool, optional
        Whether to regenerate hypercube products. Default False.
    runSwitch : str, optional
        Switch to control the run mode; either 'gen' the hypercube or 'run' the
            fitting. Default 'gen'.
    **kwargs : dict, optional
        Additional keyword arguments for the function.
    """

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

    # Data spectra
    if vbSpec.is_file():
        VB = cu.Load.lzma(vbSpec)
        binNum = VB['binNum']
        binCounts = VB['binCounts']
        binFlux = VB['binFlux']
        del VB
    else:
        raise RuntimeError(f"No binned spectra.\n{'': <4s}{vbSpec}")

    warm_start = kwargs.pop('warm', 'zeros')

    with logger.capture_all_output():
        decDir, cDirs, cKeys, nComp, teLL, lnGrid, histBinSize, dataVelScale,\
            RZ, spLL, laGrid, lmin, lmax, umetals, uages, ualphas, pixOff = \
            cu._oneTimeSpec(galaxy=galaxy, mPath=mPath, decDir=decDir,
            nCuts=nCuts, proj=proj, SN=SN, full=full, slope=slope, IMF=IMF,
            iso=iso, weighting=weighting, lOrder=lOrder, rescale=rescale,
            lsf=lsf, specRange=specRange, band=band, method=method,
            varIMF=varIMF, source=source, **kwargs)
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
    apDir = bDir/bLKey/'nn_aphist.out'
    maDir = (bDir/bLKey).parent/'datfil'/'mass_aper.dat'
    nnK = bDir/bLKey/'nn_kinem.out'

    NOrbs, inds, energs, I2s, I3s, regs, types, weights, lcuts =\
        cu.Read.orbits(nnOrb)
    cWeights = np.array([
        np.ma.sum(oDict['weights'][f"{comp:{pred}d}"]) for comp in nzComp])

    kiBin = INF['kin']['nbins'][0]
    assert nbins == kiBin, 'Output does not agree with input bins\nInput:'+\
        f"{kiBin}\nOutput: {nbins}"

    wbin, hN, histBinSize, hArr = cu.Read.apertureHist(apDir)
    logger.log(f"{'Mass outside of the histograms:': <45s}"\
          f"{np.sum(hArr[:, 0] + hArr[:, wbin * 2]):5.5}")

    fullBin, fullID, fullK0 = cu.Read.massAperture(maDir)
    logger.log(f"{'Mass normalisation is:': <45s}"\
        f"{np.sum(hArr) / np.sum(fullK0):5.5}")
    if isinstance(proj, list):
        pStr = ''.join([str(f) for f in proj])
    else:
        pStr = str(proj)
    plt.close('all')
    massNorm = fullK0

    apMassFile = direc/f"apMass_i{proj}_{nComp:{pred}d}.xz"
    if apMassFile.is_file():
        aperMass = cu.Load.lzma(apMassFile)
    else:
        aperMass = np.ma.ones((nSpat, nComp), dtype=float)*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Mass', total=nComp):
            try:
                maFile = cDir/'declib_apermass.out'
                nbin, ID, k0 = cu.Read.massAperture(maFile)
                aperMass[:, cn] = k0
            except Exception as e:
                ERR += [[cDir.name, e]]
        if len(ERR) > 0:
            logger.log(ERR)
            breakpoint()
        cu.Write.lzma(apMassFile, aperMass)
    aperMass = np.ma.masked_invalid(aperMass)
    norma = np.sum(aperMass, axis=1)

    logger.log('Done.', flush=True)
    apFile = cDirs[0]/'declib_aphist.out'
    wbin, hN, histBinSize, hArr = cu.Read.apertureHist(apFile)
    # Load the parameters regardless
    apHistFile = direc/f"apHists_i{pStr}_{nComp:{pred}d}.jl"
    if apHistFile.is_file():
        logger.log('Reading histograms...', flush=True)
        apHists = cu.Load.jobl(apHistFile)
    else:
        apFile = cDirs[0]/'declib_aphist.out'
        wbin, hN, histBinSize, cArr = cu.Read.apertureHist(apFile)
        logger.log('Generating histograms...', flush=True)
        apHists = np.ma.ones((*cArr.shape, nComp))*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Components',
            total=nComp):
            try:
                apFile = cDir/'declib_aphist.out'
                wbin, hN, histBinSize, cArr = cu.Read.apertureHist(
                    apFile)
                apHists[:, :, cn] = cArr
            except Exception as e:
                ERR += [[cDir.stem, e]]
        if len(ERR) > 0:
            logger.log(ERR)
            pdb.set_trace()
        cu.Write.jobw(apHistFile, apHists)
    logger.log('Done.')
    apHists = np.ma.masked_invalid(apHists)
    nApHists = (apHists*(massNorm/norma)[:, np.newaxis, np.newaxis])
    # nApHists /= binFlux[:, np.newaxis, np.newaxis]
    hbi = wbin*2 + 1
    vbins = (np.arange(hbi)-wbin)*histBinSize
    # (nSpat, nVel, nComp)

    logger.log('Generating spectral mask...', flush=True)
    spmask = np.ones(nLSpec, dtype=bool)
    with open(dDir/'emissionLines.txt', 'r+') as emlf:
        emMask = np.genfromtxt(emlf, usecols=(0, 1))
    for emm in emMask:
        smask += [[emm[0]-emm[1]/2.0, emm[0]+emm[1]/2.0]]
    if len(smask)>0:
        for pair in smask:
            spmask[(spLL>=np.log(pair[0])) & (spLL<=np.log(pair[1]))] = False
    logger.log('Done.', flush=True)

    # --- Setup HDF5 directory ---
    hdf5Dir = plp.Path(kwargs.pop('hdf5Dir', curdir/galaxy))
    hdf5Dir.mkdir(parents=True, exist_ok=True)
    hdf5Path = (hdf5Dir/
        f"hypercube_{nComp:{pred}d}_{lOrder:02d}").with_suffix('.h5')

    # --- Initialize and load data ---
    mgr = H5Manager(hdf5Path)
    arDims = mgr.populate_from_arrays(
        losvd=nApHists,
        datacube=laGrid,
        templates=lnGrid,
        mask=spmask,
        tem_pix=copy(teLL), obs_pix=copy(spLL),
        vel_pix=copy(vbins),
        xpix=xpix, ypix=ypix,
        binnum=binNum,
        bincounts=binCounts,
        orbit_weights=cWeights,
    )
    mgr.ensure_rebin_and_resample()

    # --- 2. Precompute HyperCube ---
    if redraw and ('gen' in runSwitch):
        logger.log('[CubeFit] Calling `invalidate_done` to regenerate '\
            '/HyperCube.')
        invalidate_done(hdf5Path)

    nS, nC, nP = 128, 1, 360
    # --- Optional hard gate before any heavy work
    # Use small prefix slices if nothing is specified.
    with logger.capture_all_output():
        _ = assert_preflight_ok(hdf5Path,
            s_list=list(range(int(np.minimum(3, nS)))),
            c_list=list(range(int(np.minimum(2, nC)))),
            p_list=list(range(int(np.minimum(6, nP)))),
            # keep tolerances in sync with preflight defaults
            tol_rel=2e-3,
            tol_shift_px=0.5,
            tol_flat_valid=3e-8,
            require_rt_flat=True,
            rt_flat_tol=3e-8,
            verbose=True,
        )

        est = estimate_global_velocity_bias_prebuild(hdf5Path,
            n_spax=96, n_features=24, window_len=31, lag_px=12)

    logger.log(f"[CubeFit] Estimated global velocity bias (km/s): "\
        f"{est['vel_bias_kms']:.3f}")
    logger.log(f"[CubeFit] Building /HyperCube in {hdf5Path}...")
    with logger.capture_all_output():
        build_hypercube(
            hdf5Path,
            norm_mode="model", # choose "model" or "data"
            # "model" preserves relative contribution to both spaxel and components
            amp_mode="sum", # "sum" or "trapz"
            S_chunk=nS, C_chunk=nC, P_chunk=nP,
            vel_bias_kms=est["vel_bias_kms"]
        )
    # even if runSwitch is fit only, we want to ensure the HyperCube
    # is built, so we don't return early here.
    # Should be zero-cost if already built

    prefit_png = figDir / f"prefit_overlay_from_models_{nComp:{pred}d}.png"
    with logger.capture_all_output():
        live_prefit_snapshot_from_models(h5_path=str(hdf5Path),
            max_components=4, templates_per_pair=3,
            out_png=str(prefit_png),)
    logger.log(f"[Prefit] wrote {prefit_png}")
    if 'gen' in runSwitch:
        return

    if 'fit' not in runSwitch:
        logger.log(f"[CubeFit] runSwitch={runSwitch} is not understood; "
            "exiting.")
        raise RuntimeError("Invalid runSwitch")
    # --- 4) Run the global Kaczmarz fit (tiled; RAM-bounded) ---
    runner = PipelineRunner(hdf5Path)

    Ncpu, Nblas = kwargs.pop('cpu_processes', CPU_PROCESSES), kwargs.pop('blas_threads', BLAS_THREADS)
    best_processes, best_blas = cu.resolve_parallelism(Ncpu, Nblas)

    #####################################
    # Multi-processing Batched Kaczmarz #
    #####################################
    x_global, stats = runner.solve_all_mp_batched(
        # orbit_weights=None, # or None for “free” fit
        orbit_weights=cWeights,
        processes=best_processes,
        blas_threads=best_blas,
        reader_s_tile=128, # match /HyperCube/models chunking on S
        warm_start=warm_start,
    )

    xPath = hdf5Dir/hdf5Path.name.replace('hypercube', 'x')
    logger.log("[Pipeline] Writing final /X_global to ...")
    with open_h5(xPath, role="writer") as f_wr:

        assert x_global.ndim == 2, "Xcp must be (C, P) before writing /X_global"

        if "/X_global" in f_wr:
            del f_wr["/X_global"]

        f_wr.create_dataset(
            "/X_global",
            data=x_global.astype(np.float64),
            compression="gzip",
            compression_opts=4,
        )

        f_wr["/X_global"].attrs["layout"] = "C_P"
        f_wr["/X_global"].attrs["P"] = x_global.shape[1]

        if "known_zero_mask" in stats:
            print("[pipeline] writing KNOWN_ZERO mask to /HyperCube/known_zero_mask",
                flush=True)
            grp = f_wr.require_group("/HyperCube")
            if "known_zero_mask" in grp:
                del grp["known_zero_mask"]
            grp.create_dataset(
                "known_zero_mask",
                data=stats["known_zero_mask"].astype(bool),
                dtype="bool",
            )

    logger.log("[CubeFit] Global fit completed.")

# ------------------------------------------------------------------------------
# HDF5 helpers (added)
# ----------------------------------------------------------------------

def _coerce_h5_path(h5_or_path) -> str:
    """
    Return a filesystem path to the HDF5 file. Accepts str/Path or an
    h5py.File (uses .filename). Raises on unsupported input.
    """
    if isinstance(h5_or_path, (str, os.PathLike, plp.Path)):
        return str(h5_or_path)
    if isinstance(h5_or_path, h5py.File):
        return str(h5_or_path.filename)
    raise TypeError(f"Expected HDF5 file path or h5py.File; got {type(h5_or_path)}")

def _h5_exists(h5_path: str, key: str) -> bool:
    with open_h5(h5_path, role="reader") as f:
        return key in f

# ----------------------------------------------------------------------
# Rewritten HDF5-native functions
# ----------------------------------------------------------------------

def compute_model_batch_global(
    h5_or_path,
    batch_idx: int,
    x_global,
    nSpat: int,
):
    """
    Reconstruct a batch of model spectra from /HyperCube/models in HDF5.
    Uses the dataset's spatial chunk size as the batch length.

    Returns
    -------
    (start, Y) : (int, ndarray (m, nLSpec))
    """
    h5_path = _coerce_h5_path(h5_or_path)
    with open_h5(h5_path, role="reader") as f:
        models = f["/HyperCube/models"]
        if models.ndim == 5:
            nB, B, nC, nP, nL = models.shape
            start = batch_idx * B
            if start >= nSpat:
                return start, np.empty((0, nL), dtype=np.float64)
            m = int(min(B, nSpat - start))
            Y = np.empty((m, nL), dtype=np.float64)
            x2 = (x_global.reshape(nC, nP) if getattr(x_global, "ndim", 1) == 1 else x_global)
            x2 = np.asarray(x2, dtype=np.float64, order="C")
            for s in range(m):
                b, i = divmod(start + s, B)
                spec = np.asarray(models[b, i, :, :, :], order="C")
                Y[s, :] = np.tensordot(spec, x2, axes=([0, 1], [0, 1]))
            return start, Y
        elif models.ndim == 4:
            S, nC, nP, nL = models.shape
            B = (models.chunks[0] if getattr(models, "chunks", None) else 32)
            start = batch_idx * B
            if start >= nSpat:
                return start, np.empty((0, nL), dtype=np.float64)
            m = int(min(B, nSpat - start))
            slab = np.asarray(models[start:start + m, :, :, :], order="C")
            x2 = (x_global.reshape(nC, nP) if getattr(x_global, "ndim", 1) == 1 else x_global)
            x2 = np.asarray(x2, dtype=np.float64, order="C")
            Y = np.tensordot(slab, x2, axes=([1, 2], [0, 1]))
            return start, Y
        else:
            raise RuntimeError(f"Unexpected /HyperCube/models rank {models.ndim}")

def parallel_model_cube_global_batched(
    h5_or_path,
    x_global,
    nSpat: int,
    nLSpec: int,
    n_workers: int = 1,
    array_name: str = "ModelCube",
    spat_tile: int | None = None,
    compression: str | None = None,
    compression_opts: int | None = None,
    shuffle: bool = False,
):
    """
    Reconstruct full model cube from a global x and store (nSpat, nLSpec) f64
    into HDF5. **Single-writer** pattern:
      - Single-process path: one "r+" handle does read & write.
      - Multi-process path: main process holds the only "r+" handle and writes;
        workers open read-only and return their (s0, Y) blocks.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    h5_path = str(h5_or_path)

    # decide compression kwargs once
    comp_kwargs = {}
    if compression is None or compression is False:
        comp_kwargs = {}
    elif compression == "lzf":
        comp_kwargs = dict(compression="lzf", shuffle=shuffle)
    elif compression == "gzip":
        level = 1 if compression_opts is None else int(compression_opts)
        comp_kwargs = dict(compression="gzip", compression_opts=level, shuffle=shuffle)
    else:
        raise ValueError(f"Unsupported compression: {compression}")

    # Inspect dims/chunks via short-lived read handle
    with open_h5(h5_path, role="reader") as f_r:
        models0 = f_r["/HyperCube/models"]
        if models0.ndim == 4:
            S_disk, C_disk, P_disk, L_disk = models0.shape
            assert S_disk == nSpat, f"S mismatch: {S_disk} vs {nSpat}"
            assert L_disk == nLSpec, f"L mismatch: {L_disk} vs {nLSpec}"
            (S_chunk, C_chunk, P_chunk, L_chunk) = models0.chunks if getattr(models0, "chunks", None) else (128, C_disk, P_disk, L_disk)
        elif models0.ndim == 5:
            nB, B, C_disk, P_disk, L_disk = models0.shape
            assert nB * B == nSpat, f"S mismatch: {nB*B} vs {nSpat}"
            assert L_disk == nLSpec, f"L mismatch: {L_disk} vs {nLSpec}"
            S_chunk, C_chunk, P_chunk, L_chunk = (B, C_disk, P_disk, L_disk)
        else:
            raise RuntimeError(f"Unexpected HyperCube/models rank: {models0.ndim}")

    # Tile size along S
    if spat_tile is None:
        spat_tile = max(S_chunk * 4, 1)
    # Ensure destination dataset exists (writer)
    with open_h5(h5_path, role="writer") as f_w:
        if array_name in f_w:
            out = f_w[array_name]
            if out.shape != (nSpat, nLSpec) or str(out.dtype) != "float64":
                del f_w[array_name]
                out = f_w.create_dataset(array_name, shape=(nSpat, nLSpec),
                    dtype="f8", chunks=(min(spat_tile, nSpat), nLSpec),
                    **comp_kwargs)
        else:
            out = f_w.create_dataset(array_name, shape=(nSpat, nLSpec),
                dtype="f8", chunks=(min(spat_tile, nSpat), nLSpec),
                **comp_kwargs)

    # Precompute 2-D view of x (C,P) for GEMMs
    nC, nP = int(C_disk), int(P_disk)
    x_cp = (x_global.reshape(nC, nP) if getattr(x_global, "ndim", 1) == 1 else x_global)
    x_cp = np.asarray(x_cp, dtype=np.float64, order="C")

    # Build ranges
    ranges = []
    s = 0
    while s < nSpat:
        e = min(s + spat_tile, nSpat)
        ranges.append((s, e))
        s = e

    L_chunk = L_chunk
    print(f"[Reconstruct] S={nSpat} L={nLSpec} (L_band={L_chunk}) spat_tile={spat_tile} nTiles={len(ranges)} n_workers(requested)={n_workers}")

    # Single-process path
    if n_workers <= 1:
        with open_h5(h5_path, role="writer") as f:
            # source and destination in the same handle
            models = f["/HyperCube/models"]  # (S, C, P, L), stored as float32
            if array_name in f:
                out = f[array_name]
                if out.shape != (nSpat, nLSpec):
                    del f[array_name]
                    out = f.create_dataset(
                        array_name, shape=(nSpat, nLSpec), dtype="f8",
                        chunks=(spat_tile, nLSpec), **comp_kwargs)
            else:
                out = f.create_dataset(
                    array_name, shape=(nSpat, nLSpec), dtype="f8",
                    chunks=(spat_tile, nLSpec), **comp_kwargs)

            # reconstruct each spaxel tile without opening a second handle
            for (s0, s1) in tqdm(ranges, desc="[Reconstruct] tiles"):
                slab = np.asarray(models[s0:s1, :, :, :], dtype=np.float64, order="C")
                dS, C, P, L = slab.shape
                # A(s) is (N,L) with N=C*P.  Compute y_hat(s) = A(s)^T @ x.
                # AFTER (mirror the worker logic)
                # slab: (dS, C, P, L), x_cp: (C, P)
                Y = np.tensordot(slab, x_cp, axes=([1, 2], [0, 1])) # -> (dS, L)
                # (optional) ensure dtype/layout
                if Y.dtype != np.float64: Y = Y.astype(np.float64, copy=False)
                if not Y.flags["C_CONTIGUOUS"]: Y = np.ascontiguousarray(Y)
                out[s0:s1, :] = Y

        logger.log("[Reconstruct] Done (single-process).")
        return

    S_tile = S_chunk   # or 2*S_chunk if you have headroom

    # Build S-tiles (chunk aligned)
    ranges = [(s0, min(S_disk, s0+S_tile)) for s0 in range(0, S_disk, S_tile)]

    # Prepare output with tile-aligned chunks; uncompressed is fastest
    with open_h5(h5_path, role="writer") as f:
        if array_name in f:
            del f[array_name]
        out = f.create_dataset(array_name, shape=(S_disk, L_disk), dtype="f8",
                               chunks=(min(S_tile, S_disk), L_disk))

    # Don’t fork the parent’s big arrays
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(n_workers, len(ranges)),
                             mp_context=ctx,
                             initializer=_init_worker) as pool:
        futs = [pool.submit(_worker_compute_tile, h5_path, s0, s1, x_cp)
                for (s0, s1) in ranges]
        for fut in tqdm(
            as_completed(futs),
            total=len(ranges),
            desc="[Reconstruct] tiles",
            unit="tile",
            dynamic_ncols=True,
            miniters=1,
            leave=True,
        ):
            s0, Y = fut.result()
            s1 = s0 + Y.shape[0]
            with open_h5(h5_path, role="writer") as f:
                f[array_name][s0:s1, :] = Y
    print("[Reconstruct] Done (multi-process).")

# ------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parallel model-cube reconstruction (safe HDF5 writer in parent process)
# ---------------------------------------------------------------------------

# module globals used by worker processes
_RECON_X_CP2 = None
_RECON_WANT_DTYPE = np.float64

def _init_reconstruct_worker(
    x_cp2: np.ndarray,
    want_dtype_str: str,
    rdcc_slots: int,
    rdcc_bytes: int,
    rdcc_w0: float,
    blas_threads: int,
):
    """
    Initializer for reconstruction workers.

    Stores the fixed x_cp2 vector in a module-global so it is not pickled on
    every task. Also sets BLAS thread limits for the worker process.
    """
    global _RECON_X_CP2, _RECON_WANT_DTYPE

    _RECON_X_CP2 = np.ascontiguousarray(
        np.asarray(x_cp2, dtype=np.float64),
        dtype=np.float64,
    )
    _RECON_WANT_DTYPE = (
        np.float64 if str(want_dtype_str) == "float64" else np.float32
    )

    os.environ["OMP_NUM_THREADS"] = str(int(blas_threads))
    os.environ["OPENBLAS_NUM_THREADS"] = str(int(blas_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(blas_threads))
    os.environ["NUMEXPR_NUM_THREADS"] = str(int(blas_threads))

    # rdcc settings are applied per file handle inside the worker.

def _reconstruct_worker(args):
    """
    Worker executed in a separate process.

    Reads one spatial slab, contracts it locally with the fixed x_cp2, and
    returns only the reconstructed Y tile.

    Args tuple:
      (h5_path, s0, s1, rdcc_slots, rdcc_bytes, rdcc_w0)

    Returns
    -------
    (s0, s1, Y_tile_as_dtype)
        Y_tile has shape (dS, L).
    """
    import numpy as _np

    global _RECON_X_CP2, _RECON_WANT_DTYPE
    if _RECON_X_CP2 is None:
        raise RuntimeError(
            "Reconstruction worker not initialised with x_cp2."
        )

    h5_path, s0, s1, rdcc_slots, rdcc_bytes, rdcc_w0 = args

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        try:
            M.id.set_chunk_cache(
                int(rdcc_slots),
                int(rdcc_bytes),
                float(rdcc_w0),
            )
        except Exception:
            pass

        slab = _np.asarray(
            M[s0:s1, :, :, :],
            dtype=_np.float64,
            order="C",
        )

        Y_tile = _np.tensordot(
            slab,
            _RECON_X_CP2,
            axes=([1, 2], [0, 1]),
        )

        if _RECON_WANT_DTYPE != _np.float64:
            Y_tile = Y_tile.astype(_RECON_WANT_DTYPE, copy=False)

    return (s0, s1, Y_tile)

def reconstruct_modelcube_fast_parallel(
    h5_path: str,
    x_cp: np.ndarray,
    out_dset: str = "/ModelCube",
    s_chunk: int | None = None,
    out_dtype: str = "float64",
    rdcc_slots: int = 1_000_003,
    rdcc_bytes: int = 8 * 1024**2,
    rdcc_w0: float = 0.90,
    n_workers: int | None = None,
    blas_threads_per_worker: int | None = None,
) -> None:
    """
    Parallel reconstruction of /ModelCube with low peak memory.

    Workers read a spatial slab, contract it locally with x_cp, and return
    only the output tile. The parent performs only HDF5 writes.
    """

    if (
        n_workers is None
        or blas_threads_per_worker is None
    ):
        cpu_processes, blas_threads = (
            cu.resolve_parallelism(CPU_PROCESSES, BLAS_THREADS)
        )

        if n_workers is None:
            n_workers = cpu_processes

        if blas_threads_per_worker is None:
            blas_threads_per_worker = blas_threads

    want_dtype = np.float64 if str(out_dtype) == "float64" else np.float32
    x_in = np.ascontiguousarray(
        np.asarray(x_cp, dtype=np.float64).ravel(),
        dtype=np.float64,
    )

    with open_h5(h5_path, role="writer") as f:
        if "/HyperCube/models" not in f:
            raise RuntimeError("No /HyperCube/models found.")
        M = f["/HyperCube/models"]
        try:
            M.id.set_chunk_cache(
                int(rdcc_slots),
                int(rdcc_bytes),
                float(rdcc_w0),
            )
        except Exception:
            pass

        if M.ndim != 4:
            raise RuntimeError(f"Unexpected models rank {M.ndim}")

        S, C, P, L = map(int, M.shape)

        if s_chunk is None:
            # conservative default: ~8 GiB slabs per worker
            bytes_per_s = C * P * L * np.dtype(np.float64).itemsize
            target_worker_gib = 8.0
            s_chunk = max(
                1,
                int((target_worker_gib * 1024**3) // bytes_per_s),
            )

        S_blk = max(1, min(int(s_chunk), S))
        n_tiles = math.ceil(S / S_blk)

        ds = f.get(out_dset, None)
        if ds is not None:
            ok = (tuple(ds.shape) == (S, L) and str(ds.dtype) == str(want_dtype))
            if not ok:
                del f[out_dset]
                ds = None

        if ds is None:
            ds = f.create_dataset(
                out_dset,
                shape=(S, L),
                dtype=want_dtype,
                chunks=(S_blk, L),
                compression=None,
                shuffle=False,
            )
        out_ds = ds

    x_cp2 = x_in.reshape(C, P).astype(np.float64, copy=False)

    if n_workers is None:
        n_workers = 2
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        pbar = tqdm(total=n_tiles, desc="[Reconstruct]", mininterval=1.5)
        with open_h5(h5_path, role="writer") as f:
            M = f["/HyperCube/models"]
            out_ds = f[out_dset]
            for s0 in range(0, S, S_blk):
                s1 = min(S, s0 + S_blk)
                slab = np.asarray(
                    M[s0:s1, :, :, :],
                    dtype=np.float64,
                    order="C",
                )
                Y_tile = np.tensordot(
                    slab,
                    x_cp2,
                    axes=([1, 2], [0, 1]),
                )
                if want_dtype != np.float64:
                    Y_tile = Y_tile.astype(want_dtype, copy=False)
                out_ds[s0:s1, :] = Y_tile
                pbar.update(1)
        pbar.close()
        return

    ctx = mp.get_context("spawn")
    bt = int(blas_threads_per_worker)

    jobs = [
        (
            h5_path,
            s0,
            min(S, s0 + S_blk),
            int(rdcc_slots),
            int(rdcc_bytes),
            float(rdcc_w0),
        )
        for s0 in range(0, S, S_blk)
    ]

    pbar = tqdm(total=n_tiles, desc="[Reconstruct]", mininterval=1.5)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_reconstruct_worker,
        initargs=(x_cp2, out_dtype, rdcc_slots, rdcc_bytes, rdcc_w0, bt),
    ) as exe:
        futures = {exe.submit(_reconstruct_worker, arg): arg for arg in jobs}

        with open_h5(h5_path, role="writer") as f:
            out_ds = f[out_dset]
            for fut in as_completed(futures):
                s0, s1, Y_tile = fut.result()
                out_ds[s0:s1, :] = Y_tile
                pbar.update(1)

    pbar.close()
    print("[Reconstruct] Done (parallel).")

# ------------------------------------------------------------------------------

def reconstruct_modelcube_fast(
    h5_path: str,
    x_cp: np.ndarray,
    out_dset: str = "/ModelCube",
    s_chunk: int | None = None,
    out_dtype: str = "float64",
    rdcc_slots: int = 1_000_003,
    rdcc_bytes: int = 512 * 1024**2,
    rdcc_w0: float = 0.90,
) -> None:
    """
    Fast reconstruction of /ModelCube using full-L vectorised contraction.

    Computes:

        ModelCube[s, λ] = Σ_{c,p} x[c,p] * models[s,c,p,λ]

    using a single tensordot per spatial tile.

    This is the fastest possible CPU implementation without
    changing the HDF5 storage layout.

    Memory footprint per tile:
        ~ S_blk * C * P * L * 8 bytes  (float64 slab)

    """

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    want_dtype = np.float64 if str(out_dtype) == "float64" else np.float32

    with open_h5(h5_path, role="writer") as f:

        if "/HyperCube/models" not in f:
            raise RuntimeError("No /HyperCube/models found.")

        M = f["/HyperCube/models"]  # (S, C, P, L)

        try:
            M.id.set_chunk_cache(int(rdcc_slots), int(rdcc_bytes), float(rdcc_w0))
        except Exception:
            pass

        if M.ndim != 4:
            raise RuntimeError(f"Unexpected models rank {M.ndim}")

        S, C, P, L = map(int, M.shape)

        # --- Choose spatial tile size ---
        m_chunks = M.chunks or (S, 1, P, L)
        S_chunk_file = int(m_chunks[0])

        S_blk = S_chunk_file if s_chunk is None else int(s_chunk)
        S_blk = max(1, min(S_blk, S))

        # --- Prepare output dataset ---
        ds = f.get(out_dset, None)
        if ds is not None:
            ok = (
                tuple(ds.shape) == (S, L)
                and str(ds.dtype) == str(want_dtype)
            )
            if not ok:
                del f[out_dset]
                ds = None

        if ds is None:
            ds = f.create_dataset(
                out_dset,
                shape=(S, L),
                dtype=want_dtype,
                chunks=(S_blk, L),
                compression=None,
                shuffle=False,
            )

        out = ds

        # --- Validate and reshape weights ---
        x_in = np.asarray(x_cp, np.float64).ravel(order="C")
        if x_in.size != C * P:
            raise ValueError(f"x_cp length {x_in.size} != C*P={C*P}")

        x_cp = np.ascontiguousarray(
            x_in.reshape(C, P),
            dtype=np.float64,
        )

        # --- Progress ---
        n_tiles = math.ceil(S / S_blk)
        pbar = tqdm(total=n_tiles, desc="[Reconstruct]", mininterval=1.5)

        # ============================================================
        # Main reconstruction loop
        # ============================================================
        for s0 in range(0, S, S_blk):
            s1 = builtins.min(S, s0 + S_blk)

            # Load full slab for this tile
            slab = np.asarray(
                M[s0:s1, :, :, :],
                dtype=np.float64,
                order="C",
            )  # shape (dS, C, P, L)

            # Contract over (C,P)
            Y_tile = np.tensordot(
                slab,
                x_cp,
                axes=([1, 2], [0, 1]),
            )  # -> (dS, L)

            if want_dtype != np.float64:
                Y_tile = Y_tile.astype(want_dtype, copy=False)

            out[s0:s1, :] = Y_tile

            pbar.update(1)

        pbar.close()

    print("[Reconstruct] Done.")

# ------------------------------------------------------------------------------

def _x_digest(x) -> str:
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    h   = hashlib.sha1()
    h.update(x64.tobytes())
    h.update(str(x64.shape).encode("utf-8"))
    return h.hexdigest()

def modelcube_status(h5_path: str, x_global=None, require_float64: bool = True,
    redraw: bool = False):
    """
    Returns (ok: bool, msg: str). ok=True means you can safely skip rebuild.
    Checks presence, shape, dtype; if 'x_digest' attr is present, also checks currency vs x_global.
    """
    if redraw:
        return (False, "Forced re-creation")
    with open_h5(h5_path, role="reader") as f:
        if "/ModelCube" not in f:
            return (False, "missing /ModelCube")

        ds = f["/ModelCube"]
        # Determine expected (S,L) from DataCube if possible; else fall back to HyperCube/models
        if "/DataCube" in f:
            S, L = map(int, f["/DataCube"].shape)
        elif "/HyperCube/models" in f:
            M = f["/HyperCube/models"]
            S, L = int(M.shape[0]), int(M.shape[-1])
        else:
            return (False, "cannot infer (S,L) — missing /DataCube and /HyperCube/models")

        if ds.shape != (S, L):
            return (False, f"wrong shape {ds.shape} != ({S},{L})")

        if require_float64 and ds.dtype != np.float64:
            return (False, f"dtype {ds.dtype} is not float64")

        # If present, verify mask length is consistent
        if "/Mask" in f:
            mask_len = int(f["/Mask"].shape[0])
            if mask_len != L:
                return (False, f"/Mask length {mask_len} != L={L}")

        # If the dataset has a digest, compare with current x (if provided)
        ds_digest = ds.attrs.get("x_digest", None)
        ds_xshape = tuple(ds.attrs.get("x_shape", ()))
        if (x_global is not None) and (ds_digest is not None):
            xshape = np.asarray(x_global).shape
            if ds_xshape and tuple(ds_xshape) != xshape:
                return (False, f"x_shape mismatch: file {tuple(ds_xshape)} vs current {xshape}")
            cur = _x_digest(x_global)
            if cur != ds_digest:
                return (False, "digest mismatch: /ModelCube built with different x")
            return (True, "present, shape/dtype ok, digest matches")

        # No digest to check; accept but note it’s unverified against x
        return (True, "present, shape/dtype ok (no digest to verify)")

# ------------------------------------------------------------------------------

def parallel_spectrum_plots(
    h5_or_path: str,
    fit_metric: np.ndarray,
    n: int,
    plot_dir: plp.Path | str,
    n_workers: int,
    tag: str,
    mask: np.ndarray | None = None,
):
    """
    Memory-safe plotting:
      - Reads only needed rows from /DataCube and /ModelCube.
      - Closes every figure immediately.
      - Small thread pool (I/O bound).

    Style:
      - Data in black, model in red (lw=0.8).
      - Residuals (data - model) as green diamonds at every pixel,
        vertically offset so they don't overlap the spectra.
      - A solid green line at the residual zero (i.e., the offset
        baseline), and thin dashed green lines at ±1σ (σ computed on
        masked residuals).
      - If /Mask exists (or 'mask' provided), masked regions are shaded
        with semi-transparent grey bands.
    """
    pDir = plp.Path(plot_dir)
    pDir.mkdir(parents=True, exist_ok=True)

    n = int(np.maximum(1, n))
    fit_metric = np.asarray(fit_metric, dtype=np.float64)
    S = int(fit_metric.shape[0])

    # Pick indices (worst/best by fit_metric)
    order_desc = np.argsort(-fit_metric)
    order_asc  = np.argsort( fit_metric)
    idx_worst  = order_desc[:n]
    idx_best   = order_asc[:n]
    picks      = np.unique(np.concatenate([idx_worst, idx_best])).astype(int)

    # Read only the selected rows + metadata
    with open_h5(str(h5_or_path), role="reader") as f:
        if "/ModelCube" not in f:
            raise RuntimeError("Expected /ModelCube (S,L) for plotting. Reconstruct first.")
        data_ds  = f["/DataCube"]    # (S,L)
        model_ds = f["/ModelCube"]   # (S,L)

        L = int(model_ds.shape[1])
        obs = f["/ObsPix"][...] if "/ObsPix" in f else np.arange(L, dtype=np.float64)

        # Prefer provided mask; else load /Mask; else keep-all
        if mask is None and "/Mask" in f:
            m = np.asarray(f["/Mask"][...], dtype=bool).ravel()
            mask = m if int(m.size) == L else None
        if mask is None:
            mask = np.ones(L, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if int(mask.size) != L:
                raise ValueError(f"Mask length {mask.size} != L={L}")

        print(
            "[Plots] picks={} L={} mem≈{:.1f} MB for data+model rows"
            .format(int(picks.size), L, float(picks.size * L * 16.0 / 1e6))
        )

        data_sel  = np.empty((int(picks.size), L), dtype=np.float64)
        model_sel = np.empty((int(picks.size), L), dtype=np.float64)
        for j, s in enumerate(picks):
            data_sel[j, :]  = data_ds[int(s), :]
            model_sel[j, :] = model_ds[int(s), :]

    # Precompute masked bands as contiguous intervals where mask == False
    masked = ~mask
    if np.any(masked):
        pad = np.concatenate((
            np.array([0], dtype=np.int8),
            masked.view(np.int8),
            np.array([0], dtype=np.int8)
        ))
        edges = np.diff(pad)
        starts = np.nonzero(edges == 1)[0]
        ends   = np.nonzero(edges == -1)[0]
        mask_spans = list(zip(starts, ends))  # intervals [start, end)
    else:
        mask_spans = []

    # Small plotting worker: operates on compact row views
    def _plot_one(s_idx: int, rank_tag: str):
        j = int(np.where(picks == s_idx)[0][0])

        dat = data_sel[j, :]
        mod = model_sel[j, :]

        # --------------------------------------------------------
        # Fractional residual:
        #
        #              D - M
        # r = -----------------------
        #          0.5 * (D + M)
        #
        # Plot 100*r in percent.
        # --------------------------------------------------------
        denom = 0.5 * (dat + mod)

        valid = (
            mask
            & np.isfinite(dat)
            & np.isfinite(mod)
            & np.isfinite(denom)
            & (denom > 0.0)
        )

        frac_resid_pct = np.full(L, np.nan, dtype=np.float64,)

        if np.any(valid):
            positive = denom[valid]
            rel_floor = max(float(np.nanpercentile(positive,1.0) * 1e-6), 1e-30)
            frac_resid_pct[valid] = 100.0 * (dat[valid] - mod[valid])\
                / np.maximum(denom[valid], rel_floor)

        # --------------------------------------------------------
        # Figure
        # --------------------------------------------------------
        fig = plt.figure(figsize=(8, 4.5))
        gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.0)
        ax_spec = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_spec)

        # --------------------------------------------------------
        # Spectrum
        # --------------------------------------------------------
        ax_spec.plot(obs[mask], dat[mask], lw=0.8, color='k', label='Data')

        ax_spec.plot(obs[mask], mod[mask], lw=0.8, color='tab:red',
            label='Model')

        # --------------------------------------------------------
        # Fractional residual
        # --------------------------------------------------------
        ax_resid.plot(obs[valid], frac_resid_pct[valid], lw=0.75,
            color="tab:green",)
        ax_resid.axhline(0.0, lw=0.55, color="tab:green", alpha=0.7)

        # --------------------------------------------------------
        # Masked regions
        # --------------------------------------------------------
        if mask_spans:
            for a, b in mask_spans:
                x0 = float(obs[int(a)])
                x1 = float(obs[int(max(a, b - 1))])

                if int(b) < L:
                    x1 = float(obs[int(b)])

                ax_spec.axvspan(x0, x1, color="0.2", alpha=0.12, zorder=0)
                ax_resid.axvspan(x0, x1, color="0.2", alpha=0.12, zorder=0)

        # --------------------------------------------------------
        # Q_s annotation
        # --------------------------------------------------------
        q_s = float(fit_metric[int(s_idx)])
        ax_spec.text(0.015, 0.94, rf"$s={int(s_idx):d}\qquad Q_s={q_s:.2f}\%$",
            transform=ax_spec.transAxes, ha="left", va="top", color="k")

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------
        ax_spec.set_ylabel(r"$F_\lambda$ (arb. units)")
        ax_spec.legend(loc="upper right", frameon=False)
        ax_spec.tick_params(axis="x", labelbottom=False,)

        ax_resid.set_ylabel(r"$100\,r_{s,\lambda}\,[\%]$")
        ax_resid.set_xlabel("log(\u03bb [\u212B])")

        # Use a symmetric residual range so positive and negative
        # discrepancies have identical visual significance.
        finite_abs = np.abs(frac_resid_pct[np.isfinite(frac_resid_pct)])

        if finite_abs.size > 0:
            resid_lim = float(np.nanpercentile(finite_abs, 99.5,))
            resid_lim = max(resid_lim, 1e-3,)
            ax_resid.set_ylim(-1.10 * resid_lim, +1.10 * resid_lim)

        fig.savefig(pDir/f"{rank_tag}_{tag}_spax{int(s_idx):05d}.png",
            dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Tiny pool; ≤4 for I/O friendliness
    pool_n = int(np.minimum(np.maximum(1, int(n_workers)), 6))
    jobs = [(int(s), "worst") for s in idx_worst] + \
           [(int(s), "best")  for s in idx_best]

    with ThreadPoolExecutor(max_workers=pool_n) as pool:
        list(pool.map(lambda args: _plot_one(*args), jobs))

    del data_sel, model_sel

# ------------------------------------------------------------------------------

def plot_best_worst_spectrum_fits_stacked(
    h5_or_path: str,
    fit_metric: np.ndarray,
    n_each: int = 3,
    plot_path: plp.Path | str | None = None,
    mask: np.ndarray | None = None,
    title: str | None = None,
    tag: str = "best_worst",
) -> plp.Path:
    """
    Make a publication-quality stacked comparison of best and worst spectral
    fits on one figure.

    Each spectrum is normalized by its own robust amplitude estimate and then
    vertically offset by a constant amount, so every panel-like trace has the
    same effective height in plot units.

    Parameters
    ----------
    h5_or_path : str
        HDF5 file path.
    fit_metric : ndarray
        Per-spaxel fractional residual NMAD in percent. Smaller is better.
    n_each : int, optional
        Number of best and worst spectra to show.
    plot_path : Path or str, optional
        Output PNG path. If omitted, a default path is created next to the HDF5
        file.
    mask : ndarray, optional
        Boolean spectral mask. True means keep the wavelength pixel.
    title : str, optional
        Optional figure title.
    tag : str, optional
        Name fragment used when plot_path is not supplied.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    h5_path = str(h5_or_path)
    fit_metric = np.asarray(fit_metric, dtype=np.float64).ravel()
    if fit_metric.size == 0:
        raise ValueError("fit_metric is empty.")

    n_each = int(max(1, n_each))
    n_each = min(n_each, int(fit_metric.size))

    idx_best = np.argsort(fit_metric)[:n_each]
    idx_worst = np.argsort(fit_metric)[::-1][:n_each]

    picks = []
    seen = set()
    for label, idxs in (("worst", idx_worst), ("best", idx_best)):
        for s in idxs:
            s = int(s)
            if s not in seen:
                picks.append((label, s))
                seen.add(s)

    with open_h5(h5_path, role="reader") as f:
        if "/DataCube" not in f or "/ModelCube" not in f:
            raise RuntimeError("Expected /DataCube and /ModelCube in HDF5.")

        data_ds = f["/DataCube"]
        model_ds = f["/ModelCube"]

        if data_ds.ndim != 2 or model_ds.ndim != 2:
            raise RuntimeError("Expected /DataCube and /ModelCube to be 2-D.")

        if data_ds.shape != model_ds.shape:
            raise RuntimeError(
                f"Shape mismatch: {data_ds.shape} vs {model_ds.shape}"
            )

        L = int(model_ds.shape[1])
        obs = (
            np.asarray(f["/ObsPix"][...], dtype=np.float64)
            if "/ObsPix" in f
            else np.arange(L, dtype=np.float64)
        )

        if mask is None and "/Mask" in f:
            m = np.asarray(f["/Mask"][...], dtype=bool).ravel()
            mask = m if int(m.size) == L else None

        if mask is None:
            mask = np.ones(L, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool).ravel()
            if int(mask.size) != L:
                raise ValueError(f"Mask length {mask.size} != L={L}")

        data_sel = np.empty((len(picks), L), dtype=np.float64)
        model_sel = np.empty((len(picks), L), dtype=np.float64)
        fit_metric_sel = np.empty((len(picks),), dtype=np.float64)

        for j, (_, s) in enumerate(picks):
            data_sel[j, :] = np.asarray(data_ds[int(s), :], dtype=np.float64)
            model_sel[j, :] = np.asarray(model_ds[int(s), :], dtype=np.float64)
            fit_metric_sel[j] = float(fit_metric[int(s)])

    # Normalise each spectrum independently so every trace has the same
    # effective vertical height in plot units.
    plot_amp = 0.42
    band_step = 1.0
    offsets = np.arange(len(picks), dtype=np.float64) * band_step

    data_plot = np.empty_like(data_sel)
    model_plot = np.empty_like(model_sel)
    scales = np.empty((len(picks),), dtype=np.float64)
    centers = np.empty((len(picks),), dtype=np.float64)

    for j in range(len(picks)):
        vals = np.concatenate((data_sel[j, mask], model_sel[j, mask]))
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            center = 0.0
            scale = 1.0
        else:
            center = float(np.nanmedian(vals))
            spread = np.abs(vals - center)
            scale = float(np.nanpercentile(spread, 99.0))
            scale = max(scale, 1e-30)

        centers[j] = center
        scales[j] = scale

        data_plot[j, :] = plot_amp * (data_sel[j, :] - center) / scale
        model_plot[j, :] = plot_amp * (model_sel[j, :] - center) / scale

    # Masked wavelength intervals.
    masked = ~mask
    mask_spans = []
    if np.any(masked):
        pad = np.concatenate((
            np.array([0], dtype=np.int8),
            masked.view(np.int8),
            np.array([0], dtype=np.int8),
        ))
        edges = np.diff(pad)
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0]
        mask_spans = list(zip(starts, ends))

    if plot_path is None:
        base = plp.Path(h5_path)
        plot_path = base.with_name(
            base.stem + f"_{tag}_stacked.png"
        )
    else:
        plot_path = plp.Path(plot_path)

    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Prepare spectra and fractional residuals
    # ------------------------------------------------------------
    n_picks = len(picks)

    plot_amp = 0.34
    residual_amp = 0.14
    residual_offset = 0.38
    band_step = 1.0

    offsets = np.arange(n_picks, dtype=np.float64,) * band_step

    data_plot = np.empty_like(data_sel)
    model_plot = np.empty_like(model_sel)

    frac_resid_pct = np.full_like(data_sel, np.nan, dtype=np.float64)

    for j in range(n_picks):
        dat = data_sel[j, :]
        mod = model_sel[j, :]

        # --------------------------------------------------------
        # Robust display normalization for the spectrum.
        # This affects only its visual vertical amplitude.
        # --------------------------------------------------------
        vals = np.concatenate((dat[mask], mod[mask],))
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            center = 0.0
            scale = 1.0
        else:
            center = float(np.nanmedian(vals))

            spread = np.abs(vals - center)

            scale = float(np.nanpercentile(spread, 99.0,))

            scale = max(scale, 1e-30,)

        data_plot[j, :] = plot_amp * (dat - center) / scale
        model_plot[j, :] = plot_amp * (mod - center) / scale

        # --------------------------------------------------------
        # Fractional residual:
        #
        #              D - M
        # r = -----------------------
        #          0.5 * (D + M)
        #
        # Store 100*r so the plotted residual is in percent.
        # --------------------------------------------------------
        denom = 0.5 * (dat + mod)

        valid = (
            mask
            & np.isfinite(dat)
            & np.isfinite(mod)
            & np.isfinite(denom)
            & (denom > 0.0)
        )

        if np.any(valid):
            positive = denom[valid]

            rel_floor = max(float(np.nanpercentile(positive, 1.0) * 1e-6
                ), 1e-30)

            frac_resid_pct[j, valid] = 100.0 * (dat[valid] - mod[valid])\
                / np.maximum(denom[valid], rel_floor)

    # ------------------------------------------------------------
    # Use ONE residual scale for every stacked spectrum.
    #
    # Therefore an n-percent residual has the same visual
    # displacement for every spatial bin.
    # ------------------------------------------------------------
    finite_resid = np.abs(frac_resid_pct[np.isfinite(frac_resid_pct)])

    if finite_resid.size > 0:
        residual_scale = max(float(np.nanpercentile(finite_resid, 99.0)), 1e-6)
    else:
        residual_scale = 1.0

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=plt.figaspect(3.0 / n_picks))

    data_c = 'k'
    model_c = 'tab:red'
    resid_c = 'tab:green'

    for j, ((label, s), off) in enumerate(zip(picks, offsets)):
        dat = data_plot[j, :] + off
        mod = model_plot[j, :] + off

        # --------------------------------------------------------
        # Data: D_{s,lambda}
        # --------------------------------------------------------
        ax.plot(obs[mask], dat[mask], lw=1.0, color=data_c,
            solid_capstyle='round')
        # --------------------------------------------------------
        # Model: M_{s,lambda}
        # --------------------------------------------------------
        ax.plot(obs[mask],mod[mask], lw=1.0, color=model_c,
            alpha=0.95, solid_capstyle='round')
        # --------------------------------------------------------
        # Fractional residual: 100 r_{s,lambda}
        # --------------------------------------------------------
        resid_off = off - residual_offset
        resid = frac_resid_pct[j, :]
        valid_resid = mask & np.isfinite(resid)

        if np.any(valid_resid):
            resid_y = resid_off + residual_amp * resid / residual_scale
            ax.plot(obs[valid_resid], resid_y[valid_resid], lw=0.75,
                color=resid_c, alpha=0.95)
        # r = 0 baseline.
        ax.axhline(resid_off, lw=0.45, color=resid_c, alpha=0.65)

        # --------------------------------------------------------
        # Masked wavelength regions
        # --------------------------------------------------------
        if mask_spans:
            for a, b in mask_spans:
                x0 = float(obs[int(a)])
                x1 = float(obs[int(max(a, b - 1))])

                if int(b) < L:
                    x1 = float(obs[int(b)])

                ax.axvspan(x0, x1, color="0.2", alpha=0.08, zorder=0)

        # --------------------------------------------------------
        # Q_s annotation
        # --------------------------------------------------------
        txt = rf"$s={int(s):d}\qquad Q_s={fit_metric_sel[j]:.2f}\%$"
        ax.text(0.01, off + 0.29, txt, ha='left', va='center',
            transform=ax.get_yaxis_transform(), color='k', path_effects=[
                PathEffects.withStroke(linewidth=2.0, foreground='white',)],)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    ax.plot([], [], color=data_c, lw=1.2, label='Data')
    ax.plot([], [], color=model_c, lw=1.2, label='Model')
    ax.plot([], [], color=resid_c, lw=0.8, label=r"$100\,r_{s,\lambda}$")

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------
    ax.set_xlim(float(np.nanmin(obs[mask])), float(np.nanmax(obs[mask])))
    ax.set_ylim(
        offsets[0] - 0.60,
        offsets[-1] + 0.55*1.5, # add extra for the legend
    )
    ax.set_xlabel("log(\u03bb [\u212B])")
    ax.set_ylabel("Normalized flux and fractional residual")
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False,)
    ax.legend(loc="upper right", frameon=False, fontsize=9,)

    if title is not None:
        ax.set_title(title, fontsize=13, pad=10,)

    fig.savefig(plot_path, dpi=300, bbox_inches="tight",)
    plt.close(fig)

    return picks

# ------------------------------------------------------------------------------

def ceil_div(a,b): return (a + b - 1)//b
def round_down_to_multiple(x,m): return (x//m)*m

def choose_spat_tile_fast(S, n_workers, s_chunk, k=2):
    """
    Try for ~k*n_workers tiles. If chunk-aligning would give too few tiles,
    drop alignment to keep all cores busy.
    """
    target_tiles = max(1, k * n_workers)
    raw = max(1, S // target_tiles)          # integer floor

    # First try: chunk-aligned (round DOWN)
    tile = round_down_to_multiple(raw, s_chunk)
    if tile < s_chunk:
        tile = s_chunk
    n_tiles = ceil_div(S, tile)

    # If alignment leaves us with fewer tiles than workers, drop alignment
    if n_tiles < n_workers:
        tile = max(1, S // target_tiles)     # non-aligned raw
        n_tiles = ceil_div(S, tile)

    return tile, n_tiles

def _init_worker(blas_threads: int):
    """
    Worker initializer: set BLAS threading environment variables.
    Must be top-level and pickleable.
    """
    os.environ["OMP_NUM_THREADS"] = str(int(blas_threads))
    os.environ["OPENBLAS_NUM_THREADS"] = str(int(blas_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(blas_threads))
    os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, int(blas_threads) // 2))
    try:
        # runtime guard in case threads were already initialized
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)  # BLAS/OpenMP libraries → 1 thread
    except Exception:
        pass

# ------------------------------------------------------------------------------

def loadCubeFit(galaxy, mPath, decDir=None, nCuts=None, proj='i', SN=90,
    full=False, slope=1.30, IMF='KB', iso='pad', weighting='luminosity',
    lOrder=4, rescale=False, specRange=None, lsf=False, band='r', smask=None,
    method='fsf', varIMF=False, source='ppxf', redraw=False,
    pplots=['sfh', 'spec', 'mw', 'proj'], **kwargs):
    """
    Load the CubeFit data for a given galaxy and model path.
    """
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
 
    pfs = pDir/galaxy/f"pixels_SN{SN:02d}.xz"
    vbSpec = pDir/galaxy/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    infn = bDir/'infil.xz'

    INF = cu.Load.lzma(infn)
    PA = INF['angle'][0]
    if vbSpec.is_file():
        VB = cu.Load.lzma(vbSpec)
        binNum = VB['binNum']
        binCounts = VB['binCounts']
        del VB
    else:
        raise RuntimeError(f"No binned spectra.\n{'': <4s}{vbSpec}")

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
    xbin, ybin = INF['kin']['moms'][0]['x'], INF['kin']['moms'][0]['y']
    xbin, ybin = GEO.rotate2D(xbin, ybin, PA)

    with logger.capture_all_output():
        decDir, cDirs, cKeys, nComp, teLL, lnGrid, histBinSize, dataVelScale,\
            RZ, spLL, laGrid, lmin, lmax, umetals, uages, ualphas, pixOff = \
            cu._oneTimeSpec(galaxy=galaxy, mPath=mPath, decDir=decDir,
            nCuts=nCuts, proj=proj, SN=SN, full=full, slope=slope, IMF=IMF,
            iso=iso, weighting=weighting, lOrder=lOrder, rescale=rescale,
            lsf=lsf, specRange=specRange, band=band, method=method,
            varIMF=varIMF, source=source, **kwargs)
    nLSpec, nSpat = laGrid.shape
    nTSpec, nMetals, nAges, nAlphas = lnGrid.shape
    nSSP = int(np.prod((nMetals, nAges, nAlphas), dtype=int))
    pred = f"0{len(repr(nComp)):d}"
    nComp = int(nComp)
    print(RZ)
    par = INF['parameters']
    dataMax = np.max(INF['dataMax'])
    rLogMin, rLogMax = par['rLogMin'], par['rLogMax']
    nMom = INF['kin']['nMom']
    grid = np.array(INF['bins'][0]['grid'], dtype=int).T.ravel()-1
    nbins = np.max(grid).astype(int)+1
    ss = np.where(grid >= 0)[0]
    GRIDS = grid[ss]

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

    pc = RZ.getPC()
    km = RZ.getKM()

    sRadius = np.logspace(par['rLogMin'], par['rLogMax'], par['nE']) #spherical
    refML = par['gpML']
    tMGE = par['tMGE']
    sMGE = par['sMGE']
    minQ = np.min([sMGE.q.min(), tMGE.q.min()])
    bTheta, bPhi, bPsi = Cu.oneQPUtoTPP(fpd['q'], fpd['p'], fpd['u'], minQ)
    cRadius = sRadius * np.sin(np.radians(bPhi))  # Spherical -> Cylindrical
    wc = np.nonzero(cRadius <= 0.1)[0]
    inner = cRadius[wc]
    cRadius = np.delete(cRadius, wc)
    cRadius = np.append(np.nanmean(inner), cRadius)
    nCRad = cRadius.size
    wc = np.nonzero(sRadius <= 0.1)[0]
    inner = sRadius[wc]
    sRadius = np.delete(sRadius, wc)
    sRadius = np.append(np.nanmean(inner), sRadius)
    nSRad = sRadius.size
    print(f"# Radial bins: {nCRad} cylindrical, {nSRad} spherical")

    apDir = bDir/bLKey/'nn_aphist.out'
    maDir = (bDir/bLKey).parent/'datfil'/'mass_aper.dat'

    NOrbs, inds, energs, I2s, I3s, regs, types, weights, lcuts =\
        cu.Read.orbits(nnOrb)
    cWeights = np.array([
        np.ma.sum(oDict['weights'][f"{comp:{pred}d}"]) for comp in nzComp])

    kiBin = INF['kin']['nbins'][0]
    assert nbins == kiBin, 'Output does not agree with input bins\nInput:'+\
        f"{kiBin}\nOutput: {nbins}"

    wbin, hN, histBinSize, hArr = cu.Read.apertureHist(apDir)
    logger.log(f"{'Mass outside of the histograms:': <45s}"\
          f"{np.sum(hArr[:, 0] + hArr[:, wbin * 2]):5.5}")

    fullBin, fullID, fullK0 = cu.Read.massAperture(maDir)
    logger.log(f"{'Mass normalisation is:': <45s}"\
        f"{np.sum(hArr) / np.sum(fullK0):5.5}")
    if isinstance(proj, list):
        pStr = ''.join([str(f) for f in proj])
    else:
        pStr = str(proj)
    plt.close('all')
    massNorm = fullK0

    apMassFile = direc/f"apMass_i{proj}_{nComp:{pred}d}.xz"
    if apMassFile.is_file():
        aperMass = cu.Load.lzma(apMassFile)
    else:
        aperMass = np.ma.ones((nSpat, nComp), dtype=float)*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Mass', total=nComp):
            try:
                maFile = cDir/'declib_apermass.out'
                nbin, ID, k0 = cu.Read.massAperture(maFile)
                aperMass[:, cn] = k0
            except Exception as e:
                ERR += [[cDir.stem, e]]
        if len(ERR) > 0:
            logger.log(ERR)
            breakpoint()
        cu.Write.lzma(apMassFile, aperMass)
    aperMass = np.ma.masked_invalid(aperMass)
    norma = np.sum(aperMass, axis=1)

    logger.log('Done.', flush=True)
    apFile = cDirs[0]/'declib_aphist.out'
    wbin, hN, histBinSize, hArr = cu.Read.apertureHist(apFile)
    # Load the parameters regardless
    apHistFile = direc/f"apHists_i{pStr}_{nComp:{pred}d}.jl"
    if apHistFile.is_file():
        logger.log('Reading histograms...', flush=True)
        apHists = cu.Load.jobl(apHistFile)
    else:
        apFile = cDirs[0]/'declib_aphist.out'
        wbin, hN, histBinSize, cArr = cu.Read.apertureHist(apFile)
        logger.log('Generating histograms...', flush=True)
        apHists = np.ma.ones((*cArr.shape, nComp))*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Components',
            total=nComp):
            try:
                apFile = cDir/'declib_aphist.out'
                wbin, hN, histBinSize, cArr = cu.Read.apertureHist(
                    apFile)
                apHists[:, :, cn] = cArr
            except Exception as e:
                ERR += [[cDir.stem, e]]
        if len(ERR) > 0:
            logger.log(ERR)
            pdb.set_trace()
        cu.Write.jobw(apHistFile, apHists)
    logger.log('Done.')
    apHists = np.ma.masked_invalid(apHists)
    nApHists = (apHists*(massNorm/norma)[:, np.newaxis, np.newaxis])

    intFN = direc/'intrinsicData.xz'
    if intFN.is_file():
        angle, intData = cu.Load.lzma(intFN)
    else:
        imDir = direc/f"ii_{1:{pred}d}"/'declib_intrinsic_moments.out'
        nMoms, nPh, nTh, nLr, phiBound, thBound, rBound, phi, theta, rr,\
            mgeMass, fitMass, errMass, XX, YY, ZZ, vX, vY, vZ, vX2, vY2, vZ2,\
            vXvY, vYvZ, vZvX, orbLong, orbShort, orbBox = \
            cu.Read.intrMoments(imDir)
        angles = np.ma.ones((3, nComp, XX.size), dtype=int)*np.nan
        intData = np.ma.ones((18, nComp, XX.size), dtype=float)*np.nan
        for comp in tqdm(range(nComp), desc='intrData', total=nComp):
            imDir = direc/f"ii_{comp+1:{pred}d}"/'declib_intrinsic_moments.out'
            nMoms, nPh, nTh, nLr, phiBound, thBound, rBound, phi, theta, rr,\
                mgeMass, fitMass, errMass, XX, YY, ZZ, vX, vY, vZ, vX2, vY2,\
                vZ2, vXvY, vYvZ, vZvX, orbLong, orbShort, orbBox = \
                cu.Read.intrMoments(imDir)
            angles[:, comp, :] = np.vstack((rr, theta, phi))
            intData[:, comp, :] = np.vstack((mgeMass, fitMass, errMass, XX,
                YY, ZZ, vX, vY, vZ, vX2, vY2, vZ2, vXvY, vYvZ, vZvX, orbLong,
                orbShort, orbBox))
        cu.Write.lzma(intFN, [angles, intData], preset=2)
    # mgeMass fitMass errMass XX YY ZZ vX vY vZ vX2 vY2 vZ2 vXvY vYvZ vZvX
    #    0       1       2     3  4  5  6  7  8  9   10  11  12   13   14
    # orbLong orbShort orbBox
    #    15      16      17
    intData[1, :, :] /= np.ma.sum(intData[1, :, :])
    ftmMask = np.broadcast_to(np.ma.getmaskarray(np.ma.masked_equal(
        intData[1, :, :], 0.0))[np.newaxis, :, :], intData.shape)
    intData = np.ma.masked_array(intData, mask=ftmMask)

    # --- Setup HDF5 directory ---
    hdf5Dir = plp.Path(kwargs.pop('hdf5Dir', curdir/galaxy))
    hdf5Dir.mkdir(parents=True, exist_ok=True)
    hdf5Path = (hdf5Dir/
        f"hypercube_{nComp:{pred}d}_{lOrder:02d}").with_suffix('.h5')
    
    # Read dims & X_global using robust reader
    with open_h5(hdf5Path, role="reader") as f:
        if "/X_global" not in f:
            raise RuntimeError("No /X_global found — run the fit first.")
        x_global = f["/X_global"][...]

        if "/HyperCube/models" not in f:
            raise RuntimeError("No /HyperCube/models found — build the HyperCube first.")
        models = f["/HyperCube/models"]
        models_chunks = models.chunks  # may be None
        if models.ndim == 4:
            nSpat, nComp, nPop, nLSpec = map(int, models.shape)
            s_chunk = (models_chunks[0] if models_chunks is not None else 32)
        elif models.ndim == 5:
            nB, B, nComp, nPop, nLSpec = map(int, models.shape)
            nSpat = nB * B
            s_chunk = B
        else:
            raise RuntimeError(f"Unexpected /HyperCube/models rank {models.ndim}")

        # optional input data for plots
        has_mask = ("/Mask" in f)
        mask_arr = np.asarray(f["/Mask"][...], bool) if has_mask else None
        obs = f["/ObsPix"][...] if "/ObsPix" in f else np.arange(nLSpec)

    best_processes, best_blas = cu.resolve_parallelism(CPU_PROCESSES,
        BLAS_THREADS)
    spat_tile, nTiles = choose_spat_tile_fast(nSpat, best_processes, s_chunk,
        k=2)
    nProcs = builtins.min(best_processes, nTiles, 12) 
    # don’t spawn more processes than tiles

    ok, why = modelcube_status(str(hdf5Path), x_global=x_global, require_float64=True, redraw=redraw)
    logger.log(f"[ModelCube] status: {why}")
    if ok:
        logger.log("[ModelCube] Skipping reconstruction.")
    else:
        logger.log("[ModelCube] Reconstructing…")
        # reconstruct_model_cube_single(  # or your parallel version
        try:
            if best_processes <= 1:
                reconstruct_modelcube_fast(
                    h5_path=str(hdf5Path),
                    x_cp=x_global,
                    s_chunk=spat_tile,
                    out_dtype="float64",
                )
            else:
                reconstruct_modelcube_fast_parallel(
                    h5_path=str(hdf5Path),
                    x_cp=x_global,
                    s_chunk=56,
                    out_dtype="float64",
                    rdcc_slots=1_000_003,
                    rdcc_bytes=8 * 1024**2,
                    rdcc_w0=0.90,
                    n_workers=builtins.min(best_processes, nTiles, 4),
                    blas_threads_per_worker=best_blas // max(1, best_processes)
                )
        except Exception as e:
            logger.log(
                f"[ModelCube] Error: Could not reconstruct ModelCube")
            raise e
        # Stamp digest so future runs can skip confidently
        try:
            with open_h5(str(hdf5Path), role="writer") as f:
                xdig = _x_digest(x_global)
                ds = f["/ModelCube"]
                ds.attrs["x_digest"] = xdig
                ds.attrs["x_shape"]  = np.asarray(x_global).shape
                ds.attrs["dtype_math"] = "float64"
                ds.attrs["generator"] = "reconstruct_modelcube_fast"
        except Exception as e:
            logger.log(f"[ModelCube] Warning: could not stamp digest ({e})")

    with open_h5(hdf5Path, role="reader") as f:
        data_cube  = np.asarray(f["/DataCube"][...], np.float64)
        model_cube = np.asarray(f["/ModelCube"][...], np.float64)

    # chi^2 per spaxel
    if mask_arr is None:
        mask_arr = np.ones(nLSpec, dtype=bool)
    assert mask_arr.shape[0] == data_cube.shape[1]

    with open_h5(hdf5Path, role="reader") as f:
        D = np.asarray(f["/DataCube"][...], np.float64)      # (nSpat, nLam)
        M = np.asarray(f["/ModelCube"][...], np.float64)     # (nSpat, nLam)
        mask = np.asarray(f["/Mask"][...], bool).ravel()
        nSpat, nLam = D.shape
    
    # ------------------------------------------------------------
    # Reconstruct the hard-prior target used by the constrained fit
    # ------------------------------------------------------------
    orbit_shape = np.asarray(
        cWeights,
        dtype=np.float64,
    ).ravel(order="C")

    if orbit_shape.size != nComp:
        raise ValueError(
            "cWeights must contain one value per orbit component."
        )

    orbit_shape = np.maximum(
        orbit_shape,
        0.0,
    )

    orbit_shape_sum = float(
        np.sum(orbit_shape)
    )

    if (
        not np.isfinite(orbit_shape_sum)
        or orbit_shape_sum <= 0.0
    ):
        raise ValueError(
            "cWeights must have positive finite total weight."
        )

    # This is the exact shape passed to the constrained solver.
    orbit_shape /= orbit_shape_sum

    x_global_cp = np.asarray(
        x_global,
        dtype=np.float64,
    )

    if x_global_cp.ndim == 1:
        if x_global_cp.size != nComp * nPop:
            raise ValueError(
                "x_global has the wrong number of coefficients."
            )
        x_global_cp = x_global_cp.reshape(
            nComp,
            nPop,
            order="C",
        )
    elif x_global_cp.shape != (nComp, nPop):
        raise ValueError(
            "x_global must have shape "
            f"({nComp}, {nPop})."
        )

    # Because orbit_shape has unit sum, the fitted global amplitude is the
    # total physical coefficient mass.
    alpha_fit = float(
        np.sum(
            x_global_cp,
            dtype=np.float64,
        )
    )

    if (
        not np.isfinite(alpha_fit)
        or alpha_fit < 0.0
    ):
        raise RuntimeError(
            "The fitted global orbit amplitude is invalid."
        )

    orbit_target_mass = (
        alpha_fit
        * orbit_shape
    )
    
    arSOL = x_global.reshape(nComp, nMetals, nAges, nAlphas, order='C')
    M = np.zeros((NOrbs, nComp), dtype=np.float64)
    for c, comp in enumerate(nzComp):
        w = oDict['weights'][f"{comp:{pred}d}"]
        mask = oDict['wheres'][f"{comp:{pred}d}"]
        if w.shape != mask.shape:
            raise ValueError('weight array and mask must have the same shape')
        w = np.where(mask, w, 0.0)
        s = w.sum()
        if s > 0.0:
            M[:, c] = w / s
    orbSOL = np.tensordot(M, arSOL, axes=(1, 0))
    compDisp = np.ma.ones((nComp, nCRad), dtype=float)*np.nan
    compVel = np.ma.ones((nComp, nSRad), dtype=float)*np.nan
    for nc in range(nComp):
        cnData = np.take(intData, nc, axis=1).reshape(18, -1)
        print(orbSOL.shape)
        print(cnData.shape)
        drad, svR, svRe, dweights, dcirc = Cgh.broadBetaCompsCyl(cnData,
            np.append(cRadius, 1e15), 'z', -np.inf, 0.0)
        _drad, vMean, vErr, _dweights, _dcirc = Cgh.broadVelCompsSph(cnData,
            np.append(sRadius, 1e15), -np.inf, 0.0)
        compDisp[nc, :] = svR
        compVel[nc, :] = vMean[:, 2] # v_phi

    solution_orbit_mass = np.sum(
        x_global_cp,
        axis=1,
    )

    orbit_resid = (
        solution_orbit_mass
        - orbit_target_mass
    )

    print(
        "[Orbit-prior plot] "
        f"alpha_fit={alpha_fit:.6e} "
        f"shape_sum={np.sum(orbit_shape):.6e} "
        f"solution_sum={np.sum(solution_orbit_mass):.6e} "
        f"target_sum={np.sum(orbit_target_mass):.6e} "
        f"L1={np.sum(np.abs(orbit_resid)):.6e} "
        f"Linf={np.max(np.abs(orbit_resid)):.6e}",
        flush=True,
    )

    # ---------------------------------------------
    # FIT RESIDUALS
    # ---------------------------------------------
    data_fit = np.asarray(
        data_cube[:, mask_arr],
        dtype=np.float64,
    )
    model_fit = np.asarray(
        model_cube[:, mask_arr],
        dtype=np.float64,
    )

    # Residual spectrum in the native flux units.
    resid_fit = data_fit - model_fit

    # ---------------------------------------------
    # RAW-FLUX SPACE (solver space)
    # ---------------------------------------------
    data_raw = np.sum(
        data_fit,
        axis=1,
    )
    model_raw = np.sum(
        model_fit,
        axis=1,
    )

    rms_resid_raw = np.sqrt(
        np.mean(
            resid_fit**2,
            axis=1,
        )
    )

    mean_resid_raw = np.mean(
        resid_fit,
        axis=1,
    )

    # ---------------------------------------------
    # SURFACE-BRIGHTNESS SPACE (interpretation)
    # ---------------------------------------------
    data_sb = data_raw / binCounts
    model_sb = model_raw / binCounts

    data_sb = np.ma.masked_invalid(
        np.ma.masked_less_equal(
            data_sb,
            0.0,
        )
    )
    model_sb = np.ma.masked_invalid(
        np.ma.masked_less_equal(
            model_sb,
            0.0,
        )
    )

    rms_resid_sb = (
        rms_resid_raw / binCounts
    )
    mean_resid_sb = (
        mean_resid_raw / binCounts
    )

    rms_resid_sb = np.ma.masked_invalid(
        rms_resid_sb
    )
    mean_resid_sb = np.ma.masked_invalid(
        mean_resid_sb
    )

    # ---------------------------------------------
    # SYMMETRIC FRACTIONAL RESIDUAL
    # ---------------------------------------------
    denom = 0.5 * (data_fit + model_fit)

    positive = denom[np.isfinite(denom) & (denom > 0.0)]

    if positive.size > 0:
        rel_floor = max(float(np.percentile(positive, 1.0) * 1e-6), 1e-30)
    else:
        rel_floor = 1e-30

    frac_resid = resid_fit / np.maximum(denom, rel_floor)

    # ---------------------------------------------
    # ROBUST FRACTIONAL ABSOLUTE RESIDUAL
    # ---------------------------------------------
    # Primary goodness-of-fit metric.
    #
    # Unlike NMAD, this retains sensitivity to a systematic
    # offset between the data and model while remaining robust
    # against a small number of pathological pixels.
    abs_frac_resid = np.abs(frac_resid)
    median_abs_frac_resid = 100.0 * np.nanmedian(abs_frac_resid, axis=1,)
    median_abs_frac_resid = np.ma.masked_invalid(median_abs_frac_resid)
    p999_abs_frac_resid = 100.0 * np.nanpercentile(abs_frac_resid, 99.9,
        axis=1,)

    # ---------------------------------------------
    # FRACTIONAL RMS RESIDUAL
    # ---------------------------------------------
    rms_resid_frac = 100.0 * np.sqrt(
        np.nanmean(
            frac_resid**2,
            axis=1,
        )
    )
    rms_resid_frac = np.ma.masked_invalid(
        rms_resid_frac
    )

    # ---------------------------------------------
    # ROBUST FRACTIONAL RESIDUAL: NMAD
    # ---------------------------------------------
    frac_median = np.nanmedian(
        frac_resid,
        axis=1,
    )
    frac_abs_dev = np.abs(
        frac_resid
        - frac_median[:, None]
    )
    mad_frac = np.nanmedian(
        frac_abs_dev,
        axis=1,
    )
    # 1.4826 converts MAD to the Gaussian-equivalent sigma.
    nmad_resid_frac = (
        100.0
        * 1.4826
        * mad_frac
    )
    nmad_resid_frac = np.ma.masked_invalid(
        nmad_resid_frac
    )
    median_frac_resid = (
        100.0 * frac_median
    )
    median_frac_resid = np.ma.masked_invalid(
        median_frac_resid
    )

    # ---------------------------------------------
    # FIT-QUALITY METRIC
    # ---------------------------------------------
    fit_metric = np.asarray(
        median_abs_frac_resid,
        dtype=np.float64,
    )
    finite_metric = np.isfinite(
        fit_metric
    )
    fitLabel = r'$Q_s\ [\%]$'
    metric_median = np.nan
    metric_std = np.nan
    if np.any(finite_metric):
        worst = int(np.nanargmax(np.where(finite_metric, fit_metric, -np.inf)))
        best = int(np.nanargmin(np.where(finite_metric, fit_metric, np.inf)))
        metric_mean = float(np.nanmean(fit_metric))
        metric_median = float(np.nanmedian(fit_metric))
        metric_std = float(np.nanstd(fit_metric))

        print("Median absolute fractional residual: "
            f"mean={metric_mean:.3f}% "
            f"median={metric_median:.3f}% "
            f"std={metric_std:.3f}%",
            flush=True)

        nmad_finite = np.asarray(
            nmad_resid_frac,
            dtype=np.float64,
        )
        nmad_finite = nmad_finite[
            np.isfinite(nmad_finite)
        ]

        if nmad_finite.size > 0:
            print(
                "Fractional NMAD: "
                f"mean={np.mean(nmad_finite):.3f}% "
                f"median={np.median(nmad_finite):.3f}%",
                flush=True,
            )
        print(
            f"Worst fit: aperture {worst} "
            f"(median |fractional residual|="
            f"{fit_metric[worst]:.3f}%)",
            flush=True,
        )

        print(
            f"Best fit: aperture {best} "
            f"(median |fractional residual|="
            f"{fit_metric[best]:.3f}%)",
            flush=True,
        )

    # ---------------------------------------------
    # PLOTTING LIMITS
    # ---------------------------------------------
    rmax_abs = np.nanpercentile(rms_resid_sb, 99)
    rmax_abs = max(float(rmax_abs), 1.0)
    rmax_frac = round(np.nanpercentile(rms_resid_frac, 99)/10.0)*10.0
    rmax_frac = max(float(rmax_frac), 1.0)

    print('[CubeFit] '+'-'*(80-10))
    print('[CubeFit] PLOTTING')
    print('[CubeFit] '+'-'*(80-10))
    print(f"[CubeFit] All plots and maps saved in {str(figDir)}")

    compare_orbit_vs_solution_absolute(h5_path=str(hdf5Path),
        orbit_target_mass=orbit_target_mass, x_global=x_global_cp,
        save=figDir/f"compare_prior_abs_{nComp:{pred}d}_i{proj}_{lOrder:02d}.png"
    )

    plt.figure(figsize=(6, 4))
    plt.hist(fit_metric[finite_metric], bins=40)
    plt.xlabel(r'Fractional NMAD $[\%]$')
    plt.ylabel(r'$N$')
    plt.savefig(figDir/'fractional_nmad_hist.png')
    plt.close()

    print(f"Mean reduced χ²: {np.mean(rms_resid_raw):.2f} ± {np.std(rms_resid_raw):.2f}")

    divcmap = colormaps.get_cmap('GECKOSdr')
    if isinstance(divcmap, mcolors.ListedColormap):
        divcmap = mcolors.LinearSegmentedColormap.from_list(
            f"{divcmap.name}_fixed", divcmap.colors, N=256)
    heat = colormaps.get_cmap('cet_fire')
    if not hasattr(heat, "n_variates"):
        heat.n_variates = 1
    
    picks = []

    # ---------------------------------------------
    # Plot
    # ---------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(yLen / xLen) * 0.75)
    ax = fig.add_subplot(111)
    # Symmetric colour scale around zero
    vlim = np.percentile(np.abs(mean_resid_sb), 99)
    cnt = dbi(xpix, ypix, mean_resid_sb[binNum], pixelsize=pixs, angle=PA,
        cmap=divcmap, vmin=-vlim, vmax=+vlim)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cax = POT.attachAxis(ax, "top", 0.1, mid=True)
    cb = plt.colorbar(cnt, cax=cax, orientation="horizontal")
    cax.text(0.5, 0.5, r"$\langle D-M\rangle_\lambda / N_{\rm pix}$",
        ha="center", va="center", color=POT.pgreen, transform=cax.transAxes,
        path_effects=[PathEffects.withStroke(linewidth=1.5, foreground='k')])
    cb.set_ticks([])
    ax.set_xlabel(r"$x\ [{\rm arcsec}]$")
    ax.set_ylabel(r"$y\ [{\rm arcsec}]$")
    plt.savefig(figDir/\
        f"signed_residual_SB_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png"
    )
    plt.close(fig)

    if 'mw' in pplots:

        fmin, fmax = np.log10(np.min(data_sb)), np.log10(np.max(data_sb))
        pren = 2
        fLabel = r"$\log_{10}\ L\ [{\rm L_\odot\ pc^{-2}}]$"
        miText = POT.prec(pren, fmin)
        maText = POT.prec(pren, fmax)
        gs = gridspec.GridSpec(3, 1, hspace=0., wspace=0.)
        fig = plt.figure(figsize=plt.figaspect((yLen*3.)/xLen)*0.75)
        ax = fig.add_subplot(gs[0])
        cnt = dbi(xpix, ypix, np.log10(data_sb[binNum]), pixelsize=pixs,
            angle=PA, cmap=heat, vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))
        cax = POT.attachAxis(ax, 'top', 0.1)
        cb = plt.colorbar(cnt, cax=cax, orientation='horizontal')
        lT = cax.text(0.5, 0.5, fr"$L\ [{UTS.lsun}]$", va='center', ha='center',
            color=POT.pgreen, transform=cax.transAxes)
        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
            foreground='k')])
        cax.text(5e-3, 0.5, miText, va='center', ha='left', color='white',
            transform=cax.transAxes)
        cax.text(1.0-5e-3, 0.5, maText, va='center', ha='right', color='black',
            transform=cax.transAxes)
        cb.set_ticks([])
        ax = fig.add_subplot(gs[1])
        dbi(xpix, ypix, np.log10(model_sb[binNum]), pixelsize=pixs, angle=PA,
            cmap=heat, vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))

        maText = POT.prec(0, rmax_frac)
        ax = fig.add_subplot(gs[2])
        cnt = dbi(xpix, ypix, fit_metric[binNum], pixelsize=pixs, angle=PA,
            cmap=moncmap, vmin=0.0, vmax=rmax_frac)
        cax = POT.attachAxis(ax, 'top', 0.1, mid=True)
        cb = plt.colorbar(cnt, cax=cax, orientation='horizontal')
        lT = cax.text(0.5, 0.5, fitLabel, va='center', ha='center',
            color=POT.pgreen, transform=cax.transAxes)
        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
            foreground='k')])
        cax.text(5e-3, 0.5, '0', va='center', ha='left', color='white',
            transform=cax.transAxes)
        cax.text(1.0-5e-3, 0.5, maText, va='center', ha='right', color='k',
            transform=cax.transAxes)
        cb.set_ticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))

        BIG = fig.add_subplot(gs[:])
        BIG.set_frame_on(False)
        BIG.set_xticks([])
        BIG.set_yticks([])
        BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=20)
        BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)

        plt.savefig(figDir/\
            f"modelCube_sb_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")

        # ---------------------------------------------
        # FIGURE: data / model / absolute residual / fractional residual
        # ---------------------------------------------
        fig = plt.figure(figsize=plt.figaspect(yLen / xLen) * 0.75)
        gs = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.0)

        # Panel 1: data
        pren = 2
        miText = POT.prec(pren, fmin)
        maText = POT.prec(pren, fmax)

        ax = fig.add_subplot(gs[0])
        cnt = dbi(xpix, ypix, np.log10(data_sb[binNum]), pixelsize=pixs,
            angle=PA, cmap=heat, vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Panel 2: model
        ax = fig.add_subplot(gs[1])
        dbi(xpix, ypix, np.log10(model_sb[binNum]), pixelsize=pixs, angle=PA,
            cmap=heat, vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cax = POT.attachAxis(ax, 'right', 0.1)
        cb = plt.colorbar(cnt, cax=cax, orientation='vertical')
        lT = cax.text(0.5, 0.5, fr"$L\ [{UTS.lsun}]$", va='center', ha='center',
            rotation=270., color=POT.pgreen, transform=cax.transAxes,
            path_effects=[PathEffects.withStroke(linewidth=1.5, foreground='k')]
            )
        cax.text(0.5, 5e-3, miText, va='bottom', ha='center', color='white',
            transform=cax.transAxes, rotation=270.)
        cax.text(0.5, 1.0 - 5e-3, maText, va='top', ha='center', color='black',
            transform=cax.transAxes, rotation=270.)
        cb.set_ticks([])

        # Panel 3: absolute RMS residual
        ax = fig.add_subplot(gs[2])
        cnt = dbi(xpix, ypix, rms_resid_sb[binNum], pixelsize=pixs, angle=PA,
            cmap=moncmap, vmin=0.0, vmax=rmax_abs)
        cax = POT.attachAxis(ax, 'right', 0.1, mid=True)
        cb = plt.colorbar(cnt, cax=cax, orientation='vertical')
        cax.text(0.5, 0.5, r"${\rm RMS}(D-M)\ /\ N_{\rm pix}$", va='center',
            ha='center', rotation=270., color=POT.pgreen,
            transform=cax.transAxes, path_effects=
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.5, 5e-3, '0.0', va='bottom', ha='center', color='white',
            transform=cax.transAxes, rotation=270.)
        cax.text(0.5, 1.0 - 5e-3, f"{rmax_abs:.1f}", va='top', ha='center',
            color='k', transform=cax.transAxes, rotation=270.)
        cb.set_ticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Panel 4: fractional RMS residual
        ax = fig.add_subplot(gs[3])
        cnt = dbi(xpix, ypix, rms_resid_frac[binNum], pixelsize=pixs, angle=PA,
            cmap=moncmap, vmin=0.0, vmax=rmax_frac)
        ax.set_yticklabels([])
        cax = POT.attachAxis(ax, 'right', 0.1)
        cb = plt.colorbar(cnt, cax=cax, orientation='vertical')
        cax.text(0.5, 0.5,
            r"${\rm RMS}\!\left[\frac{D-M}{(D+M)/2}\right]\ [\%]$",
            va='center', ha='center', rotation=270.,color=POT.pgreen,
            transform=cax.transAxes, path_effects=
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.5, 5e-3, '0.0', va='bottom', ha='center', color='white',
            transform=cax.transAxes, rotation=270.)
        cax.text(0.5, 1.0 - 5e-3, f"{rmax_frac:.1f}",
            va='top', ha='center', color='k',
            transform=cax.transAxes, rotation=270.)
        cb.set_ticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        BIG = fig.add_subplot(gs[:])
        BIG.set_frame_on(False)
        BIG.set_xticks([])
        BIG.set_yticks([])
        BIG.set_xlabel(r"$x\ [{\rm arcsec}]$", labelpad=20)
        BIG.set_ylabel(r"$y\ [{\rm arcsec}]$", labelpad=20)

        plt.savefig(figDir/\
            f"modelCube_sb_grid_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")

        fmin, fmax = np.log10(np.min(data_raw)), np.log10(np.max(data_raw))
        gs = gridspec.GridSpec(3, 1, hspace=0., wspace=0.)
        fig = plt.figure(figsize=plt.figaspect((yLen*3.)/xLen)*0.75)
        ax = fig.add_subplot(gs[0])
        cnt = dbi(xpix, ypix, np.log10(data_raw[binNum]), pixelsize=pixs,
            angle=PA, cmap=heat, vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))
        cax = POT.attachAxis(ax, 'top', 0.1)
        cb = plt.colorbar(cnt, cax=cax, orientation='horizontal')
        lT = cax.text(0.5, 0.5, fr"$L\ [{UTS.lsun}]$", va='center', ha='center',
            color=POT.pgreen, transform=cax.transAxes)
        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
            foreground='k')])
        cax.text(1e-3, 0.5, miText, va='center', ha='left', color='white',
            transform=cax.transAxes)
        cax.text(1.0-1e-3, 0.5, maText, va='center', ha='right', color='black',
            transform=cax.transAxes)
        cb.set_ticks([])
        ax = fig.add_subplot(gs[1])
        dbi(xpix, ypix, np.log10(model_raw[binNum]), pixelsize=pixs, angle=PA,
            cmap=heat, vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))

        rmax = np.percentile(rms_resid_raw, 99)
        rmax = 200
        maText = POT.prec(pren, rmax)
        ax = fig.add_subplot(gs[2])
        cnt = dbi(xpix, ypix, rms_resid_raw[binNum], pixelsize=pixs, angle=PA,
            cmap=divcmap, vmin=0.0, vmax=rmax)
        cax = POT.attachAxis(ax, 'top', 0.1, mid=True)
        cb = plt.colorbar(cnt, cax=cax, orientation='horizontal')
        lT = cax.text(0.5, 0.5,
            r"${\rm RMS}(D-M)\ /\ N_{\rm pix}$", va='center', ha='center',
            color=POT.pgreen, transform=cax.transAxes)
        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
            foreground='k')])
        cax.text(1e-3, 0.5, '0.0', va='center', ha='left', color='white',
            transform=cax.transAxes)
        cax.text(1.0-1e-3, 0.5, maText, va='center', ha='right', color='white',
            transform=cax.transAxes)
        cb.set_ticks([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))

        BIG = fig.add_subplot(gs[:])
        BIG.set_frame_on(False)
        BIG.set_xticks([])
        BIG.set_yticks([])
        BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
        BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)

        plt.savefig(figDir/\
            f"modelCube_flux_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")

    if 'spec' in pplots:
        logger.log("Generating spectrum plots...")
        with logger.capture_all_output():
            if figDir.exists():
                for prefix in ("best", "worst"):
                    for f in figDir.glob(f"{prefix}_{nComp:{pred}d}_spax*.png"):
                        f.unlink()
            parallel_spectrum_plots(h5_or_path=str(hdf5Path),
                fit_metric=fit_metric, n=50, plot_dir=str(figDir),
                n_workers=best_processes, tag=f"{nComp:{pred}d}", mask=mask_arr)
            _ = plot_best_worst_spectrum_fits_stacked(h5_or_path=str(hdf5Path),
                fit_metric=p999_abs_frac_resid, n_each=2,
                plot_path=(figDir/\
                    f"outlier_spectrum_fits_stacked_{nComp:{pred}d}.png"),
                mask=mask_arr)
            picks = plot_best_worst_spectrum_fits_stacked(
                h5_or_path=str(hdf5Path), fit_metric=fit_metric, n_each=2,
                plot_path=(figDir/f"spectrum_fits_stacked_{nComp:{pred}d}.png"),
                mask=mask_arr)

    if 'otype' not in oDict['cutOn']:
        return # only do orbital SFH if orbital decomposition
    if oDict['cuts'] and len(oDict['cuts'])>0:
        # determine which components belong to which orbital categories
        allCuts = np.array([oDict['cuts'][key] for key in oDict['cuts'].keys()])
        uCuts, uCounts = np.unique(allCuts, axis=0, return_counts=True)
        # assert that every elemnt of uCounts is equal
        assert np.unique(uCounts).size == 1
        notypes = np.max(uCounts)
        obins = np.arange(1, notypes+1) * uCuts.shape[0]
        otypes = np.digitize(nzComp, bins=obins, right=True)

        diskIdx = np.where(allCuts[:, 2][nzComp[otypes==0]] > 0.5)[0]
        diskComps = nzComp[diskIdx]
        bulgeComps = np.setdiff1d(np.arange(nComp), diskComps)
    else:
        otypes = copy(nzComp)-1 # zero-indexed
        diskComps = bulgeComps = None

    satube = (otypes == 0) # group short-axis tubes
    latube = (otypes == 1)
    boxess = (otypes == 2)
    if 'sfh' in pplots:
        try:
            coSFH = np.zeros((nMetals, nAges, nAlphas), dtype=float)
            laSFH = np.zeros((nMetals, nAges, nAlphas), dtype=float)
            boSFH = np.zeros((nMetals, nAges, nAlphas), dtype=float)
            if satube.sum() > 0:
                coSFH = arSOL[satube, :, :, :].sum(axis=0)/np.sum(arSOL)
            if latube.sum() > 0:
                laSFH = arSOL[latube, :, :, :].sum(axis=0)/np.sum(arSOL)
            if boxess.sum() > 0:
                boSFH = arSOL[boxess, :, :, :].sum(axis=0)/np.sum(arSOL)
            diskSFH = np.full_like(coSFH, 0.0)
            bulgeSFH = np.full_like(coSFH, 0.0)
            if not isinstance(diskComps, type(None)) and diskComps.size > 0:
                diskSFH = arSOL[diskComps, :, :, :].sum(axis=0)/np.sum(arSOL)
            if not isinstance(bulgeComps, type(None)) and bulgeComps.size > 0:
                bulgeSFH = arSOL[bulgeComps, :, :, :].sum(axis=0)/np.sum(arSOL)

            coSFH = np.ma.masked_less_equal(coSFH, 0.0)
            laSFH = np.ma.masked_less_equal(laSFH, 0.0)
            boSFH = np.ma.masked_less_equal(boSFH, 0.0)
            diskSFH = np.ma.masked_less_equal(diskSFH, 0.0)
            bulgeSFH = np.ma.masked_less_equal(bulgeSFH, 0.0)

            minZ, maxZ = np.min(umetals), np.max(umetals)
            minT, maxT = np.min(uages), np.max(uages)
            minA, maxA = np.min(ualphas), np.max(ualphas)

            wmax = np.log10(np.max((
                np.ma.max(coSFH[coSFH>0]) if np.ma.any(coSFH>0) else 1e-5,
                np.ma.max(laSFH[laSFH>0]) if np.ma.any(laSFH>0) else 1e-5,
                np.ma.max(boSFH[boSFH>0]) if np.ma.any(boSFH>0) else 1e-5)))
            sfhMin = np.log10(np.min((
                np.ma.min(coSFH[coSFH>0]) if np.ma.any(coSFH>0) else 1e10,
                np.ma.min(laSFH[laSFH>0]) if np.ma.any(laSFH>0) else 1e10,
                np.ma.min(boSFH[boSFH>0]) if np.ma.any(boSFH>0) else 1e10)))
            wmin = np.max((sfhMin, -12))
            print(f"SFH plot limits: {wmin:.2f} ({sfhMin:.2f}) to {wmax:.2f}")

            fig = plt.figure(figsize=plt.figaspect(3./4.))
            gs = gridspec.GridSpec(3, nAlphas, hspace=0., wspace=0.)
            # one column per alpha, 3 orbit types
            for ali in range(nAlphas):
                ax = fig.add_subplot(gs[0, ali])
                if nAlphas > 1 and ax.get_subplotspec().is_first_col() and \
                    ax.get_subplotspec().is_first_row():
                    ax.text(1e-2, 1.05, r'$[\alpha/Fe]=$', va='bottom',
                        ha='right', color=POT.pgreen, transform=ax.transAxes,
                        rotation=0, path_effects=[
                        PathEffects.withStroke(linewidth=1.5, foreground='k')])
                cnt = ax.imshow(np.ma.log10(coSFH[:, :, ali]),
                    extent=[minT, maxT, minZ, maxZ],
                    aspect='auto', interpolation='none', origin='lower',
                    cmap=moncmapr, norm=Normalize(vmin=wmin, vmax=wmax))
                if not ax.get_subplotspec().is_last_row():
                    ax.set_xticklabels([])
                if not ax.get_subplotspec().is_first_col():
                    ax.set_yticklabels([])
                if ax.get_subplotspec().is_first_col():
                    lT = ax.text(1e-2, 1e-2, r'$z$ Tubes', va='bottom',
                        ha='left', color=POT.pgreen, transform=ax.transAxes)
                    lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                        foreground='k')])
                if nAlphas > 1:
                    lT = ax.text(0.5, 1.05, rf"${ualphas[ali]:.2f}$",
                        va='bottom', ha='center', color=POT.pgreen,
                        transform=ax.transAxes, path_effects=[
                        PathEffects.withStroke(linewidth=1.5, foreground='k')])
                ax = fig.add_subplot(gs[1, ali])
                ax.imshow(np.ma.log10(laSFH[:, :, ali]),
                    extent=[minT, maxT, minZ, maxZ],
                    aspect='auto', interpolation='none', origin='lower',
                    cmap=moncmapr, norm=Normalize(vmin=wmin, vmax=wmax))
                if not ax.get_subplotspec().is_last_row():
                    ax.set_xticklabels([])
                if not ax.get_subplotspec().is_first_col():
                    ax.set_yticklabels([])
                if ax.get_subplotspec().is_first_col():
                    lT = ax.text(1e-2, 1e-2, r'$x$ Tubes', va='bottom',
                        ha='left', color=POT.pgreen, transform=ax.transAxes)
                    lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                        foreground='k')])
                ax = fig.add_subplot(gs[2, ali])
                ax.imshow(np.ma.log10(boSFH[:, :, ali]),
                    extent=[minT, maxT, minZ, maxZ],
                    aspect='auto', interpolation='none', origin='lower',
                    cmap=moncmapr, norm=Normalize(vmin=wmin, vmax=wmax))
                if not ax.get_subplotspec().is_last_row():
                    ax.set_xticklabels([])
                if not ax.get_subplotspec().is_first_col():
                    ax.set_yticklabels([])
                if ax.get_subplotspec().is_first_col():
                    lT = ax.text(1e-2, 1e-2, r'Box', va='bottom', ha='left',
                        color=POT.pgreen, transform=ax.transAxes)
                    lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                        foreground='k')])

            BIG = fig.add_subplot(gs[:])
            BIG.set_frame_on(False)
            BIG.set_xticks([])
            BIG.set_yticks([])
            BIG.set_xlabel(r'$t\ [{\rm Gyr}]$', labelpad=20)
            BIG.set_ylabel(r'$[Z/H]$', labelpad=35)
            cax = POT.attachAxis(BIG, 'right', 0.05)
            cb = plt.colorbar(cnt, cax=cax, orientation='vertical')
            lT = cax.text(0.5, 0.5, r'$\log_{10}{\text{Mass Fraction}}$',
                va='center', ha='center', color=POT.pgreen,
                transform=cax.transAxes, rotation=270)
            lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                foreground='k')])
            pren = 1
            miText = POT.prec(pren, wmin)
            maText = POT.prec(pren, wmax)
            cax.text(0.45, 5e-3, miText, va='bottom', ha='center',
                color='k', transform=cax.transAxes, rotation=270)
            cax.text(0.45, 1.0-5e-3, maText, va='top', ha='center',
                color='w', transform=cax.transAxes, rotation=270)
            cb.set_ticks([])

            plt.savefig(figDir/\
                f"orbitSFH_full_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")


            if (np.ma.any(diskSFH>0) or np.ma.any(bulgeSFH>0)):
                dbmax = np.log10(np.max((
                    np.ma.max(diskSFH[diskSFH>0]) if np.ma.any(diskSFH>0) else 1e-5,
                    np.ma.max(bulgeSFH[bulgeSFH>0]) if np.ma.any(bulgeSFH>0) else 1e-5)))
                dbsMin = np.log10(np.min((
                    np.ma.min(diskSFH[diskSFH>0]) if np.ma.any(diskSFH>0) else 1e10,
                    np.ma.min(bulgeSFH[bulgeSFH>0]) if np.ma.any(bulgeSFH>0) else 1e10)))
                dbmin = np.max((dbsMin, -12))
                print(f"DB plot limits: {dbmin:.2f} ({dbsMin:.2f}) to "
                    f"{dbmax:.2f}")
                fig = plt.figure(figsize=plt.figaspect(3./4.))
                gs = gridspec.GridSpec(2, nAlphas, hspace=0., wspace=0.)
                # one column per alpha, 3 orbit types
                print(nAlphas, ualphas)
                for ali in range(nAlphas):
                    ax = fig.add_subplot(gs[0, ali])
                    if nAlphas > 1 and ax.get_subplotspec().is_first_col() and \
                        ax.get_subplotspec().is_first_row():
                        ax.text(1e-2, 1.05, r'$[\alpha/Fe]=$', va='bottom', ha='right', color=POT.pgreen,
                            transform=ax.transAxes, rotation=0,
                            path_effects=[PathEffects.withStroke(linewidth=1.5,
                                foreground='k')])
                    cnt = ax.imshow(np.log10(diskSFH[:, :, ali]),
                        extent=[minT, maxT, minZ, maxZ],
                        aspect='auto', interpolation='none', origin='lower',
                        cmap=moncmapr, norm=Normalize(vmin=dbmin, vmax=dbmax))
                    if not ax.get_subplotspec().is_last_row():
                        ax.set_xticklabels([])
                    if not ax.get_subplotspec().is_first_col():
                        ax.set_yticklabels([])
                    if ax.get_subplotspec().is_first_col():
                        lT = ax.text(1e-2, 1.0-1e-2, r'Disk', va='top',
                            ha='left', color=POT.pgreen, transform=ax.transAxes)
                        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                            foreground='k')])
                    if nAlphas > 1:
                        lT = ax.text(0.5, 1.05,
                            rf"${ualphas[ali]:.2f}$",
                            va='bottom', ha='center', color=POT.pgreen,
                            transform=ax.transAxes,
                            path_effects=[PathEffects.withStroke(linewidth=1.5, foreground='k')])
                    ax = fig.add_subplot(gs[1, ali])
                    ax.imshow(np.log10(bulgeSFH[:, :, ali]),
                        extent=[minT, maxT, minZ, maxZ],
                        aspect='auto', interpolation='none', origin='lower',
                        cmap=moncmapr, norm=Normalize(vmin=dbmin, vmax=dbmax))
                    if not ax.get_subplotspec().is_last_row():
                        ax.set_xticklabels([])
                    if not ax.get_subplotspec().is_first_col():
                        ax.set_yticklabels([])
                    if ax.get_subplotspec().is_first_col():
                        lT = ax.text(1e-2, 1.0-1e-2, r'Bulge', va='top', ha='left',
                            color=POT.pgreen, transform=ax.transAxes)
                        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                            foreground='k')])

                BIG = fig.add_subplot(gs[:])
                BIG.set_frame_on(False)
                BIG.set_xticks([])
                BIG.set_yticks([])
                BIG.set_xlabel(r'$t\ [{\rm Gyr}]$', labelpad=20)
                BIG.set_ylabel(r'$[Z/H]$', labelpad=35)
                cax = POT.attachAxis(BIG, 'right', 0.05)
                cb = plt.colorbar(cnt, cax=cax, orientation='vertical')
                lT = cax.text(0.5, 0.5, r'$\log_{10}{\text{Mass Fraction}}$',
                    va='center', ha='center', color=POT.pgreen,
                    transform=cax.transAxes, rotation=270)
                lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                    foreground='k')])
                pren = 1
                miText = POT.prec(pren, dbmin)
                maText = POT.prec(pren, dbmax)
                cax.text(0.45, 5e-3, miText, va='bottom', ha='center',
                    color='k', transform=cax.transAxes, rotation=270)
                cax.text(0.45, 1.0-5e-3, maText, va='top', ha='center',
                    color='w', transform=cax.transAxes, rotation=270)
                cb.set_ticks([])

                plt.savefig(figDir/\
                    f"orbitSFH_diskbulge_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")
                
        except AssertionError as e:
            print(f"Could not make orbital SFH plot: {e}")
        
        # also make a 3-panel plot of metallicity (x) vs alpha (y)
        try:
            # collapse SFH over ages -> shape (nMetals, nAlphas)
            coZalpha = coSFH.sum(axis=1)
            laZalpha = laSFH.sum(axis=1)
            boZalpha = boSFH.sum(axis=1)
            # **SFH are already normalised

            # compute log limits across panels, ignore zeros
            vals = np.hstack([
                coZalpha[coZalpha>0].ravel() if np.ma.any(coZalpha>0) else np.array([]),
                laZalpha[laZalpha>0].ravel() if np.ma.any(laZalpha>0) else np.array([]),
                boZalpha[boZalpha>0].ravel() if np.ma.any(boZalpha>0) else np.array([]),
            ])
            if vals.size > 0:
                vmin2 = float(np.log10(np.max((np.min(vals), -12.0))))
                vmax2 = float(np.log10(np.max(vals)))
            else:
                vmin2, vmax2 = -12.0, -8.0

            print(f"Zα plot limits: {vmin2:.2f} to {vmax2:.2f}")

            fig2 = plt.figure(figsize=plt.figaspect(1./3.)*0.75)
            gs2 = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0)
            panels = [(coZalpha, 'Short-axis Tubes'), (laZalpha, 'Long-axis Tubes'), (boZalpha, 'Boxes')]
            for pi, (arr, title) in enumerate(panels):
                ax = fig2.add_subplot(gs2[0, pi])
                # arr shape (nMetals, nAlphas) -> transpose for imshow so
                # y=alpha
                im = ax.imshow(np.log10(np.ma.masked_invalid(arr.T)),
                    extent=[minZ, maxZ, minA, maxA],
                    aspect='auto', origin='lower', cmap=moncmapr,
                    norm=Normalize(vmin=vmin2, vmax=vmax2))
                lT = ax.text(1e-2, 1e-2, title, va='bottom', ha='left',
                    color=POT.pgreen, transform=ax.transAxes)
                lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                    foreground='k')])
                if pi > 0:
                    ax.set_yticklabels([])

            BIG2 = fig2.add_subplot(gs2[:])
            BIG2.set_frame_on(False)
            BIG2.set_xticks([])
            BIG2.set_yticks([])
            BIG2.set_xlabel(r'$[Z/H]$', labelpad=20)
            BIG2.set_ylabel(r'$[\alpha/Fe]$', labelpad=35)
            cax2 = POT.attachAxis(BIG2, 'right', 0.03)
            cb2 = plt.colorbar(im, cax=cax2, orientation='vertical')
            lT2 = cax2.text(0.5, 0.5, r'$\log_{10}{\text{Mass Fraction}}$',
                va='center', ha='center', color=POT.pgreen,
                transform=cax2.transAxes, rotation=270)
            lT2.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                foreground='k')])
            pren = 1
            miText = POT.prec(pren, vmin2)
            maText = POT.prec(pren, vmax2)
            cax2.text(0.45, 5e-3, miText, va='bottom', ha='center',
                color='k', transform=cax2.transAxes, rotation=270)
            cax2.text(0.45, 1.0-5e-3, maText, va='top', ha='center',
                color='w', transform=cax2.transAxes, rotation=270)
            cb2.set_ticks([])

            fig2.savefig(figDir/\
                f"orbitSFH_alphaMetal_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")
        
        except Exception as e:
            print(f"Could not make Z-α plot: {e}")
            pass

        # metallicity vs alpha, per age
        try:
            chemSFH = arSOL.sum(axis=0)/np.sum(arSOL)
            # shape (nMetals, nAges, nAlphas)

            vmin3 = float(np.log10(np.max((np.min(chemSFH[chemSFH>0]), -12.0))))
            vmax3 = float(np.log10(np.max(chemSFH[chemSFH>0])))
            print(f"Zαt plot limits: {vmin3:.2f} to {vmax3:.2f}")

            fig3 = plt.figure(figsize=plt.figaspect(3.)*0.75)
            gs3 = gridspec.GridSpec(3, 1, wspace=0.0, hspace=0.0)
            cuts = [(uages < 6.0), (uages >= 10.0), (uages <= 14.0)]
            labels = ['Age < 6 Gyr', 'Age ≥ 10 Gyr', 'Age < 14 Gyr']
            for pi, mask in enumerate(cuts):
                ax = fig3.add_subplot(gs3[pi, 0])
                # arr shape (nMetals, nAlphas) -> transpose for imshow so
                # y=alpha
                im = ax.imshow(np.log10(np.compress(mask, chemSFH,
                    axis=1).sum(axis=1).T),
                    extent=[minZ, maxZ, minA, maxA],
                    aspect='auto', origin='lower',
                    cmap=moncmapr, norm=Normalize(vmin=vmin3, vmax=vmax3))
                lT = ax.text(1e-2, 1e-2, labels[pi], va='bottom', ha='left',
                    color=POT.pgreen, transform=ax.transAxes)
                lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                    foreground='k')])
                if not ax.get_subplotspec().is_last_row():
                    ax.set_xticklabels([])

            BIG3 = fig3.add_subplot(gs3[:])
            BIG3.set_frame_on(False)
            BIG3.set_xticks([])
            BIG3.set_yticks([])
            BIG3.set_xlabel(r'$[Z/H]$', labelpad=20)
            BIG3.set_ylabel(r'$[\alpha/Fe]$', labelpad=35)
            cax3 = POT.attachAxis(BIG3, 'right', 0.1)
            cb3 = plt.colorbar(im, cax=cax3, orientation='vertical')
            lT3 = cax3.text(0.5, 0.5, r'$\log_{10}{\text{Mass Fraction}}$',
                va='center', ha='center', color=POT.pgreen,
                transform=cax3.transAxes, rotation=270)
            lT3.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                foreground='k')])
            pren = 1
            miText = POT.prec(pren, vmin3)
            maText = POT.prec(pren, vmax3)
            cax3.text(0.45, 5e-3, miText, va='bottom', ha='center',
                color='k', transform=cax3.transAxes, rotation=270)
            cax3.text(0.45, 1.0-5e-3, maText, va='top', ha='center',
                color='w', transform=cax3.transAxes, rotation=270)
            cb3.set_ticks([])

            fig3.savefig(figDir/\
                f"orbitSFH_alphaMetalAge_{nComp:{pred}d}_i{proj}"
                f"{tag}_{lOrder:02d}.png")
        except Exception as e:
            print(f"Could not make Z-t-α plot: {e}")
            pass
        
        # 3D corner
        try:
            ageAlpha, metalAlpha, metalAge = [chemSFH.sum(axis=i).T for i in
                range(3)]
            vmin4 = np.log10(np.min([np.min(x[x>0]) for x in
                [ageAlpha, metalAlpha, metalAge]]))
            vmax4 = np.log10(np.max([np.max(x[x>0]) for x in
                [ageAlpha, metalAlpha, metalAge]]))
            print(f"corner plot limits: {vmin4:.2f} to {vmax4:.2f}")
            fig4 = plt.figure(figsize=plt.figaspect(1.0)*0.75)
            gs4 = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0)
            ax = fig4.add_subplot(gs4[2])
            # arr shape (nMetals, nAlphas) -> transpose for imshow so y=alpha
            im = ax.imshow(np.log10(metalAlpha),
                extent=[minZ, maxZ, minA, maxA],
                aspect='auto', origin='lower',
                cmap=moncmapr, norm=Normalize(vmin=vmin4, vmax=vmax4))
            ax.set_xlabel(r'$[Z/H]$')
            ax.set_ylabel(r'$[\alpha/Fe]$')
            ax = fig4.add_subplot(gs4[0])
            # (nMetal, nAge) -> (nAge, nMetal)
            im = ax.imshow(np.log10(metalAge),
                extent=[minZ, maxZ, minT, maxT],
                aspect='auto', origin='lower',
                cmap=moncmapr, norm=Normalize(vmin=vmin4, vmax=vmax4))
            ax.set_xticklabels([])
            ax.set_ylabel(rf"$t\ [{UTS.gyr}]$")
            ax = fig4.add_subplot(gs4[3])
            # (nAge, nAlpha) -> (nAlpha, nAge)
            im = ax.imshow(np.log10(ageAlpha),
                extent=[minT, maxT, minA, maxA],
                aspect='auto', origin='lower',
                cmap=moncmapr, norm=Normalize(vmin=vmin4, vmax=vmax4))
            ax.set_yticklabels([])
            ax.set_xlabel(rf"$t\ [{UTS.gyr}]$")

            ax = fig4.add_subplot(gs4[1])
            ax.axis('off')
            cbWidth = 0.9
            cbHeight = 0.15
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax4 = inset_axes(ax, width='100%', height='100%', loc=10,
                bbox_to_anchor=((1.0-cbWidth)/2.0, (1.0-cbHeight)/2.0,
                cbWidth, cbHeight), bbox_transform=ax.transAxes, borderpad=0.)
            cb4 = plt.colorbar(im, cax=cax4, orientation='horizontal')
            lT4 = ax.text(0.5, (1.0-cbHeight)/2.0+(cbHeight)+1e-3,
                r'$\log_{10}{\text{Mass Fraction}}$',
                va='bottom', ha='center', color=POT.pgreen,
                transform=ax.transAxes)
            lT4.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                foreground='k')])
            pren = 1
            miText = POT.prec(pren, vmin4)
            maText = POT.prec(pren, vmax4)
            cax4.text(5e-3, 0.5, miText, va='center', ha='left',
                color='k', transform=cax4.transAxes)
            cax4.text(1.0-5e-3, 0.5, maText, va='center', ha='right',
                color='w', transform=cax4.transAxes)
            cb4.set_ticks([])

            fig4.savefig(figDir/\
                f"orbitSFH_corner_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")
        except Exception as e:
            print(f"Could not make corner plot: {e}")
            pass
    
    if 'proj' in pplots and nComp > 3:
        logger.log("Generating projected maps...")
        with logger.capture_all_output():
            try:
                saSOL = arSOL[satube, :, :, :]
                laSOL = arSOL[latube, :, :, :]
                boSOL = arSOL[boxess, :, :, :]

                saAge = np.nansum(np.nansum(saSOL, axis=(1,3))*\
                    uages[np.newaxis, :] / np.nansum(saSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)
                laAge = np.nansum(np.nansum(laSOL, axis=(1,3))*\
                    uages[np.newaxis, :] / np.nansum(laSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)
                boAge = np.nansum(np.nansum(boSOL, axis=(1,3))*\
                    uages[np.newaxis, :] / np.nansum(boSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)

                saMetal = np.nansum(np.nansum(saSOL, axis=(2,3))*\
                    umetals[np.newaxis, :] / np.nansum(saSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)
                laMetal = np.nansum(np.nansum(laSOL, axis=(2,3))*\
                    umetals[np.newaxis, :] / np.nansum(laSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)
                boMetal = np.nansum(np.nansum(boSOL, axis=(2,3))*\
                    umetals[np.newaxis, :] / np.nansum(boSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)

                saAlpha = np.nansum(np.nansum(saSOL, axis=(1,2))*\
                    ualphas[np.newaxis, :] / np.nansum(saSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)
                laAlpha = np.nansum(np.nansum(laSOL, axis=(1,2))*\
                    ualphas[np.newaxis, :] / np.nansum(laSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)
                boAlpha = np.nansum(np.nansum(boSOL, axis=(1,2))*\
                    ualphas[np.newaxis, :] / np.nansum(boSOL, axis=(1,2,3)
                    )[:, np.newaxis], axis=1)

                maps = dict(
                    age=dict(
                        sa=_MWProp(saAge, np.compress(satube, aperMass, axis=1)),
                        la=_MWProp(laAge, np.compress(latube, aperMass, axis=1)),
                        bo=_MWProp(boAge, np.compress(boxess, aperMass, axis=1))
                    ),
                    metal=dict(
                        sa=_MWProp(saMetal, np.compress(satube, aperMass,
                            axis=1)),
                        la=_MWProp(laMetal, np.compress(latube, aperMass,
                            axis=1)),
                        bo=_MWProp(boMetal, np.compress(boxess, aperMass,
                            axis=1))
                    ),
                    alpha=dict(
                        sa=_MWProp(saAlpha, np.compress(satube, aperMass,
                            axis=1)),
                        la=_MWProp(laAlpha, np.compress(latube, aperMass,
                            axis=1)),
                        bo=_MWProp(boAlpha, np.compress(boxess, aperMass,
                        axis=1))
                    )
                )
                
                orbKeys = ['sa', 'la']
                orbSpecs = [r'$z$ Tubes', r'$x$ Tubes']
                amin = np.min([np.nanmin(maps['age'][otype]) for otype in
                    orbKeys])
                amax = np.max([np.nanmax(maps['age'][otype]) for otype in
                    orbKeys])
                mmin = np.min([np.nanmin(maps['metal'][otype]) for otype in
                    orbKeys])
                mmax = np.max([np.nanmax(maps['metal'][otype]) for otype in
                    orbKeys])
                lmin = np.min([np.nanmin(maps['alpha'][otype]) for otype in
                    orbKeys])
                lmax = np.max([np.nanmax(maps['alpha'][otype]) for otype in
                    orbKeys])
                print(f"Age map limits: {amin:.2f} to {amax:.2f}")
                print(f"Metal map limits: {mmin:.2f} to {mmax:.2f}")
                print(f"Alpha map limits: {lmin:.2f} to {lmax:.2f}")
                propSpecs = [
                    ('metal', r"$[Fe/H]$", mmin, mmax),
                    ('age', rf"$t\ [{UTS.gyr}]$", amin, amax),
                    ('alpha', r"$[\alpha/Fe]$", lmin, lmax),
                ]


                fig = plt.figure(figsize=plt.figaspect((yLen/1.1/xLen)*\
                    (len(propSpecs)/len(orbKeys))*1.1))
                gs = gridspec.GridSpec(len(propSpecs), len(orbKeys), hspace=0.0,
                    wspace=0.0)

                for ri, (prop, label, vmin, vmax) in enumerate(propSpecs):
                    mappable = None
                    for oi, (orb_key, otype) in enumerate(zip(orbKeys,
                        orbSpecs)):
                        ax = fig.add_subplot(gs[ri, oi])
                        arr = maps[prop][orb_key]
                        arr = np.ma.masked_invalid(arr)[binNum]
                        # vmin = np.ma.min(arr) if np.ma.any(arr) else vmin
                        # vmax = np.ma.max(arr) if np.ma.any(arr) else vmax
                        mappable = dbi(xpix, ypix, arr, pixelsize=pixs, angle=PA,
                            cmap=moncmap, vmin=vmin, vmax=vmax)
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)
                        pren = 1
                        miText = POT.prec(pren, vmin)
                        maText = POT.prec(pren, vmax)
                        # ax.text(0.99, 0.99, f"{miText}/{maText}", va="top",
                        #     ha="right", color=POT.pgreen, transform=ax.transAxes,
                        #     rotation=0, path_effects=[
                        #         PathEffects.withStroke(linewidth=1.5,
                        #         foreground="k")])

                        if not ax.get_subplotspec().is_last_row():
                            ax.set_xticklabels([])
                        if not ax.get_subplotspec().is_first_col():
                            ax.set_yticklabels([])
                        if ax.get_subplotspec().is_first_row():
                            ax.text(1e-2, 1e-2, otype,
                                va="bottom", ha="left", color=POT.pgreen,
                                transform=ax.transAxes,
                                path_effects=[PathEffects.withStroke(
                                linewidth=1.5, foreground="k")],)
                        if ax.get_subplotspec().is_last_col():
                            cax = POT.attachAxis(ax, "right", 0.1)
                            cb = plt.colorbar(mappable, cax=cax,
                                orientation="vertical")
                            lT = cax.text(0.5, 0.5, label, va="center",
                                ha="center", color=POT.pgreen,
                                transform=cax.transAxes, rotation=270,
                                path_effects=[PathEffects.withStroke(
                                    linewidth=1.5, foreground="k")])
                            cax.text(0.45, 5e-3, miText, va="bottom",
                                ha="center", color="w", transform=cax.transAxes,
                                rotation=270,)
                            cax.text(0.45, 1.0 - 5e-3, maText, va="top",
                                ha="center", color="k", transform=cax.transAxes,
                                rotation=270,)
                            cb.set_ticks([])

                BIG = fig.add_subplot(gs[:])
                BIG.set_frame_on(False)
                BIG.set_xticks([])
                BIG.set_yticks([])
                BIG.set_xlabel(r"$x\ [{\rm arcsec}]$", labelpad=20)
                BIG.set_ylabel(r"$y\ [{\rm arcsec}]$", labelpad=30)

                fig.savefig(figDir/
                    f"orbitMaps_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")
            except Exception as e:
                print(f"Could not make projection plots: {e}")
                traceback.print_exc()
                pass
    
    return picks

# ------------------------------------------------------------------------------

def plot_sparse_spectra_from_x(
    h5_or_path: str,
    x_global: np.ndarray | None = None,
    *,
    picks: np.ndarray | list[int] | None = None,
    chi2: np.ndarray | None = None,
    n: int = 6,
    plot_dir: str = ".",
    tag: str = "",
    mask: np.ndarray | None = None,
):
    """
    Plot a few diagnostic spectra without building /ModelCube.
    Computes y_hat for selected spaxels directly from /HyperCube/models and x_global.

    Args
    ----
    h5_or_path : str
        Path to HDF5 with /HyperCube/models and /DataCube.
    x_global : array-like
        Global weights (C*P,) or (C,P). Internally cast to float32.
    picks : array-like of int, optional
        Explicit spaxel indices to plot. If None, use `chi2` & `n`.
    chi2 : array-like, optional
        Per-spaxel RMSE/chi2 to pick best/worst examples from.
    n : int
        If using `chi2`, number of best and worst to show (unique combined).
    plot_dir : str
        Where to save PNGs.
    tag : str
        Small tag to include in filenames.
    mask : 1-D bool array, optional
        Wavelength mask to apply to both data & model for plotting.
    """
    os.makedirs(plot_dir, exist_ok=True)

    with open_h5(h5_or_path, role="reader") as f:
        M = f["/HyperCube/models"]      # (S, C, P, L) float32
        DC = f["/DataCube"]             # (S, L)
        S, C, P, L = map(int, M.shape)
        obs = f["/ObsPix"][...] if "/ObsPix" in f else np.arange(L, dtype=int)

        # Load x_global if not provided
        if x_global is None:
            if "/X_global" not in f:
                raise RuntimeError("x_global not provided and /X_global not found in file.")
            x_global = np.asarray(f["/X_global"][...], dtype=np.float64)

        # Choose picks if not explicitly provided
        if picks is None:
            if chi2 is None:
                raise ValueError("Provide `picks` or (`chi2` and `n`).")
            chi2 = np.asarray(chi2, dtype=np.float64)
            if chi2.shape[0] != S:
                raise ValueError(f"chi2 length {chi2.shape[0]} != S={S}.")
            worst = np.argsort(-chi2)[:int(max(1, n))]
            best  = np.argsort( chi2)[:int(max(1, n))]
            picks = np.unique(np.concatenate([worst, best])).astype(int)
        else:
            picks = np.asarray(picks, dtype=int)

        # Mask sanity
        if mask is None:
            mask = np.ones(L, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape[0] != L:
                raise ValueError(f"Mask length {mask.shape[0]} != L={L}.")

        # Weights as (C,P) float32 for speed; accumulation stays float64
        x_cp = np.asarray(x_global)
        if x_cp.ndim == 1:
            if x_cp.size != C * P:
                raise ValueError(f"x_global length {x_cp.size} != C*P={C*P}.")
            x_cp = x_cp.reshape(C, P)
        elif x_cp.shape != (C, P):
            raise ValueError(f"x_global shape {x_cp.shape} != (C,P)=({C},{P}).")
        x32 = np.asarray(x_cp, dtype=np.float32, order="C")

        # Respect storage layout to keep I/O small
        chunks = M.chunks or (min(S, 32), 1, min(P, 256), L)
        S_chunk, C_chunk, P_chunk, L_chunk = map(int, chunks)

        print(f"[DiagSparse] S={S} C={C} P={P} L={L} | chunks={chunks}")
        print(f"[DiagSparse] picks={picks.size} → reads per pick ≈ C·ceil(P/P_chunk)={C*math.ceil(P/max(1,P_chunk))}")

        def _predict_row(s_idx: int) -> np.ndarray:
            # produce y = sum_{c,p} x[c,p] * A[s_idx, c, p, :]
            y = np.zeros(L, dtype=np.float64, order="C")
            # iterate over c blocks
            for c0 in range(0, C, max(1, C_chunk)):
                c1 = min(C, c0 + max(1, C_chunk))
                # read the slab for all p-blocks in one go if memory permits
                # shape => (1, nC_block, P, L)
                slab = np.asarray(M[s_idx:s_idx+1, c0:c1, :, :], dtype=np.float32, order="C")
                # slab[0] shape (nC_block, P, L)
                # multiply each component block by its x weights
                for ci in range(c1 - c0):
                    c_index = c0 + ci
                    # choose p blocks to respect P_chunk if necessary
                    # but simple dot over full P is easiest and often fastest
                    A_cp = slab[0, ci, :, :]    # (P, L) float32
                    w = x32[c_index, :]         # (P,)
                    # accumulate: (A_cp.T @ w) => (L,)
                    y += (A_cp.T @ w).astype(np.float64, copy=False)
            return y

        for s in tqdm(picks, desc="[DiagSparse] spaxels", dynamic_ncols=True, mininterval=1.5):
            s = int(s)
            data  = np.asarray(DC[s, :], dtype=np.float64, order="C")
            model = _predict_row(s)

            fig = plt.figure(figsize=(8, 3.5))
            ax  = fig.add_subplot(111)
            ax.plot(obs[mask], data[mask],  lw=1.0, label="data")
            ax.plot(obs[mask], model[mask], lw=1.0, alpha=0.9, label="model (sparse)")
            ax.set_title(f"spaxel {s}")
            ax.set_xlabel("λ (log space)")
            ax.set_ylabel("flux")
            ax.legend(loc="best", fontsize=8)
            fn = os.path.join(plot_dir, f"diag_sparse_{tag}_spax{int(s):05d}.png")
            fig.savefig(fn, dpi=120)
            plt.close(fig)

        print(f"[DiagSparse] wrote {picks.size} plots to {plot_dir}")

# ------------------------------------------------------------------------------

def compare_orbit_vs_solution_absolute(
    h5_path: str,
    *,
    orbit_target_mass: np.ndarray | None = None,
    orbit_weights: np.ndarray | None = None,
    x_global: np.ndarray | None = None,
    save: str | None = None,
):
    """
    Compare the fitted per-orbit masses against the exact hard-prior target.

    For the flexible-amplitude constrained formulation, the target is

        target[c] = alpha_fit * orbit_shape[c],

    where orbit_shape is fixed a priori and normalized to unit sum, while
    alpha_fit is the fitted global coefficient mass.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    orbit_target_mass : ndarray, optional
        Absolute orbit-mass target vector of shape (C,).
        This is the preferred input for the augmented-row formulation.
    orbit_weights : ndarray, optional
        Backward-compatible alias. If provided and `orbit_target_mass` is None,
        it is used directly as the absolute target vector.
    x_global : ndarray, optional
        Solution vector of shape (C*P,) or (C, P). If omitted, read /X_global.
    save : str, optional
        If provided, save the plot to this path.

    Returns
    -------
    dict
        Diagnostic summary with absolute and relative residuals.

    Raises
    ------
    RuntimeError
        If the target vector or solution is unavailable.
    ValueError
        If array shapes are incompatible.
    """
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        _, C, P, _ = map(int, M.shape)

        if x_global is None:
            if "/X_global" not in f:
                raise RuntimeError(
                    "No /X_global in HDF5 and x_global not provided."
                )
            x_global = np.asarray(f["/X_global"][...], dtype=np.float64)

    x = np.asarray(x_global, dtype=np.float64)
    if x.ndim == 1:
        if x.size != C * P:
            raise ValueError(
                f"x_global has length {x.size}, expected C*P={C*P}."
            )
        x = x.reshape(C, P)
    elif x.ndim == 2:
        if x.shape != (C, P):
            raise ValueError(
                f"x_global shape {x.shape}, expected (C,P)=({C},{P})."
            )
    else:
        raise ValueError("x_global must be 1-D or 2-D")

    if orbit_target_mass is None:
        if orbit_weights is None:
            raise RuntimeError(
                "Provide orbit_target_mass for absolute comparison."
            )
        orbit_target_mass = np.asarray(orbit_weights, dtype=np.float64).ravel()
    else:
        orbit_target_mass = np.asarray(orbit_target_mass, dtype=np.float64).ravel()

    if orbit_target_mass.size == C * P:
        orbit_target_mass = orbit_target_mass.reshape(C, P).sum(axis=1)
    elif orbit_target_mass.size != C:
        raise ValueError(
            f"orbit_target_mass has size {orbit_target_mass.size}, "
            f"expected C={C} or C*P={C*P}."
        )

    sol_mass = np.sum(x, axis=1)
    resid = sol_mass - orbit_target_mass

    abs_l1 = float(np.sum(np.abs(resid)))
    abs_l2 = float(np.linalg.norm(resid))
    abs_linf = float(np.max(np.abs(resid))) if resid.size else 0.0

    rel_l2 = float(abs_l2 / (np.linalg.norm(orbit_target_mass) + 1e-30))
    rel_linf = float(
        abs_linf / (np.max(np.abs(orbit_target_mass)) + 1e-30)
    )

    eps = 1e-30
    log_t = np.log10(np.maximum(orbit_target_mass, eps))
    log_s = np.log10(np.maximum(sol_mass, eps))

    fig = plt.figure(figsize=(9.5, 4.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32)

    # Left: absolute target vs solution
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(log_t, log_s, "o", alpha=0.75, ms=5)
    lo = float(np.min([log_t.min(), log_s.min()])) if log_t.size else -1.0
    hi = float(np.max([log_t.max(), log_s.max()])) if log_t.size else 1.0
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax0.set_xlabel(r"$\log_{10}(M_{\rm target})$")
    ax0.set_ylabel(r"$\log_{10}(M_{\rm solution})$")
    ax0.set_title("Absolute orbit mass comparison")

    txt = (
        rf"$||s-t||_1={abs_l1:.3e}$" "\n"
        rf"$||s-t||_2={abs_l2:.3e}$" "\n"
        rf"$||s-t||_\infty={abs_linf:.3e}$" "\n"
        rf"rel$_2$={rel_l2:.3e}, rel$_\infty$={rel_linf:.3e}"
    )
    ax0.text(
        1.0-0.03,
        1.0-0.97,
        txt,
        transform=ax0.transAxes,
        va='bottom',
        ha='right',
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Right: residual vector
    ax1 = fig.add_subplot(gs[0, 1])
    idx = np.arange(C, dtype=int)
    ax1.axhline(0.0, color="k", lw=1.0)
    ax1.bar(idx, resid, width=0.8)
    ax1.set_xlabel("orbit index")
    ax1.set_ylabel(r"$M_{\rm solution}-M_{\rm target}$")
    ax1.set_title("Absolute residual per orbit")

    if C <= 24:
        ax1.set_xticks(idx)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=140)
    plt.close(fig)

    return {
        "abs_l1": abs_l1,
        "abs_l2": abs_l2,
        "abs_linf": abs_linf,
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "orbit_target_mass": orbit_target_mass,
        "orbit_mass": sol_mass,
        "orbit_resid": resid,
    }

# ------------------------------------------------------------------------------
