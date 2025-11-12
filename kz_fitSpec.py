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

import os, pdb, math, ctypes, sys
import numpy as np
import hashlib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize
from copy import copy
import h5py
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed,\
    ThreadPoolExecutor
from typing import Tuple, Sequence
from plotbin.display_pixels import display_pixels as dbi

from CubeFit.hdf5_manager import H5Manager, H5Dims, open_h5,\
    live_prefit_snapshot_from_models, invalidate_done
from CubeFit.hypercube_builder import build_hypercube, assert_preflight_ok,\
    estimate_global_velocity_bias_prebuild
from CubeFit.pipeline_runner   import PipelineRunner
from CubeFit import cube_utils as cu
from muse import tri_fitSpec as tf
from muse import tri_utils as uu
from dynamics.IFU.Constants import Constants, Units, UnitStr
from dynamics.IFU.Functions import Plot, Geometric

mDir = curdir.parent/'muse'
dDir = uu._ddir()

UTS = UnitStr()
UTT = Units()
CTS = Constants()
POT = Plot()
GEO = Geometric()

divcmap = 'GECKOSdr'
moncmap = 'inferno'
moncmapr = 'inferno_r'

os.environ["FITTRACKER_START"] = "fork"

# ------------------------------------------------------------------------------

def _worker_reconstruct_readonly(h5_path: str, s0: int, s1: int, x_cp_2d, *, band_L: int = 128):
    """
    4-D only: compute Y for spaxels [s0:s1) without ever loading (ΔS,C,P,L).
    Compute the contraction in float64 as requested.
    """

    x64 = np.asarray(x_cp_2d, dtype=np.float64, order="C")  # (C,P)

    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"] # (S,C,P,L) float32
        S, C, P, L = map(int, M.shape)
        dS = s1 - s0
        Y  = np.empty((dS, L), dtype=np.float64)

        # λ-banding: never hold (ΔS,C,P,L) in RAM
        for i, s in enumerate(range(s0, s1)):
            y = np.empty(L, dtype=np.float64)
            for l0 in range(0, L, band_L):
                l1   = min(L, l0 + band_L)
                A32  = M[s, :, :, l0:l1][...] # (C,P,Lb), f32
                A64  = A32.astype(np.float64, copy=False) # cast just the band
                y[l0:l1] = np.tensordot(x64, A64, axes=([0,1],[0,1]))
            Y[i, :] = y

    if not Y.flags.c_contiguous:
        Y = np.ascontiguousarray(Y)
    return (s0, Y)

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

# ------------------------------------------------------------------------------

def genCubeFit(galaxy, mPath, decDir=None, nCuts=None, proj='i', SN=90,
    full=False, slope=1.30, IMF='KB', iso='pad', weighting='luminosity',
    lOrder=4, rescale=False, specRange=None, lsf=False, band='r', smask=None,
    method='fsf', varIMF=False, source='ppxf', redraw=False, runSwitch='gen',
    **kwargs):
    """
    _summary_

    Parameters
    ----------
    galaxy : str
        Name of the galaxy to process.
    mPath : str
        Path to the model directory.
    decDir : str, optional
        Decomposition directory name, by default None.
    nCuts : int, optional
        Number of cuts for decomposition, by default None.
    proj : str, optional
        Projection type, by default 'i'.
    SN : int, optional
        Signal-to-noise ratio, by default 90.
    full : bool, optional
        Whether to use full data or truncated, by default False.
    slope : float, optional
        Slope for the IMF, by default 1.30.
    IMF : str, optional
        Initial mass function type, by default 'KB'.
    iso : str, optional
        Isochrones type, by default 'pad'.
    weighting : str, optional
        Weighting scheme, by default 'luminosity'.
    nProcs : int, optional
        Number of processes to use, by default 1.
    lOrder : int, optional
        Polynomial order for the fit, by default 4.
    fit : str, optional
        Type of fit to perform, by default 'cube'.
    rescale : bool, optional
        Whether to rescale the data, by default False.
    specRange : tuple, optional
        Spectral range to consider, by default None.
    lsf : bool, optional
        Whether to apply LSF (Line Spread Function), by default False.
    band : str, optional
        Band to use for the fit, by default 'r'.
    smask : str, optional
        Mask for the spectra, by default None.
    method : str, optional
        Method to use for the fit, by default 'fsf'.
    varIMF : bool, optional
        Whether to use variable IMF, by default False.
    source : str, optional
        Source of the data, by default 'ppxf'.
    **kwargs : dict, optional
        Additional keyword arguments for the function.
    """

    # Directories
    bDir = mDir/'tri_models'/mPath
    pDir = curdir.parent/'pxf'
    spDir = bDir/'SPDec'
    MKDIRS = [bDir, pDir, spDir]
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

    INF = uu.Load.lzma(infn)
    PA = INF['angle'][0]

    xpix, ypix, sele, pixs = uu.Load.lzma(pfs)
    # saur,goods = uu.Load.lzma(sfs)
    # del saur
    xbix, ybix = GEO.rotate2D(xpix, ypix, PA)
    pfn = dDir.parent/'muse'/'obsData'/f"{galaxy}-poly-rot.xz"
    polyProps = dict(ec=POT.brown, linestyle='--', fill=False, zorder=100,
        lw=0.75, salpha=0.5)
    if pfn.is_file():
        aShape = uu.Load.lzma(pfn)
        aShape, pPatch = POT.polyPatch(POLYGON=aShape, Xpo=xbix, Ypo=ybix,
            **polyProps)
    else:
        aShape, pPatch = POT.polyPatch(Xpo=xbix, Ypo=ybix, **polyProps)
        uu.Write.lzma(pfn, aShape)
    xmin, xmax = np.amin(xbix), np.amax(xbix)
    ymin, ymax = np.amin(ybix), np.amax(ybix)
    xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels

    saur, goods = uu.Load.lzma(pDir/galaxy/f"selection_SN{SN:02d}_{tEnd}.xz")
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    xbix = np.compress(goods, xbix)
    ybix = np.compress(goods, ybix)

    # Data spectra
    if vbSpec.is_file():
        VB = uu.Load.lzma(vbSpec)
        binNum = VB['binNum']
        binCounts = VB['binCounts']
        binFlux = VB['binFlux']
        del VB
    else:
        raise RuntimeError(f"No binned spectra.\n{'': <4s}{vbSpec}")

    with logger.capture_all_output():
        decDir, cDirs, cKeys, nComp, teLL, lnGrid, histBinSize, dataVelScale,\
            RZ, spLL, laGrid, lmin, lmax, umetals, uages, ualphas, pixOff = \
            tf._oneTimeSpec(galaxy=galaxy, mPath=mPath, decDir=decDir,
            nCuts=nCuts, proj=proj, SN=SN, full=full, slope=slope, IMF=IMF,
            iso=iso, weighting=weighting, lOrder=lOrder, rescale=rescale,
            lsf=lsf, specRange=specRange, band=band, method=method,
            varIMF=varIMF, source=source, **kwargs)
    nLSpec, nSpat = laGrid.shape
    nTSpec, nMetals, nAges, nAlphas = lnGrid.shape
    nSSP = int(np.prod((nMetals, nAges, nAlphas), dtype=int))
    pred = f"0{len(repr(nComp)):d}"
    nComp = int(nComp)
    afDir = spDir/'apers'/f"C{nComp:04d}"
    sfDir = spDir/'SSPFit'/f"C{nComp:04d}"
    MKDIRS = [afDir, sfDir]
    [DIR.mkdir(parents=True, exist_ok=True) for DIR in MKDIRS]
    lAPF = "{}_{}_c.npy"
    globAper = afDir.rglob(lAPF.format('*', f"{lOrder:02}"))
    runAper = np.arange(nSpat)

    oDict = uu.Load.lzma(direc/f"decomp_{nCuts:d}.plt")
    if 'binFN' not in oDict.keys():
        oDict['binFN'] = 'bins_0.dat'
        oDict['apFN'] = 'aperture_0.dat'
        uu.Write.lzma(direc/f"decomp_{nCuts:d}.plt", oDict)
    binFN = oDict['binFN']
    apFN = oDict['apFN']
    dnPix, dgrid = uu.Read.bins(bDir/'infil'/binFN)
    dnbins = int(np.max(dgrid))
    dgrid -= 1
    dss = np.where(dgrid >= 0)[0]
    dx0, dx1, dnx, dy0, dy1, dny, dtheta = uu.Read.aperture(
        bDir/'infil'/apFN)
    ddx = np.abs((dx1-dx0)/dnx)
    ddy = np.abs((dy1-dy0)/dny)
    dpixs = np.min([ddx, ddy])
    dxr = np.arange(dnx)*dpixs + dx0 + 0.5*dpixs
    dyr = np.arange(dny)*dpixs + dy0 + 0.5*dpixs
    dxtss = uu._hash(dxr, np.full_like(dyr, 1)).ravel()[dss]
    dytss = uu._hash(np.full_like(dxr, 1), dyr).ravel()[dss]
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
    bLKey = uu.keySep.join([nnOrb.parent.parent.name, nnOrb.parent.name])
    bLKey = uu.rReplace(bLKey, uu.keySep, os.sep, 1)
    nnOrb = plp.Path(bDir, decDir, nnOrb.parent.name, nnOrb.name)
    oClass = plp.Path(bDir, decDir, oClass.parent.name, oClass.name)
    obClass = plp.Path(bDir, decDir, obClass.parent.name, obClass.name)
    fpd = uu._deetExtr(bLKey)
    apDir = bDir/bLKey/'nn_aphist.out'
    maDir = (bDir/bLKey).parent/'datfil'/'mass_aper.dat'
    nnK = bDir/bLKey/'nn_kinem.out'

    NOrbs, inds, energs, I2s, I3s, regs, types, weights, lcuts =\
        uu.Read.orbits(nnOrb)
    cWeights = np.array([
        np.ma.sum(oDict['weights'][f"{comp:{pred}d}"]) for comp in nzComp])

    kiBin = INF['kin']['nbins'][0]
    assert nbins == kiBin, 'Output does not agree with input bins\nInput:'+\
        f"{kiBin}\nOutput: {nbins}"

    wbin, hN, histBinSize, hArr = uu.Read.apertureHist(apDir)
    logger.log(f"{'Mass outside of the histograms:': <45s}"\
          f"{np.sum(hArr[:, 0] + hArr[:, wbin * 2]):5.5}")

    fullBin, fullID, fullK0 = uu.Read.massAperture(maDir)
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
        aperMass = uu.Load.lzma(apMassFile)
    else:
        aperMass = np.ma.ones((nSpat, nComp), dtype=float)*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Mass', total=nComp):
            try:
                maFile = cDir/'declib_apermass.out'
                nbin, ID, k0 = uu.Read.massAperture(maFile)
                aperMass[:, cn] = k0
            except:
                ERR += [cDir.name]
        if len(ERR) > 0:
            logger.log(ERR)
            breakpoint()
        uu.Write.lzma(apMassFile, aperMass)
    aperMass = np.ma.masked_invalid(aperMass)
    norma = np.sum(aperMass, axis=1)

    logger.log('Done.', flush=True)
    apFile = cDirs[0]/'declib_aphist.out'
    wbin, hN, histBinSize, hArr = uu.Read.apertureHist(apFile)
    # Load the parameters regardless
    apHistFile = direc/f"apHists_i{pStr}_{nComp:{pred}d}.jl"
    if apHistFile.is_file():
        logger.log('Reading histograms...', flush=True)
        apHists = uu.Load.jobl(apHistFile)
    else:
        apFile = cDirs[0]/'declib_aphist.out'
        wbin, hN, histBinSize, cArr = uu.Read.apertureHist(apFile)
        logger.log('Generating histograms...', flush=True)
        apHists = np.ma.ones((*cArr.shape, nComp))*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Components',
            total=nComp):
            try:
                apFile = cDir/'declib_aphist.out'
                wbin, hN, histBinSize, cArr = uu.Read.apertureHist(
                    apFile)
                apHists[:, :, cn] = cArr
            except:
                ERR += [cDir.stem]
        if len(ERR) > 0:
            logger.log(ERR)
            pdb.set_trace()
        uu.Write.jobw(apHistFile, apHists)
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
    hdf5Path = (hdf5Dir/f"{galaxy}_{lOrder:02d}").with_suffix('.h5')

    # --- Initialize and load data ---
    # mgr = H5Manager(hdf5Path, tem_pix=copy(teLL), obs_pix=copy(spLL))
    mgr = H5Manager(hdf5Path)
    # mgr.set_velocity_grid(copy(vbins))
    arDims = mgr.populate_from_arrays(
        losvd=nApHists,
        datacube=laGrid,
        templates=lnGrid,
        mask=spmask,
        tem_pix=copy(teLL), obs_pix=copy(spLL),
        vel_pix=copy(vbins),
        xpix=xpix, ypix=ypix,
        binnum=binNum
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
        _ = assert_preflight_ok(
            hdf5Path,
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
    
    with logger.capture_all_output():
        est = estimate_global_velocity_bias_prebuild(hdf5Path,
            n_spax=96, n_features=24, window_len=31, lag_px=12)
    logger.log(f"[CubeFit] Estimated global velocity bias (km/s): "\
        f"{est['vel_bias_kms']:.3f}")
    logger.log(f"[CubeFit] Building /HyperCube in {hdf5Path}...")
    with logger.capture_all_output():
        build_hypercube(
            hdf5Path,
            norm_mode="data", # choose "model" or "data"
            # "model" preserves relative contribution to both spaxel and components
            amp_mode="sum", # "sum" or "trapz"
            S_chunk=nS, C_chunk=nC, P_chunk=nP,
            vel_bias_kms=est["vel_bias_kms"]
        )
    # even if runSwitch is fit only, we want to ensure the HyperCube
    # is built, so we don't return early here.
    # Should be zero-cost if already built

    prefit_png = spDir / f"prefit_overlay_from_models_C{nComp:03d}.png"
    live_prefit_snapshot_from_models(
        h5_path=str(hdf5Path),
        max_components=4,
        templates_per_pair=3,
        out_png=str(prefit_png),
    )
    logger.log(f"[Prefit] wrote {prefit_png}")
    if 'gen' in runSwitch:
        return

    if 'fit' not in runSwitch:
        logger.log(f"[CubeFit] runSwitch={runSwitch} is not understood; "
            "exiting.")
        raise RuntimeError("Invalid runSwitch")
    # --- 4) Run the global Kaczmarz fit (tiled; RAM-bounded) ---
    runner = PipelineRunner(hdf5Path)

    ncpuset, mask = cu.cpuset_count()
    print(f"[guard] cpuset mask: {mask}  cores: {ncpuset}")
    logger.log(f"cpuset cores: {len(os.sched_getaffinity(0))}")
    from threadpoolctl import threadpool_info
    logger.log(f"BLAS pools: {threadpool_info()}")
    # look for openblas 
    try:
        with open("/proc/self/cgroup") as f:
            print("[guard] cgroups:")
            for line in f:
                if "cpuset" in line or "cpu" in line:
                    print(" ", line.strip())
    except Exception:
        pass

    print("[env] SLURM_JOB_ID=", os.environ.get("SLURM_JOB_ID"),
        " SLURM_STEP_ID=", os.environ.get("SLURM_STEP_ID"))

    ########################
    # Non-batched Kaczmarz #
    ########################
    # x_global, stats = runner.solve_all(
    #     epochs=1,
    #     pixels_per_aperture=4096,
    #     lr=0.25,
    #     project_nonneg=True,
    #     # orbit_weights=cWeights,
    #     orbit_weights=None,
    #     ratio_use=False,
    #     reader_s_tile=nS, reader_c_tile=nC, reader_p_tile=nP,
    #     verbose=True,
    #     warm_start='jacobi',  # 'zeros', 'random', 'jacobi'
    #     row_order='sequential',
    #     block_rows=2048,
    #     blas_threads=48,
    #     progress_interval_sec=900,
    #     tracker_mode='off',
    # # Optional ratio controls
    # # ratio_use=True, ratio_anchor="auto", ratio_eta=0.05, ratio_prob=0.02,
    # # ratio_batch=2, ratio_min_weight=1e-4,
    # # progress_interval_sec=60,  # if you want periodic on_progress ticks
    # )

    #####################################
    # Multi-processing Batched Kaczmarz #
    #####################################
    RC = cu.RatioCfg(
        anchor="x0",
        eta=0.4,
        gamma=1.3,
        prob=1.0,
        batch=0,
    )
    x_global, stats = runner.solve_all_mp_batched(
        epochs=1,
        lr=0.01,
        project_nonneg=True,
        # orbit_weights=None,     # or None for “free” fit
        orbit_weights=cWeights,
        processes=4,                # 4 workers
        blas_threads=12,            # 12 BLAS threads each → 48 total
        reader_s_tile=128,          # match /HyperCube/models chunking on S
        verbose=True,
        warm_start='resume',  # 'zeros', 'resume', 'jacobi',
        seed_cfg=dict(Ns=24, L_sub=1200, K_cols=768, per_comp_cap=24),
        ratio_cfg=RC,
    )

    logger.log("[CubeFit] Global fit completed.")
    # # --- Save the global solution vector
    # logger.log("[CubeFit] Saving fit results...")
    # with open_h5(hdf5Path, role="writer") as f:
    #     ds = f.get("/X_global", None)
    #     if ds is not None and ds.shape == (x_global.size,):
    #         ds[...] = np.asarray(x_global, np.float64)
    #         logger.log("[CubeFit] Dataset already exists; overwritten.")
    #     else:
    #         if "/X_global" in f:
    #             del f["/X_global"]
    #         f.create_dataset(
    #             "/X_global",
    #             data=np.asarray(x_global, np.float64),
    #             chunks=(min(8192, x_global.size),),
    #             compression="gzip", compression_opts=4, shuffle=True,
    #         )
    #         logger.log("[CubeFit] Results stored.")

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

def reconstruct_model_cube_single(h5_path: str,
                                  x_global,
                                  array_name: str = "ModelCube",
                                  blas_threads: int = 8,
                                  S_tile: int | None = None,
                                  L_band: int = 128) -> None:
    """
    Chunk-aligned, streaming reconstruction of Y = (HyperCube) · x.

    Reads the model cube in the same order it is chunked:
      S in blocks of S_chunk, C in blocks of 1 (C_chunk=1), and P in
      blocks of P_chunk. For each (S_chunk, C=1) slab it contracts over
      P against x[c, :] using BLAS on small 2-D views, accumulating into
      a (S_chunk, L) float64 tile. Optionally processes L in bands to
      keep the GEMV working set small. Never materializes a (ΔS,C,P,L)
      float64 slab.

    Parameters
    ----------
    h5_path : str
        Path to the base HDF5 file.
    x_global : array-like
        Global weights, shape (C*P,) or (C,P). Internally used in f32
        for speed; accumulation into output stays f64.
    array_name : str
        Name of output dataset to create (S,L) float64.
    blas_threads : int
        Number of BLAS threads to use inside this process.
    S_tile : Optional[int]
        If given, caps the S block size. Otherwise uses S_chunk.
    L_band : int
        Optional λ-banding for GEMV working set (does not change how
        HDF5 reads; it just limits the temporary 2-D views).
    """

    # Try to limit BLAS threads for predictability on clusters
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    t = int(os.environ.get("SLURM_CPUS_PER_TASK", blas_threads))
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = \
        os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    if threadpool_limits:
        try:
            threadpool_limits(t)
        except Exception:
            pass

    # Small helper to return free heap to the OS (keeps RSS flat).
    try:
        libc = ctypes.CDLL("libc.so.6")
        def trim_heap(): libc.malloc_trim(0)
    except Exception:
        def trim_heap(): pass

    # Optional dataset chunk-cache tuning (local to this dataset)
    rdcc_slots = int(os.environ.get("CUBEFIT_RDCC_SLOTS", "1000003"))
    rdcc_bytes = int(os.environ.get("CUBEFIT_RDCC_BYTES", str(256 * 1024**2)))
    rdcc_w0    = float(os.environ.get("CUBEFIT_RDCC_W0", "0.90"))

    # Force unbuffered writes so tqdm appears immediately on HPC logs
    try:
        sys.stdout.reconfigure(line_buffering=True)  # py3.7+
    except Exception:
        pass
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    with open_h5(h5_path, role="writer") as f:
        M = f["/HyperCube/models"]              # (S,C,P,L) float32
        S, C, P, L = map(int, M.shape)
        try:
            M.id.set_chunk_cache(rdcc_slots, rdcc_bytes, rdcc_w0)
        except Exception:
            pass

        chunks = M.chunks or (S, 1, P, L)
        S_chunk, C_chunk, P_chunk, L_chunk = map(int, chunks)

        # Choose S block aligned to storage chunk (or user cap).
        S_blk = S_chunk if S_tile is None else max(1, int(S_tile))
        S_blk = min(S_blk, S_chunk, S)

        # Prepare output (tile-aligned; uncompressed is fastest).
        if array_name in f:
            del f[array_name]
        out = f.create_dataset(array_name, shape=(S, L), dtype="f8",
                               chunks=(S_blk, L))

        # Weights as (C,P) — math in f32 for speed; sum in f64.
        x_cp = np.asarray(x_global)
        if x_cp.ndim == 1:
            x_cp = x_cp.reshape(C, P)
        if x_cp.shape != (C, P):
            raise ValueError(f"x shape {x_cp.shape} != (C,P)=({C},{P})")
        x32 = np.asarray(x_cp, dtype=np.float32, order="C")

        # Progress — write to stdout & flush so it shows immediately
        n_tiles = math.ceil(S / S_blk)
        header = (f"[Recon] S_chunk={S_chunk} C_chunk={C_chunk} "
                  f"P_chunk={P_chunk} L_chunk={L_chunk} S_blk={S_blk} "
                  f"L_band={L_band}")
        print(header, flush=True)

        pbar = tqdm(total=n_tiles,
                    desc="[Reconstruct] tiles",
                    unit="tile",
                    dynamic_ncols=True,
                    mininterval=2.0,
                    miniters=1,
                    smoothing=0.0,
                    file=sys.stdout,
                    leave=True)
        pbar.refresh(); sys.stdout.flush()

        # --- main loop: S in storage-aligned tiles -------------------
        for s0 in range(0, S, S_blk):
            s1 = min(S, s0 + S_blk)
            dS = s1 - s0

            # Tile accumulator in f64
            Y_tile = np.zeros((dS, L), dtype=np.float64, order="C")

            # Iterate components in storage order (C_chunk == 1)
            for c0 in range(0, C, max(1, C_chunk)):
                c1 = min(C, c0 + max(1, C_chunk))
                c = c0  # C_chunk is 1 in our files

                # Optional λ-banding for small GEMV views
                for l0 in range(0, L, max(1, L_band)):
                    l1  = min(L, l0 + max(1, L_band))
                    Lb  = l1 - l0

                    # Accumulator for this (c, L band)
                    band_acc = np.zeros((dS, Lb), dtype=np.float64, order="C")

                    # Contract over P in P_chunk steps
                    for p0 in range(0, P, max(1, P_chunk)):
                        p1  = min(P, p0 + max(1, P_chunk))
                        Pb  = p1 - p0

                        # Read one storage-aligned slab: (dS, 1, Pb, Lb) f32
                        A32 = M[s0:s1, c:c1, p0:p1, l0:l1][...]
                        A32 = np.asarray(A32, dtype=np.float32, order="C")

                        # Make a (dS*Lb, Pb) 2-D view for GEMV
                        A2D = A32[:, 0, :, :].swapaxes(1, 2).reshape(dS * Lb, Pb)

                        # Multiply by weights for this (c, p-block) in f32
                        w32 = x32[c, p0:p1]          # (Pb,)
                        tmp = A2D @ w32              # (dS*Lb,) f32

                        # Accumulate into f64 band
                        band_acc += tmp.reshape(dS, Lb).astype(np.float64, copy=False)

                    # Add this component's band into the tile
                    Y_tile[:, l0:l1] += band_acc

            # Write the finished S-tile
            out[s0:s1, :] = Y_tile

            # Make progress visible right now
            pbar.update(1)
            pbar.refresh()
            try:
                out.id.flush()    # SWMR-friendly: expose new tile
            except Exception:
                pass
            try:
                f.flush()
            except Exception:
                pass
            sys.stdout.flush()
            trim_heap()

        pbar.close()

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
    chi2: np.ndarray,
    n: int,
    plot_dir: str,
    n_workers: int,
    tag: str,
    mask: np.ndarray | None = None,
):
    """
    Memory-safe plotting:
      - Reads only needed rows from /DataCube and /ModelCube.
      - Never touches /HyperCube/models.
      - Closes every figure immediately.
      - Small thread pool (I/O bound).

    Style:
      - Data in black, model in red (lw=0.8).
      - Residuals (data - model) as green diamonds at every pixel,
        vertically offset (same offset policy as before) so they don’t
        overlap the spectra.
      - A solid green line at the residual zero (i.e., the offset
        baseline), and thin dashed green lines at ±1σ (σ computed on
        masked residuals).
      - If /Mask exists (or 'mask' provided), masked regions are shaded
        with semi-transparent grey bands.
    """
    import os
    import pathlib as plp
    from concurrent.futures import ThreadPoolExecutor

    plp.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    n = int(np.maximum(1, n))
    chi2 = np.asarray(chi2, dtype=np.float64)
    S = int(chi2.shape[0])

    # Pick indices (worst/best by chi^2)
    order_desc = np.argsort(-chi2)
    order_asc  = np.argsort( chi2)
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
        res = dat - mod

        # Keep the SAME residual offset policy as before
        y_lo = float(np.nanmin(np.concatenate((dat[mask], mod[mask]))))
        y_hi = float(np.nanmax(np.concatenate((dat[mask], mod[mask]))))
        y_rng = float(np.maximum(y_hi - y_lo, 1.0))
        y_off = y_lo - 0.25 * y_rng

        # σ from masked residuals
        sigma = float(np.nanstd(res[mask])) if np.any(mask) else 0.0

        fig = plt.figure(figsize=(8, 3.5))
        ax  = fig.add_subplot(111)

        # Data/model lines (thin)
        ax.plot(obs[mask], dat[mask], lw=0.8, color="k", label="data")
        ax.plot(obs[mask], mod[mask], lw=0.8, color="r", label="model")

        # Residuals as green diamonds at every pixel (offset)
        ax.scatter(
            obs[mask], (res[mask] + y_off),
            s=8, marker="D", edgecolors="none", color="g", alpha=0.9,
            label="residual (offset)"
        )

        # Residual baseline (solid) and ±1σ (dashed)
        ax.axhline(y_off,             ls="-",  lw=0.7, color="g")          # zero residual
        if sigma > 0.0 and np.isfinite(sigma):
            ax.axhline(y_off + sigma, ls="--", lw=0.6, color="g")
            ax.axhline(y_off - sigma, ls="--", lw=0.6, color="g")

        # Shade masked regions as semi-transparent grey bands
        if mask_spans:
            y0, y1 = ax.get_ylim()
            for a, b in mask_spans:
                x0 = float(obs[int(a)])
                x1 = float(obs[int(np.maximum(a, b - 1))])
                if int(b) < L:
                    x1 = float(obs[int(b)])
                ax.axvspan(x0, x1, color="0.2", alpha=0.12, zorder=0)
            ax.set_ylim(y0, y1)

        # ax.set_title(f"spaxel {int(s_idx)}  χ={chi2[int(s_idx)]:.3f}")
        ax.set_xlabel(rf"$\log(\lambda\ [\AA])$")
        ax.set_ylabel("$F_\lambda$ (arb. units)")
        # ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            plot_dir, f"{rank_tag}_{tag}_spax{int(s_idx):05d}.png"
        ), dpi=120)
        plt.close(fig)

    # Tiny pool; ≤4 for I/O friendliness (NumPy math, not builtins)
    pool_n = int(np.minimum(np.maximum(1, int(n_workers)), 4))
    jobs = [(int(s), "worst") for s in idx_worst] + \
           [(int(s), "best")  for s in idx_best]

    with ThreadPoolExecutor(max_workers=pool_n) as pool:
        list(pool.map(lambda args: _plot_one(*args), jobs))

    del data_sel, model_sel

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

def _init_worker():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    try:
        # runtime guard in case threads were already initialized
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)  # BLAS/OpenMP libraries → 1 thread
    except Exception:
        pass

# ------------------------------------------------------------------------------

def loadCubeFit(galaxy, mPath, decDir=None, nCuts=None, proj='i', SN=90,
    full=False, slope=1.30, IMF='KB', iso='pad', weighting='luminosity',
    nProcs=1, lOrder=4, rescale=False, specRange=None, lsf=False,
    band='r', smask=None, method='fsf', varIMF=False,
    source='ppxf', pplots=['sfh', 'spec', 'mw'], redraw=False, **kwargs):
    """
    Load the CubeFit data for a given galaxy and model path.
    """
    # Directories
    bDir = mDir/'tri_models'/mPath
    pDir = curdir.parent/'pxf'
    spDir = bDir/'SPDec'
    MKDIRS = [bDir, pDir, spDir]
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

    INF = uu.Load.lzma(infn)
    PA = INF['angle'][0]
    if vbSpec.is_file():
        VB = uu.Load.lzma(vbSpec)
        binNum = VB['binNum']
        binCounts = VB['binCounts']
        del VB
    else:
        raise RuntimeError(f"No binned spectra.\n{'': <4s}{vbSpec}")

    xpix, ypix, sele, pixs = uu.Load.lzma(pfs)
    # saur,goods = uu.Load.lzma(sfs)
    # del saur
    xbix, ybix = GEO.rotate2D(xpix, ypix, PA)
    pfn = dDir.parent/'muse'/'obsData'/f"{galaxy}-poly-rot.xz"
    polyProps = dict(ec=POT.brown, linestyle='--', fill=False, zorder=100,
        lw=0.75, salpha=0.5)
    if pfn.is_file():
        aShape = uu.Load.lzma(pfn)
        aShape, pPatch = POT.polyPatch(POLYGON=aShape, Xpo=xbix, Ypo=ybix,
            **polyProps)
    else:
        aShape, pPatch = POT.polyPatch(Xpo=xbix, Ypo=ybix, **polyProps)
        uu.Write.lzma(pfn, aShape)
    xmin, xmax = np.amin(xbix), np.amax(xbix)
    ymin, ymax = np.amin(ybix), np.amax(ybix)
    xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels

    saur, goods = uu.Load.lzma(pDir/galaxy/f"selection_SN{SN:02d}_{tEnd}.xz")
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    xbix = np.compress(goods, xbix)
    ybix = np.compress(goods, ybix)

    with logger.capture_all_output():
        decDir, cDirs, cKeys, nComp, teLL, lnGrid, histBinSize, dataVelScale,\
            RZ, spLL, laGrid, lmin, lmax, umetals, uages, ualphas, pixOff = \
            tf._oneTimeSpec(galaxy=galaxy, mPath=mPath, decDir=decDir,
            nCuts=nCuts, proj=proj, SN=SN, full=full, slope=slope, IMF=IMF,
            iso=iso, weighting=weighting, lOrder=lOrder, rescale=rescale,
            lsf=lsf, specRange=specRange, band=band, method=method,
            varIMF=varIMF, source=source, **kwargs)
    nLSpec, nSpat = laGrid.shape
    nTSpec, nMetals, nAges, nAlphas = lnGrid.shape
    nSSP = int(np.prod((nMetals, nAges, nAlphas), dtype=int))
    pred = f"0{len(repr(nComp)):d}"
    nComp = int(nComp)

    oDict = uu.Load.lzma(direc/f"decomp_{nCuts:d}.plt")
    if 'binFN' not in oDict.keys():
        oDict['binFN'] = 'bins_0.dat'
        oDict['apFN'] = 'aperture_0.dat'
        uu.Write.lzma(direc/f"decomp_{nCuts:d}.plt", oDict)
    binFN = oDict['binFN']
    apFN = oDict['apFN']
    dnPix, dgrid = uu.Read.bins(bDir/'infil'/binFN)
    dnbins = int(np.max(dgrid))
    dgrid -= 1
    dss = np.where(dgrid >= 0)[0]
    dx0, dx1, dnx, dy0, dy1, dny, dtheta = uu.Read.aperture(
        bDir/'infil'/apFN)
    ddx = np.abs((dx1-dx0)/dnx)
    ddy = np.abs((dy1-dy0)/dny)
    dpixs = np.min([ddx, ddy])
    dxr = np.arange(dnx)*dpixs + dx0 + 0.5*dpixs
    dyr = np.arange(dny)*dpixs + dy0 + 0.5*dpixs
    dxtss = uu._hash(dxr, np.full_like(dyr, 1)).ravel()[dss]
    dytss = uu._hash(np.full_like(dxr, 1), dyr).ravel()[dss]
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
    bLKey = uu.keySep.join([nnOrb.parent.parent.name, nnOrb.parent.name])
    bLKey = uu.rReplace(bLKey, uu.keySep, os.sep, 1)
    nnOrb = plp.Path(bDir, decDir, nnOrb.parent.name, nnOrb.name)
    oClass = plp.Path(bDir, decDir, oClass.parent.name, oClass.name)
    obClass = plp.Path(bDir, decDir, obClass.parent.name, obClass.name)
    fpd = uu._deetExtr(bLKey)
    apDir = bDir/bLKey/'nn_aphist.out'
    maDir = (bDir/bLKey).parent/'datfil'/'mass_aper.dat'

    NOrbs, inds, energs, I2s, I3s, regs, types, weights, lcuts =\
        uu.Read.orbits(nnOrb)
    cWeights = np.zeros(nComp)
    for jk, comp in enumerate(nzComp):
        cWeights[jk] = np.ma.sum(weights[oDict['wheres'][f"{comp:{pred}d}"]])

    kiBin = INF['kin']['nbins'][0]
    assert nbins == kiBin, 'Output does not agree with input bins\nInput:'+\
        f"{kiBin}\nOutput: {nbins}"

    wbin, hN, histBinSize, hArr = uu.Read.apertureHist(apDir)
    logger.log(f"{'Mass outside of the histograms:': <45s}"\
          f"{np.sum(hArr[:, 0] + hArr[:, wbin * 2]):5.5}")

    fullBin, fullID, fullK0 = uu.Read.massAperture(maDir)
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
        aperMass = uu.Load.lzma(apMassFile)
    else:
        aperMass = np.ma.ones((nSpat, nComp), dtype=float)*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Mass', total=nComp):
            try:
                maFile = cDir/'declib_apermass.out'
                nbin, ID, k0 = uu.Read.massAperture(maFile)
                aperMass[:, cn] = k0
            except:
                ERR += [cDir.name]
        if len(ERR) > 0:
            logger.log(ERR)
            breakpoint()
        uu.Write.lzma(apMassFile, aperMass)
    aperMass = np.ma.masked_invalid(aperMass)
    norma = np.sum(aperMass, axis=1)

    logger.log('Done.', flush=True)
    apFile = cDirs[0]/'declib_aphist.out'
    wbin, hN, histBinSize, hArr = uu.Read.apertureHist(apFile)
    # Load the parameters regardless
    apHistFile = direc/f"apHists_i{pStr}_{nComp:{pred}d}.jl"
    if apHistFile.is_file():
        logger.log('Reading histograms...', flush=True)
        apHists = uu.Load.jobl(apHistFile)
    else:
        apFile = cDirs[0]/'declib_aphist.out'
        wbin, hN, histBinSize, cArr = uu.Read.apertureHist(apFile)
        logger.log('Generating histograms...', flush=True)
        apHists = np.ma.ones((*cArr.shape, nComp))*np.nan
        ERR = []
        for cn, cDir in tqdm(enumerate(cDirs), desc='Components',
            total=nComp):
            try:
                apFile = cDir/'declib_aphist.out'
                wbin, hN, histBinSize, cArr = uu.Read.apertureHist(
                    apFile)
                apHists[:, :, cn] = cArr
            except:
                ERR += [cDir.stem]
        if len(ERR) > 0:
            logger.log(ERR)
            pdb.set_trace()
        uu.Write.jobw(apHistFile, apHists)
    logger.log('Done.')
    apHists = np.ma.masked_invalid(apHists)
    nApHists = (apHists*(massNorm/norma)[:, np.newaxis, np.newaxis])

    # --- Setup HDF5 directory ---
    hdf5Dir = plp.Path(kwargs.pop('hdf5Dir', curdir/galaxy))
    hdf5Dir.mkdir(parents=True, exist_ok=True)
    hdf5Path = (hdf5Dir/f"{galaxy}_{lOrder:02d}").with_suffix('.h5')
    
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
        mask_arr = f["/Mask"][...] if has_mask else None
        obs = f["/ObsPix"][...] if "/ObsPix" in f else np.arange(nLSpec)

    spat_tile, nTiles = choose_spat_tile_fast(nSpat, nProcs, s_chunk, k=2)
    nProcs = min(nProcs, nTiles, 12)  # don’t spawn more processes than tiles

    ok, why = modelcube_status(str(hdf5Path), x_global=x_global, require_float64=True, redraw=redraw)
    logger.log(f"[ModelCube] status: {why}")
    if ok:
        logger.log("[ModelCube] Skipping reconstruction.")
    else:
        logger.log("[ModelCube] Reconstructing…")
        reconstruct_model_cube_single(  # or your parallel version
            h5_path=str(hdf5Path),
            x_global=x_global,
            array_name="ModelCube",
            blas_threads=nProcs,
        )
        # Stamp digest so future runs can skip confidently
        try:
            with open_h5(str(hdf5Path), role="writer") as f:
                xdig = _x_digest(x_global)
                ds = f["/ModelCube"]
                ds.attrs["x_digest"] = xdig
                ds.attrs["x_shape"]  = np.asarray(x_global).shape
                ds.attrs["dtype_math"] = "float64"
                ds.attrs["generator"] = "reconstruct_model_cube_single"
        except Exception as e:
            logger.log(f"[ModelCube] Warning: could not stamp digest ({e})")

    with open_h5(hdf5Path, role="reader") as f:
        data_cube = f["/DataCube"][...]
        model_cube = f["/ModelCube"][...]  # (nSpat, nLSpec)

    # chi^2 per spaxel
    if mask_arr is None:
        mask_arr = np.ones(nLSpec, dtype=bool)
    resid = (data_cube - model_cube)[:, mask_arr]
    rchi2 = np.sqrt((resid * resid).mean(axis=1))

    plt.figure(figsize=(6, 4))
    plt.hist(rchi2, bins=40, alpha=0.7)
    plt.xlabel(r"${\rm Norm}/\sqrt{N_{\rm pix}}$")
    plt.ylabel("Number of apertures")
    plt.title("Distribution of fit quality")
    plt.tight_layout()
    plt.savefig(spDir/"chi2_hist.png")
    plt.close()

    if 'spec' in pplots:
        logger.log("Generating spectrum plots...")
        with logger.capture_all_output():
            parallel_spectrum_plots(
                h5_or_path=str(hdf5Path),
                chi2=rchi2,
                n=10,
                plot_dir=str(spDir),
                n_workers=min(12, max(1, nProcs)),
                tag=f"C{nComp:04d}",
                mask=mask_arr,
            )

    if 'mw' in pplots:
        laGrid = np.ma.masked_less_equal(laGrid, 0.0)
        model_cube = np.ma.masked_less_equal(model_cube, 0.0)
        flux = np.ma.masked_invalid(
            # (np.ma.sum(laGrid, axis=0)/binCounts)[binNum], 0.))
            (np.ma.sum(laGrid, axis=0)/binCounts))
        modSB = np.ma.masked_array( # re-scale to original data levels
            (np.ma.sum(model_cube, axis=1)/binCounts),
            # (np.ma.sum(model_cube, axis=0)*laScales/binCounts)[binNum],
            mask=np.ma.getmaskarray(flux))
        fmin, fmax = np.log10(np.ma.min(flux)), np.log10(np.ma.max(flux))
        pren = 2
        miText = POT.prec(pren, fmin)
        maText = POT.prec(pren, fmax)
        gs = gridspec.GridSpec(3, 1, hspace=0., wspace=0.)
        fig = plt.figure(figsize=plt.figaspect((yLen*3.)/xLen)*0.75)
        ax = fig.add_subplot(gs[0])
        cnt = dbi(xpix, ypix, np.log10(flux[binNum]), pixelsize=pixs, angle=PA,
            cmap='gist_heat', vmin=fmin, vmax=fmax)
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
        dbi(xpix, ypix, np.log10(modSB[binNum]), pixelsize=pixs, angle=PA,
            cmap='cet_fire', vmin=fmin, vmax=fmax)
        ax.set_xticklabels([])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.add_patch(copy(pPatch))

        delta = (flux-modSB)/flux
        ax = fig.add_subplot(gs[2])
        cnt = dbi(xpix, ypix, delta[binNum], pixelsize=pixs, angle=PA,
            cmap=divcmap, vmin=-0.1, vmax=0.1)
        cax = POT.attachAxis(ax, 'top', 0.1, mid=True)
        cb = plt.colorbar(cnt, cax=cax, orientation='horizontal')
        lT = cax.text(0.5, 0.5, r'$(D-M)/D$', va='center', ha='center',
            color=POT.pgreen, transform=cax.transAxes)
        lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
            foreground='k')])
        cax.text(1e-3, 0.5, '-0.1', va='center', ha='left', color='white',
            transform=cax.transAxes)
        cax.text(1.0-1e-3, 0.5, '0.1', va='center', ha='right', color='white',
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

        plt.savefig(spDir/\
            f"modelCube_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")

    if 'otype' not in oDict['cutOn']:
        return # only do orbital SFH if orbital decomposition
    if len(oDict['cuts'])>0:
        # determine which components belong to which orbital categories
        allCuts = np.array([oDict['cuts'][key] for key in oDict['cuts'].keys()])
        uCuts, uCounts = np.unique(allCuts, axis=0, return_counts=True)
        # assert that every elemnt of uCounts is equal
        assert np.unique(uCounts).size == 1
        notypes = np.max(uCounts)
        obins = np.arange(1, notypes+1) * uCuts.shape[0]
        otypes = np.digitize(nzComp, bins=obins, right=True)
    else:
        otypes = copy(nzComp)

    satube = (otypes == 0) # group short-axis tubes
    latube = (otypes == 1)
    boxess = (otypes == 2)
    arSOL = x_global.reshape(nComp, nMetals, nAges, nAlphas, order='C')
    coSFH = arSOL[satube, :, :, :].sum(axis=0)
    laSFH = arSOL[latube, :, :, :].sum(axis=0)
    boSFH = arSOL[boxess, :, :, :].sum(axis=0)

    minT, maxT = np.min(uages), np.max(uages)
    minZ, maxZ = np.min(umetals), np.max(umetals)

    wmax = np.max(np.array([coSFH, laSFH, boSFH]))

    if 'sfh' in pplots:
        fig = plt.figure(figsize=plt.figaspect(3./4.))
        gs = gridspec.GridSpec(3, nAlphas, hspace=0., wspace=0.)
        # one column per alpha, 3 orbit types
        print(nAlphas, ualphas)
        for ali in range(nAlphas):
            ax = fig.add_subplot(gs[0, ali])
            ax.imshow(coSFH[:, :, ali], extent=[minT, maxT, minZ, maxZ],
                aspect='auto', interpolation='bicubic', origin='lower',
                cmap=moncmapr, norm=Normalize(vmin=0, vmax=wmax))
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if ax.get_subplotspec().is_first_col():
                lT = ax.text(1e-2, 1e-2, r'$z$ Tubes', va='bottom', ha='left',
                    color=POT.pgreen, transform=ax.transAxes)
                lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                    foreground='k')])
            if nAlphas > 1:
                lT = ax.text(0.5, 1.05, rf"$[\alpha/Fe]={ualphas[ali]:.2f}$",
                    va='bottom', ha='center', color=POT.pgreen,
                    transform=ax.transAxes)
                lT.set_path_effects(
                    [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            ax = fig.add_subplot(gs[1, ali])
            ax.imshow(laSFH[:, :, ali], extent=[minT, maxT, minZ, maxZ],
                aspect='auto', interpolation='bicubic', origin='lower',
                cmap=moncmapr, norm=Normalize(vmin=0, vmax=wmax))
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if ax.get_subplotspec().is_first_col():
                lT = ax.text(1e-2, 1e-2, r'$x$ Tubes', va='bottom', ha='left',
                    color=POT.pgreen, transform=ax.transAxes)
                lT.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                    foreground='k')])
            ax = fig.add_subplot(gs[2, ali])
            ax.imshow(boSFH[:, :, ali], extent=[minT, maxT, minZ, maxZ],
                aspect='auto', interpolation='bicubic', origin='lower',
                cmap=moncmapr, norm=Normalize(vmin=0, vmax=wmax))
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
        plt.savefig(spDir/\
            f"orbitSFH_{nComp:{pred}d}_i{proj}{tag}_{lOrder:02d}.png")

    # 7. Print summary
    print(f"Mean reduced χ²: {np.mean(rchi2):.2f} ± {np.std(rchi2):.2f}")
    worst = np.argmax(rchi2)
    best = np.argmin(rchi2)
    print(f"Worst fit: aperture {worst} (χ² = {rchi2[worst]:.2f})")
    print(f"Best fit:  aperture {best} (χ² = {rchi2[best]:.2f})")
    print(f"[CubeFit] All plots and maps saved in {str(spDir)}")

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
            y = np.zeros(L, dtype=np.float64, order="C")
            for c0 in range(0, C, max(1, C_chunk)):
                c1 = min(C, c0 + max(1, C_chunk))
                c = c0  # C_chunk==1 in our files
                for p0 in range(0, P, max(1, P_chunk)):
                    p1  = min(P, p0 + max(1, P_chunk))
                    A32 = M[s_idx:s_idx+1, c:c1, p0:p1, :][...].astype(np.float32, copy=False)
                    A2D = A32[0, 0, :, :]          # (Pb, L)
                    w32 = x32[c, p0:p1]            # (Pb,)
                    y  += (A2D.T @ w32).astype(np.float64, copy=False)
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
            fig.tight_layout()
            fn = os.path.join(plot_dir, f"diag_sparse_{tag}_spax{int(s):05d}.png")
            fig.savefig(fn, dpi=120)
            plt.close(fig)

        print(f"[DiagSparse] wrote {picks.size} plots to {plot_dir}")

# ------------------------------------------------------------------------------

def compare_orbit_vs_solution(
    h5_path: str,
    *,
    orbit_weights: np.ndarray | None = None,   # shape (C,), raw or normalized
    x_global: np.ndarray | None = None,        # shape (C*P,) or (C,P)
    title: str | None = None,
    save: str | None = None,
    show: bool = True,
):
    """
    Visualize how the learned per-component mass (sum over P) compares to the
    input orbit_weights used by the ratio penalty.
    """
    # --- read C,P and x_global if needed
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]
        S, C, P, L = map(int, M.shape)
        if x_global is None:
            if "/X_global" not in f:
                raise RuntimeError("No /X_global in HDF5 and x_global not provided.")
            x_global = np.asarray(f["/X_global"][...], dtype=np.float64)

    x = np.asarray(x_global, dtype=np.float64)
    if x.ndim == 1:
        if x.size != C*P:
            raise ValueError(f"x_global has length {x.size}, expected C*P={C*P}.")
        x = x.reshape(C, P)  # (C,P)
    elif x.ndim == 2:
        if x.shape != (C, P):
            raise ValueError(f"x_global shape {x.shape}, expected (C,P)=({C},{P}).")
    else:
        raise ValueError("x_global must be 1-D or 2-D")

    # --- per-component totals and normalization
    eps = 1e-18
    sol_tot = np.maximum(0.0, x).sum(axis=1)       # (C,)
    sol_sum = float(sol_tot.sum()) or 1.0
    sol_pdf = (sol_tot / sol_sum)

    # ------------------- ratio penalty setup (simple) -------------------
    have_ratio = False
    w_full = None  # per-component step scaling

    # If not passed explicitly, read from HDF5 only (no other fallbacks)
    if orbit_weights is None:
        ow_dset = os.environ.get("CUBEFIT_ORBIT_WEIGHTS_DSET",
                                 "/HyperCube/norm/orbit_weights")
        with open_h5(h5_path, role="reader") as f:
            if ow_dset not in f:
                raise RuntimeError(f"orbit_weights requested but dataset "
                                   f"'{ow_dset}' not found in {h5_path}")
            w_in = np.asarray(f[ow_dset][...], dtype=np.float64).ravel(order="C")
    else:
        w_in = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")

    if w_in.size != C:
        raise ValueError(f"orbit_weights length {w_in.size} != C={C}")
    if not np.all(np.isfinite(w_in)):
        raise ValueError("orbit_weights contains non-finite values")
    w_sum = float(np.sum(w_in))
    if w_sum <= 0.0:
        raise ValueError("orbit_weights sum must be > 0")

    # Normalized component prior (probabilities)
    w_c = (w_in / w_sum).astype(np.float64, copy=False)

    # Enable penalty and keep your existing knobs
    have_ratio   = True
    _ratio_eta   = 0.02
    _ratio_prob  = 0.02
    _ratio_batch = 2
    _ratio_minw  = 1e-4
    rng = np.random.default_rng()

    # Per-component step scaling (mean -> 1.0), same semantics you had
    m = float(np.mean(w_c)) or 1.0
    w_full = (w_c / m).astype(np.float64, copy=False)

    def _ratio_update_in_place(x_mat: np.ndarray) -> None:
        s = x_mat.sum(axis=1)
        eps = 1e-12
        active = (w_c >= _ratio_minw) | (s > 0)
        if not np.any(active):
            return
        a = int(np.argmax(w_c * active))
        sa = float(np.max((s[a], eps)))
        wa = float(np.max((w_c[a], eps)))
        pool = np.flatnonzero(active & (np.arange(C) != a))
        if pool.size == 0:
            return
        sel = pool[rng.choice(pool.size, size=np.min((_ratio_batch, pool.size)), replace=False)]
        if sel.size > 1 and _ratio_prob < 1.0:
            keep_mask = rng.random(sel.size) < _ratio_prob
            sel = sel[keep_mask]
            if sel.size == 0:
                return
        for c in sel:
            sc = float(np.max((s[c], eps)))
            rc = float(np.max((w_c[c], eps)))
            e = math.log((sc/sa) / (rc/wa))
            if e == 0.0:
                continue
            delta_sc = -_ratio_eta * e * sc
            delta_sa = +_ratio_eta * e * sa
            pc = x_mat[c, :] / sc
            pa = x_mat[a, :] / sa
            if not np.all(np.isfinite(pc)):
                pc = np.full(P, 1.0 / P, dtype=np.float64)
            if not np.all(np.isfinite(pa)):
                pa = np.full(P, 1.0 / P, dtype=np.float64)
            x_mat[c, :] += delta_sc * pc
            x_mat[a, :] += delta_sa * pa
        if cfg.project_nonneg:
            np.maximum(x_mat, 0.0, out=x_mat)


    # --- diagnostics
    l1 = float(np.sum(np.abs(w_in - sol_pdf)))
    cos = float(np.dot(w_in, sol_pdf) / (np.linalg.norm(w_in) * np.linalg.norm(sol_pdf) + eps))
    # KLs with epsilon-smoothing
    p = np.clip(w_in, eps, 1.0); p /= p.sum()
    q = np.clip(sol_pdf, eps, 1.0); q /= q.sum()
    kl_pq = float(np.sum(p * (np.log(p) - np.log(q))))
    kl_qp = float(np.sum(q * (np.log(q) - np.log(p))))
    print(f"[ratio vs solution] C={C}, P={P}")
    print(f"  L1 distance        : {l1:.4f}")
    print(f"  Cosine similarity  : {cos:.6f}")
    print(f"  KL(p||q), KL(q||p) : {kl_pq:.6f}, {kl_qp:.6f}")

    # --- plotting
    idx = np.arange(C)
    width = 0.45

    fig = plt.figure(figsize=(10, 4.2))
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(idx - width/2, w_in,  width, label="input prior (orbit_weights)")
    ax1.bar(idx + width/2, sol_pdf, width, label="solution ∑_P x[c,p]")
    ax1.set_xlabel("component c")
    ax1.set_ylabel("normalized mass / probability")
    ttl = title or "Component weights: prior vs solution"
    ax1.set_title(ttl)
    ax1.legend(frameon=False, fontsize=9)

    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(w_in, sol_pdf, s=14)
    lim = (0.0, max(1.0/C*5, float(max(w_in.max(), sol_pdf.max()))*1.05))
    ax2.plot(lim, lim, lw=1.0)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("input prior w_in[c]")
    ax2.set_ylabel("solution mass fraction")
    ax2.set_title(f"scatter vs y=x  (cos={cos:.3f}, L1={l1:.3f})")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=140)
        print(f"[saved] {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # return raw arrays if you want to post-process
    return dict(
        prior=w_in, solution=sol_pdf, sol_tot=sol_tot,
        L1=l1, cosine=cos, KL_pq=kl_pq, KL_qp=kl_qp
    )
