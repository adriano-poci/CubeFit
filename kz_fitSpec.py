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

import os, pdb, math, ctypes
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
    live_prefit_snapshot_from_models
from CubeFit.hypercube_builder import build_hypercube
from CubeFit.pipeline_runner   import PipelineRunner
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
    nProcs=1, lOrder=4, rescale=False, specRange=None, lsf=False,
    band='r', smask=None, method='fsf', varIMF=False,
    source='ppxf', **kwargs):
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

    cont = kwargs.pop('cont', False)

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
        losvd=apHists,
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
    with logger.capture_all_output():
        nS, nC, nP = 128, 1, 360
        build_hypercube(
            hdf5Path,
            S_chunk=nS, C_chunk=nC, P_chunk=nP,
        )

    prefit_png = spDir / f"prefit_overlay_from_models_C{nComp:03d}.png"
    live_prefit_snapshot_from_models(
        h5_path=str(hdf5Path),
        max_components=4,
        templates_per_pair=3,
        out_png=str(prefit_png),
    )
    logger.log(f"[Prefit] wrote {prefit_png}")

    # --- 4) Run the global Kaczmarz fit (tiled; RAM-bounded) ---
    runner = PipelineRunner(hdf5Path)

    x_global, stats = runner.solve_all(
        epochs=1,
        pixels_per_aperture=256,
        lr=0.25,
        project_nonneg=True,
        orbit_weights=cWeights,
        verbose=True,
        warm_start='jacobi',
    )
    logger.log("[CubeFit] Global fit completed.")
    # --- Save the global solution vector
    logger.log("[CubeFit] Saving fit results...")
    with open_h5(hdf5Path, role="writer") as f:
        ds = f.get("/X_global", None)
        if ds is not None and ds.shape == (x_global.size,):
            ds[...] = np.asarray(x_global, np.float64)
        else:
            if "/X_global" in f:
                del f["/X_global"]
            f.create_dataset(
                "/X_global",
                data=np.asarray(x_global, np.float64),
                chunks=(min(8192, x_global.size),),
                compression="gzip", compression_opts=4, shuffle=True,
            )

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
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    # Respect Slurm if present; keep BLAS threads moderate.
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

    with open_h5(h5_path, role="writer") as f:
        M = f["/HyperCube/models"]              # (S,C,P,L) float32
        S, C, P, L = map(int, M.shape)
        chunks = M.chunks or (S, 1, P, L)
        S_chunk, C_chunk, P_chunk, L_chunk = map(int, chunks)

        # Choose S block aligned to storage chunk (or user cap).
        S_blk = S_chunk if S_tile is None else max(1, int(S_tile))
        S_blk = min(S_blk, S_chunk)             # never exceed chunk
        S_blk = min(S_blk, S)                   # clamp to S

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

        # Progress
        n_tiles = math.ceil(S / S_blk)
        print(f"[Recon] S_chunk={S_chunk} C_chunk={C_chunk} "
              f"P_chunk={P_chunk} L_chunk={L_chunk} S_blk={S_blk} "
              f"L_band={L_band}")
        pbar = tqdm(total=n_tiles, desc="[Reconstruct] tiles",
                    unit="tile", dynamic_ncols=True)

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
                    band_acc = np.zeros((dS, Lb), dtype=np.float64,
                                        order="C")

                    # Contract over P in P_chunk steps (exactly aligned)
                    for p0 in range(0, P, max(1, P_chunk)):
                        p1  = min(P, p0 + max(1, P_chunk))
                        Pb  = p1 - p0

                        # Read one storage-aligned slab:
                        # (dS, 1, Pb, Lb) float32
                        A32 = M[s0:s1, c:c1, p0:p1, l0:l1][...]
                        A32 = np.asarray(A32, dtype=np.float32, order="C")

                        # Make a (dS*Lb, Pb) 2-D view for BLAS gemv
                        # by swapping axes (dS, Pb, Lb) → (dS, Lb, Pb)
                        A2D = A32[:, 0, :, :].swapaxes(1, 2) \
                                           .reshape(dS * Lb, Pb)

                        # Multiply by weights for this (c, p-block) in f32
                        w32 = x32[c, p0:p1]           # (Pb,)
                        tmp = A2D @ w32               # (dS*Lb,) f32

                        # Accumulate into f64 band
                        band_acc += tmp.reshape(dS, Lb).astype(np.float64,
                                                               copy=False)

                    # Add this component's band into the tile
                    Y_tile[:, l0:l1] += band_acc

            # Write the finished S-tile
            out[s0:s1, :] = Y_tile
            trim_heap()
            pbar.update(1)

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
      - Reads only the needed rows from /DataCube and /ModelCube.
      - Never touches /HyperCube/models.
      - Closes every figure immediately.
      - Keeps thread pool tiny (I/O bound).
    """
    plp.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    n = int(max(1, n))
    chi2 = np.asarray(chi2, dtype=np.float64)
    S = chi2.shape[0]

    # Pick indices
    idx_worst = np.argsort(-chi2)[:n]
    idx_best  = np.argsort( chi2)[:n]
    picks     = np.unique(np.concatenate([idx_worst, idx_best])).astype(int)


    # Read only those spaxels (rows) from /DataCube and /ModelCube
    with open_h5(str(h5_or_path), role="reader") as f:
        # a modest raw chunk cache keeps mem small but avoids metadata churn
        if "/ModelCube" not in f:
            raise RuntimeError("Expected /ModelCube (S,L) for plotting. Reconstruct first.")
        data_ds  = f["/DataCube"]    # (S,L) float64 (or f32)
        model_ds = f["/ModelCube"]   # (S,L) float64
        obs      = f["/ObsPix"][...] if "/ObsPix" in f else np.arange(model_ds.shape[1])
        L = int(model_ds.shape[1])

        print(f"[Plots] picks={picks.size} L={L} mem≈{picks.size*L*8*2/1e6:.1f} MB for data+model rows")
        if mask is None:
            mask = np.ones(L, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape[0] != L:
                raise ValueError(f"Mask length {mask.shape[0]} != L={L}")

        # Pull just the rows we need into compact arrays
        data_sel  = np.empty((picks.size, L), dtype=np.float64)
        model_sel = np.empty((picks.size, L), dtype=np.float64)
        for j, s in enumerate(picks):
            # These slices are small; using [...] is fine here
            data_sel[j, :]  = data_ds[s, :]
            model_sel[j, :] = model_ds[s, :]

    # Small plotting worker: takes *row views*, not whole cubes
    def _plot_one(s_idx: int, rank_tag: str):
        j   = int(np.where(picks == s_idx)[0][0])
        dat = data_sel[j, :]
        mod = model_sel[j, :]

        fig = plt.figure(figsize=(8, 3.5))
        ax  = fig.add_subplot(111)
        ax.plot(obs[mask], dat[mask], lw=1.0, label="data")
        ax.plot(obs[mask], mod[mask], lw=1.0, alpha=0.85, label="model")
        ax.set_title(f"spaxel {s_idx}  χ={chi2[s_idx]:.3f}")
        ax.set_xlabel("λ (log space)")
        ax.set_ylabel("flux")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{rank_tag}_{tag}_spax{int(s_idx):05d}.png"), dpi=120)
        plt.close(fig)  # critical: free figure memory immediately

    # Launch a tiny pool; 2–4 is plenty
    pool_n = max(1, min(n_workers, 4))
    jobs = [(int(s), "worst") for s in idx_worst] + [(int(s), "best") for s in idx_best]
    with ThreadPoolExecutor(max_workers=pool_n) as pool:
        list(pool.map(lambda args: _plot_one(*args), jobs))

    # Help GC release the small arrays (paranoid but cheap)
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
    source='ppxf', pplots=['spec', 'mw'], redraw=False, **kwargs):
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

    cont = kwargs.get('cont', False)

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
    arSOL = x_global.reshape(nComp, nMetals, nAges, nAlphas)
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