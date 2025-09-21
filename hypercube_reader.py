# -*- coding: utf-8 -*-
r"""
    hypercube_reader.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Manages Zarr data storage for CubeFit pipeline, including creation, loading,
    and validation of large, chunked arrays (templates, data cube, LOSVD, weights).
    Supports buffered template grids for safe convolution.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   Initial design and validation. 14 August 2025
v1.1:   Complete re-write to use HDF5. 7 September 2025

Hypercube reader that streams planes for Kaczmarz solving.

Contract with the solver:
  - read_spaxel_plane(s) -> (A, y)
      A: (N, L_eff) float32   with N = C*P
      y: (L_eff,)   float64   observed spectrum
  - Mask (if present at /Mask) is applied to BOTH A and y (True = keep)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple

import numpy as np

from CubeFit.hdf5_manager import open_h5


@dataclass
class ReaderCfg:
    s_tile: Optional[int] = None
    c_tile: Optional[int] = None
    p_tile: Optional[int] = None
    dtype_models: str = "float32"   # how to read models; math stays float64
    apply_mask: bool = True


class HyperCubeReader:
    """
    Minimal, robust reader for /HyperCube/models.
    Provides tiles and (s,c) planes as float32 (cast to float64 in solver).
    """

    def __init__(self, h5_path: str, cfg: ReaderCfg | None = None):
        self._ctx = open_h5(h5_path, "reader")   # keep the context manager
        self._f = self._ctx.__enter__()          # get the h5py.File
        self._closed = False

        f = self._f
        if "/HyperCube/models" not in f:
            roots = list(f.keys())
            try:
                hc = list(f["/"].keys())
            except Exception:
                hc = roots
            path = getattr(f, "filename", h5_path)
            raise RuntimeError(
                f"Missing /HyperCube/models in '{path}'. "
                f"Root groups present: {hc}. Did you build the same file?"
            )

        self._models = f["/HyperCube/models"]             # (S,C,P,L)
        self._done   = f["/HyperCube/_done"]              # (Sgrid,Cgrid,Pgrid)
        self._Y      = f["/DataCube"]                     # (S,L)
        self._Mask   = f["/Mask"] if "/Mask" in f else None

        self.nSpat, self.nComp, self.nPop, self.nLSpec = map(int, self._models.shape)
        self._chunks = self._models.chunks or (self.nSpat, 1, self.nPop, self.nLSpec)

        # config
        cfg = cfg or ReaderCfg()
        self._model_dtype = np.dtype(cfg.dtype_models).type
        self._apply_mask = bool(cfg.apply_mask)

        # default tile sizes fall back to storage chunks
        self.s_tile = int(cfg.s_tile or self._chunks[0])
        self.c_tile = int(cfg.c_tile or self._chunks[1])
        self.p_tile = int(cfg.p_tile or self._chunks[2])

        # cache mask as boolean (if present)
        self._mask = None
        if self._Mask is not None and self._apply_mask:
            self._mask = np.asarray(self._Mask[...], dtype=bool)

    # ----------------------------- lifecycle ---------------------------------

    def close(self):
        if not self._closed:
            try:
                self._ctx.__exit__(None, None, None)
            finally:
                self._closed = True

    # ----------------------------- helpers -----------------------------------

    def spaxel_tiles(self) -> Iterator[Tuple[int, int]]:
        """Yield (s0, s1) ranges across S using s_tile."""
        S = self.nSpat
        for s0 in range(0, S, self.s_tile):
            s1 = min(s0 + self.s_tile, S)
            yield s0, s1

    def _read_obs_row(self, s: int) -> np.ndarray:
        return np.asarray(self._Y[s, :], dtype=np.float64)

    # ----------------------------- public API --------------------------------

    def read_spaxel_plane(self, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (A, y) for a single spaxel:
            A: (N=LHS rows = C*P,   L columns) in dtype_models
            y: (L,) float64
        Mask (if present and apply_mask=True) is applied at the end.
        """
        if not (0 <= s < self.nSpat):
            raise IndexError(f"spaxel {s} out of bounds [0,{self.nSpat})")

        C, P, L = self.nComp, self.nPop, self.nLSpec
        slab = np.asarray(self._models[s, :, :, :], dtype=self._model_dtype, order="C")  # (C,P,L)
        A = slab.reshape(C * P, L)
        y = self._read_obs_row(s)

        if self._mask is not None:
            A = A[:, self._mask]
            y = y[self._mask]
        return A, y

    def read_spaxel_range(self, s0: int, s1: int, *, flatten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (A_slab, Y_slab) for s in [s0, s1):
            if flatten=True:
                A_slab: (ΔS, N=C*P, L) in dtype_models
                Y_slab: (ΔS, L)       in float64
            else:
                A_slab: (ΔS, C, P, L), Y_slab: (ΔS, L)
        """
        if not (0 <= s0 < s1 <= self.nSpat):
            raise IndexError(f"spaxel range [{s0},{s1}) invalid for S={self.nSpat}")

        C, P, L = self.nComp, self.nPop, self.nLSpec
        dS = s1 - s0

        # models slab (ΔS, C, P, L)
        A = np.empty((dS, C, P, L), dtype=self._model_dtype)
        for k, s in enumerate(range(s0, s1)):
            A[k, :, :, :] = np.asarray(self._models[s, :, :, :], dtype=self._model_dtype, order="C")

        # observed slab (ΔS, L)
        Y = np.empty((dS, L), dtype=np.float64)
        for k, s in enumerate(range(s0, s1)):
            Y[k, :] = self._read_obs_row(s)

        if self._mask is not None:
            A = A[..., self._mask]
            Y = Y[..., self._mask]

        if flatten:
            return A.reshape(dS, C * P, A.shape[-1]), Y
        else:
            return A, Y
