# -*- coding: utf-8 -*-
r"""
    hypercube_builder.py
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
v1.1:   Added Zarr v3 sharding support and safe-direct writes. 5 September 2025
v1.2:   Complete re-write to use HDF5. 7 September 2025


Hypercube builder (LOSVD convolution in log-λ, then rebin to observed grid)

Writes /HyperCube/models with shape (S, C, P, L) float32, chunked, resumable.
Requires the HDF5 to already contain:
  /Templates  (P, T) float64          -- flattened populations on native template grid
  /TemPix     (T,)   float64          -- template grid in log-λ (natural log)
  /ObsPix     (L,)   float64          -- observed wavelength grid (not used directly here)
  /R_T        (T,L) or (L,T)          -- rebin operator (stored by H5Manager)
  /LOSVD      (S,V,C) float64         -- per-spaxel LOSVD histograms
  /VelPix     (V,)   float64          -- velocity bins (km/s) corresponding to LOSVD axis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable
import math
import numpy as np
from tqdm import tqdm

from dynamics.IFU.Constants import Constants
from CubeFit.hdf5_manager import open_h5

CTS = Constants()
C_KMS = CTS.c

# ------------------------- small utilities ------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())

def _choose_nfft(T_len: int, k_support: int) -> int:
    """Linear conv length >= T + m - 1; pick power-of-two for speed."""
    L = T_len + k_support - 1
    return _next_pow2(L)

def _make_done_bitmap(f, S: int, C: int, P: int,
                      S_chunk: int, C_chunk: int, P_chunk: int):
    """
    Create or open the /HyperCube/_done bitmap with shape
    (ceil(S/S_chunk), ceil(C/C_chunk), ceil(P/P_chunk)).
    """
    g = f.require_group("/HyperCube")
    grid = (math.ceil(S / S_chunk), math.ceil(C / C_chunk), math.ceil(P / P_chunk))
    if "_done" in g:
        ds = g["_done"]
        if tuple(ds.shape) != grid:
            del g["_done"]
            ds = g.create_dataset("_done", shape=grid, dtype="u1", chunks=True)
    else:
        ds = g.create_dataset("_done", shape=grid, dtype="u1", chunks=True)
    return ds, grid

def _done_get(ds, idx3: Tuple[int, int, int]) -> int:
    try:
        return int(ds[idx3])
    except Exception:
        # some h5py builds dislike 3D scalar reads; slice as 1x1x1
        i, j, k = idx3
        return int(ds[i:i+1, j:j+1, k:k+1][0, 0, 0])

def _done_set(ds, idx3: Tuple[int, int, int]) -> None:
    i, j, k = idx3
    ds[i, j, k] = np.uint8(1)

def _iter_slices(total: int, step: int) -> Iterable[Tuple[int, int]]:
    x = 0
    while x < total:
        y = min(x + step, total)
        yield x, y
        x = y

# ------------------------- LOSVD → pixel-kernel mapping -----------------------

@dataclass(frozen=True)
class KernelMap:
    # Integer pixel offsets on the template grid where the kernel is sampled
    k_offsets: np.ndarray        # (m,) int64
    # For interpolation: left/right indices into VelPix and t in [0,1]
    il: np.ndarray               # (m,) int64
    ir: np.ndarray               # (m,) int64
    t: np.ndarray                # (m,) float64
    # Mask for offsets whose mapped velocity is outside VelPix range
    out_mask: np.ndarray         # (m,) bool
    # Length of kernel support m and centered index (k==0)
    m: int
    center_idx: int

def _precompute_kernel_map(tem_loglam: np.ndarray, vel_pix: np.ndarray) -> KernelMap:
    """
    Precompute mapping from velocity grid (km/s) to integer pixel shifts on the
    template's log-λ grid. Returns interpolation weights to build Hk quickly
    for each LOSVD row without calling np.interp in the inner loop.
    """
    logL = np.asarray(tem_loglam, dtype=np.float64)
    dlog = np.median(np.diff(logL))
    V = np.asarray(vel_pix, dtype=np.float64)
    # Map the full velocity range to pixel shifts
    k_min = math.floor(np.log1p(V.min() / C_KMS) / dlog)
    k_max = math.ceil (np.log1p(V.max() / C_KMS) / dlog)
    k_offsets = np.arange(k_min, k_max + 1, dtype=np.int64)
    # Velocity corresponding to integer pixel shifts
    v_for_k = C_KMS * np.expm1(k_offsets * dlog)

    # Linear interpolation weights: for each v_for_k find bracketing VelPix
    il = np.searchsorted(V, v_for_k, side="right") - 1
    ir = il + 1
    # Out-of-range mask (will be set to 0 contribution)
    out_mask = (il < 0) | (ir >= V.size)
    il = np.clip(il, 0, V.size - 1)
    ir = np.clip(ir, 0, V.size - 1)
    # Fraction t toward the right point
    denom = (V[ir] - V[il])
    denom[denom == 0] = 1.0
    t = (v_for_k - V[il]) / denom
    # Center index where k == 0
    center_idx = int(np.searchsorted(k_offsets, 0))

    return KernelMap(
        k_offsets=k_offsets,
        il=il.astype(np.int64),
        ir=ir.astype(np.int64),
        t=t.astype(np.float64),
        out_mask=out_mask.astype(bool),
        m=k_offsets.size,
        center_idx=center_idx,
    )

def _build_kernel_from_losvd(H_native: np.ndarray, km: KernelMap) -> np.ndarray:
    """
    Given one LOSVD histogram H_native (length V on VelPix grid),
    produce the kernel Hk on integer pixel shifts (length m), normalized and >=0.
    Uses the precomputed linear interpolation weights.
    """
    H = np.asarray(H_native, dtype=np.float64, order="C")
    # Linear interpolation: Hk = (1-t)*H[il] + t*H[ir]
    Hk = (1.0 - km.t) * H[km.il] + km.t * H[km.ir]
    # zero outside the VelPix range
    Hk[km.out_mask] = 0.0
    # Non-negative & unit-area
    np.maximum(Hk, 0.0, out=Hk)
    s = Hk.sum()
    if s > 0:
        Hk /= s
    return Hk  # (m,)


# ------------------------- main builder ---------------------------------------

def build_hypercube(
    base_h5: str,
    *,
    S_chunk: int = 128,
    C_chunk: int = 1,
    P_chunk: int = 360,
    compression: str | None = None,   # keep None for speed; compress later
) -> None:
    """
    Build /HyperCube/models with LOSVD convolution on the template grid
    followed by rebinning to the observed grid.

    Tiles are (S_chunk, C_chunk, P_chunk, L) and resume is supported via /HyperCube/_done.
    """
    # -------- Fast preflight: exit early if fully complete (reader-only)
    with open_h5(base_h5, "reader") as f_rd:
        # core dims (also needed later if we do write)
        P, T = map(int, f_rd["/Templates"].shape)
        S, V, C = map(int, f_rd["/LOSVD"].shape)
        L = int(f_rd["/DataCube"].shape[1])

        g = f_rd.get("/HyperCube", None)
        if g is not None and "models" in g and "_done" in g:
            done = np.asarray(g["_done"][...], dtype=np.uint8)  # small (e.g. few K)
            all_done = (done.size > 0 and int(done.sum()) == int(done.size))
            if bool(g.attrs.get("complete", False)) and all_done:
                print("[HyperCube] already complete; skip build (no writer open).")
                return

        # Read grids/operators/templates once here (reader), reuse later
        tem_loglam = np.asarray(f_rd["/TemPix"][...], dtype=np.float64)
        vel_pix    = np.asarray(f_rd["/VelPix"][...], dtype=np.float64)
        R_any      = np.asarray(f_rd["/R_T"][...])
        if R_any.shape == (T, L):
            R_T = R_any.astype(np.float32, copy=False)
        elif R_any.shape == (L, T):
            R_T = R_any.T.astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"/R_T shape {R_any.shape} incompatible with T={T}, L={L}")
        Templates = np.asarray(f_rd["/Templates"][...], dtype=np.float64, order="C")

    # Precompute velocity→pixel kernel mapping (shared for all (s,c))
    km = _precompute_kernel_map(tem_loglam, vel_pix)
    n_fft = _choose_nfft(T, km.m)

    # Template FFT at n_fft (zero-padded for linear convolution)
    T_fft = np.fft.rfft(Templates, n=n_fft, axis=1).astype(np.complex64, copy=False)  # (P, rfft)

    # -------- Prepare destination dataset and resume bitmap
    with open_h5(base_h5, "writer") as f:
        g = f.require_group("/HyperCube")
        # create models if absent
        if "models" not in g:
            chunks = (min(S_chunk, S), min(C_chunk, C), min(P_chunk, P), L)
            g.create_dataset(
                "models",
                shape=(S, C, P, L),
                dtype="f4",
                chunks=chunks,
                compression=compression or None,
            )
        models = g["models"]
        done, grid = _make_done_bitmap(f, S, C, P, S_chunk, C_chunk, P_chunk)
        # Check how many tiles remain BEFORE enabling SWMR
        # (cheap: _done grid is small)
        done_arr = np.asarray(done[...], dtype=np.uint8)
        remaining = int(done_arr.size - done_arr.sum())
        if remaining <= 0:
            # Stamp/refresh completion and return without enabling SWMR
            g.attrs["complete"] = True
            g.attrs["shape"] = (S, C, P, L)
            g.attrs["chunks"] = models.chunks
            try:
                f.flush()
            except Exception:
                pass
            print("[HyperCube] already complete; nothing to write.")
            return

        # Keep a LOSVD handle for streaming once we *know* we will write
        losvd_ds = f["/LOSVD"]

        # --- SWMR: expose incremental writes to concurrent read-only processes
        try:
            f.flush()            # make sure metadata is on disk
            f.swmr_mode = True   # switch this file handle into SWMR writer mode
            print("[SWMR] writer mode enabled.")
        except Exception as e:
            # If SWMR is not available in this HDF5/h5py build, just continue.
            print(f"[SWMR] could not enable writer mode: {e}")

        # --- Build the worklist (iterate tiles; skip the ones marked done)
        def _iter_all_tiles():
            for p0 in range(0, P, P_chunk):
                p1 = min(P, p0 + P_chunk)
                for s0 in range(0, S, S_chunk):
                    is_s = s0 // S_chunk
                    s1 = min(S, s0 + S_chunk)
                    for c0 in range(0, C, C_chunk):
                        ic = c0 // C_chunk
                        c1 = min(C, c0 + C_chunk)
                        ip = p0 // P_chunk
                        yield (p0, p1, s0, s1, c0, c1, (is_s, ic, ip))

        # Count only remaining tiles for progress bar
        total_tiles = remaining
        pbar = tqdm(total=total_tiles, desc="[HyperCube] tiles", mininterval=2.0)
        
        for (p0, p1, s0, s1, c0, c1, idx3) in _iter_all_tiles():
            if _done_get(done, idx3) != 0:
                continue  # skip completed tile

            # Frequency-domain template slice for this P-block
            T_fft_slice = T_fft[p0:p1, :]  # (ΔP, rfft_len)

            # For each (s,c) in this tile:
            for s_idx in range(s0, s1):
                for c_idx in range(c0, c1):
                    # 1) Build centered kernel FFT for this (s,c)
                    H_native = np.asarray(losvd_ds[s_idx, :, c_idx], dtype=np.float64, order="C")  # (V,)
                    Hk = _build_kernel_from_losvd(H_native, km)  # (m,)
                    h0 = np.roll(Hk, -km.center_idx)             # center zero-lag at index 0
                    H_fft = np.fft.rfft(h0, n=n_fft).astype(np.complex64, copy=False)

                    # 2) Multiply in frequency domain for all ΔP templates
                    Y_fft = T_fft_slice * H_fft[None, :]  # (ΔP, rfft)

                    # 3) Linear convolution via irfft at padded length
                    conv_full = np.fft.irfft(Y_fft, n=n_fft, axis=1)  # (ΔP, n_fft)

                    # 4) Centered 'same' crop on template grid
                    start = int(km.center_idx)
                    stop  = start + T
                    conv_td = conv_full[:, start:stop]
                    if conv_td.shape[1] != T:  # rare edge guard
                        tmp = np.zeros((conv_td.shape[0], T), dtype=conv_td.dtype)
                        w = min(T, max(0, conv_td.shape[1]))
                        tmp[:, :w] = conv_td[:, :w]
                        conv_td = tmp
                    conv_td = conv_td.astype(np.float32, copy=False)  # (ΔP, T)

                    # 5) Rebin to observed grid AFTER convolution; clamp tiny round-off
                    Ycp = conv_td @ R_T  # (ΔP, L)
                    np.maximum(Ycp, 0.0, out=Ycp)

                    # 6) Write
                    models[s_idx, c_idx, p0:p1, 0:L] = Ycp

            _done_set(done, idx3)

            # Make this tile visible to SWMR readers
            try:
                models.id.flush()
            except Exception:
                pass
            try:
                done.id.flush()
            except Exception:
                pass
            try:
                f.flush()
            except Exception:
                pass

            pbar.update(1)

        pbar.close()
        g.attrs["complete"] = True
        g.attrs["shape"] = (S, C, P, L)
        g.attrs["chunks"] = models.chunks