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
from typing import Iterable, Optional, Dict, Any, List, Tuple
import os, math
import numpy as np
from tqdm import tqdm

from dynamics.IFU.Constants import Constants
from CubeFit.hdf5_manager import open_h5
from CubeFit.logger import get_logger

logger = get_logger()

CTS = Constants()
C_KMS = CTS.c

# ------------------------- small utilities ------------------------------------

class _P2Median:
    """
    Streaming P² median estimator (Jain & Chlamtac, 1985) with ~5 markers.

    Stores only O(1) state per tracked series; good accuracy for large S.
    """
    __slots__ = ("_n", "_q", "_nq", "_dn")

    def __init__(self) -> None:
        self._n = 0
        self._q = np.zeros(5, dtype=np.float64)
        self._nq = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        self._dn = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """
        Vectorized: accept a batch of scalars as a 1D array.
        Initializes each series lazily on first five samples.
        """
        # For simplicity with many series, we call per-scalar here.
        # The outer code will call update() with scalars in a tight loop.
        for v in np.ravel(x):
            n = self._n
            if n < 5:
                self._q[n] = float(v)
                self._n += 1
                if self._n == 5:
                    self._q.sort()
                continue
            # Find cell k such that q[k] <= v < q[k+1]
            if v < self._q[0]:
                self._q[0] = v
                k = 0
            elif v >= self._q[4]:
                self._q[4] = v
                k = 3
            else:
                k = int(np.searchsorted(self._q, v)) - 1
            self._nq[:5] += self._dn
            self._nq[k+1:5] += 1.0
            # Desired marker positions for median (p=0.5)
            m = np.array([0, 0.5*(self._n), 1*self._n, 1.5*(self._n), 2*self._n],
                         dtype=np.float64) / 2.0
            self._n += 1
            # Adjust interior markers (1..3)
            for i in (1, 2, 3):
                d = m[i] - self._nq[i]
                if (d >= 1 and self._nq[i+1] - self._nq[i] > 1) or \
                   (d <= -1 and self._nq[i-1] - self._nq[i] < -1):
                    d = np.sign(d)
                    qi = self._q[i]
                    qip = self._q[i+1]
                    qim = self._q[i-1]
                    di = (self._q[i+1] - qi)/(self._nq[i+1] - self._nq[i]) if self._nq[i+1] != self._nq[i] else 0.0
                    dm = (qi - self._q[i-1])/(self._nq[i] - self._nq[i-1]) if self._nq[i] != self._nq[i-1] else 0.0
                    qnew = qi + d*((self._nq[i] - self._nq[i-1] + d)*dm +
                                   (self._nq[i+1] - self._nq[i] - d)*di) / \
                                   (self._nq[i+1] - self._nq[i-1])
                    # If monotonicity breaks, fall back to linear.
                    if not (qim <= qnew <= qip):
                        qnew = qi + d * (self._q[i + int(d)] - qi) / \
                               (self._nq[i + int(d)] - self._nq[i])
                    self._q[i] = qnew
                    self._nq[i] += d

    def median(self) -> float:
        n = self._n
        if n == 0:
            return 0.0
        if n <= 5:
            return float(np.median(self._q[:n]))
        return float(self._q[2])


def _ensure_mask_indices(f, L: int) -> Optional[np.ndarray]:
    """
    Return wavelength mask indices (int64) if /Mask exists and is same L.
    Otherwise None (use full λ range).
    """
    if "/Mask" in f:
        m = np.asarray(f["/Mask"][...], dtype=bool, order="C")
        if m.ndim == 1 and m.size == L:
            return np.nonzero(m)[0].astype(np.int64)
    return None

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

# ------------------------- normalization helpers -----------------------------

def _ensure_norm_group(f):
    """
    Create and return /HyperCube/norm group. Never compress tiny norm arrays.
    """
    g = f.require_group("/HyperCube")
    ng = g.require_group("norm")
    return ng

def _compute_and_store_losvd_amplitudes(f, *, amp_mode: str) -> tuple[str, str]:
    """
    One streaming pass over /LOSVD to compute:
      A[s,c]     := LOSVD amplitude per (spaxel, component)
      A_sum[s]   := sum_c A[s,c]   (per-spaxel total)

    Stores datasets and returns their HDF5 paths.

    amp_mode:
      "sum"   -> amplitude = sum(H_native)
      "trapz" -> amplitude = ∫ H_native dV   (requires /VelPix)
    """
    S, V, C = map(int, f["/LOSVD"].shape)
    ng = _ensure_norm_group(f)

    if "losvd_amp" in ng:
        del ng["losvd_amp"]
    if "losvd_amp_sum" in ng:
        del ng["losvd_amp_sum"]

    A = ng.create_dataset("losvd_amp", shape=(S, C), dtype="f8",
                          chunks=(min(64, S), min(32, C)))
    A_sum = ng.create_dataset("losvd_amp_sum", shape=(S,), dtype="f8",
                              chunks=(min(64, S),))

    if amp_mode == "trapz":
        vel_pix = np.asarray(f["/VelPix"][...], np.float64)
    else:
        vel_pix = None

    # stream over spaxels to bound RAM
    losvd = f["/LOSVD"]
    step = max(1, 64)
    for s0 in range(0, S, step):
        s1 = min(S, s0 + step)
        slab = np.asarray(losvd[s0:s1, :, :], np.float64, order="C")  # (dS,V,C)
        if amp_mode == "trapz":
            a = np.trapz(slab, vel_pix, axis=1)                       # (dS,C)
        else:
            a = slab.sum(axis=1)                                       # (dS,C)
        A[s0:s1, :] = a
        A_sum[s0:s1] = a.sum(axis=1)
    try:
        A.id.flush(); A_sum.id.flush(); f.flush()
    except Exception:
        pass

    return (A.name, A_sum.name)

# ------------------------------------------------------------------------------
# ------------------------- canonical kernel/conv helpers ----------------------

def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())

@dataclass(frozen=True)
class _KM:
    k_offsets: np.ndarray
    il: np.ndarray
    ir: np.ndarray
    t: np.ndarray
    out_mask: np.ndarray
    center_idx: int
    m: int
    dlog: float

def _kernel_map_from_grids(tem_loglam: np.ndarray,
                           vel_pix: np.ndarray) -> _KM:
    logL = np.asarray(tem_loglam, dtype=np.float64)
    dlog = float(np.median(np.diff(logL)))
    V = np.asarray(vel_pix, dtype=np.float64)

    k_min = int(np.floor(np.log1p(np.min(V) / C_KMS) / dlog))
    k_max = int(np.ceil (np.log1p(np.max(V) / C_KMS) / dlog))
    k_offsets = np.arange(k_min, k_max + 1, dtype=np.int64)

    v_for_k = C_KMS * np.expm1(k_offsets * dlog)

    il = np.searchsorted(V, v_for_k, side="right") - 1
    ir = il + 1
    out_mask = (il < 0) | (ir >= V.size)
    il = np.clip(il, 0, V.size - 1)
    ir = np.clip(ir, 0, V.size - 1)

    denom = (V[ir] - V[il]).astype(np.float64, copy=False)
    denom[denom == 0.0] = 1.0
    t = (v_for_k - V[il]) / denom

    center_idx = int(np.searchsorted(k_offsets, 0))

    return _KM(
        k_offsets=k_offsets,
        il=il.astype(np.int64, copy=False),
        ir=ir.astype(np.int64, copy=False),
        t=t.astype(np.float64, copy=False),
        out_mask=out_mask.astype(bool, copy=False),
        center_idx=center_idx,
        m=int(k_offsets.size),
        dlog=dlog,
    )

def _losvd_to_unit_kernel(H_native: np.ndarray, km: _KM) -> np.ndarray:
    H = np.asarray(H_native, dtype=np.float64, order="C")
    Hk = (1.0 - km.t) * H[km.il] + km.t * H[km.ir]
    Hk[km.out_mask] = 0.0
    np.maximum(Hk, 0.0, out=Hk)
    s = float(np.sum(Hk))
    if s == 0.0:
        Hk.fill(0.0)
        Hk[int(km.center_idx)] = 1.0
    else:
        Hk /= s
    return Hk

def _fft_conv_centered(T_fft_slice: np.ndarray,
                       Hk_unit: np.ndarray,
                       km: _KM,
                       n_fft: int,
                       T: int,
                       phase_shift: np.ndarray | None = None) -> np.ndarray:
    """
    FFT-based centered convolution of templates with a unit-area LOSVD kernel,
    followed by a 'same' crop aligned on km.center_idx.

    If `phase_shift` is provided (shape (rfft_len,), complex128), the output is
    additionally shifted by a fractional number of samples `shift_T` via the
    Fourier shift theorem (i.e., multiply the kernel FFT by the phase).
    This implements a global velocity bias without any extra FFTs.

    Parameters
    ----------
    T_fft_slice : (ΔP, rfft_len) complex128
        rFFT of the selected template rows (already zero-padded to n_fft).
    Hk_unit : (m,) float64
        Unit-area kernel on the template grid for a given (s,c).
    km : _KM
        Kernel map (contains center_idx and support length).
    n_fft : int
        FFT length used for the linear convolution.
    T : int
        Template length (pre-convolution).
    phase_shift : (rfft_len,) complex128 or None
        Optional precomputed exp(-2πi * f * shift_T). If None, no shift.

    Returns
    -------
    (ΔP, T) float64
        Convolved (and optionally globally shifted) templates, cropped in the
        centered 'same' sense.
    """
    H = np.maximum(np.asarray(Hk_unit, np.float64, order="C"), 0.0)
    s = float(np.sum(H))
    if s == 0.0:
        H[:] = 0.0
        H[int(km.center_idx)] = 1.0
    else:
        H /= s

    # Kernel FFT (complex128). Keep precision parity with template FFT.
    H_fft = np.fft.rfft(H, n=int(n_fft))

    # Apply optional fractional shift (global Δv) in frequency domain.
    if phase_shift is not None:
        H_fft *= phase_shift  # elementwise, shape (rfft_len,)

    # Standard conv in frequency domain and inverse transform
    Y_fft = T_fft_slice * H_fft[None, :]           # (ΔP, rfft_len)
    conv_f = np.fft.irfft(Y_fft, n=int(n_fft), axis=1)

    # Centered 'same' crop
    start = int(km.center_idx)
    stop  = int(start + T)
    return conv_f[:, start:stop].astype(np.float64, copy=False)

def _direct_centered_conv(Templates: np.ndarray,
                          Hk_unit: np.ndarray,
                          km: _KM) -> np.ndarray:
    X = np.asarray(Templates, np.float64, order="C")
    H = np.maximum(np.asarray(Hk_unit, np.float64, order="C"), 0.0)
    s = float(np.sum(H))
    if s == 0.0:
        H[:] = 0.0
        H[int(km.center_idx)] = 1.0
    else:
        H /= s

    dP, T = X.shape
    y = np.zeros((dP, T), dtype=np.float64)
    for k in range(int(km.m)):
        shift = int(k) - int(km.center_idx)
        if shift >= 0:
            if shift < T:
                y[:, shift:] += H[k] * X[:, :T-shift]
        else:
            sh = -shift
            if sh < T:
                y[:, :T-sh] += H[k] * X[:, sh:]
    return y

def _flat_response_errors(Hk_unit: np.ndarray,
                          km: _KM,
                          n_fft: int,
                          T: int) -> Tuple[float, float, float]:
    """
    Return (err_valid, err_left, err_right) for conv(1, Hk) on the
    centered 'same' crop. Only the interior 'valid' region can be ~1.
    """
    H = np.maximum(np.asarray(Hk_unit, np.float64, order="C"), 0.0)
    s = float(np.sum(H))
    if s == 0.0:
        H[:] = 0.0
        H[int(km.center_idx)] = 1.0
    else:
        H /= s

    flat = np.ones((1, int(T)), dtype=np.float64)
    T_fft = np.fft.rfft(flat, n=int(n_fft), axis=1)
    H_fft = np.fft.rfft(H, n=int(n_fft))  # keep complex128 for precision
    flat_full = np.fft.irfft(T_fft * H_fft[None, :], n=int(n_fft), axis=1)

    start = int(km.center_idx)
    stop = start + int(T)
    y = flat_full[:, start:stop].ravel()

    # Fully overlapped interior indices for the centered crop:
    # n in [ (m-1)-c , T-1 - c ]  (inclusive), clipped to [0, T-1]
    lo = int(np.maximum(0, (km.m - 1) - km.center_idx))
    hi = int(np.minimum(T - 1, (T - 1) - km.center_idx))
    if lo <= hi:
        err_valid = float(np.max(np.abs(y[lo:hi+1] - 1.0)))
    else:
        err_valid = float(np.nan)  # no interior exists (pathological)

    err_left = float(np.max(np.abs(y[:lo] - 1.0))) if lo > 0 else 0.0
    err_right = float(np.max(np.abs(y[hi+1:] - 1.0))) if hi < (T - 1) else 0.0
    return err_valid, err_left, err_right

def _xcorr_int_shift(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.asarray(a, np.float64).ravel()
    bb = np.asarray(b, np.float64).ravel()
    T = int(aa.size)
    corr = np.correlate(aa, bb, mode="full")
    return int(np.argmax(corr) - (T - 1))

def preflight_hypercube_convolution(
    h5_path: str,
    s_list: Optional[Iterable[int]] = None,
    c_list: Optional[Iterable[int]] = None,
    p_list: Optional[Iterable[int]] = None,
    max_spax: int = 3,
    max_comp: int = 2,
    max_pop: int = 6,
    tol_rel: float = 2e-3,
    tol_shift_px: float = 0.5,
    tol_flat_valid: float = 3e-8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Sanity-check the convolution path on a tiny subset *before* building
    the full HyperCube. It proves that the FFT-based convolution (no
    roll + centered crop) exactly matches a direct time-domain
    reference, and that the kernel is unit-area (flat response ~ 1).

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 input. Must contain /Templates, /TemPix,
        /LOSVD, /VelPix, and /R_T (for a flat-response rebin sanity).
    s_list, c_list, p_list : iterable of int, optional
        Explicit indices to test. If any are None, the function picks
        small initial slices from the available ranges.
    max_spax, max_comp, max_pop : int
        Caps used when lists are not provided.
    tol_rel : float
        Relative L2 tolerance for FFT≡Direct on the template grid.
    tol_shift_px : float
        Allowed integer-lag shift (pixels) from cross-correlation.

    Returns
    -------
    out : dict
        Summary with fields:
          - 'passes': list[bool] per triple (s, c, p)
          - 'triples': list[tuple] of tested indices
          - 'rel_err': list[float] relative errors
          - 'shift_px': list[int] integer-lag shifts
          - 'flat_err': list[float] flat-response errors
          - 'all_pass': bool, True if all triples passed
          - 'rt_flat_check': float, max|R_T @ 1 - 1| on obs grid

    Exceptions
    ----------
    RuntimeError
        If required datasets are missing or have incompatible shapes.

    Examples
    --------
    >>> res = preflight_hypercube_convolution("NGC4365_01.h5",
    ...                                       s_list=[24,25,26],
    ...                                       c_list=[124,197],
    ...                                       p_list=[42,44,88,89])
    >>> res["all_pass"]
    True
    """
    with open_h5(h5_path, role="reader") as f:
        if "/Templates" not in f or "/TemPix" not in f:
            raise RuntimeError("Missing /Templates or /TemPix.")
        if "/LOSVD" not in f or "/VelPix" not in f:
            raise RuntimeError("Missing /LOSVD or /VelPix.")
        if "/R_T" not in f:
            raise RuntimeError("Missing /R_T for flat rebin sanity.")

        Templates = np.asarray(f["/Templates"][...], np.float64, order="C")
        TemPix = np.asarray(f["/TemPix"][...], np.float64)
        LOSVD = np.asarray(f["/LOSVD"][...], np.float64, order="C")
        VelPix = np.asarray(f["/VelPix"][...], np.float64)

        P, T = map(int, Templates.shape)
        S, V, C = map(int, LOSVD.shape)

        R_any = np.asarray(f["/R_T"][...])
        L = int(f["/DataCube"].shape[1])
        if R_any.shape == (T, L):
            R_T = R_any.astype(np.float32, copy=False)
        elif R_any.shape == (L, T):
            R_T = R_any.T.astype(np.float32, copy=False)
        else:
            raise RuntimeError("Incompatible /R_T shape.")

    s_sel = list(s_list) if s_list is not None else list(range(0, int(np.minimum(max_spax, S))))
    c_sel = list(c_list) if c_list is not None else list(range(0, int(np.minimum(max_comp, C))))
    p_sel = list(p_list) if p_list is not None else list(range(0, int(np.minimum(max_pop, P))))

    km = _kernel_map_from_grids(TemPix, VelPix)

    p_arr = np.array(p_sel, dtype=np.int64)
    T_slice = Templates[p_arr, :]  # (ΔP, T)

    triples: List[Tuple[int, int, int]] = []
    rel_err: List[float] = []
    shift_px: List[int] = []
    flat_valid: List[float] = []
    flat_left: List[float] = []
    flat_right: List[float] = []
    passes: List[bool] = []

    # Rebin flatness on obs grid
    ones_T = np.ones((1, T), dtype=np.float64)
    flat_obs = ones_T @ R_T
    rt_flat_check = float(np.max(np.abs(flat_obs - 1.0)))

    for s in s_sel:
        for c in c_sel:
            H = LOSVD[s, :, c]
            Hk = _losvd_to_unit_kernel(H, km)

            n_fft = _next_pow2(int(T) + int(km.m) - 1)
            T_fft = np.fft.rfft(T_slice, n=int(n_fft), axis=1)

            y_fft = _fft_conv_centered(T_fft, Hk, km, n_fft, int(T))
            y_dir = _direct_centered_conv(T_slice, Hk, km)

            a = y_fft[0]
            b = y_dir[0]
            num = float(np.linalg.norm(a - b))
            den = float(np.maximum(np.linalg.norm(b), 1.0e-30))
            rel = num / den
            sh = _xcorr_int_shift(a, b)
            fr_valid, fr_left, fr_right = _flat_response_errors(Hk, km, n_fft, int(T))

            ok = (rel <= float(tol_rel)) and \
                 (abs(sh) <= float(tol_shift_px)) and \
                 (np.isnan(fr_valid) or (fr_valid <= float(tol_flat_valid)))

            triples.append((int(s), int(c), int(p_arr[0])))
            rel_err.append(rel)
            shift_px.append(int(sh))
            flat_valid.append(fr_valid)
            flat_left.append(fr_left)
            flat_right.append(fr_right)
            passes.append(bool(ok))

            if verbose:
                # expected pixel shift from LOSVD first moment
                Hpos = np.maximum(H, 0.0)
                denom = float(np.trapezoid(Hpos, VelPix))
                if denom > 0.0:
                    mu_v = float(np.trapezoid(Hpos * VelPix, VelPix) / denom)
                    exp_px = float(np.log1p(mu_v / 299792.458) / km.dlog)
                else:
                    exp_px = 0.0
                print(
                    "[preflight] s={:4d} c={:3d}  rel(FFT≡DIR)={:6.2e}  "
                    "shift={:+4.0f}px  flat_valid={:6.2e}  "
                    "edgeL={:6.2e} edgeR={:6.2e}  exp_px={:+6.2f}  ok={}"
                    .format(s, c, rel, sh, fr_valid, fr_left, fr_right, exp_px, ok)
                )

    out = dict(
        passes=passes,
        triples=triples,
        rel_err=rel_err,
        shift_px=shift_px,
        flat_valid=flat_valid,
        flat_edge_left=flat_left,
        flat_edge_right=flat_right,
        all_pass=bool(np.all(np.asarray(passes, bool))),
        rt_flat_check=rt_flat_check,
    )
    return out

# ------------------------------------------------------------------------------

def estimate_global_velocity_bias_prebuild(h5_path: str,
                                           n_spax: int = 96,
                                           n_features: int = 24,
                                           window_len: int = 31,
                                           lag_px: int = 12,
                                           use_mask: bool = True,
                                           amp_mode: str = "trapz",
                                           progress: bool = True) -> dict:
    """
    Estimate a single global velocity bias (Δv in km/s) *before* building the
    hypercube, by cross-correlating a synthesized reference spectrum (built
    directly from Templates+LOSVD+R_T) against the observed data.

    Method:
      1) Build a global reference model M_ref on the observed grid:
         - Compute per-component mean LOSVD kernels by stacking LOSVD over
           spaxels with amplitude weights (trapz or sum).
         - Convolve the average template with each mean kernel on the template
           grid, then rebin to the observed grid via R_T.
         - Weight components by global LOSVD amplitudes and sum.
      2) Continuum-normalize via a running-average subtraction (odd window).
      3) Select the deepest absorption features from the reference only; form a
         boolean mask over ±window_len//2 around each selected feature.
      4) FFT cross-correlate each masked, detrended data spectrum with the
         masked, detrended reference over small lags ±lag_px; collect per-spaxel
         integer-pixel shifts and take the median → global shift; convert to km/s.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 with /Templates, /TemPix, /LOSVD, /VelPix, /R_T, /DataCube,
        /ObsPix, and optionally /Mask.
    n_spax : int
        Number of spaxels to sample evenly across S.
    n_features : int
        Number of deepest features (from reference) to keep.
    window_len : int
        Odd length for running-average baseline and feature half-width*2+1.
    lag_px : int
        Max tested shift (±lag_px pixels) around zero.
    use_mask : bool
        Apply /Mask when present.
    amp_mode : {"trapz","sum"}
        LOSVD amplitude mode for global weighting.
    progress : bool
        Show tqdm progress bars.

    Returns
    -------
    dict
        {
          "vel_bias_kms": float,
          "shift_px": float,
          "median_px": float,
          "mad_px": float,
          "n_spax_used": int,
          "L_eff": int,
          "dlog_obs_med": float,
          "n_features": int,
          "window_len": int,
          "quality": "ok" | "warn",
        }
    """
    # ---------- read static arrays into RAM ----------
    with open_h5(h5_path, role="reader") as f:
        Templates = np.asarray(f["/Templates"][...], dtype=np.float64, order="C")
        TemPix    = np.asarray(f["/TemPix"][...],    dtype=np.float64)
        LOSVD     = np.asarray(f["/LOSVD"][...],     dtype=np.float64, order="C")
        VelPix    = np.asarray(f["/VelPix"][...],    dtype=np.float64)
        R_any     = np.asarray(f["/R_T"][...])
        ObsPix    = np.asarray(f["/ObsPix"][...],    dtype=np.float64)

        S, V, C = LOSVD.shape
        P, T    = Templates.shape
        L       = int(f["/DataCube"].shape[1])

        if R_any.shape == (T, L):
            R_T = R_any.astype(np.float64, copy=False)
        elif R_any.shape == (L, T):
            R_T = R_any.T.astype(np.float64, copy=False)
        else:
            raise RuntimeError("Incompatible /R_T shape vs Templates/DataCube.")

        mask = None
        if use_mask and "/Mask" in f:
            m = np.asarray(f["/Mask"][...], dtype=bool).ravel()
            if m.size == L and np.any(m):
                mask = m

    if mask is None:
        keep_idx = None
        L_eff = int(L)
    else:
        keep_idx = np.flatnonzero(mask)
        L_eff = int(keep_idx.size)

    # ---------- build global reference spectrum ----------
    Tbar = np.mean(Templates, axis=0).astype(np.float64, copy=False)

    if amp_mode == "trapz":
        dH = np.trapezoid(LOSVD, VelPix, axis=1)   # (S, C)
    else:
        dH = np.sum(LOSVD, axis=1)             # (S, C)

    dH = np.maximum(dH, 0.0)
    H_sum_s = np.sum(dH, axis=1)               # (S,)
    H_sum_s = np.where(H_sum_s > 0.0, H_sum_s, 1.0)
    w_s = H_sum_s / np.sum(H_sum_s)

    w_c = np.sum(dH * w_s[:, None], axis=0)    # (C,)
    w_c = np.where(np.sum(w_c) > 0.0, w_c / np.sum(w_c), w_c)

    km = _kernel_map_from_grids(TemPix, VelPix)
    n_fft = 1 << (int(2 * T + km.m - 1 - 1).bit_length())

    Tbar_fft = np.fft.rfft(Tbar[None, :], n=int(n_fft), axis=1)  # (1, rfft_len)

    M_ref = np.zeros((L,), dtype=np.float64)
    for c in range(int(C)):
        Hc = np.tensordot(w_s, LOSVD[:, :, c], axes=(0, 0))      # (V,)
        Hk = _losvd_to_unit_kernel(Hc, km)                       # (m,)
        yT = _fft_conv_centered(Tbar_fft, Hk, km, int(n_fft), int(T)).ravel()
        Mc = yT @ R_T                                            # (L,)
        M_ref += w_c[c] * np.maximum(Mc, 0.0)

    if keep_idx is not None:
        M_ref = M_ref[keep_idx]

    # ---------- continuum removal & feature mask ----------
    def _detrend(y: np.ndarray, win: int) -> np.ndarray:
        k = int(np.maximum(1, win))
        if (k % 2) == 0:
            k = int(k + 1)
        ker = np.ones((k,), dtype=np.float64) / np.float64(k)
        baseline = np.convolve(y, ker, mode="same")
        return y - baseline

    M_ref_dt = _detrend(M_ref, int(window_len))

    order = np.argsort(M_ref_dt)  # deepest (most negative) first
    n_pick = int(np.minimum(int(n_features), M_ref_dt.size))
    pick = order[:n_pick]

    half = int(window_len // 2)
    mask_feat = np.zeros((L_eff,), dtype=bool)
    for p in pick:
        a = int(np.maximum(0, p - half))
        b = int(np.minimum(L_eff, p + half + 1))
        mask_feat[a:b] = True

    M_ref_f = np.where(mask_feat, M_ref_dt, 0.0)

    # ---------- per-spaxel xcorr, file open DURING the loop ----------
    if n_spax >= S:
        s_sel = np.arange(S, dtype=np.int64)
    else:
        s_sel = np.linspace(0, S - 1, int(n_spax), dtype=np.int64)

    n_corr = 1 << int((2 * L_eff - 1 - 1).bit_length())

    Mr = M_ref_f - np.mean(M_ref_f)
    nr = np.linalg.norm(Mr)
    Mr = Mr / (nr if nr > 0.0 else 1.0)

    pad_ref = np.zeros((n_corr,), dtype=np.float64)
    pad_ref[:L_eff] = Mr
    F_ref = np.fft.rfft(pad_ref)

    shifts = np.empty((s_sel.size,), dtype=np.float64)
    used = 0

    with open_h5(h5_path, role="reader") as f:
        DataCube = f["/DataCube"]
        try:
            # give the dataset a friendlier chunk cache if h5py supports it
            DataCube.id.set_chunk_cache(521, 8 * 1024 * 1024, 1.0)
        except Exception:
            pass

        itS = tqdm(s_sel, desc="[bias] xcorr", dynamic_ncols=True) if progress else s_sel
        for i, s in enumerate(itS):
            y = np.asarray(DataCube[int(s), :], dtype=np.float64, order="C")
            if keep_idx is not None:
                y = y[keep_idx]
            y = np.where(np.isfinite(y), y, 0.0)

            y_dt = _detrend(y, int(window_len))
            y_f  = np.where(mask_feat, y_dt, 0.0)

            y0 = y_f - np.mean(y_f)
            ny = np.linalg.norm(y0)
            if not np.isfinite(ny) or ny == 0.0:
                continue
            y0 = y0 / ny

            pad_dat = np.zeros((n_corr,), dtype=np.float64)
            pad_dat[:L_eff] = y0
            F_dat = np.fft.rfft(pad_dat)

            cc = np.fft.irfft(F_dat * np.conj(F_ref), n_corr)

            lo = int(0)
            hi = int(np.minimum(n_corr, int(lag_px) + 1))
            left  = cc[n_corr - int(lag_px):] if int(lag_px) > 0 else np.empty((0,))
            right = cc[lo:hi]
            win   = np.concatenate([left, right], axis=0)

            j = int(np.argmax(win))
            shift = j - int(lag_px)
            shifts[used] = np.float64(shift)
            used += 1

    if used == 0:
        raise RuntimeError("No usable spaxels for bias estimation.")

    shifts = shifts[:used]
    med = np.median(shifts)
    mad = np.median(np.abs(shifts - med))

    dlog = np.median(np.diff(ObsPix if keep_idx is None else ObsPix[keep_idx]))
    v_per_px = C_KMS * np.expm1(dlog)
    vel_bias = med * v_per_px

    with open_h5(h5_path, role="writer") as f:
        f["/HyperCube"].attrs["vel_bias_kms"] = float(vel_bias)
        f["/HyperCube"].attrs["vel_bias_method"] = "feature_xcorr_v1"

    return dict(
        vel_bias_kms=float(vel_bias),
        shift_px=float(np.median(shifts)),
        median_px=float(med),
        mad_px=float(mad),
        n_spax_used=int(used),
        L_eff=int(L_eff),
        dlog_obs_med=float(dlog),
        n_features=int(n_pick),
        window_len=int(window_len),
        quality="ok" if np.isfinite(vel_bias) else "warn",
    )

# ------------------------------------------------------------------------------

def _frac_shift_last_axis(X: np.ndarray, shift: float) -> np.ndarray:
    """
    Fractional circular shift of each row of X along the last axis by `shift`
    samples using the Fourier shift theorem. X is (..., T).
    """
    T = int(X.shape[-1])
    F = np.fft.rfft(X, n=T, axis=-1)
    freq = np.fft.rfftfreq(T)                 # cycles/sample
    phase = np.exp(-2j * np.pi * freq * float(shift))  # (rfft_len,)
    Y = np.fft.irfft(F * phase[(...,) + (None,)* (F.ndim-1 - 1)], n=T, axis=-1)
    return Y

# ------------------------- main builder ---------------------------------------

def build_hypercube(
    base_h5: str,
    *,
    norm_mode: str = "data",        # "model" or "data"
    amp_mode: str = "sum",           # "sum" or "trapz" for LOSVD amplitude
    cp_flux_ref_mode: str = "median",   # "median" | "mean" | "off"
    floor: float = 1e-12,
    S_chunk: int = 128,
    C_chunk: int = 1,
    P_chunk: int = 360,
    compression: str | None = None,  # keep None for speed; compress later
    vel_bias_kms: float = 0.0,
) -> None:
    """
    Build /HyperCube/models with LOSVD convolution on the template grid,
    rebin to observed grid, and scale once according to the selected
    normalization. The output is "ready to use" by the solver; no runtime
    scaling is needed.

    Additionally, this version computes the global column energy on-the-fly:
        E[c,p] = sum_{s, λ (masked if present)} models[s,c,p,λ]^2
    and stores it at /HyperCube/col_energy (float64) at the end of the build.
    """

    if norm_mode not in ("model", "data"):
        raise ValueError("norm_mode must be 'model' or 'data'.")
    if cp_flux_ref_mode not in ("median", "mean", "off"):
        raise ValueError("cp_flux_ref_mode must be 'median', 'mean', or 'off'.")


    # -------- Fast preflight: exit early if fully complete (reader-only)
    with open_h5(base_h5, "reader") as f_rd:
        P, T = map(int, f_rd["/Templates"].shape)
        S, V, C = map(int, f_rd["/LOSVD"].shape)
        L = int(f_rd["/DataCube"].shape[1])

        g = f_rd.get("/HyperCube", None)
        if g is not None and "models" in g and "_done" in g:
            done = np.asarray(g["_done"][...], dtype=np.uint8)
            all_done = (done.size > 0 and int(done.sum()) == int(done.size))
            if bool(g.attrs.get("complete", False)) and all_done:
                print("[HyperCube] already complete; skip build (no writer).")
                return

        # Read grids/operators/templates once (reader), reuse later
        tem_loglam = np.asarray(f_rd["/TemPix"][...], dtype=np.float64)
        vel_pix    = np.asarray(f_rd["/VelPix"][...], dtype=np.float64)
        R_any      = np.asarray(f_rd["/R_T"][...])
        if R_any.shape == (T, L):
            R_T = R_any.astype(np.float32, copy=False)
        elif R_any.shape == (L, T):
            R_T = R_any.T.astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"/R_T shape {R_any.shape} incompatible with "
                f"T={T}, L={L}")
        Templates = np.asarray(f_rd["/Templates"][...], dtype=np.float64,
            order="C")

        # capture mask once for later energy accumulation
        keep_idx = None
        if "/Mask" in f_rd:
            m = np.asarray(f_rd["/Mask"][...], dtype=bool).ravel()
            if m.size == L and np.any(m):
                keep_idx = np.flatnonzero(m)


        # Optional λ-weights for energy/statistics (apply same floor, use √w)
        w_lam_sqrt = None
        lamw_floor = 1e-6
        if "/HyperCube/lambda_weights" in f_rd:
            _w = np.asarray(f_rd["/HyperCube/lambda_weights"][...],
                            dtype=np.float64).ravel()
            if _w.size == L:
                _w = np.clip(_w, lamw_floor, None)
                if keep_idx is not None:
                    _w = _w[keep_idx]
                w_lam_sqrt = np.sqrt(_w).astype(np.float64, copy=False)  # (Lk,) or (L,)

    # Precompute velocity→pixel mapping (shared for all (s,c))
    km = _kernel_map_from_grids(tem_loglam, vel_pix)
    n_fft = _choose_nfft(T, km.m)

    if vel_bias_kms != 0.0:
        dlog_T = float(np.median(np.diff(tem_loglam)))
        shift_T = float(np.log1p(float(vel_bias_kms) / C_KMS) / dlog_T)  # samples
        # Precompute phase vector once; reuse for all (s,c) in the build.
        freq = np.fft.rfftfreq(int(n_fft))                               # (rfft_len,)
        phase_shift = np.exp(-2j * np.pi * freq * shift_T).astype(np.complex128)
    else:
        phase_shift = None

    # Template FFT at n_fft (zero-padded for linear convolution)
    # Keep complex128 for numerical parity with preflight.
    T_fft = np.fft.rfft(Templates, n=int(n_fft), axis=1)

    # -------- Prepare destination dataset, normalization, and resume bitmap
    with open_h5(base_h5, "writer") as f:
        g = f.require_group("/HyperCube")
        # create models if absent
        if "models" not in g:
            chunks = (min(S_chunk, S), min(C_chunk, C), min(P_chunk, P), L)
            g.create_dataset("models", shape=(S, C, P, L), dtype="f4",
                             chunks=chunks, compression=compression or None)
        models = g["models"]
        # Load per-spaxel data-flux mean (required in data mode)
        if norm_mode == "data":
            if "/HyperCube/data_flux" not in f:
                raise RuntimeError("Missing /HyperCube/data_flux; compute it "
                                   "first (masked mean per spaxel, shape (S,)).")
            L_ds = f["/HyperCube/data_flux"]
            if L_ds.shape != (S,):
                raise RuntimeError(
                    f"/HyperCube/data_flux shape {L_ds.shape} invalid; "
                    f"expected (S,)={S}. It must be the masked mean flux "
                    f"per spaxel."
                )
            L_vec = np.asarray(L_ds[...], dtype=np.float64)  # (S,)
        else:
            L_vec = None

        # normalization datasets: A[s,c], A_sum[s], and optionally data_flux
        ng = _ensure_norm_group(f)
        A_path, A_sum_path = _compute_and_store_losvd_amplitudes(f,
            amp_mode=amp_mode)
        A = f[A_path] # (S,C) float64
        A_sum = f[A_sum_path] # (S,)  float64

        # record attributes once
        g.attrs["norm.mode"] = norm_mode
        g.attrs["losvd_amplitude_mode"] = amp_mode
        g.attrs["kernel_unit_area"] = True

        done, grid = _make_done_bitmap(f, S, C, P, S_chunk, C_chunk, P_chunk)

        # How many tiles remain BEFORE enabling SWMR
        rem = int(np.asarray(done[...], np.uint8).size
                  - np.asarray(done[...], np.uint8).sum())
        if rem <= 0:
            g.attrs["complete"] = True
            g.attrs["shape"] = (S, C, P, L)
            g.attrs["chunks"] = models.chunks
            try:
                f.flush()
            except Exception:
                pass
            print("[HyperCube] already complete; nothing to write.")
            return

        losvd_ds = f["/LOSVD"]

        masked_flag = bool(keep_idx is not None)
        # create dataset & attrs up front (metadata writes pre-SWMR)
        if "/HyperCube/col_energy" in f:
            del f["/HyperCube/col_energy"]
        E_ds = g.create_dataset("col_energy", shape=(C, P), dtype="f8",
                                chunks=(min(C,256), min(P,1024)), compression="gzip")
        # set attrs before swmr_mode
        E_ds.attrs["masked"] = bool(masked_flag)
        E_ds.attrs["lambda_weights"] = bool(w_lam_sqrt is not None)
        E_ds.attrs["lambda_weights_floor"] = 1e-6
        E_ds.attrs["shape"] = (int(C), int(P))
        E_ds.attrs["provenance"] = "sum_{s,λ} models^2 (mask applied)" if masked_flag else "sum_{s,λ} models^2"
        if masked_flag and (w_lam_sqrt is not None):
            E_ds.attrs["provenance"] = "sum_{s,λ} (√w·models)^2 over masked λ"
        elif masked_flag:
            E_ds.attrs["provenance"] = "sum_{s,λ} models^2 over masked λ"
        elif (w_lam_sqrt is not None):
            E_ds.attrs["provenance"] = "sum_{s,λ} (√w·models)^2 over all λ"
        else:
            E_ds.attrs["provenance"] = "sum_{s,λ} models^2 over all λ"

        # --- cp_flux_ref accumulators
        do_ref = (cp_flux_ref_mode != "off")
        if do_ref:
            if cp_flux_ref_mode == "mean":
                ref_sum = np.zeros((C, P), dtype=np.float64)
                ref_cnt = np.zeros((C, P), dtype=np.int64)
            else:  # "median"
                ref_acc = np.empty((C, P), dtype=object)
                for c_ in range(C):
                    for p_ in range(P):
                        ref_acc[c_, p_] = _P2Median()

        # --- SWMR writer mode
        try:
            f.flush()
            # f.swmr_mode = True
            print("[SWMR] writer mode enabled for HyperCube build.")
        except Exception as e:
            print(f"[SWMR] could not enable writer mode: {e}")

        # Initialize global column energy accumulator E[c,p] float64
        E_acc = np.zeros((C, P), dtype=np.float64)

        # --- iterate tiles; skip ones marked done
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

        total_tiles = rem
        pbar = tqdm(total=total_tiles, desc="[HyperCube] tiles",
                    mininterval=2.0)

        eps = 1e-30  # avoids 0/0 in data mode
        for (p0, p1, s0, s1, c0, c1, idx3) in _iter_all_tiles():
            if _done_get(done, idx3) != 0:
                continue

            # Frequency-domain template slice for this P-block
            T_fft_slice = T_fft[p0:p1, :]  # (ΔP, rfft_len)

            # For each (s,c) in this tile:
            for s_idx in range(s0, s1):
                a_sum = float(A_sum[s_idx])  # scalar
                for c_idx in range(c0, c1):
                    # 1) build unit-area kernel for (s,c)
                    H_native = np.asarray(losvd_ds[s_idx, :, c_idx],
                                         dtype=np.float64, order="C")
                    Hk_unit = _losvd_to_unit_kernel(H_native, km)

                    # 2) FFT conv (centered 'same' crop on template grid)
                    conv_td = _fft_conv_centered(T_fft_slice, Hk_unit, km, n_fft, int(T),
                             phase_shift=phase_shift)

                    # 3) rebin AFTER convolution
                    Ycp = conv_td @ R_T  # (ΔP, L)
                    np.maximum(Ycp, 0.0, out=Ycp)

                    # 4) apply FINAL scale according to norm_mode
                    if norm_mode == "model":
                        scale = float(A[s_idx, c_idx])
                    else:
                        if a_sum <= 0.0 or L_vec[s_idx] <= 0.0:
                            scale = 0.0
                        else:
                            frac = float(A[s_idx, c_idx]) / np.maximum(a_sum, 1.0e-30)
                            scale = float(L_vec[s_idx]) * frac

                    if scale != 0.0:
                        Ycp *= np.float32(scale)
                    else:
                        Ycp.fill(0.0)

                    # 6b) accumulate global energy E[c,p] with same λ-view:
                    # mask if present, and apply √w if lambda_weights exist.
                    if masked_flag:
                        Yv = Ycp[:, keep_idx]  # (ΔP, Lk)
                    else:
                        Yv = Ycp               # (ΔP, L or Lk)

                    # Compute per-population contribution for this (s_idx, c_idx, P-block)
                    if w_lam_sqrt is not None:
                        # weighted: sum_λ (√w·Y)^2 = sum_λ (w * Y^2)
                        e_local = np.sum(
                            np.square(Yv, dtype=np.float64) * w_lam_sqrt[None, :],
                            axis=1
                        )  # (ΔP,)
                    else:
                        # unweighted: sum_λ Y^2
                        e_local = np.sum(
                            np.square(Yv, dtype=np.float64),
                            axis=1
                        )  # (ΔP,)

                    E_acc[c_idx, p0:p1] += e_local

                    # --- cp_flux_ref: per-(c,p) masked λ-sum, one sample per spaxel
                    if do_ref:
                        if masked_flag:
                            s_lambda = np.sum(Ycp[:, keep_idx], axis=1, dtype=np.float64)  # (ΔP,)
                        else:
                            s_lambda = np.sum(Ycp, axis=1, dtype=np.float64)              # (ΔP,)
                        if cp_flux_ref_mode == "mean":
                            ref_sum[c_idx, p0:p1] += s_lambda
                            ref_cnt[c_idx, p0:p1] += 1
                        else:
                            # median: update sketches per p
                            for dp, val in enumerate(s_lambda):
                                ref_acc[c_idx, p0 + dp].update(float(val))

                    # 7) write
                    models[s_idx, c_idx, p0:p1, 0:L] = Ycp


            _done_set(done, idx3)

            # SWMR visibility
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

        E_ds[...] = E_acc

        # --- finalize cp_flux_ref
        if do_ref:
            if "/HyperCube/norm/cp_flux_ref" in f:
                del f["/HyperCube/norm/cp_flux_ref"]
            if cp_flux_ref_mode == "mean":
                ref = np.divide(ref_sum, np.maximum(ref_cnt, 1, dtype=np.int64),
                                dtype=np.float64)
            else:
                ref = np.empty((C, P), dtype=np.float64)
                for c_ in range(C):
                    for p_ in range(P):
                        ref[c_, p_] = ref_acc[c_, p_].median()
            # sanitize + floor
            ref = np.where(np.isfinite(ref), ref, 0.0)
            ref = np.maximum(ref, float(floor))
            g_norm = f.require_group("/HyperCube/norm")
            g_norm.create_dataset(
                "cp_flux_ref", data=ref.astype(np.float64),
                chunks=(min(C,256), min(P,1024)), compression="gzip"
            )
            g_norm.attrs["cp_flux_ref.mode"] = cp_flux_ref_mode
            g_norm.attrs["masked"] = bool(masked_flag)
            g_norm.attrs["definition"] = (
                "per-(c,p) statistic of sum_λ models[s,c,p,λ] over spaxels; "
                "mask applied to λ if present"
            )


        g.attrs["complete"] = True
        if vel_bias_kms is not None:
            g.attrs["vel_bias_kms"] = float(vel_bias_kms)
            g.attrs["vel_bias_note"] = "global LOSVD shift applied at build"
        g.attrs["shape"] = (S, C, P, L)
        g.attrs["chunks"] = models.chunks
        g.attrs["cp_flux_ref_mode"] = cp_flux_ref_mode


        # help h5py close cleanly
        del models, losvd_ds, done, g, E_ds
        try: f.flush()
        except: pass

# ------------------------------------------------------------------------------

def ensure_global_column_energy(
    h5_path: str,
    apply_mask: bool = True,
    dset_name: str = "/HyperCube/col_energy",
    *,
    rdcc_slots: int = 1_000_003,
    rdcc_bytes: int = 256 * 1024**2,
    rdcc_w0: float = 0.90,
) -> np.ndarray:
    """
    Compute/store E[c,p] = sum_{s,λ (masked if requested)} models[s,c,p,λ]^2 in float64.
    Streamed in (S_chunk, C_chunk=1, P_chunk, L) to match dataset chunking.
    """
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]  # (S,C,P,L) f32
        S, C, P, L = map(int, M.shape)

        # help the dataset cache a bit
        try:
            M.id.set_chunk_cache(int(rdcc_slots), int(rdcc_bytes), float(rdcc_w0))
        except Exception:
            pass

        keep_idx = None
        if apply_mask and "/Mask" in f:
            m = np.asarray(f["/Mask"][...], bool).ravel()
            keep_idx = np.flatnonzero(m) if (m.size == L and np.any(m)) else None

        S_chunk = int(M.chunks[0]) if (M.chunks and M.chunks[0]) else 128
        C_chunk = int(M.chunks[1]) if (M.chunks and M.chunks[1]) else 1
        P_chunk = int(M.chunks[2]) if (M.chunks and M.chunks[2]) else P

        E = np.zeros((C, P), dtype=np.float64)

        nS = int(np.ceil(S / S_chunk))
        nC = int(np.ceil(C / C_chunk))
        total = int(nS * nC)

        pbar = tqdm(total=total, desc="[col_energy]", mininterval=1.5, dynamic_ncols=True)

        for s0 in range(0, S, S_chunk):
            s1 = int(np.min((S, s0 + S_chunk)))
            for c0 in range(0, C, C_chunk):
                c1 = int(np.min((C, c0 + C_chunk)))

                # Read only the band that aligns with chunking
                X = np.asarray(M[s0:s1, c0:c1, :, :], dtype=np.float32, order="C")  # (Sblk,Cb,P,L)
                if keep_idx is not None:
                    X = X[:, :, :, keep_idx]  # (Sblk,Cb,P,Lk)

                # Sum of squares over s and λ → (Cb,P)
                # Avoids giant temporaries by working on the small Cb-band (usually 1)
                e_local = np.sum(np.square(X, dtype=np.float64), axis=(0, 3))  # (Cb,P)
                E[c0:c1, :] += e_local

                pbar.update(1)

        pbar.close()

    # Write once
    with open_h5(h5_path, role="writer") as f:
        if dset_name in f:
            del f[dset_name]
        ds = f.create_dataset(dset_name, data=E, dtype="f8", compression="gzip", compression_opts=1)
        ds.attrs["masked"] = bool(apply_mask)
        ds.attrs["note"] = "sum_{s,λ} models^2 (mask applied if masked=True)"
    return E

# ------------------------------------------------------------------------------

def read_global_column_energy(
    h5_path: str,
    dset_name: str = "/HyperCube/col_energy",
    strict: bool = True,
) -> np.ndarray | None:
    """
    Read E[c,p] from `dset_name`. Validates shape against /HyperCube/models.
    Returns (C,P) float64, or None if missing and strict=False.
    """
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]      # (S,C,P,L)
        _, C, P, _ = map(int, M.shape)

        if dset_name not in f:
            if strict:
                raise FileNotFoundError(
                    f"{dset_name} not found. Run compute_and_store_global_column_energy(...) once first."
                )
            return None

        E = np.asarray(f[dset_name][...], dtype=np.float64)
        if E.shape != (C, P):
            raise ValueError(f"{dset_name} has shape {E.shape}, expected {(C, P)}.")
        if not np.all(np.isfinite(E)):
            raise ValueError(f"{dset_name} contains non-finite values.")
        print(f"[col_energy] read {dset_name} with shape {E.shape}.")
        return E

# ------------------------------------------------------------------------------

def convert_hypercube_norm(h5_path: str,
                           to_mode: str,
                           *,
                           recompute_energy: bool = True) -> None:
    """
    In-place convert /HyperCube/models normalization between 'data' and 'model'
    using only per-spaxel scaling, without loading huge slabs into RAM.
    After rescaling, optionally recompute /HyperCube/col_energy by calling
    ensure_global_column_energy(apply_mask=True) — which will read /Mask itself.

    Semantics match the original function; only the I/O pattern is chunked.
    """
    to_mode = str(to_mode).lower()
    if to_mode not in ("data", "model"):
        raise ValueError("to_mode must be 'data' or 'model'.")

    with open_h5(h5_path, role="writer") as f:
        g = f["/HyperCube"]
        M = g["models"]                              # (S,C,P,L) float32
        S, C, P, L = map(int, M.shape)
        chunks = M.chunks or (min(S,128), 1, min(P,360), L)
        S_chunk, C_chunk, P_chunk, L_chunk = map(int, chunks)

        cur_mode = str(g.attrs.get("norm.mode", "model")).lower()
        if cur_mode == to_mode:
            print(f"[Norm] Already in '{to_mode}' mode; nothing to do.")
            return

        # Per-spaxel factors (identical to your original logic)
        ng = _ensure_norm_group(f)
        A_sum = np.asarray(ng["losvd_amp_sum"][...], dtype=np.float64)  # (S,)

        need_L_vec = (cur_mode == "model" and to_mode == "data") or \
                     (cur_mode == "data"  and to_mode == "model")
        if need_L_vec:
            if "/HyperCube/data_flux" not in f:
                raise RuntimeError("Missing /HyperCube/data_flux required for 'data' normalization.")
            L_vec = np.asarray(f["/HyperCube/data_flux"][...], dtype=np.float64)  # (S,)
        else:
            L_vec = None

        eps = 1e-30
        if   cur_mode == "data"  and to_mode == "model":
            F = (A_sum / np.maximum(L_vec, eps))
        elif cur_mode == "model" and to_mode == "data":
            F = (L_vec / np.maximum(A_sum, eps))
        else:
            raise RuntimeError(f"Unexpected conversion {cur_mode} → {to_mode}")

        # Clean degenerate entries (leave those spaxels unchanged)
        bad = ~np.isfinite(F) | (F <= 0.0)
        if np.any(bad):
            F = F.copy()
            F[bad] = 1.0

        # Nicely sized chunk cache to keep I/O smooth
        try:
            M.id.set_chunk_cache(521, 8*1024*1024, 1.0)  # slots, bytes, w0
        except Exception:
            pass

        # Stream in storage order: (S, C, P, L) with λ-banding = L_chunk
        from math import ceil
        total_steps = ceil(S/S_chunk) * ceil(C/C_chunk) * ceil(P/P_chunk) * ceil(L/L_chunk)
        pbar = tqdm(total=total_steps, desc=f"[Norm] {cur_mode}→{to_mode}",
                    mininterval=1.0, dynamic_ncols=True)

        for s0 in range(0, S, S_chunk):
            s1 = min(S, s0 + S_chunk)
            scale = F[s0:s1].astype(np.float32).reshape(-1, 1, 1, 1)  # (dS,1,1,1)

            for c0 in range(0, C, C_chunk):
                c1 = min(C, c0 + C_chunk)
                for p0 in range(0, P, P_chunk):
                    p1 = min(P, p0 + P_chunk)
                    for l0 in range(0, L, L_chunk):
                        l1 = min(L, l0 + L_chunk)

                        slab = M[s0:s1, c0:c1, p0:p1, l0:l1][...]  # f32
                        np.multiply(slab, scale, out=slab)        # rescale temp slab
                        M[s0:s1, c0:c1, p0:p1, l0:l1] = slab      # write back

                        pbar.update(1)

                # occasional flush keeps SWMR readers happy
                try:
                    M.id.flush(); f.flush()
                except Exception:
                    pass

        pbar.close()

        # Update the mode stamp
        g.attrs["norm.mode"] = to_mode
        try: f.flush()
        except Exception: pass

    # Keep semantics identical to your original: recompute col_energy VIA the helper
    if recompute_energy:
        # This helper should read /Mask internally (apply_mask=True),
        # so this function doesn’t need to load the mask itself.
        ensure_global_column_energy(h5_path, apply_mask=True)

# ------------------------------------------------------------------------------

def assert_preflight_ok(
    h5_path: str,
    s_list: Optional[Iterable[int]] = None,
    c_list: Optional[Iterable[int]] = None,
    p_list: Optional[Iterable[int]] = None,
    *,
    max_spax: int = 3,
    max_comp: int = 2,
    max_pop: int = 6,
    tol_rel: float = 2e-3,
    tol_shift_px: float = 0.5,
    tol_flat_valid: float = 3e-8,
    require_rt_flat: bool = True,
    rt_flat_tol: float = 3e-8,
    verbose: bool = True,
) -> Dict[str, Any]:
    r"""
    Run the convolution preflight on a tiny subset and raise a hard error
    if any check fails. This is intended to be called immediately before
    launching a full HyperCube build.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    s_list, c_list, p_list : iterable of int, optional
        Explicit indices to test. If any are None, a small prefix slice is
        chosen using the *max_* caps.
    max_spax, max_comp, max_pop : int
        Caps for auto-selection when lists are not provided.
    tol_rel : float
        Relative L2 tolerance for FFT≡Direct on the template grid.
    tol_shift_px : float
        Allowed integer-lag shift (pixels) from cross-correlation.
    tol_flat_valid : float
        Max |conv(1,Hk)−1| on the fully overlapped interior region of the
        centered 'same' crop.
    require_rt_flat : bool
        If True, also require that the rebin operator satisfies
        max|R_T @ 1 − 1| ≤ rt_flat_tol on the observed grid.
    rt_flat_tol : float
        Tolerance for the R_T flatness check.
    verbose : bool
        If True, prints the per-case lines emitted by the preflight.

    Returns
    -------
    Dict[str, Any]
        The result dict returned by
        ``preflight_hypercube_convolution(...)``. Always returned on
        success. On failure a ``RuntimeError`` is raised.

    Exceptions
    ----------
    RuntimeError
        If any per-case check fails (FFT vs direct, shift, or flat-valid),
        or if ``require_rt_flat`` is True and the rebin flatness exceeds
        ``rt_flat_tol``.

    Examples
    --------
    >>> assert_preflight_ok("NGC4365/NGC4365_01.h5",
    ...                     s_list=[24, 25, 26],
    ...                     c_list=[124, 197],
    ...                     p_list=[42, 44, 88, 89])
    """
    res = preflight_hypercube_convolution(
        h5_path,
        s_list=s_list,
        c_list=c_list,
        p_list=p_list,
        max_spax=int(max_spax),
        max_comp=int(max_comp),
        max_pop=int(max_pop),
        tol_rel=float(tol_rel),
        tol_shift_px=float(tol_shift_px),
        tol_flat_valid=float(tol_flat_valid),
        verbose=bool(verbose),
    )

    # Optional global rebin flatness guard
    if require_rt_flat and (float(res["rt_flat_check"]) > float(rt_flat_tol)):
        raise RuntimeError(
            "[preflight] R_T flatness failed: max|R_T @ 1 − 1| = "
            f"{float(res['rt_flat_check']):.3e} > {float(rt_flat_tol):.3e}"
        )

    if bool(res["all_pass"]):
        return res

    # Summarize failures
    passes = np.asarray(res["passes"], bool)
    rel = np.asarray(res["rel_err"], float)
    sh = np.asarray(res["shift_px"], int)
    fv = np.asarray(res["flat_valid"], float)
    triples = list(res["triples"])

    bad = np.flatnonzero(~passes)
    worst_rel = float(np.max(rel)) if rel.size else float("nan")
    worst_abs_shift = int(np.max(np.abs(sh))) if sh.size else 0
    # Handle all-NaN interior cases gracefully
    fv_clean = fv[~np.isnan(fv)] if fv.size else np.array([], float)
    worst_flat = float(np.max(fv_clean)) if fv_clean.size else float("nan")

    # Show up to 8 failing triples for quick triage
    show = bad[: np.minimum(bad.size, 8)]
    lines = []
    for i in show:
        lines.append(
            f"  case {i}: (s,c,p0)={triples[i]}  "
            f"rel={rel[i]:.3e}  shift={sh[i]:+d}px  "
            f"flat_valid={fv[i]:.3e}"
        )

    msg = [
        "[preflight] convolution checks failed.",
        f"  worst rel={worst_rel:.3e}  worst |shift|={worst_abs_shift:d}px  "
        f"worst flat_valid={worst_flat:.3e}",
    ]
    if lines:
        msg.append("  examples:")
        msg.extend(lines)

    raise RuntimeError("\n".join(msg))

# ------------------------------------------------------------------------------