# thread_utils.py
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Sequence
import numpy as np
import os, glob, time, re
import pathlib as plp
from dataclasses import dataclass
import matplotlib.pyplot as plt

from dynamics.IFU.Constants import Constants
from CubeFit.hypercube_reader import HyperCubeReader, ReaderCfg
from CubeFit.hdf5_manager import open_h5

CTS = Constants()
C_KMS = CTS.c

@contextmanager
def blas_threads_ctx(n: Optional[int]):
    """
    Limit BLAS threads to `n` inside the context using threadpoolctl.
    If n is None, do nothing and let environment (e.g. OMP_NUM_THREADS)
    control threading.
    """
    if n is None:
        # respect environment; do not override
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        # threadpoolctl not available; best-effort no-op
        yield
        return
    # Ensure positive int; treat <=0 as 'no limit' (no-op)
    n_int = int(n)
    if n_int <= 0:
        yield
        return
    with threadpool_limits(n_int):
        yield

def _cfg_float(cfg, name, default):
    v = getattr(cfg, name, None)
    return default if v is None else float(v)

def _cfg_int(cfg, name, default):
    v = getattr(cfg, name, None)
    return default if v is None else int(v)

def _cfg_bool(cfg, name, default):
    v = getattr(cfg, name, None)
    return default if v is None else bool(v)

# ------------------------------------------------------------------------------

def _find_latest_sidecar(main_path: str) -> str | None:
    pat = f"{os.fspath(main_path)}.fit.*.h5"
    cand = glob.glob(pat)
    if not cand:
        return None
    return max(cand, key=os.path.getmtime)

def _default_sidecar_path(main_h5: str) -> str:
    """
    Compose a unique, stable sidecar path next to the main HDF5.
    Example: /path/file.h5.fit.<JOBID or pid>.<unix_ts>.h5
    """
    ts = int(time.time())
    jid = os.environ.get("SLURM_JOB_ID")
    sid = os.environ.get("SLURM_STEP_ID")
    tag = f"{jid}.{sid}" if jid else f"pid{os.getpid()}"
    return f"{str(main_h5)}.fit.{tag}.{ts}.h5"

# ------------------------------------------------------------------------------

def cpuset_count():
    """
    Return (n_cores, mask_string) for current process cpuset.

    Tries, in order:
      1) /proc/self/cgroup → cpuset slice → {v1,v2} cpuset.cpus(.effective)
      2) root-level fallbacks
      3) sched_getaffinity(0)

    On failure returns (None, None).
    """
    import os

    def _parse_mask(s: str) -> tuple[int, str]:
        s = s.strip().strip("\n")
        if not s:
            return 0, ""
        n = 0
        out_parts = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                if b < a:
                    a, b = b, a
                n += (b - a + 1)
                out_parts.append(f"{a}-{b}")
            else:
                a = int(part)
                n += 1
                out_parts.append(str(a))
        return n, ",".join(out_parts)

    # --- discover cpuset cgroup path for this process
    cg_cpuset = None
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                # format: "<id>:<controllers>:<path>"
                parts = line.strip().split(":", 2)
                if len(parts) == 3:
                    ctrls = parts[1].split(",")
                    if "cpuset" in ctrls:
                        cg_cpuset = parts[2] or "/"
                        break
    except Exception:
        cg_cpuset = None

    candidates = []
    if cg_cpuset:
        # v1 mount style
        candidates.append(f"/sys/fs/cgroup/cpuset{cg_cpuset}/cpuset.cpus.effective")
        candidates.append(f"/sys/fs/cgroup/cpuset{cg_cpuset}/cpuset.cpus")
        # v2 unified style
        candidates.append(f"/sys/fs/cgroup{cg_cpuset}/cpuset.cpus.effective")
        candidates.append(f"/sys/fs/cgroup{cg_cpuset}/cpuset.cpus")

    # root-level fallbacks
    candidates += [
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ]

    for p in candidates:
        try:
            with open(p) as f:
                s = f.read().strip()
            if s:
                return _parse_mask(s)
        except Exception:
            continue

    # last resort: affinity
    try:
        cores = sorted(os.sched_getaffinity(0))
        if not cores:
            return None, None
        # compress to "0-3,5,7-8"
        ranges = []
        start = prev = None
        for c in cores:
            if start is None:
                start = prev = c
            elif c == prev + 1:
                prev = c
            else:
                ranges.append((start, prev))
                start = prev = c
        if start is not None:
            ranges.append((start, prev))
        parts = [f"{a}-{b}" if a != b else str(a) for a, b in ranges]
        return len(cores), ",".join(parts)
    except Exception:
        return None, None

# ------------------------------------------------------------------------------

def ensure_lambda_weights(
    h5_path: str,
    dset_name: str = "/HyperCube/lambda_weights",
    *,
    apply_mask: bool = True,
    floor: float = 1e-6,
    min_w: float = 0.2,
    smooth1: int = 7,
    smooth2: int = 31,
    pct: float = 95.0,
    alpha: float = 0.9,   # modest contrast; set 1.0 to disable
) -> np.ndarray:
    """Build line-emphasis weights w(λ) from the median data spectrum.
       w = min_w + (1-min_w) * normalize(|boxcar(med, s1)-boxcar(med, s2)|)^alpha
       → clip to [floor, 1] and write to HDF5.
    """
    smooth1 = int(max(1, smooth1))
    smooth2 = int(max(smooth1+2, smooth2))
    alpha   = float(alpha)
    min_w   = float(min_w)
    floor   = float(floor)

    def _boxcar(x, win):
        k = np.ones(win, dtype=np.float64) / win
        return np.convolve(x, k, mode="same")

    with open_h5(h5_path, role="reader") as f:
        DC = f["/DataCube"]  # (S,L)
        S, L = map(int, DC.shape)
        mask = None
        if apply_mask and "/Mask" in f:
            m = np.asarray(f["/Mask"][...], bool).ravel()
            if m.size == L:
                mask = m

        med = np.median(np.asarray(DC[...], np.float64), axis=0)  # (L,)
        if mask is not None:
            med = med * mask

    lp1 = _boxcar(med, smooth1)
    lp2 = _boxcar(med, smooth2)
    dog = np.abs(lp1 - lp2)

    # robust scale to [0,1]
    pos = dog[dog > 0]
    s   = np.percentile(pos, pct) if pos.size else 1.0
    z   = np.clip(dog / (s + 1e-12), 0.0, 1.0)

    w = min_w + (1.0 - min_w) * (z ** alpha)
    if mask is not None:
        w = w * mask  # masked λ -> 0 (will be clipped to floor)
    w = np.clip(w, floor, 1.0).astype(np.float64)

    with open_h5(h5_path, role="writer") as f:
        if dset_name in f:
            del f[dset_name]
        ds = f.create_dataset(dset_name, data=w, dtype="f8", compression="gzip")
        ds.attrs.update(dict(method="median-DoG", min_w=min_w, alpha=alpha,
                             smooth1=smooth1, smooth2=smooth2, pct=float(pct),
                             masked=bool(apply_mask)))
    return w

def read_lambda_weights(h5_path: str, dset_name: str = "/HyperCube/lambda_weights",
                        floor: float = 1e-6) -> np.ndarray:
    with open_h5(h5_path, role="reader") as f:
        if dset_name not in f:
            raise RuntimeError(f"Missing {dset_name}")
        w = np.asarray(f[dset_name][...], dtype=np.float64).ravel()
    return np.maximum(w, float(floor))

# --------------------------- HDF5 helpers ------------------------------

def _get_mask(f):
    """Return /Mask as boolean with strict semantics: True == keep."""
    if "/Mask" not in f:
        return None
    return np.asarray(f["/Mask"][...], dtype=bool).ravel()

# ------------------------------------------------------------------------------

def _bin_edges_from_loglam(loglam: np.ndarray) -> np.ndarray:
    loglam = np.asarray(loglam, float)
    mid = 0.5 * (loglam[1:] + loglam[:-1])
    edges = np.empty(loglam.size + 1, float)
    edges[1:-1] = mid
    # linear extrapolation at ends
    edges[0]  = loglam[0]  - (mid[0]  - loglam[0])
    edges[-1] = loglam[-1] + (loglam[-1] - mid[-1])
    return edges

def _delta_lambda_from_loglam(loglam: np.ndarray) -> np.ndarray:
    edges = _bin_edges_from_loglam(loglam)
    lam_edges = np.exp(edges)
    dlam = lam_edges[1:] - lam_edges[:-1]
    return dlam

def _apply_RT(t_row: np.ndarray, R_any: np.ndarray) -> np.ndarray:
    # /R_T is (T,L) or (L,T). We want result (L,)
    if R_any.ndim != 2:
        raise ValueError(f"/R_T must be 2D, got {R_any.shape}")
    T, = t_row.shape
    if R_any.shape == (T,):
        raise ValueError("Degenerate /R_T")
    if R_any.shape[0] == T:
        return (t_row.astype(np.float64) @ R_any.astype(np.float64))  # (L,)
    elif R_any.shape[1] == T:
        return (R_any.astype(np.float64) @ t_row.astype(np.float64))  # (L,)
    else:
        raise ValueError(f"/R_T shape {R_any.shape} incompatible with T={T}")

def _xcorr_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Return subpixel shift (in pixels) to align b to a: shift>0 means b→right."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    corr = np.correlate(a, b, mode="full")
    k0 = int(np.argmax(corr)) - (a.size - 1)
    i  = k0 + (a.size - 1)
    # quadratic refinement
    if 1 <= i < corr.size - 1:
        y0, y1, y2 = corr[i-1], corr[i], corr[i+1]
        denom = (y0 - 2*y1 + y2)
        delta = 0.5*(y0 - y2)/denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return float(k0 + delta)

def _vel_to_pix_shift(vel: float, dlog: float, c_kms: float) -> float:
    # pixels ≈ ln(1+v/c)/dlog (exact on log-λ grid)
    return float(np.log1p(vel / c_kms) / dlog)

def run_wavelength_checks(h5_path: str,
                          s: int,
                          c: int,
                          p_list: Optional[Sequence[int]] = None,
                          max_p_auto: int = 6) -> None:
    """
    Step-by-step wavelength alignment checks for one (s, c) and a handful of p.
    - Validates grids and /R_T
    - Compares rebin(Templates[p]) vs models[s,c,p,:]
    - Prints expected vs measured pixel shift
    """
    with open_h5(h5_path, "reader") as f:
        TPT = np.asarray(f["/Templates"][...], float)        # (P,T)
        Tem = np.asarray(f["/TemPix"][...], float)           # (T,) log-λ (natural)
        Obs = np.asarray(f["/ObsPix"][...], float)           # (L,) log-λ (natural)
        DCs = np.asarray(f["/DataCube"][s, :], float)        # (L,)
        R_any = np.asarray(f["/R_T"][...])
        Msc = np.asarray(f["/HyperCube/models"][s, c, :, :], float)  # (P,L)
        H   = np.asarray(f["/LOSVD"][s, :, c], float)        # (V,)
        Vel = np.asarray(f["/VelPix"][...], float)           # (V,) km/s

    # 0) Grids sanity
    dtem = np.diff(Tem); dobs = np.diff(Obs)
    print("[grids] TemPix monotonic:", bool(np.all(dtem > 0)),
          "  ObsPix monotonic:", bool(np.all(dobs > 0)))
    print("[grids] median Δlogλ: tem =", float(np.median(dtem)), " obs =", float(np.median(dobs)))
    # Data vs ObsPix: they share the L axis; this confirms *wavelengths* are consistent inputs

    # 1) Flux-conserving /R_T sanity
    dlam_T = _delta_lambda_from_loglam(Tem)   # per-bin λ width
    dlam_O = _delta_lambda_from_loglam(Obs)
    # Push a flat-in-λ flux vector through R_T: input proportional to Δλ_T → expect Δλ_O
    y_ref = _apply_RT(dlam_T, R_any)                     # (L,)
    rel_err = float(np.max(np.abs(y_ref - dlam_O)) / np.maximum(np.max(dlam_O), 1e-30))
    print(f"[R_T] max rel error mapping Δλ_T→Δλ_O = {rel_err:.3e} (should be ≲1e-12–1e-8)")

    # 2) Choose populations p to test
    if p_list is None:
        # Auto-pick by column energy at this component, if available
        with open_h5(h5_path, "reader") as f:
            if "/HyperCube/col_energy" in f:
                E = np.asarray(f["/HyperCube/col_energy"][c, :], float)  # (P,)
                cand = np.argsort(E)[::-1][:int(np.minimum(max_p_auto, E.size))]
                p_list = list(map(int, cand))
            else:
                p_list = list(range(int(np.minimum(max_p_auto, TPT.shape[0]))))
    print(f"[pick] testing p indices: {p_list}")

    # 3) LOSVD mean velocity and expected pixel shift (sign!)
    amp = float(np.sum(H)) if np.sum(H) > 0 else 0.0
    mu_v = float(np.sum(H * Vel) / np.sum(H)) if amp > 0 else 0.0  # km/s
    dlog_tem = float(np.median(np.diff(Tem)))
    k_exp = _vel_to_pix_shift(mu_v, dlog_tem, C_KMS)
    print(f"[losvd] sum={amp:.6g}  mean(v)={mu_v:.3f} km/s  expected pixel shift ≈ {k_exp:+.3f} px")

    # 4) For each p: rebin template (no convolution) vs stored model column
    for p in p_list:
        t_row = TPT[p, :]                        # (T,)
        pre = _apply_RT(t_row, R_any)            # (L,)
        post = Msc[p, :]                         # (L,)

        # Remove overall scale for a clean shape comparison
        a = pre - np.nanmean(pre);  b = post - np.nanmean(post)
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na > 0: a = a / na
        if nb > 0: b = b / nb

        k_meas = _xcorr_shift(a, b)
        # also align and report residual
        shift_px = int(np.round(k_meas))
        if shift_px > 0:
            b_al = np.pad(b, (shift_px, 0), mode="constant")[:b.size]
        elif shift_px < 0:
            b_al = np.pad(b, (0, -shift_px), mode="constant")[-shift_px:]
            b_al = np.pad(b_al, (0, b.size - b_al.size), mode="constant")
        else:
            b_al = b

        rmse_al = float(np.sqrt(np.mean((a - b_al)**2)))
        print(f"[p={int(p):3d}] xcorr shift={k_meas:+.3f} px   rmse(after int-align)={rmse_al:.4f}")

    # 5) Quick data vs best pre-convolution template similarity (sanity)
    #    (not a fit — just to ensure there isn't a huge calibration drift)
    # find template row whose pre-rebin resembles the data the most
    # (cheap cosine similarity)
    with open_h5(h5_path, "reader") as f:
        # use a small random subset if P is huge for speed; otherwise do all
        P = int(TPT.shape[0])
        take = np.arange(P) if P <= 2000 else np.random.default_rng(7).choice(P, 2000, replace=False)
    S_pre = []
    for p in take:
        pre = _apply_RT(TPT[p, :], R_any)
        u = pre - np.nanmean(pre)
        v = DCs  - np.nanmean(DCs)
        num = float(np.dot(u, v))
        den = float(np.linalg.norm(u) * np.linalg.norm(v)) or 1.0
        S_pre.append(num/den)
    S_pre = np.asarray(S_pre, float)
    print(f"[data↔pre-rebin] best cosine over {S_pre.size} templates: "
          f"{float(np.max(S_pre)):.4f}  median={float(np.median(S_pre)):.4f}")

    # stochastic row/column test (depending on orientation)
    T, L = TPT.shape[1], len(Obs)
    ones_T = np.ones(T)
    r = _apply_RT(ones_T, R_any)     # rebin a flat-in-pixel spectrum
    print("max|r-1| =", float(np.max(np.abs(r - 1.0))))

# ------------------------------------------------------------------------------

def estimate_global_velocity_bias_features(
    h5_path: str,
    *,
    sidecar: str | None = None,
    sample_spaxels: int = 192,
    feature_spaxels: int = 96,
    n_features: int = 24,
    half_window: int = 15,
    search_range_px: int = 6,
    cont_window: int | None = None,
    use_mask: bool = True,
    seed: int = 1234,
    show_progress: bool = True,
) -> dict:
    """
    Estimate a single global velocity bias (km/s) by correlating observed
    spectra with model predictions over *only* the strongest absorption
    features. This preserves pixel-level accuracy (no spectral downsampling),
    but dramatically reduces work by restricting to compact windows around
    deep features and by sampling fewer spaxels.

    The method:
      1) Build a robust reference line-depth profile from a subset of
         spaxels. Use a smoothed continuum to get residuals; pick the K
         deepest feature centers with a minimum spacing.
      2) For a (larger) set of spaxels, compute normalized dot products
         between observed spectra and model spectra over the union of
         those feature windows for integer pixel shifts k ∈ [-R, +R].
      3) Pick the best shift by aggregated score, then refine to sub-pixel
         via a quadratic fit around the peak. Report a per-spax sanity
         distribution (median/MAD) from the same procedure.

    Notes
    -----.
    * Preserves the solver's masked grid exactly (no regridding).
    * The search range should cover your suspected offset (e.g., ±6 px).
    * Feature windows are *global* and identical for all spaxels (no
      per-spax elastic shifts).

    Parameters
    ----------
    h5_path : str
        Path to the main HDF5 file containing ``/DataCube`` and
        ``/HyperCube/models``.
    sidecar : str or None, optional
        Path to a live-fit sidecar for reading ``x`` (prefers
        ``/Fit/x_best`` → ``/Fit/x_last`` → ``/Fit/x_epoch_last`` →
        ``/Fit/x_snapshots``). Falls back to main file (``/Fit/x_latest``,
        then ``/X_global``).
    sample_spaxels : int, default 192
        Number of spaxels used to accumulate the *global* correlation score.
    feature_spaxels : int, default 96
        Number of spaxels used to select deepest features and build the
        reference residual profile. Can be smaller than ``sample_spaxels``.
    n_features : int, default 24
        Number of feature centers (global) to select.
    half_window : int, default 15
        Half-width (in pixels) for each feature window; total window length
        is ``2*half_window + 1``. Windows are unioned.
    search_range_px : int, default 6
        Integer shift search half-range. Candidate shifts are
        ``[-search_range_px, ..., +search_range_px]``.
    cont_window : int or None, optional
        Boxcar width for continuum smoothing (pixels). If ``None``, uses
        a default of ~L/128 rounded to an odd integer. The window is
        clipped to at least 5 and at most L-1.
    use_mask : bool, default True
        If ``True``, apply ``/Mask`` so A and y match the solver grid.
    seed : int, default 1234
        RNG seed for reproducible spaxel selection.
    show_progress : bool, default True
        If ``True``, show tqdm progress bars.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``vel_bias_kms`` : float
            The global velocity bias (km/s), positive = model needs to be
            redshifted relative to data by this amount.
        - ``shift_px`` : float
            Estimated sub-pixel shift on the masked observed grid.
        - ``median_px`` : float
            Median of per-spax shifts from the sanity subset.
        - ``mad_px`` : float
            Median absolute deviation (MAD) of per-spax shifts.
        - ``n_spax_used`` : int
            Number of spaxels used for the aggregate correlation.
        - ``L_eff`` : int
            Effective masked spectral length.
        - ``dlog_obs_med`` : float
            Median spacing of ``ln λ`` on the masked observed grid.
        - ``n_features`` : int
            Number of feature centers used.
        - ``window_len`` : int
            Window length per feature (``2*half_window + 1``).
        - ``quality`` : str
            ``"ok"`` if finite; otherwise ``"bad"``.

    Raises
    ------
    RuntimeError
        If no solution vector is found or the masked length is zero.

    Examples
    --------
    >>> est = estimate_global_velocity_bias_features(
    ...     "galaxy.h5",
    ...     sidecar="galaxy.h5.fit.123.h5",
    ...     sample_spaxels=160,
    ...     feature_spaxels=80,
    ...     n_features=20,
    ...     half_window=12,
    ...     search_range_px=6,
    ...     show_progress=True,
    ... )
    >>> est["shift_px"]
    1.2
    """

    # tqdm (optional, soft dependency)
    try:
        from tqdm import tqdm as _tqdm
    except Exception:  # pragma: no cover
        def _tqdm(iterable=None, **kwargs):
            return iterable if iterable is not None else range(
                int(kwargs.get("total", 0))
            )

    # ------------------------------- helpers ---------------------------------
    def _read_latest_x(side_f, main_f):
        def _row_or_vec(dset):
            arr = np.asarray(dset[...], dtype=float)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return arr[-1, :].astype(float, copy=False)
            return arr.ravel().astype(float, copy=False)

        if side_f is not None:
            for key in ("/Fit/x_best", "/Fit/x_last", "/Fit/x_epoch_last",
                        "/Fit/x_snapshots", "/Fit/x_hist"):
                if key in side_f:
                    return _row_or_vec(side_f[key])
        for key in ("/Fit/x_latest", "/X_global"):
            if key in main_f:
                return _row_or_vec(main_f[key])
        raise RuntimeError("No solution vector x found in sidecar or main HDF5.")

    def _choose_spaxels(S: int, k: int, rng: np.random.Generator) -> np.ndarray:
        k_eff = int(np.minimum(k, S))
        if k_eff >= S:
            return np.arange(S, dtype=np.int64)
        # quasi-uniform linspace, then add a tiny jitter for diversity
        base = np.linspace(0, S - 1, k_eff, dtype=np.int64)
        if k_eff > 4:
            # jitter within ±1 (clipped), unique to avoid duplicates
            jitter = rng.integers(-1, 2, size=k_eff, dtype=np.int64)
            cand = np.clip(base + jitter, 0, S - 1)
            return np.unique(cand)
        return base

    def _moving_avg_same(x: np.ndarray, w: int) -> np.ndarray:
        # 1D boxcar (odd length) with symmetric pad via reflection
        L = int(x.size)
        half = int((w - 1) // 2)
        # Reflect padding
        pre = x[1:half+1][::-1] if half > 0 else np.empty(0, x.dtype)
        suf = x[L-half-1:L-1][::-1] if half > 0 else np.empty(0, x.dtype)
        z = np.concatenate([pre, x, suf])
        k = np.ones((w,), dtype=x.dtype) / np.float64(w)
        y = np.convolve(z, k, mode="valid")  # length L
        return y

    def _pick_feature_centers(resid_ref: np.ndarray,
                              n_feat: int,
                              min_gap: int) -> np.ndarray:
        # pick most negative residuals with spacing >= min_gap
        order = np.argsort(resid_ref)  # ascending: most negative first
        sel = np.empty(0, dtype=np.int64)
        for j in order:
            if sel.size == 0:
                sel = np.append(sel, j)
                if sel.size >= n_feat:
                    break
                continue
            # enforce separation
            if np.all(np.abs(j - sel) >= min_gap):
                sel = np.append(sel, j)
                if sel.size >= n_feat:
                    break
        # sort centers
        return np.sort(sel)

    def _norm_win(y: np.ndarray, idx_win: np.ndarray) -> np.ndarray:
        # z-score on windowed region only
        ww = y[idx_win]
        mu = float(np.mean(ww))
        sd = float(np.std(ww))
        sd = sd if (np.isfinite(sd) and sd > 0.0) else 1.0
        return (y - mu) / sd

    def _dot_windows(y: np.ndarray, m: np.ndarray,
                     centers: np.ndarray,
                     half_w: int,
                     shift: int) -> float:
        # Sum of dot products over disjoint windows for a given integer shift.
        total = 0.0
        L = int(y.size)
        for c in centers:
            a = int(np.maximum(0, c - half_w))
            b = int(np.minimum(L, c + half_w + 1))  # [a, b)
            # valid overlap under shift (shift applied to model m index)
            a_m = int(a - shift)
            b_m = int(b - shift)
            # clip range for both signals
            a_eff = int(np.maximum(a, 0))
            b_eff = int(np.minimum(b, L))
            a_m = int(np.maximum(a_m, 0))
            b_m = int(np.minimum(b_m, L))
            # adjust to keep lengths equal
            len_ab = int(b_eff - a_eff)
            len_m  = int(b_m - a_m)
            if (len_ab <= 0) or (len_m <= 0):
                continue
            d = int(np.minimum(len_ab, len_m))
            total += float(np.dot(y[a_eff:a_eff+d], m[a_m:a_m+d]))
        return total

    def _best_shift_for_pair(y: np.ndarray, m: np.ndarray,
                             centers: np.ndarray, half_w: int,
                             R: int) -> float:
        # window-normalize both sequences (using all windows together)
        # Build a single index vector for window union to estimate stats.
        L = int(y.size)
        mask = np.zeros(L, dtype=bool)
        for c in centers:
            a = int(np.maximum(0, c - half_w))
            b = int(np.minimum(L, c + half_w + 1))
            mask[a:b] = True
        idx = np.flatnonzero(mask)
        y_n = _norm_win(y, idx)
        m_n = _norm_win(m, idx)

        ks = np.arange(-R, R + 1, dtype=np.int64)
        scores = np.empty(ks.size, dtype=np.float64)
        for i, k in enumerate(ks):
            scores[i] = _dot_windows(y_n, m_n, centers, half_w, int(k))

        # parabolic subpixel around best integer k
        i_best = int(np.argmax(scores))
        k_best = float(ks[i_best])
        if (i_best > 0) and (i_best < (scores.size - 1)):
            y0 = scores[i_best - 1]
            y1 = scores[i_best]
            y2 = scores[i_best + 1]
            denom = (y0 - 2.0 * y1 + y2)
            frac = 0.0 if (np.abs(denom) < 1.0e-30) else 0.5 * (y0 - y2) / denom
            return float(k_best + frac)
        return float(k_best)

    # -------------------------- read dims/mask/x ------------------------------
    with open_h5(h5_path, role="reader", swmr=True) as F:
        L_obs = int(F["/DataCube"].shape[1])

        keep = None
        if bool(use_mask) and ("/Mask" in F):
            msk = np.asarray(F["/Mask"][...], dtype=bool).ravel()
            if (msk.size == L_obs) and np.any(msk):
                keep = np.flatnonzero(msk)

        ObsPix = np.asarray(F["/ObsPix"][...], dtype=np.float64)
        ObsPix_used = ObsPix if keep is None else ObsPix[keep]
        if ObsPix_used.size == 0:
            raise RuntimeError("Masked ObsPix has zero length.")

        dlog_obs_med = float(np.median(np.diff(ObsPix_used)))

        # read x (prefer sidecar)
        if sidecar and plp.Path(sidecar).exists():
            with open_h5(sidecar, role="reader", swmr=True) as G:
                x = _read_latest_x(G, F)
        else:
            x = _read_latest_x(None, F)

        S = int(F["/DataCube"].shape[0])
        C = int(F["/HyperCube/models"].shape[1])
        P = int(F["/HyperCube/models"].shape[2])

    if x.size != (C * P):
        raise RuntimeError(f"x has length {x.size}, expected C*P={C*P}.")

    rng = np.random.default_rng(int(seed))

    # ----------------------------- sampling -----------------------------------
    spx_feat = _choose_spaxels(S, int(feature_spaxels), rng)
    spx_corr = _choose_spaxels(S, int(sample_spaxels),  rng)

    reader = HyperCubeReader(
        h5_path,
        ReaderCfg(dtype_models="float32", apply_mask=bool(use_mask))
    )

    try:
        # One read to get L_eff
        A0, y0 = reader.read_spaxel_plane(int(spx_feat[0]))
        L_eff = int(y0.size)

        # continuum window default (odd)
        if cont_window is None:
            cw = int(np.maximum(5, np.minimum(L_eff - 1, np.round(L_eff / 128.0))))
            if (cw % 2) == 0:
                cw = int(cw + 1)
        else:
            cw_clip = int(np.clip(cont_window, 5, np.maximum(5, L_eff - 1)))
            cw = int(cw_clip + (1 - (cw_clip % 2)))  # make odd

        # ------------------ (1) build reference residual profile ---------------
        accum = np.zeros(L_eff, dtype=np.float64)
        count = 0

        it_feat = spx_feat
        if show_progress:
            it_feat = _tqdm(
                spx_feat,
                desc="[bias] building feature profile",
                mininterval=0.5,
                dynamic_ncols=True,
            )

        for s in it_feat:
            _, y = reader.read_spaxel_plane(int(s))
            # simple continuum via boxcar; residual = y - cont
            cont = _moving_avg_same(y, cw)
            r = y - cont
            # normalize residual per spaxel by its robust scale
            sig = float(np.std(r))
            sig = sig if (np.isfinite(sig) and sig > 0.0) else 1.0
            accum += (r / sig)
            count += 1

        resid_ref = accum / np.maximum(1.0, float(count))

        # pick centers: deepest negative residuals with spacing ≥ half_window
        centers = _pick_feature_centers(
            resid_ref,
            int(n_features),
            int(np.maximum(half_window, 1)),
        )
        # if not enough due to spacing, relax by 1 px spacing until filled
        gap = int(np.maximum(half_window, 1))
        while centers.size < int(n_features) and gap > 1:
            gap = int(gap - 1)
            centers = _pick_feature_centers(resid_ref, int(n_features), gap)

        if centers.size == 0:
            # fallback: use evenly spaced centers
            centers = np.linspace(0, L_eff - 1, int(np.maximum(4, n_features)), dtype=np.int64)

        # ------------------ (2) aggregated integer-shift scores ----------------
        R = int(np.maximum(1, search_range_px))
        ks = np.arange(-R, R + 1, dtype=np.int64)
        scores = np.zeros(ks.size, dtype=np.float64)

        it_corr = spx_corr
        if show_progress:
            it_corr = _tqdm(
                spx_corr,
                desc="[bias] accumulating correlation",
                mininterval=0.5,
                dynamic_ncols=True,
            )

        for s in it_corr:
            A, y = reader.read_spaxel_plane(int(s))
            m = (A.T @ x).astype(np.float64, copy=False)

            # pre-normalize both on union of windows (shared stats)
            mask = np.zeros(L_eff, dtype=bool)
            for c in centers:
                a = int(np.maximum(0, c - half_window))
                b = int(np.minimum(L_eff, c + half_window + 1))
                mask[a:b] = True
            idx = np.flatnonzero(mask)
            y_n = _norm_win(y, idx)
            m_n = _norm_win(m, idx)

            for i, k in enumerate(ks):
                scores[i] += _dot_windows(y_n, m_n, centers, int(half_window), int(k))

        i_best = int(np.argmax(scores))
        k_best = float(ks[i_best])
        # quadratic subpixel refine
        if (i_best > 0) and (i_best < (scores.size - 1)):
            y0 = scores[i_best - 1]
            y1 = scores[i_best]
            y2 = scores[i_best + 1]
            denom = (y0 - 2.0 * y1 + y2)
            frac = 0.0 if (np.abs(denom) < 1.0e-30) else 0.5 * (y0 - y2) / denom
            shift_px = float(k_best + frac)
        else:
            shift_px = float(k_best)

        # ------------------ (3) per-spax sanity distribution -------------------
        k_chk = int(np.minimum(np.maximum(16, centers.size), spx_corr.size))
        s_small = spx_corr[:k_chk]
        per_px = np.empty(k_chk, dtype=np.float64)

        it_chk = enumerate(s_small)
        if show_progress:
            it_chk = _tqdm(
                enumerate(s_small),
                total=k_chk,
                desc="[bias] sanity subset",
                mininterval=0.5,
                dynamic_ncols=True,
            )

        for i, s in it_chk:
            A, y = reader.read_spaxel_plane(int(s))
            m = (A.T @ x).astype(np.float64, copy=False)
            per_px[i] = _best_shift_for_pair(y, m, centers, int(half_window), int(R))

        median_px = float(np.median(per_px))
        mad_px = float(np.median(np.abs(per_px - median_px)))

    finally:
        reader.close()

    # ------------------------ pixel → ln λ → km/s -----------------------------
    dln = dlog_obs_med * float(shift_px)
    vel_bias = float(C_KMS * np.expm1(dln))

    return dict(
        vel_bias_kms=vel_bias,
        shift_px=float(shift_px),
        median_px=median_px,
        mad_px=mad_px,
        n_spax_used=int(spx_corr.size),
        L_eff=int(L_eff),
        dlog_obs_med=dlog_obs_med,
        n_features=int(centers.size),
        window_len=int(2 * half_window + 1),
        quality="ok" if np.isfinite(vel_bias) else "bad",
    )

# ------------------------------------------------------------------------------

@dataclass
class RatioCfg:
    """
    Mixture control for component usage s_c = Σ_p x[c,p].

    Parameters
    ----------
    use : bool
        Enable the mixture update.
    eta : float
        Step size for the multiplicative correction in log space.
    gamma : float
        Per-update clamp on multiplicative change; factors are clipped
        into [1/gamma, gamma] before renormalization.
    prob : float
        Probability of updating a given component in a tile (stochastic
        thinning). 1.0 = update all.
    batch : int
        If >0, cap on the number of components updated per tile.
    minw : float
        Floor for target and empirical mixtures to avoid log(0).
    anchor : str
        'target'  -> drive toward orbit_weights (normalized)
        'x0'      -> drive toward the mixture in the provided x0
        'auto'    -> use 'x0' if non-empty, else 'target'
    tile_every : int
        Apply the update once every this many tiles (1 = every tile).
    epoch_renorm : bool
        If True, recompute the per-epoch target normalization so the
        target is comparable to the current mass.

    Notes
    -----
    The update is mass-preserving and nonnegativity-preserving:
        1) compute ṡ = s / Σ s, t = target (Σ t = 1)
        2) f = exp(-eta * (log(ṡ) - log(t)))  (clipped)
        3) normalize f by ⟨ṡ, f⟩ to keep total mass
        4) x[c,:] ← f[c] * x[c,:]
    """
    use: bool = True
    eta: float = 0.8
    gamma: float = 1.3
    prob: float = 1.0
    batch: int = 0
    minw: float = 1e-6
    anchor: str = "auto"       # {'target','x0','auto'}
    tile_every: int = 1
    epoch_renorm: bool = True

    # strong epoch-end projector
    epoch_project: bool = True  # do a global pass at epoch end
    epoch_eta: float = 1.0      # stronger than tile eta
    epoch_gamma: float = 10.0   # allow larger rebalancing
    epoch_beta: float = 1.0     # mixing: 1.0=full replace, <1 = blend

# ------------------------------------------------------------------------------

def compare_usage_to_orbit_weights(h5_path: str,
                                   sidecar: str | None = None,
                                   x_dset: str | None = None,
                                   normalize: str = "unit_sum",
                                   out_png: str | None = None,
                                   *,
                                   usage_metric: str = "sum",
                                   E_cp: np.ndarray | None = None) -> dict:
    """
    Compare the final component usage (sum over populations) against the
    input orbital weights, both read from the HDF5 store.

    The function prefers solution vectors in a sidecar if present, then
    falls back to the main file. It also prefers orbital weights stored
    in the sidecar, then falls back to the main file.

    Parameters
    ----------
    h5_path : str
        Path to the main HDF5 file.
    sidecar : str or None
        Optional explicit path to a sidecar. If None, the function will
        try to auto-detect one matching '<main>.fit.*.h5' and use the
        newest by mtime.
    x_dset : str or None
        Optional explicit dataset path for X in either the sidecar or
        main file (e.g., '/Fit/x_best'). If None, the search order is:
        sidecar: '/Fit/x_best', '/Fit/x_last', '/Fit/x_epoch_last',
                 last row of '/Fit/x_snapshots', last row of '/Fit/x_hist';
        main:    '/X_global', '/Fit/x_latest'.
    normalize : {'unit_sum', 'match_sum'}
        Normalization applied before comparison. With 'unit_sum', both
        vectors are scaled to sum to 1. With 'match_sum', the usage is
        scaled so its sum matches the orbital weights sum. Use
        'unit_sum' for shape-only comparison.
    out_png : str or None
        If provided, write a small bar+marker figure overlaying usage
        and target weights.
    usage_metric : {'sum', 'energy'}, optional
        How to aggregate usage over populations. With 'sum', usage is
        sum_p X[c,p]. With 'energy', usage is sum_p X[c,p]*E[c,p], where
        E is the global column energy from the HyperCube build.
    E_cp : ndarray or None, optional
        Optional (C,P) array of global column energies. If None and
        usage_metric=='energy', this will be read from the main HDF5 via
        `read_global_column_energy(h5_path)`.

    Returns
    -------
    out : dict
        Dictionary with:
          - 'usage_raw'   : (C,) float64, aggregated usage (per metric)
          - 'weights_raw' : (C,) float64, as read
          - 'usage'       : (C,) float64, normalized
          - 'weights'     : (C,) float64, normalized
          - 'l1'          : float, L1 error (|u-w|).sum()
          - 'linf'        : float, L∞ error |u-w|.max()
          - 'cosine'      : float, cosine similarity
          - 'pearson_r'   : float, Pearson correlation (NaN if C<2)
          - 'argmax_err'  : int, index of max |u-w|
          - 'plot_path'   : str or None
    """
    # ------------------ helpers ------------------
    def _find_latest_sidecar(main_path: str) -> str | None:
        import glob  # ensure available even if caller didn't import at top
        pat = f"{os.fspath(main_path)}.fit.*.h5"
        cand = glob.glob(pat)
        if not cand:
            return None
        cand.sort(key=lambda p: os.path.getmtime(p))
        return cand[-1]

    def _read_C_P(f) -> tuple[int, int]:
        M = f["/HyperCube/models"]
        _, C_, P_, _ = map(int, M.shape)
        return C_, P_

    def _row_or_vec(ds, C: int, P: int) -> np.ndarray:
        arr = np.asarray(ds[...], dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape == (C, P):
                return arr.reshape(C * P).astype(np.float64, copy=False)
            if arr.shape == (P, C):
                return arr.T.reshape(C * P).astype(np.float64, copy=False)
            if arr.shape[1] == C * P:
                return arr[-1, :].astype(np.float64, copy=False)
            if arr.shape == (1, C * P):
                return arr[0, :].astype(np.float64, copy=False)
            if arr.shape == (C * P, 1):
                return arr[:, 0].astype(np.float64, copy=False)
        return arr.ravel().astype(np.float64, copy=False)

    def _read_X(f_side, f_main, C, P) -> np.ndarray:
        # explicit override
        if x_dset is not None:
            src = f_side if (f_side is not None and x_dset in f_side) else f_main
            if src is None or x_dset not in src:
                raise RuntimeError(f"Requested x_dset '{x_dset}' not found.")
            return _row_or_vec(src[x_dset], C, P)

        # sidecar-first
        if f_side is not None:
            for name in ("/Fit/x_best", "/Fit/x_last", "/Fit/x_epoch_last"):
                if name in f_side:
                    return _row_or_vec(f_side[name], C, P)
            for name in ("/Fit/x_snapshots", "/Fit/x_hist"):
                if name in f_side and f_side[name].shape[0] > 0:
                    return _row_or_vec(f_side[name], C, P)

        # main fallbacks
        for name in ("/X_global", "/Fit/x_latest"):
            if name in f_main:
                return _row_or_vec(f_main[name], C, P)

        raise RuntimeError("No solution vector found in sidecar or main file.")

    def _read_weights(f_side, f_main) -> np.ndarray:
        if f_side is not None and "/CompWeights" in f_side:
            return np.asarray(f_side["/CompWeights"][...], dtype=np.float64)
        if "/Fit/orbit_weights" in f_main:
            return np.asarray(f_main["/Fit/orbit_weights"][...], dtype=np.float64)
        if "/CompWeights" in f_main:
            return np.asarray(f_main["/CompWeights"][...], dtype=np.float64)
        raise RuntimeError("No orbital weights found in sidecar or main file.")

    def _safe_norm(v: np.ndarray, mode: str, ref_sum: float | None = None
                   ) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        s = float(np.sum(v))
        eps = 1.0e-30
        if mode == "unit_sum":
            if s <= eps:
                return np.zeros_like(v)
            return v / s
        # match_sum
        tgt = float(ref_sum) if ref_sum is not None else s
        if s <= eps:
            return np.zeros_like(v)
        return v * (tgt / s)

    # ------------------ open files ------------------
    if sidecar is None:
        sidecar = _find_latest_sidecar(h5_path)

    with open_h5(h5_path, role="reader", swmr=True) as F:
        C, P = _read_C_P(F)
        if sidecar is not None and os.path.exists(sidecar):
            with open_h5(sidecar, role="reader", swmr=True) as G:
                x = _read_X(G, F, C, P)
                w = _read_weights(G, F)
        else:
            x = _read_X(None, F, C, P)
            w = _read_weights(None, F)

    # ------------------ assemble & compare ------------------
    x = np.asarray(x, dtype=np.float64).ravel(order="C")
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size not in (C, C * P):
        raise ValueError(
            f"x length {x.size} is incompatible; expected C={C} or C*P={C*P}."
        )

    # Usage metric
    usage_metric = str(usage_metric).lower()
    if usage_metric not in ("sum", "energy"):
        raise ValueError("usage_metric must be 'sum' or 'energy'.")

    if x.size == C * P:
        Xcp = x.reshape(C, P, order="C")
    else:
        # If only a C-vector is provided, treat it as already aggregated.
        Xcp = None

    if usage_metric == "energy":
        # Prefer passed-in E_cp; otherwise read once from main file.
        if E_cp is None:
            from CubeFit.hypercube_builder import read_global_column_energy
            E_cp = read_global_column_energy(h5_path)  # (C,P) float64
        E = np.asarray(E_cp, np.float64, order="C")
        if E.shape != (C, P):
            raise ValueError(f"E_cp shape {E.shape} != (C,P) {(C,P)}")
        if Xcp is None:
            # x is (C,), nothing to multiply; interpret as already-aggregated.
            usage_raw = np.maximum(x.copy(), 0.0)
        else:
            usage_raw = np.maximum((Xcp * E).sum(axis=1), 0.0)
    else:
        # 'sum' metric
        if Xcp is None:
            usage_raw = np.maximum(x.copy(), 0.0)
        else:
            usage_raw = np.maximum(Xcp.sum(axis=1), 0.0)

    # Accept orbit_weights as (C,) or (C*P,)
    w_vec = np.asarray(w, dtype=np.float64).ravel(order="C")
    if w_vec.size == C:
        weights_raw = np.maximum(w_vec, 0.0)
    elif w_vec.size == C * P:
        weights_raw = np.maximum(
            w_vec.reshape(C, P, order="C").sum(axis=1), 0.0
        )
    else:
        raise ValueError(
            f"weights length {w_vec.size} is incompatible; expected C={C} "
            f"or C*P={C*P}."
        )

    if normalize not in ("unit_sum", "match_sum"):
        raise ValueError("normalize must be 'unit_sum' or 'match_sum'.")

    if normalize == "unit_sum":
        usage = _safe_norm(usage_raw, "unit_sum")
        weights = _safe_norm(weights_raw, "unit_sum")
    else:
        usage = _safe_norm(
            usage_raw, "match_sum", ref_sum=float(np.sum(weights_raw))
        )
        weights = weights_raw.copy()

    diff = usage - weights
    l1 = float(np.sum(np.abs(diff)))
    linf = float(np.max(np.abs(diff))) if diff.size else np.nan
    denom = np.linalg.norm(usage) * np.linalg.norm(weights)
    cosine = float((usage @ weights) / np.maximum(denom, 1.0e-30)) \
        if denom > 0.0 else np.nan
    pearson_r = float(np.corrcoef(usage, weights)[0, 1]) \
        if usage.size >= 2 else np.nan
    argmax_err = int(np.argmax(np.abs(diff))) if diff.size else -1

    plot_path = None
    if out_png is not None:
        fig = plt.figure(figsize=(9, 3.2))
        ax = fig.add_subplot(111)
        idx = np.arange(C, dtype=int)
        ax.bar(idx, usage, width=0.8, alpha=0.85, label="usage (sum over P)",
               color='r')
        ax.plot(idx, weights, marker="o", linestyle="--", label="target w_c",
                color='k')
        mx = np.max(weights) if C > 0 else 1.0
        ax.set_ylim(0.0, float(1.05 * mx))
        ax.set_xlabel("component c")
        ax.set_ylabel("normalized weight")
        ttl = (f"component usage vs target  L1={l1:.3g}  L∞={linf:.3g}  "
               f"cos={cosine:.3f}  r={pearson_r:.3f}")
        ax.set_title(ttl)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        plot_path = out_png

    out = dict(
        usage_raw=usage_raw,
        weights_raw=weights_raw,
        usage=usage,
        weights=weights,
        l1=l1,
        linf=linf,
        cosine=cosine,
        pearson_r=pearson_r,
        argmax_err=argmax_err,
        plot_path=plot_path,
    )
    print("[usage_vs_weights] L1={:.4g}  L∞={:.4g}  cos={:.4f}  r={:.4f}  "
          "argmax_err={}".format(l1, linf, cosine, pearson_r, argmax_err))
    return out

# ------------------------------------------------------------------------------

def apply_component_softbox(
    x_cp: np.ndarray,
    w_c: np.ndarray,
    *,
    band: float = 0.30,
    step: float = 0.25,
    min_target: float = 1e-10,
) -> None:
    """
    Softly pull per-component usage s_c = sum_p x[c,p] toward a target
    proportional to the prior weights w_c, within a (1±band) tube.
    In-place, O(C·P). Accepts w_c of length C or C*P.

    Parameters
    ----------
    x_cp : (C,P) float
        Current component×population weights (modified in place).
    w_c : (C,) or (C*P,) float
        Component priors. If length C*P, they are summed over P → (C,).
    band : float
        Allowed relative deviation around target (e.g. 0.30 → ±30%).
    step : float
        Relaxation fraction toward the band edge per call (0..1).
    min_target : float
        Floor for targets to avoid division-by-zero.
    """
    x = np.asarray(x_cp, dtype=np.float64, order="C")
    if x.ndim != 2:
        raise ValueError(f"x_cp must be 2-D (C,P); got shape {x.shape}.")
    C, P = x.shape

    w = np.asarray(w_c, dtype=np.float64).ravel(order="C")
    if w.size == C:
        wC = w
    elif w.size == C * P:
        wC = w.reshape(C, P).sum(axis=1)
    else:
        raise ValueError(f"w_c length {w.size} incompatible with C={C}, P={P}.")

    w_sum = np.sum(wC)
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return  # nothing to do

    # Targets proportional to priors, matched to total mass in x
    total = np.sum(x)
    if not np.isfinite(total) or total <= 0.0:
        return

    w_norm = wC / np.maximum(w_sum, 1.0e-300)         # (C,)
    t = np.maximum(min_target, w_norm * total)        # (C,)
    s = np.sum(x, axis=1)                             # (C,)

    # Tube bounds and which rows violate
    lo = (1.0 - band)
    hi = (1.0 + band)
    too_high = s > (hi * t)
    too_low  = s < (lo * t)
    need = (too_high | too_low)
    if not np.any(need):
        return

    # Scale factors (smooth pull toward tube edges)
    f = np.ones(C, dtype=np.float64)
    s_safe = np.maximum(s, min_target)

    # map s -> hi*t for high violators; s -> lo*t for low violators
    # blend with (1-step) to avoid shocks
    f_hi = (hi * t) / s_safe
    f_lo = (lo * t) / s_safe

    f[too_high] = (1.0 - step) + step * f_hi[too_high]
    f[too_low]  = (1.0 - step) + step * f_lo[too_low]

    # Apply only to rows that need adjustment
    idx = np.flatnonzero(need)
    x[idx, :] *= f[idx, None]
    np.maximum(x[idx, :], 0.0, out=x[idx, :])  # keep nonnegativity

# ------------------------------------------------------------------------------

def apply_component_softbox_energy(
    x_cp: np.ndarray,
    E_cp: np.ndarray,          # read_global_column_energy(h5_path) → (C,P)
    w_c: np.ndarray,           # /CompWeights: len C or C*P
    *,
    band: float = 0.30,        # allowed deviation band ±30%
    step: float = 0.20,        # fraction of the way to the band boundary
    min_target: float = 1e-10,
    row_chunk: int = 8192,
) -> None:
    """
    Soft "box" constraint in energy-weighted usage space. Scales rows of
    x_cp toward the nearest band edge around the target usage, without
    forming large temporaries or using advanced indexing.

    Parameters
    ----------
    x_cp : ndarray, shape (C, P)
        Coefficient matrix to adjust in-place.
    E_cp : ndarray, shape (C, P)
        Global column energy per (c, p).
    w_c : ndarray, shape (C,) or (C*P,)
        Component prior weights; if length C*P, reduced to component level.
    band : float, optional
        Half-width of allowed band, i.e., s ∈ [(1-band)t, (1+band)t].
    step : float, optional
        Fractional move toward the boundary (0→no-op, 1→land exactly).
    min_target : float, optional
        Floor on target weights before normalization.
    row_chunk : int, optional
        Row-chunk size for streamed scaling.

    Returns
    -------
    None
    """
    x = np.asarray(x_cp, np.float64, order="C")
    E = np.asarray(E_cp, np.float64, order="C")
    C, P = x.shape
    if E.shape != (C, P):
        raise ValueError(f"E_cp shape {E.shape} != (C,P) {(C, P)}")

    w = np.asarray(w_c, np.float64).ravel(order="C")
    if w.size == C:
        t = w
    elif w.size == C * P:
        t = w.reshape(C, P).sum(axis=1)
    else:
        raise ValueError(f"w_c length {w.size} incompatible with C={C}, P={P}.")

    # energy-weighted usage per component (proportional to flux)
    # avoids materializing x*E as a (C,P) temp
    s = np.einsum("cp,cp->c", x, E, dtype=np.float64)      # (C,)

    # normalize target & usage
    t = np.maximum(t, float(min_target))
    t /= np.sum(t)
    s_sum = float(np.sum(s))
    if not np.isfinite(s_sum) or s_sum <= 0.0:
        return
    s /= s_sum

    lo = (1.0 - float(band))
    hi = (1.0 + float(band))

    over = s > (hi * t)
    under = s < (lo * t)

    # Build a single per-row scaling vector G (defaults to 1.0)
    G = np.ones(C, dtype=np.float64)

    if np.any(over):
        # exact factor to land on upper edge, then partial move
        f_over = (hi * t[over]) / s[over]
        G[over] = (1.0 - float(step)) + float(step) * f_over

    if np.any(under):
        # exact factor to land on lower edge, then partial move
        f_under = (lo * t[under]) / s[under]
        G[under] = (1.0 - float(step)) + float(step) * f_under

    # Streamed in-place row scaling: x[i,:] *= G[i]
    for i0 in range(0, C, int(row_chunk)):
        i1 = min(C, i0 + int(row_chunk))
        x[i0:i1, :] *= G[i0:i1, None]

    np.maximum(x, 0.0, out=x)  # keep nonnegativity

# ------------------------------------------------------------------------------

def _ensure_cp_flux_ref(h5_path: str,
                        keep_idx: np.ndarray | None,
                        dset: str = "/HyperCube/norm/cp_flux_ref",
                        *,
                        max_samples: int = 256,
                        floor: float = 1e-12) -> np.ndarray:
    """
    Ensure /HyperCube/norm/cp_flux_ref (C,P) exists. If missing, compute
    a robust per-(c,p) flux reference as the **median over spaxels** of
    the λ-sum of the (c,p) slice, using the current spectral mask.

    We sample up to `max_samples` spaxels (evenly spaced) for speed.
    """
    # Fast path: already present
    with open_h5(h5_path, role="reader") as f:
        if dset in f:
            ref = np.asarray(f[dset][...], dtype=np.float64, order="C")
            return ref

    # Build (approximate) median over spaxels of sum_λ A[s,c,p,λ]
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]  # (S,C,P,L)
        if M.ndim != 4:
            raise RuntimeError(f"Unexpected /HyperCube/models rank {M.ndim}")
        S, C, P, L = map(int, M.shape)

        if keep_idx is None:
            Lk = L
        else:
            keep_idx = np.asarray(keep_idx, dtype=np.int64)
            Lk = int(keep_idx.size)

        # choose evenly spaced spaxels
        Ns = int(min(S, max_samples))
        if Ns <= 0:
            raise RuntimeError("No spaxels available for cp_flux_ref.")
        picks = np.unique(np.linspace(0, S - 1, Ns).astype(int))

        # accumulate per-sample λ-sums → (Ns, C, P)
        ref_samp = np.empty((picks.size, C, P), dtype=np.float64)
        for j, s in enumerate(picks):
            A = np.asarray(M[s, :, :, :], dtype=np.float32, order="C")  # (C,P,L)
            if keep_idx is not None:
                A = A[:, :, keep_idx]                                    # (C,P,Lk)
            # sum over λ (keep_idx) in float64
            ref_samp[j, :, :] = np.sum(A.astype(np.float64, copy=False), axis=2)

        ref = np.median(ref_samp, axis=0)  # (C,P)
        ref = np.where(np.isfinite(ref), ref, 0.0)
        ref = np.maximum(ref, float(floor))

    # persist
    with open_h5(h5_path, role="writer") as f:
        g = f.require_group("/HyperCube/norm")
        if dset in f:
            del f[dset]
        d = f.create_dataset(dset, data=ref.astype(np.float64), chunks=(1, ref.shape[1]))
        g.attrs["basis.mode"] = "cp_flux_ref"  # mark what we applied

    return ref

# ------------------------------------------------------------------------------

def kacz_weighted_tile_whitelight(h5_path, x_flat, s0=0, Sblk=128, use_mask=True):
    """
    Rebuild y_hat for Sblk spaxels using the *weighted* operator (A_w, b_w)
    exactly like the solver, then collapse to white-light with the *same*
    weights/mask. Returns (data_sum, model_sum) arrays of length dS.
    """
    with open_h5(h5_path, role="reader") as f:
        M = f["/HyperCube/models"]       # (S, C, P, L), float32
        D = f["/DataCube"]               # (S, L), data
        wlam = f["/HyperCube/lambda_weights"][...] if "/HyperCube/lambda_weights" in f else None
        mask = f["/Masks/fit_mask"][...] if (use_mask and "/Masks/fit_mask" in f) else None

        S, C, P, L = map(int, M.shape)
        dS = int(min(Sblk, S - s0))
        x = np.asarray(x_flat, np.float64).ravel(order="C")
        assert x.size == C * P, f"x size {x.size} != C*P={C*P}"
        X = x.reshape(C, P)

        # λ selection consistent with the fit
        lam_sel = np.ones(L, dtype=bool)
        if mask is not None:
            lam_sel &= (mask != 0)
        lam_sel = np.where(lam_sel)[0]
        if lam_sel.size == 0:
            raise RuntimeError("Empty lambda selection.")

        # weights used by the solver (sqrt-w per row)
        w = np.ones(lam_sel.size, dtype=np.float64)
        if wlam is not None:
            w = np.sqrt(np.asarray(wlam, np.float64)[lam_sel])

        # build weighted rhs and model on the *same* λ grid
        D_sub = np.asarray(D[s0:s0+dS, lam_sel], np.float64, order="C")
        D_w   = D_sub * w[None, :]

        Y_w = np.zeros_like(D_w)  # (dS, L_sel), weighted model
        for c in range(C):
            A_sc = np.asarray(M[s0:s0+dS, c, :, lam_sel], np.float32, order="C")  # (dS, P, L_sel)
            # tensordot over P, then apply sqrt-w
            Y_w += np.tensordot(X[c, :].astype(np.float64, copy=False),
                                A_sc.astype(np.float64, copy=False),
                                axes=(0, 1)) * w[None, :]

        # white-light collapse with the *same* weights & λ selection
        data_sum  = D_w.sum(axis=1)
        model_sum = Y_w.sum(axis=1)
        return data_sum, model_sum

# Example usage (pick a central tile)
# data_sum, model_sum = kacz_weighted_tile_whitelight(h5_path, x_global, s0=0, Sblk=256)
# print("median(model/data) =", np.median(model_sum / np.maximum(1e-30, data_sum)))

# ------------------------------------------------------------------------------

def project_to_component_weights(
    x_cp: np.ndarray,
    t_vec: np.ndarray,          # target mixture for components (len C or C*P→C)
    *,
    E_cp: np.ndarray | None = None,   # (C,P) energy weights if you want energy-weighted usage
    minw: float = 1e-12,        # floor for targets and shares
) -> None:
    """
    Mass-preserving projection of the global component mixture of x_cp onto
    t_vec in ONE pass.

    If E_cp is provided, the mixture is defined by the energy-weighted usage:
        s_c = sum_p x[c,p] * E[c,p]
    otherwise by plain sums:
        s_c = sum_p x[c,p].

    This is O(C*P) and essentially instantaneous for (C,P)≈(207,360).

    In-place update of x_cp. Numerically safe: no NaNs/Infs are left behind.
    """
    # Ensure writable, C-contiguous float64 view
    X = np.require(x_cp, dtype=np.float64, requirements=["C", "W"])
    if X.ndim != 2:
        raise ValueError("x_cp must be 2-D (C,P).")
    C, P = X.shape

    # ---------- sanitize target t_vec -> per-component target t[c] ----------
    t_raw = np.asarray(t_vec, dtype=np.float64).ravel(order="C")
    if t_raw.size == C * P:
        t_raw = t_raw.reshape(C, P, order="C").sum(axis=1)
    elif t_raw.size != C:
        raise ValueError(f"[ratio] target len {t_raw.size} not in {{C, C*P}}")

    t_raw = np.nan_to_num(t_raw, nan=0.0, posinf=0.0, neginf=0.0)
    t_raw = np.maximum(t_raw, 0.0)

    # Apply a floor so very small targets don't blow up ratios
    t = np.maximum(t_raw, float(minw))
    T = float(t.sum())
    if not np.isfinite(T) or T <= 0.0:
        # Nothing meaningful to enforce
        return

    # ---------- current usage s[c] in chosen metric (plain vs energy) ----------
    if E_cp is not None:
        E = np.asarray(E_cp, dtype=np.float64)
        if E.shape != (C, P):
            raise ValueError(f"E_cp shape {E.shape} != (C,P) {(C, P)}")
        E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        s = (X * E).sum(axis=1)
    else:
        s = X.sum(axis=1)

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.maximum(s, 0.0)
    S = float(s.sum())
    if not np.isfinite(S) or S <= 0.0:
        # Degenerate solution; nothing to rescale
        return

    # ---------- sprinkle tiny mass where s[c]==0 but t[c]>0 ----------
    need = (s <= 0.0) & (t > 0.0)
    if np.any(need):
        eps = 1.0e-12 * S
        X[need, :] = eps / float(P)
        # recompute usage in the same metric
        if E_cp is not None:
            s = (X * E).sum(axis=1)
        else:
            s = X.sum(axis=1)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.maximum(s, 1.0e-30)
        S = float(s.sum())
        if not np.isfinite(S) or S <= 0.0:
            return

    # ---------- compute safe scale factors G[c] ----------
    # We want new usage s'_c proportional to t_c, but with the same total mass S:
    #     s'_c = t_c * (S / T)
    target_usage = t * (S / T)           # (C,)
    s_safe = np.maximum(s, 1.0e-30)
    G = target_usage / s_safe            # (C,)

    # Clean any residual NaNs/Infs in G
    G = np.nan_to_num(G, nan=1.0, posinf=1.0, neginf=0.0)

    # ---------- apply scaling in-place & clean ----------
    X *= G[:, None]
    np.maximum(X, 0.0, out=X)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------------------------------------------------------------------

def project_to_component_weights_strict(
    x_cp: np.ndarray,
    orbit_weights: np.ndarray,
    *,
    E_cp: np.ndarray | None = None,
    min_target: float = 1e-10,
) -> None:
    """
    In-place rescale of X[c,p] so that component usage matches orbit_weights.

    If E_cp is provided (C,P), usage is energy-weighted:
        s_c = sum_p X[c,p] * E[c,p]
    Otherwise usage is plain:
        s_c = sum_p X[c,p]

    The projection is:
        - mass-preserving in the chosen metric (plain or energy-weighted),
        - strictly non-negative,
        - numerically safe (no NaNs / Infs).
    """
    # Ensure writable, C-contiguous float64 view
    X = np.require(x_cp, dtype=np.float64, requirements=["C", "W"])
    if X.ndim != 2:
        raise ValueError("x_cp must be 2-D (C,P).")
    C, P = X.shape

    # ---------- sanitize orbit_weights -> per-component target t_c ----------
    w = np.asarray(orbit_weights, dtype=np.float64).ravel(order="C")
    if w.size == C * P:
        w = w.reshape(C, P, order="C").sum(axis=1)
    elif w.size != C:
        raise ValueError(
            f"orbit_weights size {w.size} incompatible; expected {C} or {C*P}."
        )

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    Wsum = float(w.sum())
    if not np.isfinite(Wsum) or Wsum <= 0.0:
        # Nothing meaningful to enforce
        return

    # Target fractions per component in [min_target, 1]
    t = w / Wsum
    t = np.maximum(t, float(min_target))
    T = float(t.sum())
    if not np.isfinite(T) or T <= 0.0:
        return

    # ---------- usage in the chosen metric ----------
    if E_cp is not None:
        E = np.asarray(E_cp, dtype=np.float64)
        if E.shape != (C, P):
            raise ValueError(f"E_cp shape {E.shape} != (C,P) {(C, P)}")
        E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        s = (X * E).sum(axis=1)
    else:
        s = X.sum(axis=1)

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.maximum(s, 0.0)
    S = float(s.sum())

    if not np.isfinite(S) or S <= 0.0:
        # Degenerate solution; nothing to rescale
        return

    # ---------- sprinkle tiny mass where usage is zero but target > 0 ----------
    need = (s <= 0.0) & (t > 0.0)
    if np.any(need):
        eps = 1.0e-12 * S
        X[need, :] = eps / float(P)
        if E_cp is not None:
            s = (X * E).sum(axis=1)
        else:
            s = X.sum(axis=1)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.maximum(s, 1.0e-30)
        S = float(s.sum())

    # ---------- compute safe scale factors G[c] ----------
    # We want new usage s'_c ∝ t_c, but keep the same total mass S in this metric:
    #   s'_c = t_c * (S / T)
    target_usage = t * (S / T)              # (C,)
    s_safe = np.maximum(s, 1.0e-30)
    G = target_usage / s_safe               # (C,)

    # Clean any residual nastiness
    G = np.nan_to_num(G, nan=1.0, posinf=1.0, neginf=0.0)

    # ---------- apply in-place + final cleanup ----------
    X *= G[:, None]
    np.maximum(X, 0.0, out=X)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------------------------------------------------------------------