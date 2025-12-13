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
from pathlib import Path
from typing import Optional, Sequence

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import read_global_column_energy
from CubeFit.cube_utils import (
    project_to_component_weights,
    # project_to_component_weights_strict,
)
from CubeFit.live_fit_dashboard import (
    render_aperture_fits_with_x,
    render_sfh_from_x,
)

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
    project_to_component_weights(
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
    # project_to_component_weights_strict(
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

    if ap_idx:
        render_aperture_fits_with_x(
            h5_path,
            x_flat,
            fits_png,
            apertures=ap_idx,
            show_residual=show_residual,
            title="NNLS seed treated as final solution",
        )

    render_sfh_from_x(
        h5_path,
        x_flat,
        sfh_png,
    )

    return {
        "seed_path": str(seed_path),
        "apertures": ap_idx,
        "fits_png": str(fits_png),
        "sfh_png": str(sfh_png),
    }

# ------------------------------------------------------------------------------