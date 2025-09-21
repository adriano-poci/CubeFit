"""
example_run.py
==============

Minimal synthetic sanity‑check:

* Builds random LOSVD / templates / data that satisfy the final CubeFit
  conventions.
* Saves them into a Zarr hierarchy via ``ZarrManager``.
* Runs ``PipelineRunner.solve_all`` on a *tiny* cube (faster than
  `run_with_real_data.py`), prints the design‑matrix shape for aperture 0.

Run::

    python example_run.py
"""

from __future__ import annotations
import os
import numpy as np

from CubeFit.zarr_manager import ZarrManager
from CubeFit.pipeline_runner import PipelineRunner
from CubeFit.model_cube import ModelCube

# ------------------------------------------------------------------- #
# Synthetic cube dimensions (tiny – adjust if you want larger tests)
# ------------------------------------------------------------------- #
n_spat = 16          # spatial apertures
n_vel = 60           # LOSVD velocity bins
n_comp = 3           # dynamical components
n_pix_obs = 800      # observed spectral pixels
n_pix_tem = 1000     # template spectral pixels
n_met, n_age, n_alp = 2, 2, 2
n_pop = n_met * n_age * n_alp

rng = np.random.default_rng(0)

# ------------------------------------------------------------------- #
# Synthetic arrays in *external/user* shapes
# ------------------------------------------------------------------- #
data = np.abs(rng.normal(1.0, 0.1, (n_pix_obs, n_spat)))
templates = np.abs(
    rng.normal(1.0, 0.1, (n_pix_tem, n_met, n_age, n_alp))
)
losvd = np.abs(rng.normal(1.0, 0.1, (n_spat, n_vel, n_comp)))
tem_pix = np.linspace(4000.0, 8000.0, n_pix_tem)
obs_pix = np.linspace(4100.0, 7900.0, n_pix_obs)

# ------------------------------------------------------------------- #
# Initialise Zarr hierarchy and load arrays
# ------------------------------------------------------------------- #
z_dir = "./cubefit_synth"
os.makedirs(z_dir, exist_ok=True)

zmgr = ZarrManager(z_dir, n_spat, n_comp, n_vel, n_pop, n_pix_obs)
zmgr.load_data_from_arrays(data, templates, losvd, tem_pix, obs_pix)

# ------------------------------------------------------------------- #
# Build model matrix for aperture 0 (sanity‑check)
# ------------------------------------------------------------------- #
A0 = ModelCube(zmgr.z["Templates"][:], losvd[0], tem_pix, obs_pix).convolve()
print("Design‑matrix shape  (ap 0):", A0.shape)     # expect (800, n_comp*n_pop)

# ------------------------------------------------------------------- #
# Solve across apertures (small cube – finish quickly)
# ------------------------------------------------------------------- #
runner = PipelineRunner(
    z_dir,
    n_spat,
    n_comp,
    n_vel,
    n_pop,
    n_pix_obs,
    tem_pix=tem_pix,
    obs_pix=obs_pix,
)
runner.solve_all(n_jobs=2, tol=1e-3, max_iter=2000, verbose=True)
print("Synthetic example finished.")
