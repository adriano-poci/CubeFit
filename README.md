# CubeFit

CubeFit is a high-throughput pipeline for orbital spectral–fitting on IFU data cubes.  
It builds a **HyperCube** of convolved template populations, then solves for global
population weights with a parallel **Kaczmarz** solver, including sidecar tracking,
a live dashboard, and memory-safe reconstruction/plotting utilities.

- Fast, chunk-aligned HDF5 I/O (SWMR-friendly)
- On-the-fly or cached λ-weights and global column energy
- Robust multi-process Kaczmarz with trust-region, backtracking, optional NNLS polish
- Fit sidecar writer for live progress & EWMA metrics
- Reconstruction of `ModelCube` without materializing the full basis
- Lightweight plotting (best/worst residual spectra) and diagnostics

> **Full documentation** → [`docs/CubeFit.md`](docs/CubeFit.md)  
> **Environment variables (all knobs)** → [`docs/CubeFit.md#environment-variables`](docs/CubeFit.md#environment-variables)

---

## Installation

Requirements: Python ≥ 3.10, NumPy, SciPy, h5py, matplotlib, tqdm, threadpoolctl, (optional) Cython.

```bash
# (optional) create an environment
python -m venv .venv && source .venv/bin/activate

# install package & deps
pip install -r requirements.txt  # if present
pip install -e .
```

If your build includes Cython extensions (e.g., continuum detrending), ensure a C compiler
is available; `pip install -e .` will compile them in place.

---

## Quickstart

Below is a minimal end-to-end sketch. See the manual for full options and data prep.

```python
from CubeFit.hypercube_builder import build_hypercube
from CubeFit.pipeline_runner import PipelineRunner

H5 = "CubeFit/NGC4365/NGC4365_04.h5"

# 1) Build the HyperCube (choose normalization you’ll use in fitting)
build_hypercube(
    H5,
    norm_mode="model",   # or "data" (see docs on normalization)
    amp_mode="sum"       # or "trapz" (requires /VelPix)
)

# 2) Solve (multi-process Kaczmarz); see docs for all knobs/args
pr = PipelineRunner(H5)
x_global, stats = pr.solve_all_mp_batched(
    epochs=3,
    pixels_per_aperture=1247,
    lr=0.9,
    project_nonneg=True,
    processes=8,
    blas_threads=8,
)

# 3) Reconstruct the model cube for diagnostics/plots
from CubeFit.kz_fitSpec import reconstruct_model_cube_single
reconstruct_model_cube_single(h5_path=H5, x_global=x_global, array_name="ModelCube", blas_threads=8)
```

**Tips**
- If your HyperCube was built in the other normalization, you can convert in-place
  (streaming) without a full rebuild. See **Normalization utilities** in the manual.
- λ-weights (feature emphasis) and global column energy are read/cached automatically if present.

---

## Building the docs

This repo ships a full manual at `docs/CubeFit.md`. You can build HTML/PDF with your Makefile:

```bash
make html
make pdf
```

MkDocs config (`mkdocs.yml`) already points navigation to `docs/CubeFit.md`.

---

## Where to next?

- **Full reference & walkthrough**: [`docs/CubeFit.md`](docs/CubeFit.md)  
- **All environment variables** (solver, NNLS, tracking, mp, etc.):
  [`docs/CubeFit.md#environment-variables`](docs/CubeFit.md#environment-variables)

---

## License

See `LICENSE` (if present). If you plan to publish results, please cite the repository and any underlying data/model libraries you use (e.g., MILES/eMILES).
