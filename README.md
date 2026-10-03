# CubeFit

CubeFit is an HDF5-backed pipeline for fitting stellar-population mixtures to
spatially resolved IFU spectroscopy while retaining the spatial and kinematic
structure supplied by a Schwarzschild/orbit decomposition.

> **Documentation status:** this README and `docs/CubeFit.md` describe the
> repository snapshot packed on 3 October 2026. They replace the old
> Kaczmarz-centric documentation. The production pathway in this snapshot is
> `kz_fitSpec.genCubeFit()` -> `PipelineRunner.solve_all_mp_batched()` ->
> `streaming_nnls_constrained.solve_streaming_nnls()` ->
> `streamActiveSetNNLS()`.

## What CubeFit solves

Let

- `S` be the number of spatial bins;
- `L` the number of fitted wavelength pixels;
- `C` the number of dynamical components; and
- `P` the number of flattened SSP populations.

The HyperCube stores

```text
A[s,c,p,l] = model spectrum for spatial bin s,
             component c, population p, wavelength l
```

and CubeFit solves for a global non-negative coefficient matrix `X[c,p]`:

```text
Y_hat[s,l] = sum_c sum_p A[s,c,p,l] * X[c,p].
```

The production HyperCube is normally built with `norm_mode="model"`. In that
mode the LOSVD amplitude of every `(s,c)` component is multiplied into the
HyperCube exactly once. The relative Schwarzschild component strength is
therefore already encoded in `A`.

## Current orbit constraint

When `orbitWeights=False`, the fit is unconstrained apart from `X >= 0`.

When orbit weights are enabled, the current solver uses the v2.0
`equal_component_mass` formulation. Positive dynamical weights select the
components that participate in the constraint, but their magnitudes are not
applied to `X` a second time. Instead,

```text
sum_p X[c,p] = alpha
```

for every positive-weight component, with a common fitted `alpha >= 0`.

This is deliberate: under model normalization the relative dynamical masses
are already present in the HyperCube. The constraint removes the otherwise
arbitrary component-level rescaling available to NNLS without double-applying
the Schwarzschild weights.

## Main entry points

The high-level science interface is:

```python
from CubeFit.kz_fitSpec import genCubeFit, loadCubeFit
```

`genCubeFit()` prepares the spectral/dynamical inputs, writes the HDF5
backbone, preflights and builds the HyperCube, and optionally runs the global
streaming active-set NNLS solver.

`loadCubeFit()` loads a completed fit, reconstructs `/ModelCube` when needed,
computes residual diagnostics, and writes the scientific plots.

Lower-level entry points include:

```python
from CubeFit.hypercube_builder import build_hypercube
from CubeFit.pipeline_runner import PipelineRunner
from CubeFit.streaming_nnls_constrained import solve_streaming_nnls
```

## High-level run controls

`genCubeFit()` consumes the following important values from `**kwargs`:

```text
warm                  zeros | saved_x | resume | upscale...
regularisation_scale  non-negative float; default 1.0
cpu_processes          requested process count; default 12
blas_threads           requested BLAS threads/process; default 4
hdf5Dir                output directory
orbitWeights           False by default
validationPath         synthetic validation input HDF5
validationDataset      validation spectra dataset; default /DataCube
validationTag          required for validation runs
```

`runSwitch='gen'` builds the HyperCube and returns. `runSwitch='fit'` ensures
the HyperCube exists and then fits it.

## Parallelism

`genCubeFit()` passes `cpu_processes` and `blas_threads` through
`cube_utils.resolve_parallelism()`, which respects the process CPU affinity /
Slurm cpuset and avoids a requested `processes * BLAS_threads` total above the
available allocation.

Set numerical-library thread variables before Python starts, especially on
Slurm:

```bash
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export BLIS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
```

The worker initializers in this repository explicitly set OpenMP, OpenBLAS and
MKL thread counts. They do **not** set `BLIS_NUM_THREADS`, so BLIS users should
set it in the launch environment.

## Validation / closure tests

`validation.makeValidationCube()` creates a deliberately small validation
input from a completed `/ModelCube`. It writes only:

```text
/DataCube   synthetic spectra, (S,L), float64, gzip
/ObsPix     observed log-wavelength grid, (L,)
```

The full `/HyperCube/models` dataset is not copied.

Example:

```python
from CubeFit.validation import makeValidationCube

makeValidationCube(
    'CubeFit/NGC4365/hypercube_3_01.h5',
    'CubeFit/NGC4365/hypercube_3_01_mock-data.h5',
)
```

Then fit it with a distinct output tag:

```python
genCubeFit(
    ...,
    runSwitch='fit',
    validationPath='CubeFit/NGC4365/hypercube_3_01_mock-data.h5',
    validationDataset='/DataCube',
    validationTag='closure-01',
)
```

The new fit is written to a tagged HyperCube such as
`hypercube_3_01_closure-01.h5`; validation figures are written under
`<galaxy>/validation/closure-01/`.

Coefficient recovery can then be audited with
`validation.compare_validation_solution()`.

## Diagnostics

The active solver emits JSONL diagnostics when configured with:

```bash
export CUBEFIT_DIAG_LEVEL=2
export CUBEFIT_DIAG_STRIDE=1
export CUBEFIT_DIAG_TOPK=12
export CUBEFIT_DIAG_JSONL=/path/to/diagnostics.jsonl
export CUBEFIT_SOLVER_CHECKPOINT_EVERY=1
```

The most useful records are `setup`, `mono_setup`, `kkt`,
`exploration_trial`, `exploration_accept`, `exploration`, and the final stop
summary.

## Important current implementation notes

The source snapshot contains several historical APIs and environment knobs.
The production constrained solver does **not** use the old Kaczmarz trust
region, tile-local NNLS polish, orbit softbox, or the old proportional
orbit-weight penalty described by the previous documentation.

Likewise, several `CUBEFIT_*` variables still set by `kz_run.py` belong to
older solver paths and are not read by `streaming_nnls_constrained.py`. The
current constrained solver directly reads the diagnostic and checkpoint
variables listed above.

The current `build_hypercube()` creates a spawned process pool, but this
snapshot submits one `(s-tile,c-tile,p-block)` future and waits for it before
submitting the next tile. Consequently, `processes>1` does not create multiple
simultaneous tile jobs in this exact implementation. See the manual's
performance section before interpreting process-count benchmarks.

## Documentation

- Full manual: [`docs/CubeFit.md`](docs/CubeFit.md)
- Standalone HTML: [`docs/CubeFit.html`](docs/CubeFit.html)
- PDF manual: [`docs/CubeFit.pdf`](docs/CubeFit.pdf)

The manual includes the HDF5 schema, normalization semantics, v2.0 orbit
constraint, active-set search, dyadic exploration, regularization, resume
compatibility, validation workflow, plotting, storage management, diagnostics,
HPC operation, and a source-derived API index.
