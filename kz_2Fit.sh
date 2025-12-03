#!/bin/bash -l
# #SBATCH -A durham
# #SBATCH -D "/cosma5/data/durham/dc-poci1/muse"
# #SBATCH --output="/cosma5/data/durham/dc-poci1/muse/slurm_tri_losvdDecomp.log" --open-mode=append
# #SBATCH --error="/cosma5/data/durham/dc-poci1/muse/slurm_tri_losvdDecomp.log" --open-mode=append

# #SBATCH -A oz059
# #SBATCH -D "/fred/oz059/poci/muse"
# #SBATCH --output="/fred/oz059/poci/muse/slurm_tri_losvdDecomp.log" --open-mode=append
# #SBATCH --error="/fred/oz059/poci/muse/slurm_tri_losvdDecomp.log" --open-mode=append

#SBATCH -D "/data/phys-gal-dynamics/phys2603/muse"
#SBATCH --output="/data/phys-gal-dynamics/phys2603/CubeFit/log_2Fit.log" --open-mode=append
#SBATCH --error="/data/phys-gal-dynamics/phys2603/CubeFit/log_2Fit.log" --open-mode=append
#SBATCH -p short

#SBATCH --job-name="CubeFit_2Fit"
#SBATCH --time=0-12:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk

module purge
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0

# --- BLAS / OpenMP threading (per worker) ---
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
# For OpenBLAS, avoid accidental main-thread reuse:
export OPENBLAS_VERBOSE=0

# --- HDF5 raw-data chunk cache (reader side) ---
export CUBEFIT_RDCC_NBYTES=$((16*1024*1024*1024))  # 16 GiB
export CUBEFIT_RDCC_NSLOTS=400003                   # large-ish prime
export CUBEFIT_RDCC_W0=0.9

export HDF5_USE_FILE_LOCKING=FALSE

export CUBEFIT_NNLS_ENABLE=0

# File descriptors
ulimit -n 8192

cd /data/phys-gal-dynamics/phys2603/CubeFit

# sanity print (once) to confirm cpuset and BLAS threads
srun -n1 -c${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  python - <<'PY'
import os, json
print(f"[sanity] cpuset cores: {len(os.sched_getaffinity(0))}")
try:
    from threadpoolctl import threadpool_info
    print("[sanity] BLAS pools:", json.dumps(threadpool_info(), indent=2)[:600], "...")
except Exception as e:
    print("[sanity] threadpoolctl not available:", e)
PY

# run your job as a Slurm step (gives you the full cpuset)
srun -n1 -c${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  python -m IPython kz_run.py -- --run-switch fit