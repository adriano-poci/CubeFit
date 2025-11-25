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
#SBATCH --output="/data/phys-gal-dynamics/phys2603/CubeFit/log_3Rio.log" --open-mode=append
#SBATCH --error="/data/phys-gal-dynamics/phys2603/CubeFit/log_3Rio.log" --open-mode=append
#SBATCH -p short

#SBATCH --job-name="CubeFit_3Rio"
#SBATCH --time=0-12:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk

module purge
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0

# glibc / allocator hygiene
export MALLOC_ARENA_MAX=2

# Threading (OpenBLAS-backed NumPy)
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}      # harmless with OpenBLAS(pthreads)
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset GOMP_CPU_AFFINITY
export KMP_AFFINITY=disabled
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# HDF5 raw chunk cache (4 GiB is plenty; bump if you like)
export CUBEFIT_RDCC_NBYTES=$((4*1024*1024*1024))  # 4 GiB
export CUBEFIT_RDCC_NSLOTS=400003
export CUBEFIT_RDCC_W0=0.9

# FitTracker tweaks (reduce sidecar churn; enforce spawn)
export CUBEFIT_RMSE_STRIDE=16
export CUBEFIT_TRACKER_QSIZE=8192
export FITTRACKER_START=spawn

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
  python -m IPython kz_rio.py -- --redraw