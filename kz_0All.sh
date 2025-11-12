#!/bin/bash -l

module purge
module load Python/3.11.3-GCCcore-12.3.0
# glibc / allocator hygiene
export MALLOC_ARENA_MAX=2

# Threading (OpenBLAS-backed NumPy)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset GOMP_CPU_AFFINITY
export KMP_AFFINITY=disabled
export MKL_NUM_THREADS=1            # harmless, prevents surprise MKL use elsewhere
export NUMEXPR_NUM_THREADS=1        # avoid hidden extra threads

# File descriptors
ulimit -n 8192

# HDF5 raw chunk cache (tune if memory pressure)
export CUBEFIT_RDCC_NBYTES=$((16*1024*1024*1024))  # 8–16 GiB are sane
export CUBEFIT_RDCC_NSLOTS=400003
export CUBEFIT_RDCC_W0=0.9

cd /data/phys-gal-dynamics/phys2603/CubeFit
# declare fSGA=$(sbatch --parsable "kz_1Gen.sh")
# declare fSF=$(sbatch --parsable --dependency=afterok:"${fSGA}" "kz_2Fit.sh")
declare fSF=$(sbatch --parsable "kz_2Fit.sh")
sbatch --dependency=afterok:"${fSF}" "kz_3Rio.sh"
