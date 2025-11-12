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
#SBATCH --output="/data/phys-gal-dynamics/phys2603/CubeFit/log_1Gen.log" --open-mode=append
#SBATCH --error="/data/phys-gal-dynamics/phys2603/CubeFit/log_1Gen.log" --open-mode=append
#SBATCH -p short

#SBATCH --job-name="CubeFit_1Gen"
#SBATCH --time=0-12:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk
#SBATCH --hint=nomultithread
#SBATCH --exclusive

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
# run your job as a Slurm step (gives you the full cpuset)
srun -n1 -c${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  python -m IPython kz_run.py -- --run-switch 'gen' --redraw