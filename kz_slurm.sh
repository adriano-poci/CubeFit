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
#SBATCH --output="/data/phys-gal-dynamics/phys2603/CubeFit/log.log" --open-mode=append
#SBATCH --error="/data/phys-gal-dynamics/phys2603/CubeFit/log.log" --open-mode=append
#SBATCH -p short

#SBATCH --job-name="CubeFit"
#SBATCH --time=0-12:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk
#SBATCH --hint=nomultithread

module purge
module load Python/3.11.3-GCCcore-12.3.0
export MALLOC_ARENA_MAX=2
export MKL_DISABLE_FAST_MM=1
export KMP_BLOCKTIME=0 # snappier threads, less idle memory
export OMPI_MCA_btl=^openib
# then at runtime:
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
# good binding defaults:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ulimit -n 8192

cd /data/phys-gal-dynamics/phys2603/CubeFit
# ipython kz_run.py
ipython kz_rio.py
