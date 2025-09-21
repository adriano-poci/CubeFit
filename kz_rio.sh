#!/bin/bash
#SBATCH --job-name="fs_fitNGC4365"
#SBATCH --output="/mnt/extraspace/poci/fs_fitNGC4365.log" --open-mode=append
#SBATCH --error="/mnt/extraspace/poci/muse/fs_fitNGC4365.log" --open-mode=append
#SBATCH --time=0-12:00
#SBATCH -D "/mnt/extraspace/poci/muse"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=30G
#SBATCH -p short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk

module purge

#module load Anaconda3/2024.02-1 foss/2023a
#module load Python/3.11.3-GCCcore-12.3.0

module load gcc/13.2
module load openmpi
module load python/3.11.4
export OMPI_MCA_btl=^openib
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DISPLAY_ENV=true

# Get the directory of the script, resolving symlinks
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

Ipy='ipython --pylab --pprint --autoindent'
$Ipy ${SCRIPT_DIR}/kz_rio.py