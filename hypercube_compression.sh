#!/bin/bash
#SBATCH --job-name=cubefit_cold
#SBATCH --output=cubefit_cold_%j.out
#SBATCH --error=cubefit_cold_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# Usage:
#
#   sbatch cubefit_cold.slurm /path/to/hypercube_109.h5 zstd
#
# or:
#
#   sbatch cubefit_cold.slurm /path/to/hypercube_109.h5 7z
#
# The source HDF5 is removed only after the archive has been successfully
# created and verified by hypercube_cold().

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: sbatch $0 <hypercube.h5> [zstd|7z]"
    exit 2
fi

H5_PATH="$1"
METHOD="${2:-zstd}"

if [ ! -f "${H5_PATH}" ]; then
    echo "ERROR: HyperCube does not exist:"
    echo "  ${H5_PATH}"
    exit 1
fi

case "${METHOD}" in
    zstd|7z)
        ;;
    *)
        echo "ERROR: compression method must be 'zstd' or '7z'."
        exit 2
        ;;
esac

# Prefer the CPU allocation supplied by SLURM. Outside SLURM, use the
# number of CPUs currently available to this process.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    NCPUS="${SLURM_CPUS_PER_TASK}"
elif command -v nproc >/dev/null 2>&1; then
    NCPUS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
    NCPUS="$(getconf _NPROCESSORS_ONLN)"
else
    NCPUS=1
fi

echo "============================================================"
echo "CubeFit HyperCube cold-storage compression"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "CPUs         : ${NCPUS}"
echo "Method       : ${METHOD}"
echo "Input        : ${H5_PATH}"
echo "Started      : $(date)"
echo "============================================================"

# ----------------------------------------------------------------------
# Load your normal CubeFit environment here.
#
# Replace these lines with the environment setup you normally use on the
# cluster.
# ----------------------------------------------------------------------

module purge
module load gcc/13.2
module load hdf5

# ----------------------------------------------------------------------
# Run the Python compression function.
#
# Change the import below to wherever you place hypercube_cold().
# ----------------------------------------------------------------------

python - "${H5_PATH}" "${METHOD}" "${NCPUS}" <<'PY'
import sys

from CubeFit.cube_utils import hypercube_cold

h5_path = sys.argv[1]
method = sys.argv[2]
threads = int(sys.argv[3])

archive_path = hypercube_cold(h5_path, method=method,
    remove_original=False, overwrite=False, threads=threads)

print()
print(f"[SLURM] Cold archive ready: {archive_path}")
PY

echo "============================================================"
echo "Compression completed successfully."
echo "Finished     : $(date)"
echo "============================================================"