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
#SBATCH -p medium

#SBATCH --job-name="CubeFit_2Fit"
#SBATCH --time=2-00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=700G
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk

module purge
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0

# --- BLAS / OpenMP threading (per worker) ---
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
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

# File descriptors
ulimit -n 8192

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



# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

usage() {
    cat <<EOF
Usage: $0 GALAXY [-n N] [--ncomp=N] [--ncomp N] [positional...]
  GALAXY         galaxy name (string, required)
  -n N           short form
  --ncomp=N      long form (either form optional)
If provided, N must be a positive integer.
EOF
}

NCOMP=""
# Build a new argv array excluding any long-form --ncomp tokens
new_argv=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --ncomp=*)
            NCOMP="${1#--ncomp=}"
            shift
            ;;
        --ncomp)
            if [ "$#" -lt 2 ]; then
                echo "Error: --ncomp requires an argument." >&2
                usage; exit 2
            fi
            NCOMP="$2"
            shift 2
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                new_argv+=("$1"); shift
            done
            break
            ;;
        *)
            new_argv+=("$1")
            shift
            ;;
    esac
done

# Replace positional parameters with filtered args for getopts
set -- "${new_argv[@]:-}"

# Parse short options (-n)
while getopts ":n:" opt; do
    case "$opt" in
        n) NCOMP="$OPTARG" ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

# ------------------------------------------------------------------
# Positional arguments
# ------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
    echo "Error: GALAXY argument is required." >&2
    usage
    exit 2
fi

GALAXY="$1"
shift

# Remaining positional args (if any) are now in "$@"

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
if [ -n "${NCOMP:-}" ]; then
    if ! printf '%s' "$NCOMP" | grep -Eq '^[0-9]+$'; then
        echo "Error: ncomp must be a positive integer, got '$NCOMP'." >&2
        exit 2
    fi
    if [ "$NCOMP" -le 0 ]; then
        echo "Error: ncomp must be > 0, got '$NCOMP'." >&2
        exit 2
    fi
    echo "GALAXY = $GALAXY"
    echo "NCOMP  = $NCOMP"
else
    echo "GALAXY = $GALAXY"
    echo "NCOMP not provided; running with defaults"
fi
# ------------------------------------------------------------------------------
# /Argument parsing
# ------------------------------------------------------------------------------



# run your job as a Slurm step (gives you the full cpuset)
cd /data/phys-gal-dynamics/phys2603/CubeFit
echo "Submitted cluster: ${CF_CLUSTER:-unknown}"
srun -n1 -c${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  python -m IPython --colors=NoColor kz_run.py -- --galaxy "$GALAXY" --run-switch fit ${NCOMP:+--ncomp="$NCOMP"}