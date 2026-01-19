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
#SBATCH -p medium

#SBATCH --job-name="CubeFit_1Gen"
#SBATCH --time=0-48:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk
#SBATCH --hint=nomultithread
#SBATCH --exclusive

module purge
module load foss/2023a
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




cd /data/phys-gal-dynamics/phys2603/CubeFit
# run your job as a Slurm step (gives you the full cpuset)
srun -n1 -c${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  python -m IPython kz_run.py -- --galaxy "$GALAXY" --run-switch 'gen' --redraw ${NCOMP:+--ncomp="$NCOMP"}