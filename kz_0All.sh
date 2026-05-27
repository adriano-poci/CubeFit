#!/bin/bash -l

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
Usage: $0 --cluster CLUSTER GALAXY [-n N] [--ncomp=N] [--ncomp N]
  --cluster CLUSTER   Slurm cluster to submit to, e.g. arc or htc
  GALAXY              galaxy name (string, required)
  -n N                number of components
  --ncomp=N           same as -n
EOF
}

CLUSTER=""
NCOMP=""

# Parse long options first.
new_argv=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cluster=*)
            CLUSTER="${1#--cluster=}"
            shift
            ;;
        --cluster)
            if [ "$#" -lt 2 ]; then
                echo "Error: --cluster requires an argument." >&2
                usage
                exit 2
            fi
            CLUSTER="$2"
            shift 2
            ;;
        --ncomp=*)
            NCOMP="${1#--ncomp=}"
            shift
            ;;
        --ncomp)
            if [ "$#" -lt 2 ]; then
                echo "Error: --ncomp requires an argument." >&2
                usage
                exit 2
            fi
            NCOMP="$2"
            shift 2
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                new_argv+=("$1")
                shift
            done
            break
            ;;
        *)
            new_argv+=("$1")
            shift
            ;;
    esac
done

set -- "${new_argv[@]:-}"

while getopts ":n:" opt; do
    case "$opt" in
        n) NCOMP="$OPTARG" ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ -z "$CLUSTER" ]; then
    echo "Error: --cluster is required." >&2
    usage
    exit 2
fi

if [ "$#" -lt 1 ]; then
    echo "Error: GALAXY argument is required." >&2
    usage
    exit 2
fi

GALAXY="$1"
shift

if [ -n "${NCOMP:-}" ]; then
    if ! printf '%s' "$NCOMP" | grep -Eq '^[0-9]+$'; then
        echo "Error: ncomp must be a positive integer, got '$NCOMP'." >&2
        exit 2
    fi
    if [ "$NCOMP" -le 0 ]; then
        echo "Error: ncomp must be > 0, got '$NCOMP'." >&2
        exit 2
    fi
fi

cd /data/phys-gal-dynamics/phys2603/CubeFit

common_args=("$GALAXY")
if [ -n "${NCOMP:-}" ]; then
    common_args+=("--ncomp=$NCOMP")
fi

fSGA_raw=$(sbatch --parsable -M "$CLUSTER" \
    --export=ALL,CF_CLUSTER="$CLUSTER" \
    kz_1Gen.sh "${common_args[@]}")
fSGA=${fSGA_raw%%;*}

fSF_raw=$(sbatch --parsable -M "$CLUSTER" \
    --export=ALL,CF_CLUSTER="$CLUSTER" \
    --dependency=afterok:"$fSGA" \
    kz_2Fit.sh "${common_args[@]}")
fSF=${fSF_raw%%;*}

sbatch -M "$CLUSTER" \
    --export=ALL,CF_CLUSTER="$CLUSTER" \
    --dependency=afterok:"$fSF" \
    kz_3Rio.sh "${common_args[@]}"