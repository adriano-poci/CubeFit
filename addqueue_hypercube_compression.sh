#!/bin/bash

set -euo pipefail
curdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE='planet'
NodeMem=64
nCPU=32

memPerCPU=$(echo "scale=2; $NodeMem*0.99 / $nCPU" | bc)
#memPerCPU=$(echo "scale=2; 110.0 / $nCPU" | bc)
# memPerCPU=10
printf "Running ${nCPU} cores on ${QUEUE},\n using ${memPerCPU} GB per core.\n\n"

addqueue --sbatch --requeue --serial -n "1x${nCPU}" -q "${QUEUE}" \
    -m "${memPerCPU}" -c "hypercube_cold" -g "hypercube" \
    -o "${curdir}/hypercube_cold.log" "${curdir}/hypercube_compression.sh" \
    "${curdir}/FCC170/hypercube_134_00.h5" zstd