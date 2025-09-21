#!/bin/bash

while getopts ":g:" arg; do
  case $arg in
    g) galax=$OPTARG;;
  esac
done

curdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="cmb"
NodeMem=273
nCPU=32

memPerCPU=$(echo "scale=2; $NodeMem*0.99 / $nCPU" | bc)
#memPerCPU=$(echo "scale=2; 110.0 / $nCPU" | bc)
# memPerCPU=10
printf "Running ${nCPU} cores on ${QUEUE},\n using ${memPerCPU} GB per core.\n\n"

addqueue --sbatch --requeue --serial -n "${nCPU}" -q "${QUEUE}" \
    -m "${memPerCPU}" -c "genAper_${galax}" -g "CubeFit${galax}" \
    -o "${curdir}/log.log" "${curdir}/kz_rio.sh"
    # -o "${curdir}/log.log" "${curdir}/kz_run.sh"
