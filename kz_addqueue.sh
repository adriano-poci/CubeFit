#!/bin/bash

while getopts ":g:" arg; do
  case $arg in
    g) galax=$OPTARG;;
  esac
done

curdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="redwood"
NodeMem=350
nCPU=48

memPerCPU=$(echo "scale=2; $NodeMem*0.99 / $nCPU" | bc)
#memPerCPU=$(echo "scale=2; 110.0 / $nCPU" | bc)
# memPerCPU=10
printf "Running ${nCPU} cores on ${QUEUE},\n using ${memPerCPU} GB per core.\n\n"

printf "Running on galaxy ${galax}.\n\n"

jobid=$(addqueue --sbatch --requeue --serial -n "1x${nCPU}" -q "${QUEUE}" \
    -m "${memPerCPU}" -c "genFit_${galax}" -g "CubeFit${galax}" \
    -o "${curdir}/log.log" "${curdir}/kz_run.py" -- --galaxy "${galax}" \
    | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p')
printf "Submitted gen/fit job with ID: ${jobid}\n"
addqueue --sbatch --requeue --serial --runafter "$jobid" -n "1x${nCPU}" \
    -q "${QUEUE}" -m "${memPerCPU}" -c "loadFit_${galax}" -g "CubeFit${galax}" \
    -o "${curdir}/log.log" "${curdir}/kz_rio.py" -- --galaxy "${galax}"
