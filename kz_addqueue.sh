#!/bin/bash

while getopts ":g:" arg; do
  case $arg in
    g) galax=$OPTARG;;
  esac
done

curdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="cbass"
NodeMem=124
nCPU=20

memPerCPU=$(echo "scale=2; $NodeMem*0.99 / $nCPU" | bc)
buildCPU=$(echo $((nCPU<16 ? nCPU : 16)))
# Should eventually become 8 cores, 2 BLAS
printf "Running ${nCPU} cores on ${QUEUE},\n using ${memPerCPU} GB per core.\n"
printf "Building with ${buildCPU} cores.\n"

printf "Running on galaxy ${galax}.\n\n"
# jobidGen=$(addqueue --sbatch --requeue --serial -n "1x${buildCPU}" \
#     -q "${QUEUE}" -m "${memPerCPU}" -c "gen_${galax}" -g "CubeFit${galax}" \
#     -o "${curdir}/log.log" "${curdir}/kz_run.py" -- --galaxy "${galax}" \
#     --run-switch 'gen' --redraw | \
#     sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p')
# printf "Submitted gen job with ID: ${jobidGen}\n"
# jobidFit=$(addqueue --sbatch --requeue --serial --runafter "$jobidGen" \
jobidFit=$(addqueue --sbatch --requeue --serial \
    -n "1x${nCPU}" -q "${QUEUE}" -m "${memPerCPU}" -c "fit_${galax}" \
    -g "CubeFit${galax}" -o "${curdir}/log.log" "${curdir}/kz_run.py" -- \
    --galaxy "${galax}" --run-switch 'fit' | \
    sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p')
printf "Submitted fit job with ID: ${jobidFit}\n"
addqueue --sbatch --requeue --serial --runafter "$jobidFit" -n "1x${nCPU}" \
    -q "${QUEUE}" -m "${memPerCPU}" -c "rio_${galax}" -g "CubeFit${galax}" \
    -o "${curdir}/log.log" "${curdir}/kz_rio.py" -- --galaxy "${galax}"
