"""
#!/apps/skylake/software/core/anaconda3/5.1.0/bin/python3
#SBATCH -A oz059
#SBATCH --job-name="slurmSpecNGC4365"
#SBATCH --time=2-00:00
#SBATCH -D "/fred/oz059/poci/muse"
#SBATCH --output="/fred/oz059/poci/muse/slurmSpecNGC4365.log"
#SBATCH --error="/fred/oz059/poci/muse/slurmSpecNGC4365.log"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@students.mq.edu.au

    slurmSpecNGC4365.py
    Adriano Poci
    Durham University
    2021

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module executes some function in the `SLURM` queueing environment

    Authors
    -------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:	12 November 2021
"""

import numpy as np
import os, re
import pathlib as plp

# Custom modules
from CubeFit.kz_fitSpec import genCubeFit
from muse.fs_initNGC4365 import props

# First: Check if SLURM_CPU_PER_TASK environment variable is defined
slurm_cpu = os.environ.get('SLURM_CPUS_PER_TASK')
if slurm_cpu is not None:
    nCPU = int(slurm_cpu)
else:
    # Fallback to reading from kz_addqueue.sh as before
    curdir = plp.Path(__file__).parent
    with open(curdir/'kz_addqueue.sh') as f:
        content = f.read()
    match = re.search(r'^\s*nCPU\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        nCPU = int(match.group(1))
    else:
        print("nCPU not found")
        nCPU = 20

print(f"Setting nCPU to {nCPU} from SLURM_CPUS_PER_TASK or kz_addqueue.sh")
props['nProcs'] = nCPU

# props['zarrDir'] = plp.Path.home()/'Cube'/\
#     f"{props['galaxy']}_{props['lOrder']:02d}"
# props['nCuts'] = 3

genCubeFit(**props)
