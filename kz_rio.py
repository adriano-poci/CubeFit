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

# from site import addsitedir as sas
# import pathlib as plp
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci')))
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci', 'dynamics')))
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci', 'pxf')))
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci', 'muse')))
# do not need to add to paths, if run with
#   mpiexec -usize <nProcs+1> -n 1 ipython slurmSpecFCC170.py

import os
t = os.environ.get("SLURM_CPUS_PER_TASK", "8")
os.environ["OMP_NUM_THREADS"]      = t
os.environ["MKL_NUM_THREADS"]      = t
os.environ["OPENBLAS_NUM_THREADS"] = t
os.environ["OMP_DYNAMIC"]          = "FALSE"
os.environ["MKL_DYNAMIC"]          = "FALSE"

import numpy as np
import re
import pathlib as plp

# Custom modules
from CubeFit.kz_fitSpec import loadCubeFit
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

loadCubeFit(**props, redraw=True)

