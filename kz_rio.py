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
v1.1:   Capture exceptions around `loadCubeFit` call. 4 December 2025
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
t = os.environ.get("SLURM_CPUS_PER_TASK", "12")
os.environ["OMP_NUM_THREADS"]      = t
os.environ["MKL_NUM_THREADS"]      = t
os.environ["OPENBLAS_NUM_THREADS"] = t
os.environ["OMP_DYNAMIC"]          = "FALSE"
os.environ["MKL_DYNAMIC"]          = "FALSE"

import numpy as np
import re, sys
import pathlib as plp
import argparse

# Custom modules
from CubeFit.kz_fitSpec import loadCubeFit
from CubeFit.kz_initNGC4365 import props

def main():
    ap = argparse.ArgumentParser(description="Thin wrapper around genCubeFit")
    # boolean redraw with explicit on/off flags
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--redraw", dest="redraw", action="store_true",
                       help="Enable redraw mode")
    group.add_argument("--no-redraw", dest="redraw", action="store_false",
                       help="Disable redraw mode")
    ap.set_defaults(redraw=False)

    args = ap.parse_args()

    # Detect CPUs
    slurm_cpu = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpu is not None:
        nCPU = int(slurm_cpu)
    else:
        curdir = plp.Path(__file__).parent
        try:
            with open(curdir/'kz_addqueue.sh') as f:
                content = f.read()
            m = re.search(r'^\s*nCPU\s*=\s*(\d+)', content, re.MULTILINE)
            nCPU = int(m.group(1)) if m else 20
        except FileNotFoundError:
            nCPU = 20

    print(f"Setting nCPU to {nCPU} from SLURM_CPUS_PER_TASK or kz_addqueue.sh")
    props['nProcs'] = nCPU

    # Pass-through args
    props['redraw'] = bool(args.redraw)
    print(f"redraw = {props['redraw']}")

    try:
        loadCubeFit(**props)
    except SystemExit:
        # Let explicit sys.exit()s behave normally
        raise
    except BaseException as e:
        # Log + print the traceback explicitly
        import traceback
        print("[kz_rio] FATAL: unhandled exception in genCubeFit", file=sys.__stderr__, flush=True)
        traceback.print_exc()
        # This *forces* the interpreter to exit, even under IPython
        sys.exit(1)

if __name__ == "__main__":
    main()