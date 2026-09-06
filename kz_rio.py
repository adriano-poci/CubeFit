#!/usr/bin/env python3
"""
    kz_rio.py
    Adriano Poci
    University of Oxford
    2026

    <adriano.poci@physics.ox.ac.uk>

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Light Python wrapper around execution tasks for `CubeFit`.

    Author
    ------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:	12 November 2021
v1.1:   Capture exceptions around `loadCubeFit` call. 4 December 2025
v1.2:   Use universal `props` function. 22 July 2026
"""

import os
import numpy as np
import re, sys
import pathlib as plp
import argparse

# Custom modules
from CubeFit.kz_init import props
from CubeFit.kz_fitSpec import loadCubeFit

def _configure_solver_environment():
    """Set the solver-related environment variables used by the current wrapper."""
    t = os.environ.get("SLURM_CPUS_PER_TASK", "12")
    os.environ["OMP_NUM_THREADS"]      = t
    os.environ["MKL_NUM_THREADS"]      = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["OMP_DYNAMIC"]          = "FALSE"
    os.environ["MKL_DYNAMIC"]          = "FALSE"

def main():
    ap = argparse.ArgumentParser(description="Thin wrapper around genCubeFit")
    # boolean redraw with explicit on/off flags
    ap.add_argument('--galaxy', type=str, default=None,
        help='Galaxy name to process')
    ap.add_argument("--ncomp", type=int, default=None,
        help="Number of components to fit")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--redraw", dest="redraw", action="store_true",
        help="Enable redraw mode")
    group.add_argument("--no-redraw", dest="redraw", action="store_false",
        help="Disable redraw mode")
    ap.set_defaults(redraw=False)

    args = ap.parse_args()
    propDict = props(args.galaxy)

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
    propDict['nProcs'] = nCPU

    # Pass-through args
    propDict['redraw'] = bool(args.redraw)
    print(f"redraw = {propDict['redraw']}")

    _configure_solver_environment()

    if args.ncomp is not None:
        propDict['nCuts'] = args.ncomp
    if propDict['nCuts'] > 50:
        propDict['sspIdx'] = ([-1.5, -1.0, -0.6, -0.3, 0.0, 0.15, 0.26, 0.4],
            [3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [-0.2, 0.0, 0.2, 0.4, 0.6])
    print(propDict)

    try:
        loadCubeFit(**propDict)
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