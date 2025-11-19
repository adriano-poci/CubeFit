"""

"""

import numpy as np
import os, re
import pathlib as plp
import argparse

# Custom modules
from CubeFit.kz_fitSpec import genCubeFit
from CubeFit.kz_initNGC4365 import props


def main():
    ap = argparse.ArgumentParser(description="Thin wrapper around genCubeFit")
    ap.add_argument(
        "--run-switch",
        type=str,
        default=None,
        help="Single string passed directly as runSwitch to genCubeFit"
    )
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
    if args.run_switch is not None:
        props['runSwitch'] = args.run_switch
        print(f"runSwitch = {props['runSwitch']}")
    props['redraw'] = bool(args.redraw)
    print(f"redraw = {props['redraw']}")

    os.environ['CUBEFIT_NNLS_ENABLE']=str(0)
    os.environ['CUBEFIT_NNLS_EVERY']=str(5)
    os.environ['CUBEFIT_NNLS_MIN_IMPROVE']=str(0.9995)
    os.environ['CUBEFIT_NNLS_MAX_COLS']=str(64)
    os.environ['CUBEFIT_NNLS_SUB_L']=str(512)
    os.environ['CUBEFIT_NNLS_SOLVER']='fista'
    os.environ['CUBEFIT_NNLS_L2']=str(1e-6)

    print(props)

    genCubeFit(**props)

if __name__ == "__main__":
    main()