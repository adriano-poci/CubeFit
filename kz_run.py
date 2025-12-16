"""
History
-------
v1.0:	Capture exceptions around `genCubeFit` call. 4 December 2025
"""

import numpy as np
import os, re, sys
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
    os.environ['CUBEFIT_NNLS_SOLVER']='nnls'
    os.environ["CUBEFIT_USE_NNLS_PRIOR"] = "0"
    os.environ['CUBEFIT_NNLS_L2']=str(0)
    os.environ["CUBEFIT_LAMBDA_WEIGHTS_ENABLE"] = "1"
    os.environ["CUBEFIT_KACZ_L2"] = "0.0"
    os.environ["CUBEFIT_RMSE_PROXY_GUARD"] = "0"
    os.environ["CUBEFIT_NNLS_ENABLE"] = "0"             # no tile NNLS
    os.environ["CUBEFIT_ORBIT_BETA"] = "0.0"
    print(props)

    try:
        genCubeFit(**props)
    except SystemExit:
        # Let explicit sys.exit()s behave normally
        raise
    except BaseException as e:
        # Log + print the traceback explicitly
        import traceback
        print("[kz_run] FATAL: unhandled exception in genCubeFit", file=sys.__stderr__, flush=True)
        traceback.print_exc()
        # This *forces* the interpreter to exit, even under IPython
        sys.exit(1)

if __name__ == "__main__":
    main()