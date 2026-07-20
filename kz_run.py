"""
History
-------
v1.0:	Capture exceptions around `genCubeFit` call. 4 December 2025
"""

import numpy as np
import os, re, sys
import pathlib as plp
import argparse


def _configure_solver_environment():
    """Set the solver-related environment variables used by the current wrapper."""
    os.environ["CUBEFIT_LAMBDA_WEIGHTS_ENABLE"] = "1"
    os.environ["CUBEFIT_GLOBAL_TAU"] = "0.3"
    os.environ["CUBEFIT_GLOBAL_ENERGY_BLEND"] = str(3e-2)
    os.environ["CUBEFIT_ZERO_COL_REL"] = str(5e-5)
    os.environ["CUBEFIT_LAMBDA_AGE"] = str(9.0)
    os.environ["CUBEFIT_LAMBDA_ASMOOTH"] = str(0.0)
    os.environ["CUBEFIT_LAMBDA_FLAT"] = str(0.0)
    os.environ["CUBEFIT_LAMBDA_FLAT_MAX_SCALE"] = str(5e4)
    os.environ["CUBEFIT_LAMBDA_GROUP"] = str(1e-2)
    os.environ["CUBEFIT_LAMBDA_L1"] = "0.0"
    os.environ["CUBEFIT_LAMBDA_GROUP_SCALE"] = "mass_norm"
    os.environ["CUBEFIT_LAMBDA_L2"] = str(0.0)
    os.environ["CUBEFIT_MAX_INV_D"] = "1e6"
    os.environ["CUBEFIT_ZERO_COL_DATAFLOOR_MUL"] = "1e-8"
    os.environ["CUBEFIT_ZERO_COL_ABS"] = "1e-30"
    os.environ["CUBEFIT_ORBIT_PRIOR_DELTA"] = "1e-6"


def main():
    ap = argparse.ArgumentParser(description="Thin wrapper around genCubeFit")
    ap.add_argument('--galaxy', type=str, default=None,
        help='Galaxy name to process')
    ap.add_argument(
        "--run-switch",
        type=str,
        default=None,
        help="Single string passed directly as runSwitch to genCubeFit"
    )
    # boolean redraw with explicit on/off flags
    ap.add_argument("--ncomp", type=int, default=None,
        help="Number of components to fit")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--redraw", dest="redraw", action="store_true",
                       help="Enable redraw mode")
    group.add_argument("--no-redraw", dest="redraw", action="store_false",
                       help="Disable redraw mode")
    ap.set_defaults(redraw=False)

    args = ap.parse_args()

    if 'NGC4365' in args.galaxy:
        from CubeFit.kz_initNGC4365 import props
    elif 'FCC170' in args.galaxy:
        from CubeFit.kz_initFCC170 import props
    else:
        raise ValueError(f"Unknown galaxy '{args.galaxy}'")

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

    _configure_solver_environment()

    if args.ncomp is not None:
        props['nCuts'] = args.ncomp
    print(props)

    from CubeFit.kz_fitSpec import genCubeFit

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