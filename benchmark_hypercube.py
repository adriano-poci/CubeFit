#!/usr/bin/env python3

import os
import time
import argparse
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from CubeFit.hdf5_manager import open_h5
from CubeFit.hypercube_builder import (
    _kernel_map_from_grids,
    _choose_nfft,
    _init_hypercube_worker,
    _hypercube_worker,
)


def _read_inputs(h5_path, s_count, c_count, p_count):
    """Read representative production inputs and prepare the real kernel."""

    with open_h5(h5_path, role="reader") as f:
        P, T = map(int, f["/Templates"].shape)
        S, V, C = map(int, f["/LOSVD"].shape)
        L = int(f["/DataCube"].shape[1])

        ns = min(int(s_count), S)
        nc = min(int(c_count), C)
        npop = min(int(p_count), P)

        Templates = np.asarray(
            f["/Templates"][:npop], dtype=np.float64, order="C")
        tem_loglam = np.asarray(f["/TemPix"][...], dtype=np.float64)
        vel_pix = np.asarray(f["/VelPix"][...], dtype=np.float64)

        R_any = np.asarray(f["/R_T"][...])
        if R_any.shape == (T, L):
            R_T = R_any.astype(np.float32, copy=False)
        elif R_any.shape == (L, T):
            R_T = R_any.T.astype(np.float32, copy=False)
        else:
            raise RuntimeError(
                f"Incompatible R_T shape {R_any.shape}.")

        LOSVD = np.asarray(
            f["/LOSVD"][:ns, :, :nc],
            dtype=np.float64, order="C")

    km = _kernel_map_from_grids(tem_loglam, vel_pix)
    n_fft = _choose_nfft(T, km.m)

    T_fft = np.fft.rfft(
        Templates, n=n_fft, axis=1)

    return {
        "S_full": S,
        "C_full": C,
        "P_full": P,
        "T": T,
        "L": L,
        "ns": ns,
        "nc": nc,
        "npop": npop,
        "LOSVD": LOSVD,
        "R_T": R_T,
        "T_fft": T_fft,
        "km": km,
        "n_fft": n_fft,
    }


def _make_jobs(inp, s_chunk, c_chunk, p_chunk, repeats):
    """Construct representative worker jobs."""

    ns = inp["ns"]
    nc = inp["nc"]
    npop = inp["npop"]
    LOSVD = inp["LOSVD"]

    jobs = []

    for _ in range(int(repeats)):
        for p0 in range(0, npop, p_chunk):
            p1 = min(p0 + p_chunk, npop)

            for s0 in range(0, ns, s_chunk):
                s1 = min(s0 + s_chunk, ns)

                for c0 in range(0, nc, c_chunk):
                    c1 = min(c0 + c_chunk, nc)

                    H_tile = np.empty(
                        (s1 - s0, c1 - c0), dtype=object)

                    for i, s in enumerate(range(s0, s1)):
                        for j, c in enumerate(range(c0, c1)):
                            H_tile[i, j] = LOSVD[s, :, c].copy()

                    scales = np.ones(
                        (s1 - s0, c1 - c0),
                        dtype=np.float64)

                    jobs.append((
                        s0, s1, c0, c1,
                        p0, p1, H_tile, scales,
                    ))

    return jobs


def benchmark(args):
    """Benchmark the actual HyperCube convolution/rebin worker."""

    inp = _read_inputs(
        args.h5,
        args.spaxels,
        args.components,
        args.populations,
    )

    jobs = _make_jobs(
        inp,
        args.s_chunk,
        args.c_chunk,
        args.p_chunk,
        args.repeats,
    )

    n_sc = (
        inp["ns"] * inp["nc"] *
        args.repeats
    )

    n_models = n_sc * inp["npop"]

    output_bytes = (
        n_models * inp["L"] *
        np.dtype(np.float64).itemsize
    )

    print()
    print("Production geometry:")
    print(
        f"    S={inp['S_full']} "
        f"C={inp['C_full']} "
        f"P={inp['P_full']} "
        f"T={inp['T']} "
        f"L={inp['L']}"
    )

    print("Benchmark geometry:")
    print(
        f"    S={inp['ns']} "
        f"C={inp['nc']} "
        f"P={inp['npop']}"
    )

    print(
        f"    chunks=({args.s_chunk}, "
        f"{args.c_chunk}, {args.p_chunk})"
    )

    print(
        f"    processes={args.processes} "
        f"BLAS threads={args.blas_threads}"
    )

    print(f"    jobs={len(jobs)}")
    print(f"    spectra generated={n_models:,}")
    print()

    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=args.processes,
        mp_context=ctx,
        initializer=_init_hypercube_worker,
        initargs=(
            inp["T_fft"],
            inp["R_T"],
            inp["km"],
            inp["n_fft"],
            inp["T"],
            None,
            args.blas_threads,
        ),
    ) as executor:

        futures = [
            executor.submit(_hypercube_worker, job)
            for job in jobs
        ]

        checksum = 0.0

        for future in as_completed(futures):
            result = future.result()

            # Touch the output so the complete calculation is represented.
            for _, _, _, _, Ycp in result:
                checksum += float(Ycp[0, 0])

    elapsed = time.perf_counter() - t0

    models_per_sec = n_models / elapsed
    logical_gbs = output_bytes / elapsed / 1.0e9

    # Full-build extrapolation.
    full_models = (
        inp["S_full"] *
        inp["C_full"] *
        inp["P_full"]
    )

    projected_sec = full_models / models_per_sec

    print("Results:")
    print(f"    wall time       : {elapsed:10.3f} s")
    print(f"    models/s        : {models_per_sec:10.1f}")
    print(f"    output rate     : {logical_gbs:10.3f} GB/s")
    print(f"    checksum        : {checksum:.8e}")
    print()
    print(
        "    projected full compute time: "
        f"{projected_sec / 3600.0:.2f} h"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("h5")

    parser.add_argument(
        "--processes", type=int, default=8)
    parser.add_argument(
        "--blas-threads", type=int, default=1)

    parser.add_argument(
        "--spaxels", type=int, default=32)
    parser.add_argument(
        "--components", type=int, default=8)
    parser.add_argument(
        "--populations", type=int, default=315)

    parser.add_argument(
        "--s-chunk", type=int, default=8)
    parser.add_argument(
        "--c-chunk", type=int, default=1)
    parser.add_argument(
        "--p-chunk", type=int, default=315)

    parser.add_argument(
        "--repeats", type=int, default=1)

    args = parser.parse_args()
    benchmark(args)