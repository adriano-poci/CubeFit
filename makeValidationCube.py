#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
    makeValidationCube.py
    Adriano Poci
    University of Oxford
    2026

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Create a CubeFit validation HDF5 file in which the fitted /ModelCube
    becomes the new synthetic /DataCube.

    The original file is never modified.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   29 September 2026
"""

import argparse
import os
import shutil

import h5py
import numpy as np


def makeValidationCube(source_path, output_path, *,
    rebuild_hypercube=False):
    """
    Create a CubeFit validation input from an existing model cube.

    Parameters
    ----------
    source_path : str
        Existing CubeFit HDF5 file containing /ModelCube.
    output_path : str
        Output HDF5 path.
    rebuild_hypercube : bool, optional
        If True, remove /HyperCube so that CubeFit must regenerate the
        design hypercube. Default is False.

    Returns
    -------
    str
        Output filename.

    Raises
    ------
    FileNotFoundError
        If the source HDF5 file does not exist.
    RuntimeError
        If required datasets are missing or inconsistent.
    ValueError
        If source_path and output_path refer to the same file.

    Examples
    --------
    >>> makeValidationCube(
    ...     "hypercube_003_00.h5",
    ...     "validation_003_00.h5",
    ... )
    'validation_003_00.h5'
    """
    source_path = os.path.abspath(str(source_path))
    output_path = os.path.abspath(str(output_path))

    if source_path == output_path:
        raise ValueError(
            "source_path and output_path must be different."
        )

    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)

    with h5py.File(source_path, "r") as f:
        required = (
            "/DataCube",
            "/ModelCube",
            "/LOSVD",
            "/Templates",
            "/ObsPix",
        )

        missing = [name for name in required if name not in f]

        if missing:
            raise RuntimeError(f"Missing required datasets: {missing}")

        data_shape = tuple(f["/DataCube"].shape)
        model_shape = tuple(f["/ModelCube"].shape)

        if data_shape != model_shape:
            raise RuntimeError("DataCube/ModelCube shape mismatch: "
                f"{data_shape} != {model_shape}")

        if len(model_shape) != 2:
            raise RuntimeError(f"Expected ModelCube (S, L), got {model_shape}.")

    print(f"Copying:\n  {source_path}\n-> {output_path}")

    shutil.copy2(source_path, output_path)

    with h5py.File(output_path, "r+") as f:
        model = np.asarray(f["/ModelCube"][...], dtype=np.float64,)

        if not np.all(np.isfinite(model)):
            bad = int(np.count_nonzero(~np.isfinite(model)))

            raise RuntimeError(f"ModelCube contains {bad} non-finite values.")

        # Preserve useful properties of the existing DataCube dataset.
        old_data = f["/DataCube"]

        chunks = old_data.chunks
        compression = old_data.compression
        compression_opts = old_data.compression_opts
        shuffle = old_data.shuffle
        fletcher32 = old_data.fletcher32

        data_attrs = dict(old_data.attrs)

        del f["/DataCube"]

        kwargs = {}

        if chunks is not None:
            kwargs["chunks"] = chunks

        if compression is not None:
            kwargs["compression"] = compression
            kwargs["compression_opts"] = compression_opts

        if shuffle:
            kwargs["shuffle"] = True

        if fletcher32:
            kwargs["fletcher32"] = True

        ds = f.create_dataset("/DataCube", data=model, dtype=np.float64,
            **kwargs)

        for key, value in data_attrs.items():
            ds.attrs[key] = value

        ds.attrs["validation_source"] = "/ModelCube"
        ds.attrs["validation_source_file"] = source_path
        ds.attrs["validation_kind"] = "exact_model_closure"

        # ----------------------------------------------------------
        # Remove products belonging to the ORIGINAL fit.
        # ----------------------------------------------------------

        if "/X_global" in f:
            del f["/X_global"]

        if "/ModelCube" in f:
            del f["/ModelCube"]

        # Remove known-zero state inferred during the original solve.
        if "/HyperCube/known_zero_mask" in f:
            del f["/HyperCube/known_zero_mask"]

        # Optionally force the entire design hypercube to be rebuilt.
        if rebuild_hypercube and "/HyperCube" in f:
            del f["/HyperCube"]

        f.flush()

    # The streaming solver cache is external to the HDF5 file.
    cache_path = output_path + ".bcfused.npz"

    if os.path.isfile(cache_path):
        os.remove(cache_path)

    print()
    print("Validation input created.")
    print(f"DataCube shape: {model.shape}")
    print("DataCube range: "
        f"{np.min(model):.6e} .. {np.max(model):.6e}")

    return output_path

# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=(
        "Convert a CubeFit /ModelCube into a synthetic "
        "CubeFit validation input."))

    parser.add_argument("source", help="Completed CubeFit HDF5 file.")

    parser.add_argument("output", help="Output validation HDF5 file.")

    parser.add_argument("--rebuild-hypercube",
        action="store_true",
        help=("Remove /HyperCube from the validation file so that "
            "it must be regenerated."))

    args = parser.parse_args()

    makeValidationCube(args.source, args.output,
        rebuild_hypercube=args.rebuild_hypercube)


if __name__ == "__main__":
    main()