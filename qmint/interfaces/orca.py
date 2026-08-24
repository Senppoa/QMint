"""ORCA ExtOpt energy and gradient interface."""

from __future__ import annotations

import argparse

from ..protocol import request
from .common import read_orca_request, read_xyz, strip_suffix, write_engrad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="ORCA .extinp.tmp file")
    args = parser.parse_args()
    xyz_path, charge, multiplicity, threads, gradient_required = read_orca_request(args.file)
    count, _, _, coordinate_lines = read_xyz(xyz_path)
    result = request(
        {
            "charge": charge,
            "mult": multiplicity,
            "num_threads": threads,
            "coord": coordinate_lines,
            "hessian": False,
        }
    )
    if "error" in result:
        raise SystemExit(f"QMint calculation failed: {result['error']}")
    write_engrad(
        strip_suffix(args.file) + ".engrad", count, result["energy"], result["gradient"], gradient_required
    )
    print("\n-------------------------   --------------------")
    print(f"FINAL SINGLE POINT ENERGY   {result['energy']:>20.12f}")
    print("-------------------------   --------------------\n")


if __name__ == "__main__":
    main()
