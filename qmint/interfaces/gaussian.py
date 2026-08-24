"""Gaussian External interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..protocol import request


BOHR_TO_ANGSTROM = 0.529177210903


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit("Gaussian must call mlpint with its five External interface files")
    input_file, output_file, message_file = sys.argv[-5:-2]
    lines = Path(input_file).read_text(encoding="utf-8").splitlines()
    count, derivative, charge, multiplicity = (int(value) for value in lines[0].split())
    coordinates = []
    for line in lines[1 : count + 1]:
        parts = line.split()
        xyz = [float(value) * BOHR_TO_ANGSTROM for value in parts[1:4]]
        coordinates.append(f"{parts[0]} {xyz[0]:20.12f} {xyz[1]:20.12f} {xyz[2]:20.12f}")
    threads = int(os.environ.get("MLP_THREADS") or os.environ.get("OMP_NUM_THREADS") or 1)
    result = request(
        {
            "charge": charge,
            "mult": multiplicity,
            "num_threads": threads,
            "coord": coordinates,
            "hessian": derivative == 2,
        }
    )
    if "error" in result:
        Path(message_file).write_text(
            "---------------------Error in ASE calculator---------------------\n"
            f"{result['error']}\n",
            encoding="utf-8",
        )
        raise SystemExit(1)
    with open(output_file, "w", encoding="utf-8", newline="\n") as output:
        output.write(f"{result['energy']:20.12e}{0.0:20.12e}{0.0:20.12e}{0.0:20.12e}\n")
        if derivative > 0:
            for row in result["gradient"]:
                output.write("".join(f"{value:20.12e}" for value in row) + "\n")
        for _ in range(3 * count + 2):
            output.write(f"{0.0:20.12e}{0.0:20.12e}{0.0:20.12e}\n")
        if derivative == 2:
            matrix = result["hessian"]
            packed = [matrix[i, j] for i in range(3 * count) for j in range(i + 1)]
            for start in range(0, len(packed), 3):
                values = packed[start : start + 3]
                output.write("".join(f"{value:20.12e}" for value in values) + "\n")
    Path(message_file).write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
