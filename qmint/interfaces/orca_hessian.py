"""ORCA ExtOpt and standalone Hessian interface."""

from __future__ import annotations

import argparse
import os

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from ..protocol import request
from .common import read_orca_request, read_xyz, strip_suffix, write_engrad


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def frequencies_and_modes(hessian: np.ndarray, masses: list[float]) -> tuple[np.ndarray, np.ndarray]:
    repeated = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / np.outer(repeated, repeated)
    eigenvalues, eigenvectors = np.linalg.eigh(weighted)
    normal_modes = eigenvectors / repeated[:, None]
    amu_to_kg = 1.66053906660e-27
    hartree_to_joule = 4.3597447222071e-18
    bohr_to_metre = 0.529177210903e-10
    speed_of_light = 2.99792458e10
    conversion = np.sqrt(hartree_to_joule / (bohr_to_metre**2 * amu_to_kg)) / (
        2 * np.pi * speed_of_light
    )
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion
    return frequencies, normal_modes


def write_orca_hessian(
    filename: str,
    hessian,
    energy: float,
    symbols: list[str],
    positions: list[list[float]],
    charge: int,
    multiplicity: int,
) -> None:
    matrix = np.asarray(hessian)
    count = len(symbols)
    dimension = 3 * count
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"Expected a {(dimension, dimension)} Hessian, got {matrix.shape}")
    masses = [float(atomic_masses[atomic_numbers[symbol]]) for symbol in symbols]
    frequencies, modes = frequencies_and_modes(matrix, masses)
    with open(filename, "w", encoding="utf-8", newline="\n") as output:
        output.write("$orca_hessian_file\n\n")
        output.write(f"$act_atom\n  {count}\n\n")
        output.write("$act_coord\n  2\n\n")
        output.write(f"$act_energy\n      {energy:.12f}\n\n")
        _write_matrix(output, "$hessian", matrix)
        output.write(f"$vibrational_frequencies\n{dimension}\n")
        for index, frequency in enumerate(frequencies):
            output.write(f"{index:5d}        {frequency:12.6f}\n")
        output.write("\n")
        output.write(f"$normal_modes\n{dimension} {dimension}\n")
        _write_matrix_body(output, modes)
        output.write(f"$atoms\n{count}\n")
        for index, (symbol, mass, xyz) in enumerate(zip(symbols, masses, positions)):
            x, y, z = (value * ANGSTROM_TO_BOHR for value in xyz)
            output.write(
                f"{index:5d}   {symbol:>2s}   {mass:14.8f}"
                f"   {x:20.12f}   {y:20.12f}   {z:20.12f}\n"
            )
        output.write("\n")
        output.write(f"$ir_intensity\n{dimension}\n")
        for index in range(dimension):
            output.write(f"{index:5d}        {0.0:12.6f}\n")
        output.write("\n$thermo_info\n")
        output.write(f"  charge = {charge}\n  mult = {multiplicity}\n  temperature = 298.15\n\n")
        output.write("$end\n")


def _write_matrix(output, name: str, matrix: np.ndarray) -> None:
    output.write(f"{name}\n{matrix.shape[0]}\n")
    _write_matrix_body(output, matrix)


def _write_matrix_body(output, matrix: np.ndarray) -> None:
    rows, columns = matrix.shape
    for start in range(0, columns, 5):
        end = min(start + 5, columns)
        output.write("    " + "".join(f"{column:>20}" for column in range(start, end)) + "\n")
        for row in range(rows):
            values = "".join(f"{matrix[row, column]:20.10E}" for column in range(start, end))
            output.write(f"   {row:4d}{values}\n")
    output.write("\n")


def _calculate(
    coordinate_lines: list[str], charge: int, multiplicity: int, threads: int
) -> dict:
    result = request(
        {
            "charge": charge,
            "mult": multiplicity,
            "num_threads": threads,
            "coord": coordinate_lines,
            "hessian": True,
        }
    )
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def orca_mode(path: str) -> None:
    xyz_path, charge, multiplicity, threads, gradient_required = read_orca_request(path)
    count, symbols, positions, coordinate_lines = read_xyz(xyz_path)
    result = _calculate(coordinate_lines, charge, multiplicity, threads)
    base = strip_suffix(path)
    write_engrad(base + ".engrad", count, result["energy"], result["gradient"], gradient_required)
    write_orca_hessian(
        base + ".hess", result["hessian"], result["energy"], symbols, positions, charge, multiplicity
    )
    print(f"\nHessian written to: {base}.hess")
    print("\n-------------------------   --------------------")
    print(f"FINAL SINGLE POINT ENERGY   {result['energy']:>20.12f}")
    print("-------------------------   --------------------\n")


def standalone_mode(
    xyz_path: str, charge: int, multiplicity: int, threads: int, output_path: str | None
) -> None:
    _, symbols, positions, coordinate_lines = read_xyz(xyz_path)
    result = _calculate(coordinate_lines, charge, multiplicity, threads)
    destination = output_path or os.path.splitext(os.path.basename(xyz_path))[0] + ".hess"
    write_orca_hessian(
        destination, result["hessian"], result["energy"], symbols, positions, charge, multiplicity
    )
    print(f"Hessian written to: {destination}")
    print(f"Energy: {result['energy']:.12f} Eh")
    print("\nTo use in ORCA, add:")
    print(f'  %geom\n    InHess Read\n    InHessName "{os.path.abspath(destination)}"\n  end')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="ORCA .extinp.tmp file")
    parser.add_argument("--xyz", help="XYZ file for standalone mode")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1, help="spin multiplicity")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", "-o")
    args = parser.parse_args()
    try:
        if args.file:
            orca_mode(args.file)
        elif args.xyz:
            standalone_mode(args.xyz, args.charge, args.mult, args.threads, args.output)
        else:
            parser.error("provide an ORCA input file or --xyz")
    except RuntimeError as exc:
        raise SystemExit(f"QMint calculation failed: {exc}") from exc


if __name__ == "__main__":
    main()
