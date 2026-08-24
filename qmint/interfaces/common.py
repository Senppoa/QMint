from __future__ import annotations

from pathlib import Path


def strip_suffix(path: str, suffix: str = ".extinp.tmp") -> str:
    return path[: -len(suffix)] if path.endswith(suffix) else str(Path(path).with_suffix(""))


def remove_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_xyz(path: str) -> tuple[int, list[str], list[list[float]], list[str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    count = int(lines[0].strip())
    coordinate_lines = lines[2 : 2 + count]
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in coordinate_lines:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(value) for value in parts[1:4]])
    return count, symbols, positions, coordinate_lines


def read_orca_request(path: str) -> tuple[str, int, int, int, bool]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Invalid ORCA external input: {path}")
    return (
        remove_comment(lines[0]),
        int(remove_comment(lines[1])),
        int(remove_comment(lines[2])),
        int(remove_comment(lines[3])),
        int(remove_comment(lines[4])) == 1,
    )


def write_engrad(path: str, count: int, energy: float, gradient, required: bool) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as output:
        output.writelines(
            [
                "#\n# Number of atoms\n#\n",
                f"{count}\n",
                "#\n# The current total energy in Eh\n#\n",
                f"{energy:.12f}\n",
            ]
        )
        if required:
            output.write("#\n# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, ...\n#\n")
            for value in gradient.flatten():
                output.write(f"{value:.12f}\n")

