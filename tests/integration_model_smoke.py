#!/usr/bin/env python3
"""Run real QMint model inference against geometries from Gaussian inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qmint.calculator import calculate, load_calculator, prepare_wsl_cuda_env
from qmint.models import ModelSpec


def read_first_gaussian_geometry(path: Path) -> tuple[int, int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    route_end = next(index for index, line in enumerate(lines) if line.lstrip().startswith("#"))
    index = route_end + 1
    while index < len(lines) and lines[index].strip():
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    index += 1  # title
    while index < len(lines) and not lines[index].strip():
        index += 1
    charge, multiplicity = (int(value) for value in lines[index].split()[:2])
    coordinates: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip() or line.strip() == "--Link1--":
            break
        parts = line.split()
        if len(parts) < 4:
            break
        coordinates.append(" ".join(parts[:4]))
    if not coordinates:
        raise ValueError(f"No Cartesian geometry found in {path}")
    return charge, multiplicity, coordinates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("fairchem", "mace", "orb"))
    parser.add_argument("--head")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    if args.backend == "orb":
        prepare_wsl_cuda_env()
    spec = ModelSpec(args.name, args.backend, args.model.resolve(), head=args.head)
    started = time.perf_counter()
    calculator, load_device = load_calculator(spec, args.device)
    report = {
        "model": args.name,
        "backend": args.backend,
        "model_path": str(spec.path),
        "head": args.head,
        "device": args.device,
        "load_device": load_device,
        "load_seconds": time.perf_counter() - started,
        "inputs": [],
    }
    for path in args.inputs:
        charge, multiplicity, coordinates = read_first_gaussian_geometry(path)
        task_started = time.perf_counter()
        result = calculate(
            {
                "charge": charge,
                "mult": multiplicity,
                "num_threads": 1,
                "coord": coordinates,
                "hessian": False,
            },
            calculator,
            "numeric",
        )
        item = {
            "input": path.name,
            "atoms": len(coordinates),
            "elements": sorted({line.split()[0] for line in coordinates}),
            "seconds": time.perf_counter() - task_started,
        }
        if "error" in result:
            item["error"] = result["error"].split("\nTraceback", 1)[0]
        else:
            gradient = np.asarray(result["gradient"])
            item.update(
                {
                    "energy_hartree": float(result["energy"]),
                    "gradient_shape": list(gradient.shape),
                    "max_abs_gradient": float(np.abs(gradient).max()),
                    "finite": bool(np.isfinite(gradient).all() and np.isfinite(result["energy"])),
                }
            )
        report["inputs"].append(item)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

