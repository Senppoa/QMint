"""Lazy model loading and ASE task execution."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.vibrations import Vibrations

from .models import ModelSpec


def prepare_wsl_cuda_env() -> None:
    wsl_lib = Path("/usr/lib/wsl/lib")
    if not (wsl_lib / "libcuda.so.1").exists():
        return
    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(wsl_lib), *[p for p in paths if p != str(wsl_lib)]])


def load_calculator(spec: ModelSpec, device: str, hessian_mode: str = "numeric"):
    if spec.backend == "fairchem":
        if device.startswith("cuda:"):
            logical_id = int(device.split(":", 1)[1])
            visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                visible[logical_id] if visible else str(logical_id)
            )
            device = "cuda"
        import torch

        torch.serialization.add_safe_globals([slice])
        try:
            from fairchem.core import FAIRChemCalculator
            from fairchem.core.units.mlip_unit import load_predict_unit
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("fairchem backend is unavailable; install fairchem-core") from exc
        predictor = load_predict_unit(path=str(spec.path), device=device)
        return FAIRChemCalculator(predictor, task_name="omol"), device

    if spec.backend == "mace":
        try:
            from mace.calculators import MACECalculator
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("mace backend is unavailable; install mace-torch") from exc
        return MACECalculator(model_paths=str(spec.path), device=device), device

    if spec.backend == "orb":
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.inference.calculator import ORBCalculator
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("orb backend is unavailable; install orb-models") from exc
        if not hasattr(pretrained, "orbmol_v2"):
            raise RuntimeError("orb-models must provide pretrained.orbmol_v2")
        if hessian_mode == "analytic":
            if importlib.util.find_spec("orb_hessian") is None:
                raise RuntimeError("analytic orb Hessian requires orb-hessian")
            import orb_hessian  # noqa: F401
            precision = "float64"
        else:
            precision = "float32-high"
        forcefield, adapter = pretrained.orbmol_v2(
            weights_path=str(spec.path), device=device, compile=False, precision=precision
        )
        return ORBCalculator(forcefield, atoms_adapter=adapter, device=device), device

    raise ValueError(f"Unsupported backend: {spec.backend}")


def _atoms(task: dict[str, Any]) -> Atoms:
    symbols: list[str] = []
    numbers: list[int] = []
    positions: list[list[float]] = []
    for line in task["coord"]:
        parts = str(line).split()
        if len(parts) < 4:
            continue
        try:
            numbers.append(int(parts[0]))
        except ValueError:
            symbols.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    atoms = Atoms(numbers=numbers, positions=positions) if numbers else Atoms(symbols=symbols, positions=positions)
    atoms.info = {"charge": task.get("charge", 0), "spin": task.get("mult", 1)}
    return atoms


def _normalise_hessian(value: Any, n_atoms: int) -> np.ndarray:
    matrix = np.asarray(value)
    size = 3 * n_atoms
    if matrix.shape == (size, size):
        return matrix
    if matrix.shape == (n_atoms, 3, n_atoms, 3):
        return matrix.reshape(size, size)
    if matrix.shape == (n_atoms, n_atoms, 3, 3):
        return matrix.transpose(0, 2, 1, 3).reshape(size, size)
    if matrix.shape == (3, n_atoms, 3, n_atoms):
        return matrix.transpose(1, 0, 3, 2).reshape(size, size)
    raise ValueError(f"Unexpected Hessian shape: {matrix.shape}")


def calculate(task: dict[str, Any], calculator: Any, hessian_mode: str) -> dict[str, Any]:
    try:
        threads = max(1, int(task.get("num_threads", 1)))
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        atoms = _atoms(task)
        atoms.calc = calculator
        energy = atoms.get_potential_energy() / Hartree
        gradient = -atoms.get_forces() / (Hartree / Bohr)
        hessian = None
        if task.get("hessian"):
            if hessian_mode == "analytic":
                hessian = _normalise_hessian(calculator.get_hessian(atoms), len(atoms)) / (
                    Hartree / (Bohr * Bohr)
                )
            elif hessian_mode == "numeric":
                with tempfile.TemporaryDirectory(prefix=f"qmint-hess-{os.getpid()}-") as directory:
                    vib = Vibrations(atoms=atoms, name=str(Path(directory) / "vib"), delta=0.005)
                    try:
                        vib.run()
                        hessian = vib.get_vibrations().get_hessian_2d() / (Hartree / (Bohr * Bohr))
                    finally:
                        vib.clean()
            else:
                raise ValueError(f"Unknown Hessian mode: {hessian_mode}")
        return {"energy": energy, "gradient": gradient, "hessian": hessian}
    except Exception as exc:
        return {"error": f"{exc}\n{traceback.format_exc()}"}
