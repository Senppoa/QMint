"""Model registry and path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BACKENDS = ("fairchem", "mace", "orb")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    path: Path
    description: str = ""
    builtin: bool = False


BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "uma-s": {
        "backend": "fairchem",
        "filename": "uma-s-1p1.pt",
        "aliases": ("s", "small", "uma-s-1p1.pt"),
        "description": "UMA small 1.1",
    },
    "uma-m": {
        "backend": "fairchem",
        "filename": "uma-m-1p1.pt",
        "aliases": ("m", "medium", "middle", "uma-m-1p1.pt"),
        "description": "UMA medium 1.1",
    },
    "mace": {
        "backend": "mace",
        "filename": "MACE.model",
        "aliases": (),
        "description": "Generic MACE model",
    },
    "mace-omol": {
        "backend": "mace",
        "filename": "MACE-omol-0-extra-large-1024.model",
        "aliases": (),
        "description": "MACE OMol extra-large",
    },
    "mace-polar-m": {
        "backend": "mace",
        "filename": "MACE-POLAR-1-M.model",
        "aliases": (),
        "description": "MACE POLAR medium",
    },
    "mace-polar-l": {
        "backend": "mace",
        "filename": "MACE-POLAR-1-L.model",
        "aliases": (),
        "description": "MACE POLAR large",
    },
    "orbmol-v2": {
        "backend": "orb",
        "filename": "orbmol-v2-teqabfhg-20260523.ckpt",
        "aliases": ("orb-mol-v2", "orbmol-v2-teqabfhg-20260523.ckpt"),
        "description": "OrbMol v2",
    },
}


def model_dir(config: dict[str, Any]) -> Path:
    value = os.environ.get("MLP_MODEL_DIR") or config["model_dir"]
    return Path(value).expanduser().resolve()


def aliases() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, data in BUILTIN_MODELS.items():
        result[name] = name
        result.update({alias: name for alias in data["aliases"]})
    return result


def list_models(config: dict[str, Any]) -> list[ModelSpec]:
    root = model_dir(config)
    models = [
        ModelSpec(
            name=name,
            backend=data["backend"],
            path=root / data["filename"],
            description=data["description"],
            builtin=True,
        )
        for name, data in BUILTIN_MODELS.items()
    ]
    for name, data in sorted(config.get("custom_models", {}).items()):
        models.append(
            ModelSpec(
                name=name,
                backend=data["backend"],
                path=Path(data["path"]).expanduser().resolve(),
                description=data.get("description", "Custom model"),
            )
        )
    return models


def resolve_model(
    reference: str,
    config: dict[str, Any],
    backend: str | None = None,
) -> ModelSpec:
    canonical = aliases().get(reference)
    if canonical:
        data = BUILTIN_MODELS[canonical]
        selected_backend = backend or data["backend"]
        if selected_backend != data["backend"]:
            raise ValueError(
                f"Model {reference!r} uses backend {data['backend']!r}, not {selected_backend!r}"
            )
        return ModelSpec(
            canonical,
            selected_backend,
            model_dir(config) / data["filename"],
            data["description"],
            True,
        )

    custom = config.get("custom_models", {}).get(reference)
    if custom:
        selected_backend = backend or custom["backend"]
        if selected_backend != custom["backend"]:
            raise ValueError(
                f"Model {reference!r} uses backend {custom['backend']!r}, not {selected_backend!r}"
            )
        return ModelSpec(
            reference,
            selected_backend,
            Path(custom["path"]).expanduser().resolve(),
            custom.get("description", "Custom model"),
        )

    if backend is None:
        raise ValueError(
            f"Unknown model {reference!r}. Register it with 'qmint model add' "
            "or pass --backend with a model path."
        )
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    return ModelSpec(Path(reference).stem, backend, Path(reference).expanduser().resolve())


def add_custom_model(
    config: dict[str, Any], name: str, path: str, backend: str, description: str = ""
) -> None:
    if name in aliases():
        raise ValueError(f"{name!r} is reserved by a built-in model")
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    config.setdefault("custom_models", {})[name] = {
        "path": str(Path(path).expanduser().resolve()),
        "backend": backend,
        "description": description or "Custom model",
    }
