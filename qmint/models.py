"""Model registry and path resolution."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
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
    head: str | None = None
    download_url: str | None = None
    sha256: str | None = None


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
        "head": "omol",
        "download_url": (
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_omol_0/MACE-omol-0-extra-large-1024.model"
        ),
        "sha256": "9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469",
    },
    "mace-polar-m": {
        "backend": "mace",
        "filename": "MACE-POLAR-1-M.model",
        "aliases": (),
        "description": "MACE POLAR medium",
        "download_url": (
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_polar_1/MACE-POLAR-1-M.model"
        ),
        "sha256": "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a",
    },
    "mace-polar-l": {
        "backend": "mace",
        "filename": "MACE-POLAR-1-L.model",
        "aliases": (),
        "description": "MACE POLAR large",
        "download_url": (
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_polar_1/MACE-POLAR-1-L.model"
        ),
        "sha256": "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b",
    },
    "mace-mh-1": {
        "backend": "mace",
        "filename": "mace-mh-1.model",
        "aliases": (),
        "description": "MACE multi-head model (OMol head)",
        "head": "omol",
    },
    "deepest-os": {
        "backend": "mace",
        "filename": "deepest-os.model",
        "aliases": (),
        "description": "DeepEst-OS MACE model",
    },
    "orbmol-v2": {
        "backend": "orb",
        "filename": "orbmol-v2-teqabfhg-20260523.ckpt",
        "aliases": ("orb-mol-v2", "orbmol-v2-teqabfhg-20260523.ckpt"),
        "description": "OrbMol v2",
        "download_url": (
            "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
            "forcefields/orbmol-v2-teqabfhg-20260523.ckpt"
        ),
        "sha256": "3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8",
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
            head=data.get("head"),
            download_url=data.get("download_url"),
            sha256=data.get("sha256"),
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
                head=data.get("head"),
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
            data.get("head"),
            data.get("download_url"),
            data.get("sha256"),
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
            False,
            custom.get("head"),
        )

    if backend is None:
        raise ValueError(
            f"Unknown model {reference!r}. Register it with 'qmint model add' "
            "or pass --backend with a model path."
        )
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    return ModelSpec(Path(reference).stem, backend, Path(reference).expanduser().resolve())


def verify_model(path: Path, sha256: str | None = None) -> bool:
    """Return whether a model exists and matches its optional SHA-256 digest."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if not sha256:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == sha256.lower()


def download_model(spec: ModelSpec, destination: Path | None = None) -> Path:
    """Download a built-in model atomically and verify its digest."""
    if not spec.download_url:
        raise ValueError(
            f"{spec.name} has no automatic download URL; download it manually and place it at {spec.path}"
        )
    target = (destination or spec.path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify_model(target, spec.sha256):
        return target
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            spec.download_url, headers={"User-Agent": "QMint model downloader"}
        )
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(
            request, timeout=60
        ) as response:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not verify_model(temporary, spec.sha256):
            raise ValueError(f"Downloaded {spec.name} failed SHA-256 verification")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def add_custom_model(
    config: dict[str, Any],
    name: str,
    path: str,
    backend: str,
    description: str = "",
    head: str | None = None,
) -> None:
    if name in aliases():
        raise ValueError(f"{name!r} is reserved by a built-in model")
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    config.setdefault("custom_models", {})[name] = {
        "path": str(Path(path).expanduser().resolve()),
        "backend": backend,
        "description": description or "Custom model",
        "head": head,
    }
