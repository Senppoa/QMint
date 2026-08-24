"""Persistent user configuration for model selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CONFIG_VERSION = 1


def config_dir() -> Path:
    override = os.environ.get("QMINT_CONFIG_HOME")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "qmint"


def config_path() -> Path:
    return config_dir() / "config.json"


def default_config() -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "active_model": "uma-s",
        "model_dir": str(Path.home() / ".local" / "share" / "qmint" / "models"),
        "custom_models": {},
        "server": {
            "workers": 1,
            "gpu": None,
            "hessian": "numeric",
            "debug": False,
        },
    }


def load_config(path: Path | None = None) -> dict[str, Any]:
    target = path or config_path()
    config = default_config()
    if not target.exists():
        return config

    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read QMint config {target}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"QMint config must contain a JSON object: {target}")

    for key in ("active_model", "model_dir", "custom_models"):
        if key in loaded:
            config[key] = loaded[key]
    if isinstance(loaded.get("server"), dict):
        config["server"].update(loaded["server"])
    return config


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["version"] = CONFIG_VERSION
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return target
