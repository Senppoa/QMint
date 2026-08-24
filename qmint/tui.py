"""Dependency-free guided TUI for configuring and controlling QMint."""

from __future__ import annotations

import curses
import os

from . import __author__, __version__
from .config import config_path, load_config, save_config
from .models import ModelSpec, download_model, list_models
from .protocol import server_info, stop_server
from .server import start


ORB_HESSIAN_URL = "https://github.com/Senppoa/orb-hessian"
UMA_DOWNLOAD_URL = "https://huggingface.co/facebook/UMA"
ASCII_LOGO = (
    "  QQQ  M   M  III  N   N  TTT",
    " Q     MM MM   I   NN  N   T ",
    " Q  QQ M M M   I   N N N   T ",
    "  QQQ  M   M  III  N  NN   T ",
)


def run(first_run: bool | None = None) -> None:
    """Open QMint and always stop its model workers when the TUI exits."""
    if first_run is None:
        first_run = not config_path().exists()
    runtime = {"server_started": False}
    try:
        curses.wrapper(lambda screen: _draw(screen, bool(first_run), runtime))
    finally:
        if runtime["server_started"] and server_info() is not None:
            try:
                stop_server()
            except RuntimeError:
                pass


def _draw(screen, first_run: bool = False, runtime: dict[str, bool] | None = None) -> None:
    runtime = runtime if runtime is not None else {"server_started": False}
    config = load_config()
    if first_run:
        _initial_setup(screen, config)
    specs = list_models(config)
    if not specs:
        raise RuntimeError("No models registered")

    selected_model = next(
        (index for index, spec in enumerate(specs) if spec.name == config["active_model"]), 0
    )
    workers = max(1, int(config["server"].get("workers", 1)))
    gpu_value = config["server"].get("gpu")
    execution = "cpu" if gpu_value is None else (
        "multi" if _is_multi_gpu(gpu_value) else "single"
    )
    gpu_ids = "auto" if gpu_value in (None, "", "auto") else str(gpu_value)
    hessian = config["server"].get("hessian", "numeric")
    debug = bool(config["server"].get("debug", False))
    row = 0
    message = "Configure each field, then select Start server."

    while True:
        fields = ["model", "workers", "execution", "gpu_ids", "hessian", "debug", "start"]
        screen.erase()
        height, width = screen.getmaxyx()
        _line(screen, 0, "QMint - Quantum Machine-Learning Interface", width, curses.A_BOLD)
        _line(
            screen,
            1,
            f"v{__version__}   Author: {__author__}   Gaussian | ORCA | Extensible",
            width,
        )
        _line(screen, 3, "Guided server configuration", width, curses.A_BOLD)

        spec = specs[selected_model]
        values = [
            f"{spec.name} ({spec.backend}, {'ready' if spec.path.exists() else 'missing'})",
            str(workers),
            {"cpu": "CPU", "single": "Single GPU", "multi": "Multiple GPUs"}[execution],
            "not used" if execution == "cpu" else gpu_ids,
            hessian,
            "on" if debug else "off",
            "Start server",
        ]
        labels = ["Model", "Workers", "Compute", "GPU IDs", "Hessian", "Debug logging", "Action"]
        for index, (label, value) in enumerate(zip(labels, values)):
            marker = ">" if index == row else " "
            attributes = curses.A_REVERSE if index == row else 0
            _line(screen, 5 + index, f"{marker} {label:<16} {value}", width, attributes)

        active = server_info()
        _line(screen, 13, f"Server: {'running' if active else 'stopped'}", width, curses.A_BOLD)
        if spec.backend == "orb" and hessian == "analytic":
            _line(screen, 14, "Orb analytic Hessian requires orb-hessian:", width, curses.A_BOLD)
            _line(screen, 15, ORB_HESSIAN_URL, width)
        elif execution == "multi":
            _line(screen, 14, "Multi-GPU loads one model replica per GPU worker.", width)
            _line(screen, 15, "Set Workers >= the number of selected GPU IDs.", width)
        else:
            _line(screen, 14, spec.description, width)
            _line(screen, 15, f"Model path: {spec.path}", width)
        _line(screen, height - 3, message, width)
        _line(
            screen,
            height - 2,
            "Up/Down field  Left/Right change  Enter edit/select  s stop  q quit",
            width,
            curses.A_DIM,
        )
        screen.refresh()
        key = screen.getch()

        if key in (ord("q"), 27):
            _save_settings(config, spec, workers, execution, gpu_ids, hessian, debug)
            return
        if key in (curses.KEY_UP, ord("k")):
            row = (row - 1) % len(fields)
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            row = (row + 1) % len(fields)
            continue
        if key == ord("s"):
            try:
                stop_server()
                runtime["server_started"] = False
                message = "Server stopped."
            except RuntimeError as exc:
                message = str(exc)
            continue

        direction = -1 if key in (curses.KEY_LEFT, ord("h")) else 1
        activate = key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord("h"), ord("l"), 10, 13)
        if not activate:
            continue
        field = fields[row]
        if field == "model":
            selected_model = (selected_model + direction) % len(specs)
        elif field == "workers":
            if key in (10, 13):
                entered = _prompt(screen, "Worker processes", str(workers))
                try:
                    workers = max(1, int(entered))
                except ValueError:
                    message = "Workers must be a positive integer."
            else:
                workers = max(1, workers + direction)
        elif field == "execution":
            modes = ["cpu", "single", "multi"]
            execution = modes[(modes.index(execution) + direction) % len(modes)]
            if execution == "single" and gpu_ids == "auto":
                gpu_ids = "0"
            elif execution == "multi" and "," not in gpu_ids:
                gpu_ids = "auto"
                workers = max(workers, _available_gpu_count())
        elif field == "gpu_ids":
            if execution == "cpu":
                message = "Select Single GPU or Multiple GPUs before setting GPU IDs."
            else:
                default = "0" if execution == "single" else "auto"
                gpu_ids = _prompt(
                    screen,
                    "GPU IDs (for example 0 or 0,1; auto uses all)",
                    gpu_ids or default,
                )
                if execution == "single" and gpu_ids in ("", "auto"):
                    gpu_ids = "0"
        elif field == "hessian":
            hessian = "analytic" if hessian == "numeric" else "numeric"
            if hessian == "analytic" and spec.backend == "orb":
                message = f"Install orb-hessian for this mode: {ORB_HESSIAN_URL}"
        elif field == "debug":
            debug = not debug
        elif field == "start":
            spec = specs[selected_model]
            if not spec.path.exists():
                message = (
                    f"Model missing. Place it at {spec.path}; "
                    "downloads run only at first setup."
                )
                continue
            try:
                required_workers = _required_gpu_workers(execution, gpu_ids)
            except ValueError:
                message = "GPU IDs must be numbers, comma-separated IDs, or ranges such as 0,2-3."
                continue
            if workers < required_workers:
                message = (
                    f"Multiple GPUs require at least {required_workers} workers; "
                    f"current value is {workers}."
                )
                continue
            gpu = _gpu_argument(execution, gpu_ids)
            _save_settings(config, spec, workers, execution, gpu_ids, hessian, debug)
            try:
                start(spec, workers, gpu, hessian, debug)
                runtime["server_started"] = True
                message = f"Server started with {spec.name}."
            except (RuntimeError, ValueError, OSError) as exc:
                message = str(exc)


def _initial_setup(screen, config: dict) -> None:
    """Offer optional model downloads once, before the first TUI session."""
    downloadable = [
        spec for spec in list_models(config) if spec.download_url and not spec.path.exists()
    ]
    selected = [True] * len(downloadable)
    row = 0
    message = "Space toggles. Enter downloads selected models; s skips."
    while downloadable:
        screen.erase()
        height, width = screen.getmaxyx()
        _logo(screen, width)
        _line(screen, 6, "QMint first-time setup", width, curses.A_BOLD)
        _line(screen, 7, f"Author: {__author__}", width)
        _line(screen, 8, "Citation: Tang, K. (2026). QMint: Quantum Machine-Learning Interface.", width)
        _line(screen, 10, "Optional MLIP model downloads", width)
        _line(screen, 11, f"Destination: {downloadable[0].path.parent}", width)
        for index, spec in enumerate(downloadable[: max(1, height - 11)]):
            marker = ">" if index == row else " "
            checked = "x" if selected[index] else " "
            _line(
                screen,
                13 + index,
                f"{marker} [{checked}] {spec.name:<18} {spec.description}",
                width,
            )
        note_row = min(height - 5, 14 + len(downloadable))
        _line(
            screen,
            note_row,
            "UMA models are not downloaded automatically because access is gated.",
            width,
        )
        _line(screen, note_row + 1, f"Download UMA manually: {UMA_DOWNLOAD_URL}", width)
        _line(screen, note_row + 2, "Then place its .pt file in the destination above.", width)
        chosen = downloadable[row]
        _line(screen, note_row + 3, f"Selected download URL: {chosen.download_url}", width)
        _line(screen, height - 2, message, width, curses.A_DIM)
        screen.refresh()
        key = screen.getch()
        if key in (curses.KEY_UP, ord("k")):
            row = (row - 1) % len(downloadable)
        elif key in (curses.KEY_DOWN, ord("j")):
            row = (row + 1) % len(downloadable)
        elif key == ord(" "):
            selected[row] = not selected[row]
        elif key in (ord("s"), ord("q"), 27):
            break
        elif key in (10, 13):
            for index, spec in enumerate(downloadable):
                if not selected[index]:
                    continue
                screen.erase()
                _line(screen, 0, "QMint first-time setup", width, curses.A_BOLD)
                _line(screen, 2, f"Downloading {spec.name}...", width)
                _line(screen, 3, str(spec.path), width)
                screen.refresh()
                try:
                    download_model(spec)
                except (OSError, ValueError) as exc:
                    message = f"{spec.name} failed: {exc}"
            break

    if not downloadable:
        screen.erase()
        height, width = screen.getmaxyx()
        _logo(screen, width)
        _line(screen, 6, "QMint first-time setup", width, curses.A_BOLD)
        _line(screen, 7, f"Author: {__author__}", width)
        _line(screen, 8, "Citation: Tang, K. (2026). QMint: Quantum Machine-Learning Interface.", width)
        _line(screen, 10, "No automatically downloadable model is missing.", width)
        _line(
            screen,
            12,
            "UMA models are not downloaded automatically because access is gated.",
            width,
        )
        _line(screen, 13, f"Download UMA manually: {UMA_DOWNLOAD_URL}", width)
        _line(screen, height - 2, "Press any key to continue.", width, curses.A_DIM)
        screen.refresh()
        screen.getch()

    ready = next((spec for spec in list_models(config) if spec.path.exists()), None)
    active_ready = next(
        (spec.path.exists() for spec in list_models(config) if spec.name == config["active_model"]),
        False,
    )
    if ready and not active_ready:
        config["active_model"] = ready.name
    save_config(config)


def _save_settings(
    config: dict,
    spec: ModelSpec,
    workers: int,
    execution: str,
    gpu_ids: str,
    hessian: str,
    debug: bool,
) -> None:
    config["active_model"] = spec.name
    config["server"].update(
        {
            "workers": workers,
            "gpu": _gpu_argument(execution, gpu_ids),
            "hessian": hessian,
            "debug": debug,
        }
    )
    save_config(config)


def _gpu_argument(execution: str, gpu_ids: str) -> str | None:
    if execution == "cpu":
        return None
    value = gpu_ids.strip()
    if execution == "single":
        return value.split(",", 1)[0].strip() or "0"
    return value or "auto"


def _is_multi_gpu(value: object) -> bool:
    text = str(value)
    return text in ("", "auto") or "," in text or ";" in text or "-" in text


def _required_gpu_workers(execution: str, gpu_ids: str) -> int:
    if execution != "multi":
        return 1
    if gpu_ids.strip() in ("", "auto"):
        return _available_gpu_count()
    identifiers: set[int] = set()
    for block in gpu_ids.replace(";", ",").split(","):
        block = block.strip()
        if not block:
            continue
        if "-" in block:
            first, last = (int(value) for value in block.split("-", 1))
            identifiers.update(range(first, last + 1))
        else:
            identifiers.add(int(block))
    return max(1, len(identifiers))


def _available_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return max(1, len([item for item in visible.split(",") if item.strip()]))
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except ModuleNotFoundError:
        return 1


def _prompt(screen, label: str, default: str) -> str:
    height, width = screen.getmaxyx()
    screen.move(height - 4, 0)
    screen.clrtoeol()
    _line(screen, height - 4, f"{label} [{default}]: ", width, curses.A_BOLD)
    screen.refresh()
    curses.echo()
    try:
        column = min(width - 1, len(label) + len(default) + 5)
        value = screen.getstr(height - 4, column, 64)
    finally:
        curses.noecho()
    text = value.decode("utf-8", errors="replace").strip()
    return text or default


def _line(screen, row: int, text: str, width: int, attributes: int = 0) -> None:
    if row < 0 or row >= screen.getmaxyx()[0]:
        return
    try:
        screen.addnstr(row, 0, text, max(1, width - 1), attributes)
    except curses.error:
        pass


def _logo(screen, width: int) -> None:
    for row, text in enumerate(ASCII_LOGO):
        _line(screen, row, text, width, curses.A_BOLD)
