"""Dependency-free curses UI for selecting and controlling a QMint server."""

from __future__ import annotations

import curses

from . import __author__, __version__
from .config import load_config, save_config
from .models import list_models
from .protocol import server_info, stop_server
from .server import start


def run() -> None:
    curses.wrapper(_draw)


def _draw(screen) -> None:
    config = load_config()
    specs = list_models(config)
    if not specs:
        raise RuntimeError("No models registered")
    selected = next((i for i, spec in enumerate(specs) if spec.name == config["active_model"]), 0)
    gpu = config["server"].get("gpu")
    hessian = config["server"].get("hessian", "numeric")
    message = "Up/Down select  Enter start  s stop  g GPU  h Hessian  q quit"
    while True:
        screen.erase()
        height, width = screen.getmaxyx()
        _line(screen, 0, "QMint - Quantum ML Potential Router", width, curses.A_BOLD)
        _line(screen, 1, f"v{__version__}   Author: {__author__}   Gaussian | ORCA | Extensible", width)
        _line(screen, 3, "Model", width, curses.A_BOLD)
        visible_count = max(1, height - 10)
        offset = min(max(0, selected - visible_count + 1), max(0, len(specs) - visible_count))
        for row, spec in enumerate(specs[offset : offset + visible_count]):
            index = offset + row
            marker = ">" if index == selected else " "
            status = "ready" if spec.path.exists() else "missing"
            _line(screen, 4 + row, f"{marker} {spec.name:<20} {spec.backend:<9} {status:<8} {spec.description}", width)
        info_line = 5 + min(len(specs), visible_count)
        _line(screen, info_line, f"GPU: {gpu or 'CPU'}    Hessian: {hessian}", width)
        active = server_info()
        _line(screen, info_line + 1, f"Server: {'running' if active else 'stopped'}", width)
        _line(screen, height - 2, message, width, curses.A_DIM)
        screen.refresh()
        key = screen.getch()
        if key in (ord("q"), 27):
            return
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(specs)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(specs)
        elif key == ord("g"):
            gpu = "auto" if gpu is None else None
            config["server"]["gpu"] = gpu
            save_config(config)
        elif key == ord("h"):
            hessian = "analytic" if hessian == "numeric" else "numeric"
            config["server"]["hessian"] = hessian
            save_config(config)
        elif key == ord("s"):
            try:
                stop_server()
                message = "Server stopped"
            except RuntimeError as exc:
                message = str(exc)
        elif key in (curses.KEY_ENTER, 10, 13):
            spec = specs[selected]
            if not spec.path.exists():
                message = f"Missing model file: {spec.path}"
                continue
            config["active_model"] = spec.name
            save_config(config)
            try:
                start(spec, int(config["server"].get("workers", 1)), gpu, hessian, bool(config["server"].get("debug")))
                message = f"Server started with {spec.name}"
            except (RuntimeError, ValueError) as exc:
                message = str(exc)


def _line(screen, row: int, text: str, width: int, attributes: int = 0) -> None:
    if row < 0 or row >= screen.getmaxyx()[0]:
        return
    try:
        screen.addnstr(row, 0, text, max(1, width - 1), attributes)
    except curses.error:
        pass
