"""QMint command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __author__, __version__
from .config import load_config, save_config
from .models import BACKENDS, add_custom_model, list_models, resolve_model
from .protocol import server_info, stop_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmint",
        description="QMint: switch ML potentials from the terminal and serve them to quantum-chemistry programs.",
    )
    parser.add_argument("--version", action="version", version=f"QMint {__version__}")
    commands = parser.add_subparsers(dest="command")

    start_parser = commands.add_parser("start", help="start the model server")
    start_parser.add_argument("--model", "-m", help="model alias, registered name, or file path")
    start_parser.add_argument("--backend", "-b", choices=BACKENDS)
    start_parser.add_argument("--workers", "--np", "-n", type=int)
    device_group = start_parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", "-g", nargs="?", const="auto", metavar="IDS")
    device_group.add_argument("--cpu", action="store_true", help="override configured GPU use")
    start_parser.add_argument("--hessian", choices=("numeric", "analytic"))
    start_parser.add_argument("--debug", "-d", action="store_true")
    commands.add_parser("stop", aliases=["exit"], help="stop the running model server")
    commands.add_parser("status", help="show server and active model status")
    commands.add_parser("models", help="list built-in and registered models")

    use_parser = commands.add_parser(
        "use", aliases=["switch"], help="select the default model for future starts"
    )
    use_parser.add_argument("model", help="model alias or registered model name")
    use_parser.add_argument("--backend", "-b", choices=BACKENDS)

    model_commands = commands.add_parser("model", help="manage custom model registrations")
    model_subcommands = model_commands.add_subparsers(dest="model_command", required=True)
    add_parser = model_subcommands.add_parser("add", help="register a model file")
    add_parser.add_argument("name")
    add_parser.add_argument("path")
    add_parser.add_argument("--backend", "-b", choices=BACKENDS, required=True)
    add_parser.add_argument("--description", default="")
    model_subcommands.add_parser("list", help="list models")
    model_use_parser = model_subcommands.add_parser("use", help="select the active model")
    model_use_parser.add_argument("model")
    model_use_parser.add_argument("--backend", "-b", choices=BACKENDS)
    remove_parser = model_subcommands.add_parser("remove", help="remove a custom registration")
    remove_parser.add_argument("name")
    config_parser = commands.add_parser("config", help="inspect or change persistent settings")
    config_subcommands = config_parser.add_subparsers(dest="config_command", required=True)
    config_subcommands.add_parser("show")
    set_parser = config_subcommands.add_parser("set")
    set_parser.add_argument("key", choices=("model-dir", "workers", "gpu", "hessian", "debug"))
    set_parser.add_argument("value")
    commands.add_parser("tui", help="open the interactive terminal interface")
    return parser


def _print_models(config: dict) -> None:
    active = config.get("active_model")
    print("NAME                 BACKEND    STATUS   DESCRIPTION")
    print("-" * 76)
    for spec in list_models(config):
        marker = "*" if spec.name == active else " "
        status = "ready" if spec.path.exists() else "missing"
        print(f"{marker}{spec.name:<19} {spec.backend:<10} {status:<8} {spec.description}")
        print(f"  path: {spec.path}")


def _start(args: argparse.Namespace, config: dict) -> None:
    from .server import start

    reference = args.model or config["active_model"]
    spec = resolve_model(reference, config, args.backend)
    workers = args.workers if args.workers is not None else int(config["server"].get("workers", 1))
    gpu = None if args.cpu else (
        args.gpu if args.gpu is not None else config["server"].get("gpu")
    )
    hessian = args.hessian or config["server"].get("hessian", "numeric")
    debug = bool(args.debug or config["server"].get("debug", False))
    if not spec.path.exists():
        raise ValueError(
            f"Model file does not exist: {spec.path}\n"
            "Set MLP_MODEL_DIR/QMINT_CONFIG_HOME or register a custom model with 'qmint model add'."
        )
    start(spec, workers, gpu, hessian, debug)
    print(f"QMint server ready: {spec.name} ({spec.backend})")


def _set_config(config: dict, key: str, value: str) -> None:
    if key == "model-dir":
        config["model_dir"] = value
    elif key == "workers":
        config["server"]["workers"] = max(1, int(value))
    elif key == "gpu":
        config["server"]["gpu"] = None if value.lower() in ("none", "cpu", "off") else value
    elif key == "hessian":
        if value not in ("numeric", "analytic"):
            raise ValueError("hessian must be numeric or analytic")
        config["server"]["hessian"] = value
    elif key == "debug":
        config["server"]["debug"] = value.lower() in ("1", "true", "yes", "on")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.command:
        _parser().print_help()
        return 0
    config = load_config()
    try:
        if args.command == "start":
            _start(args, config)
        elif args.command in ("stop", "exit"):
            stop_server()
            print("QMint server stopped")
        elif args.command == "status":
            info = server_info()
            if info:
                public_info = {key: value for key, value in info.items() if key != "token"}
                print(json.dumps(public_info, indent=2, sort_keys=True))
            else:
                print("QMint server is not running")
        elif args.command == "models":
            _print_models(config)
        elif args.command in ("use", "switch"):
            spec = resolve_model(args.model, config, args.backend)
            config["active_model"] = spec.name if spec.builtin else args.model
            save_config(config)
            print(f"Active model: {config['active_model']} ({spec.backend})")
        elif args.command == "model" and args.model_command == "add":
            add_custom_model(config, args.name, args.path, args.backend, args.description)
            save_config(config)
            print(f"Registered model: {args.name}")
        elif args.command == "model" and args.model_command == "list":
            _print_models(config)
        elif args.command == "model" and args.model_command == "use":
            spec = resolve_model(args.model, config, args.backend)
            config["active_model"] = spec.name if spec.builtin else args.model
            save_config(config)
            print(f"Active model: {config['active_model']} ({spec.backend})")
        elif args.command == "model" and args.model_command == "remove":
            if args.name not in config.get("custom_models", {}):
                raise ValueError(f"Custom model is not registered: {args.name}")
            del config["custom_models"][args.name]
            if config.get("active_model") == args.name:
                config["active_model"] = "uma-s"
            save_config(config)
            print(f"Removed model registration: {args.name}")
        elif args.command == "config" and args.config_command == "show":
            print(json.dumps(config, indent=2, sort_keys=True))
        elif args.command == "config" and args.config_command == "set":
            _set_config(config, args.key, args.value)
            save_config(config)
            print(f"Updated {args.key}")
        elif args.command == "tui":
            from .tui import run

            run()
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"qmint: error: {exc}", file=sys.stderr)
        return 1


def server_main(argv: Sequence[str] | None = None) -> int:
    """Compatibility entry point for the historical ``server`` command."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("start", "stop", "exit"):
        command = "stop" if values[0] == "exit" else values[0]
        return main([command, *values[1:]])
    print("Use 'server start ...' or 'server exit'. For model switching use 'qmint use ...'.")
    return 2
