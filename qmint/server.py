"""Multi-worker local socket server used by Gaussian and ORCA adapters."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import select
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from queue import Empty

from .calculator import calculate, load_calculator, prepare_wsl_cuda_env
from .models import ModelSpec
from .protocol import (
    recv_authenticated,
    remove_port,
    send_pickle,
    server_info,
    write_port,
)


def parse_gpu_spec(spec: str | None, workers: int) -> list[str]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if spec is None:
        return ["cpu"] * workers
    import torch

    if spec in ("", "auto"):
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
        for block in spec.replace(",", " ").replace(";", " ").replace(":", " ").split():
            if "-" in block:
                start, end = (int(value) for value in block.split("-", 1))
                gpu_ids.extend(range(start, end + 1))
            else:
                gpu_ids.append(int(block))
    count = torch.cuda.device_count()
    if not gpu_ids or count == 0:
        raise ValueError("GPU was requested but no visible CUDA device was found")
    if any(gpu < 0 or gpu >= count for gpu in gpu_ids):
        raise ValueError(f"GPU ids must be in 0..{count - 1}")
    return [f"cuda:{gpu_ids[i]}" if i < len(gpu_ids) else "cpu" for i in range(workers)]


def _worker(
    listener_fd: int,
    exit_event: mp.synchronize.Event,
    spec: ModelSpec,
    device: str,
    hessian_mode: str,
    ready_queue,
    num_gpu_workers: int,
    gpu_busy_count,
    token: str,
):
    listener = listener_fd
    try:
        calculator, load_device = load_calculator(spec, device, hessian_mode)
        ready_queue.put({"status": "ready", "pid": os.getpid(), "device": device, "load_device": load_device})
    except Exception as exc:
        ready_queue.put({"status": "error", "pid": os.getpid(), "error": repr(exc), "traceback": traceback.format_exc()})
        return

    listener.setblocking(False)
    while not exit_event.is_set():
        try:
            if device == "cpu" and num_gpu_workers and gpu_busy_count.value < num_gpu_workers:
                time.sleep(0.01)
                continue
            readable, _, _ = select.select([listener], [], [], 0.2)
            if not readable:
                continue
            conn, _ = listener.accept()
            is_gpu = device.startswith("cuda:")
            if is_gpu:
                with gpu_busy_count.get_lock():
                    gpu_busy_count.value += 1
            try:
                with conn:
                    request = recv_authenticated(conn, token)
                    if request == "PING":
                        send_pickle(conn, {"status": "ok"})
                        continue
                    if request == "EXIT":
                        exit_event.set()
                        continue
                    send_pickle(conn, calculate(request, calculator, hessian_mode))
            finally:
                if is_gpu:
                    with gpu_busy_count.get_lock():
                        gpu_busy_count.value -= 1
        except (
            BlockingIOError,
            ConnectionError,
            EOFError,
            OSError,
            PermissionError,
            ValueError,
            pickle.PickleError,
        ):
            continue
    listener.close()


def _daemon(
    spec: ModelSpec,
    devices: list[str],
    hessian_mode: str,
    debug: bool,
    ready_fd: int | None = None,
) -> None:
    if spec.backend == "orb":
        prepare_wsl_cuda_env()
    context = mp.get_context("spawn" if any(d.startswith("cuda:") for d in devices) else "fork")
    exit_event = context.Event()
    ready_queue = context.Queue()
    num_gpu_workers = sum(device.startswith("cuda:") for device in devices)
    gpu_busy_count = context.Value("i", 0)
    import secrets

    token = secrets.token_hex(32)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    processes = []
    try:
        for device in devices:
            process = context.Process(
                target=_worker,
                args=(
                    listener,
                    exit_event,
                    spec,
                    device,
                    hessian_mode,
                    ready_queue,
                    num_gpu_workers,
                    gpu_busy_count,
                    token,
                ),
            )
            process.start()
            processes.append(process)
        for _ in processes:
            try:
                message = ready_queue.get(timeout=120)
            except Empty as exc:
                raise RuntimeError("Timed out while loading a model worker") from exc
            if message.get("status") != "ready":
                raise RuntimeError(message.get("traceback") or message.get("error", "worker failed"))
        write_port(
            listener.getsockname()[1],
            os.getpid(),
            model=spec.name,
            model_path=str(spec.path),
            backend=spec.backend,
            devices=devices,
            hessian=hessian_mode,
            token=token,
        )
        if ready_fd is not None:
            os.write(ready_fd, b"READY\n")
            os.close(ready_fd)
            ready_fd = None
        exit_event.wait()
    finally:
        remove_port()
        exit_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        listener.close()


def start(spec: ModelSpec, workers: int, gpu: str | None, hessian: str, debug: bool = False) -> None:
    if server_info() is not None:
        raise RuntimeError("A QMint server is already running for this job")
    devices = parse_gpu_spec(gpu, workers)
    state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    state_root = state_root / "qmint"
    state_root.mkdir(parents=True, exist_ok=True)
    error_log = state_root / "server.log"
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid:
        os.close(write_fd)
        readable, _, _ = select.select([read_fd], [], [], 130)
        message = os.read(read_fd, 65536).decode("utf-8", errors="replace") if readable else ""
        os.close(read_fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        if message.startswith("READY"):
            return
        if not message:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        detail = message.removeprefix("ERROR\n").strip()
        if not detail and error_log.exists():
            detail = "See " + str(error_log)
        raise RuntimeError(detail or "QMint server startup timed out")
    os.close(read_fd)
    os.setsid()
    second = os.fork()
    if second:
        os._exit(0)
    stream = error_log.open("a", encoding="utf-8")
    with stream:
        sys.stdin = open(os.devnull)
        sys.stdout = stream
        sys.stderr = stream
        try:
            _daemon(spec, devices, hessian, debug, write_fd)
        except Exception:
            detail = traceback.format_exc()
            try:
                os.write(write_fd, ("ERROR\n" + detail).encode("utf-8"))
            except OSError:
                pass
            raise
        finally:
            try:
                os.close(write_fd)
            except OSError:
                pass
