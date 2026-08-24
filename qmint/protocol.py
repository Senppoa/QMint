"""Small framed pickle protocol used between adapters and the local server."""

from __future__ import annotations

import json
import os
import pickle
import re
import secrets
import signal
import socket
import struct
import tempfile
import time
from pathlib import Path
from typing import Any


AUTH_TOKEN_SIZE = 64


def job_id() -> str:
    for name in ("MLP_JOBID", "PBS_JOBID", "LSB_JOBID", "SLURM_JOB_ID", "PJM_JOBID"):
        if os.environ.get(name):
            value = os.environ[name]
            break
    else:
        value = os.environ.get("USER") or os.environ.get("LOGNAME") or "default"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def port_file() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp")) / f"qmint_{job_id()}.json"


def send_pickle(conn: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def recv_pickle(conn: socket.socket) -> Any:
    header = _recv_exact(conn, 4)
    if not header:
        return None
    size = struct.unpack("!I", header)[0]
    if size > 256 * 1024 * 1024:
        raise ValueError(f"Refusing oversized message ({size} bytes)")
    return pickle.loads(_recv_exact(conn, size))


def send_authenticated(conn: socket.socket, token: str, value: Any) -> None:
    encoded = token.encode("ascii")
    if len(encoded) != AUTH_TOKEN_SIZE:
        raise ValueError("Invalid QMint authentication token")
    conn.sendall(encoded)
    send_pickle(conn, value)


def recv_authenticated(conn: socket.socket, token: str) -> Any:
    supplied = _recv_exact(conn, AUTH_TOKEN_SIZE).decode("ascii", errors="replace")
    if not secrets.compare_digest(supplied, token):
        raise PermissionError("Unauthorized QMint request")
    return recv_pickle(conn)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed before a complete message was received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_port() -> int:
    path = port_file()
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["port"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"QMint server is not running ({path})") from exc


def read_server_file() -> dict[str, Any]:
    path = port_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "port" not in data or "token" not in data:
            raise ValueError("missing port or token")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"QMint server is not running ({path})") from exc


def server_pid() -> int | None:
    try:
        value = json.loads(port_file().read_text(encoding="utf-8")).get("pid")
        return int(value) if value else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_port(port: int, pid: int | None = None, **metadata: Any) -> None:
    target = port_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"port": port, "pid": pid or os.getpid(), **metadata})
    descriptor, temporary_name = tempfile.mkstemp(prefix=".qmint-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def remove_port() -> None:
    try:
        port_file().unlink()
    except FileNotFoundError:
        pass


def server_info() -> dict[str, Any] | None:
    try:
        info = json.loads(port_file().read_text(encoding="utf-8"))
        with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=0.5) as conn:
            send_authenticated(conn, str(info["token"]), "PING")
            if recv_pickle(conn) == {"status": "ok"}:
                return info
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def request(task: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while not port_file().exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for QMint server: {port_file()}")
        time.sleep(0.25)
    info = read_server_file()
    with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=timeout) as conn:
        send_authenticated(conn, str(info["token"]), task)
        result = recv_pickle(conn)
    if not isinstance(result, dict):
        raise RuntimeError("QMint server returned an invalid response")
    return result


def stop_server() -> None:
    info = read_server_file()
    port = int(info["port"])
    pid = int(info["pid"]) if info.get("pid") else None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            send_authenticated(conn, str(info["token"]), "EXIT")
    except OSError as exc:
        remove_port()
        raise RuntimeError(f"Could not contact QMint server: {exc}") from exc
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                time.sleep(0.2)
        except OSError:
            remove_port()
            return
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
    remove_port()
