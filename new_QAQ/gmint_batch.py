"""Run independent Gaussian inputs concurrently, one persistent service per GPU."""

import argparse
import csv
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from utils import (cuda_device_token, get_service_name,
                   get_service_record, probe_service, read_int_list_string,
                   remove_port_file, send_pickle)


def available_gpu_count():
    import torch
    return torch.cuda.device_count()


def select_devices(gpu_spec, workers, job_count):
    """Return (display name, physical CUDA token) for each execution slot."""
    if not gpu_spec:
        return [(f'cpu-{i}', None) for i in range(min(workers or 1, job_count))]
    count = available_gpu_count()
    ids = list(range(count)) if gpu_spec == 'auto' else read_int_list_string(gpu_spec)
    if not ids:
        raise ValueError('No GPU selected or no CUDA GPU available')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate GPU indices are not allowed')
    if any(i < 0 or i >= count for i in ids):
        raise ValueError(f'GPU selection {ids} is invalid; {count} GPU(s) are visible')
    if workers is not None and workers > len(ids):
        raise ValueError('--workers cannot exceed the number of selected GPUs')
    ids = ids[:min(workers or len(ids), job_count)]
    return [(f'gpu-{i}', cuda_device_token(i)) for i in ids]


def validate_parallel_inputs(inputs, parallel):
    checkpoints = {}
    for path in inputs:
        text = path.read_text(encoding='utf-8', errors='replace')
        for command in re.findall(r'''external\s*=\s*["']([^"']*)["']''', text, re.I):
            if re.search(r'(?:^|\s)(?:--service(?:=|\s)|-s\s)', command):
                raise ValueError(f'{path.name}: remove the fixed --service/-s from External; '
                                 'the batch scheduler sets MLP_SERVICE for each GPU')
        if not parallel:
            continue
        for name in re.findall(r'^\s*%chk\s*=\s*(.+?)\s*$', text, re.I | re.M):
            name = name.strip('"\'')
            if 'GAUSS_SCRDIR' in name:
                continue  # The scheduler gives each job a separate scratch directory.
            checkpoint = Path(os.path.expandvars(name)).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = path.parent / checkpoint
            checkpoint = checkpoint.resolve()
            previous = checkpoints.get(checkpoint)
            if previous is not None and previous != path:
                raise ValueError(f'{previous.name} and {path.name} share %chk={name}; '
                                 'give parallel jobs different checkpoint paths')
            checkpoints[checkpoint] = path


def terminate_group(process):
    """Reap a process and descendants started by us with start_new_session."""
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
            if sig == signal.SIGTERM:
                # Allow Python's resource tracker to unlink IPC objects after
                # the leader closes its last pipe, then finish stray children.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.2)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(f'Could not stop process group {process.pid}')


@dataclass
class Slot:
    label: str
    token: str | None
    service: str
    env: dict
    directory: Path
    process: subprocess.Popen | None = None
    ready: bool = False


class Batch:
    def __init__(self, args):
        self.args = args
        self.project = Path(__file__).resolve().parent
        self.inputs = [Path(value).resolve() for value in args.inputs]
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError('The same input file was selected more than once')
        self.outputs = [Path(args.output).resolve() if args.output else
                        path.with_suffix('.log') for path in self.inputs]
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError('Inputs have colliding log names (for example a.gjf and a.com)')
        if any(path in self.inputs for path in self.outputs):
            raise ValueError('A log path would overwrite an input file')
        self.cancel = threading.Event()
        self.signal_number = None
        self.lock = threading.Lock()
        self.all_ready = False
        self.run_dir = self.inputs[0].parent / '.gmint' / args.job_id
        self.summary = self.run_dir / 'summary.tsv'
        self.results = [dict(input=str(path), gpu='', cuda_device='', service='',
                             status='pending', exit_code='', detail='',
                             log=str(log), model=args.model)
                        for path, log in zip(self.inputs, self.outputs)]
        devices = select_devices(args.gpu, args.workers, len(self.inputs))
        validate_parallel_inputs(self.inputs, parallel=len(devices) > 1)
        self.slots = []
        for label, token in devices:
            name = args.service if len(devices) == 1 else f'{args.service}-{label}'
            get_service_name(name)
            env = os.environ.copy()
            env.update(MLP_JOBID=args.job_id, MLP_SERVICE=name,
                       CUDA_VISIBLE_DEVICES=token if token is not None else '')
            self.slots.append(Slot(label, token, name, env, self.run_dir / label))

    def record(self, index, **changes):
        with self.lock:
            self.results[index].update(changes)
            temporary = self.summary.with_suffix('.tmp')
            with temporary.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.results[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(self.results)
            os.replace(temporary, self.summary)

    def request_cancel(self, signum, frame):
        self.signal_number = signum
        self.cancel.set()

    def start_slot(self, slot):
        slot.directory.mkdir()
        command = [sys.executable, str(self.project / 'server-multi'), 'start',
                   '--foreground', '--service', slot.service,
                   '--backend', self.args.backend, '--model', self.args.model,
                   '--np', '1', '--hessian', self.args.hessian,
                   '--startup-timeout', str(self.args.startup_timeout)]
        if slot.token is not None:
            # Each fresh server sees exactly one GPU, at logical index zero.
            command.extend(['--gpu', '0'])
        log = slot.directory / 'server.log'
        print(f'[gmint] {slot.label}: service={slot.service} '
              f'CUDA_VISIBLE_DEVICES={slot.env["CUDA_VISIBLE_DEVICES"]!r}; log={log}', flush=True)
        with log.open('w', encoding='utf-8') as stream:
            slot.process = subprocess.Popen(command, cwd=slot.directory, env=slot.env,
                                            stdout=stream, stderr=subprocess.STDOUT,
                                            start_new_session=True)
        deadline = time.monotonic() + self.args.startup_timeout
        while not self.cancel.is_set() and time.monotonic() < deadline:
            if slot.process.poll() is not None:
                raise RuntimeError(f'{slot.label}: model service exited; see {log}')
            try:
                record = get_service_record(slot.service)
                if record.get('pid') == slot.process.pid and probe_service(slot.service, timeout=0.2):
                    slot.ready = True
                    return
            except (OSError, ValueError, KeyError):
                pass
            self.cancel.wait(0.2)
        raise RuntimeError(f'{slot.label}: startup cancelled or timed out; see {log}')

    def stop_slot(self, slot):
        if slot.process is None:
            return
        if slot.ready and slot.process.poll() is None:
            try:
                record = get_service_record(slot.service)
                if record.get('pid') == slot.process.pid:
                    with socket.create_connection(('localhost', record['port']), timeout=1) as conn:
                        send_pickle(conn, 'EXIT')
                slot.process.wait(timeout=10)
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
                pass
        # This also handles interruption before a discovery file was published.
        terminate_group(slot.process)
        try:
            if get_service_record(slot.service).get('pid') == slot.process.pid:
                remove_port_file(slot.service)
        except (OSError, ValueError, KeyError):
            pass

    def run_input(self, slot, index):
        path, log = self.inputs[index], self.outputs[index]
        self.record(index, gpu=slot.label, cuda_device=slot.token or 'cpu',
                    service=slot.service, status='running')
        print(f'[gmint] {slot.label} -> {path.name}; log={log}', flush=True)
        if slot.process.poll() is not None:
            raise RuntimeError(f'{slot.label}: service stopped unexpectedly')
        env = slot.env.copy()
        # Isolate Gaussian scratch files even when many jobs share a directory.
        scratch_root = env.get('GAUSS_SCRDIR') or tempfile.gettempdir()
        with tempfile.TemporaryDirectory(prefix='gmint-g16-', dir=scratch_root) as scratch:
            env['GAUSS_SCRDIR'] = scratch
            with log.open('w', encoding='utf-8') as stream:
                process = subprocess.Popen([shutil.which('g16'), str(path)], cwd=path.parent,
                                           env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    while process.poll() is None:
                        if self.cancel.wait(0.1):
                            break
                        if slot.process.poll() is not None:
                            raise RuntimeError(f'{slot.label}: model service died during {path.name}')
                finally:
                    # Stop any remaining descendants before releasing scratch.
                    terminate_group(process)
            code = process.returncode
        if self.cancel.is_set():
            status, detail = 'cancelled', 'batch interrupted'
        else:
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                tail = stream.read().rstrip().splitlines()
            normal = bool(tail and b'Normal termination of Gaussian' in tail[-1])
            status = 'succeeded' if code == 0 and normal else 'failed'
            detail = '' if status == 'succeeded' else 'nonzero exit or missing Gaussian normal termination'
        self.record(index, status=status, exit_code=code, detail=detail)
        print(f'[gmint] {slot.label}: {path.name}: {status} (exit {code})', flush=True)

    def run_worker(self, slot, initial, pending):
        index = initial
        while not self.cancel.is_set():
            try:
                self.run_input(slot, index)
            except Exception as exc:
                self.record(index, status='failed', detail=str(exc))
                print(f'[gmint] {slot.label}: {exc}', file=sys.stderr, flush=True)
                if slot.process.poll() is not None:
                    self.cancel.set()
                    return
            if self.cancel.is_set():
                return
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return

    def run(self):
        # Do not take ownership of a pre-existing service, even with a user JOBID.
        for slot in self.slots:
            if probe_service(slot.service, timeout=0.2):
                raise ValueError(f'Service {slot.service!r} is already running for this JOBID')
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.record(0)
        print(f'[gmint] Python: {sys.executable}\n[gmint] Model: {self.args.model}\n'
              f'[gmint] Job ID: {self.args.job_id}\n'
              f'[gmint] Files: {len(self.inputs)}; parallel slots: {len(self.slots)}\n'
              f'[gmint] Summary: {self.summary}', flush=True)
        handlers = {sig: signal.signal(sig, self.request_cancel)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        startup_errors = []
        try:
            with ThreadPoolExecutor(max_workers=len(self.slots)) as pool:
                futures = [pool.submit(self.start_slot, slot) for slot in self.slots]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        startup_errors.append(str(exc))
                        self.cancel.set()
            if not startup_errors and not self.cancel.is_set():
                self.all_ready = True
                pending = queue.Queue()
                for index in range(len(self.slots), len(self.inputs)):
                    pending.put(index)
                print('[gmint] All services ready. Each GPU will take the next file when free.', flush=True)
                with ThreadPoolExecutor(max_workers=len(self.slots)) as pool:
                    futures = [pool.submit(self.run_worker, slot, index, pending)
                               for index, slot in enumerate(self.slots)]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except BaseException:
                            self.cancel.set()
                            raise
        finally:
            keep = self.args.keep_server and self.all_ready and not self.cancel.is_set()
            if keep:
                for slot in self.slots:
                    print(f'[gmint] Service kept. Stop: MLP_JOBID={self.args.job_id} '
                          f'{self.project / "server-multi"} exit --service {slot.service}', flush=True)
            else:
                with ThreadPoolExecutor(max_workers=len(self.slots)) as pool:
                    for future in [pool.submit(self.stop_slot, slot) for slot in self.slots]:
                        try:
                            future.result()
                        except Exception as exc:
                            startup_errors.append(f'cleanup failed: {exc}')
                print('[gmint] Owned model services stopped.', flush=True)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            for index, result in enumerate(self.results):
                if result['status'] in ('pending', 'running'):
                    self.record(index, status='cancelled', detail='batch interrupted or service startup failed')
        for error in startup_errors:
            print(f'[gmint] {error}', file=sys.stderr)
        passed = sum(row['status'] == 'succeeded' for row in self.results)
        print(f'[gmint] Batch summary: {passed} succeeded, '
              f'{len(self.inputs) - passed} failed/cancelled. See {self.summary}', flush=True)
        if self.signal_number:
            return 128 + self.signal_number
        return int(bool(startup_errors) or passed != len(self.inputs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['fairchem', 'mace', 'orb'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', default='')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--service', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output')
    parser.add_argument('--hessian', choices=['numeric', 'analytic'], default='numeric')
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--keep-server', action='store_true')
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args(argv)
    try:
        if args.workers is not None and args.workers < 1:
            raise ValueError('--workers must be positive')
        if args.startup_timeout <= 0:
            raise ValueError('--startup-timeout must be positive')
        if args.output and len(args.inputs) > 1:
            raise ValueError('--output requires exactly one input')
        if not args.job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in args.job_id) or args.job_id in ('.', '..'):
            raise ValueError('Invalid job ID')
        if any(not Path(path).is_file() for path in args.inputs):
            raise ValueError('An input file does not exist')
        os.environ['MLP_JOBID'] = args.job_id
        return Batch(args).run()
    except (ValueError, OSError, ImportError) as exc:
        print(f'gmint-g16: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
