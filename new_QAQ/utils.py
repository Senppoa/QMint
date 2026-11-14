#!/usr/bin/env python3
"""客户端与计算服务端共用的基础工具。

本模块负责三类事情：
1. 用作业调度器的 Job ID 隔离不同任务所使用的端口文件；
2. 定义基于 TCP 的“4 字节长度头 + pickle 负载”通信协议；
3. 提供服务守护进程的启动、退出等待和兜底清理逻辑。

注意：pickle 只能用于可信进程之间的通信。反序列化不可信数据可能执行任意代码，
因此本项目只让服务监听 localhost，不能直接把这个协议暴露到公网。
"""

import os
import pickle
import struct
import time
import socket
import json
import sys
import hashlib
import tempfile
from typing import Optional


DEFAULT_SERVICE = 'default'
SERVICE_ENV = 'MLP_SERVICE'
RUNTIME_DIR_ENV = 'MLP_RUNTIME_DIR'
MAX_MESSAGE_SIZE = 512 * 1024 * 1024


def cuda_device_token(index, environ=None):
    """把当前进程的逻辑 GPU 编号映射回 CUDA_VISIBLE_DEVICES 中的设备标识。"""
    env = os.environ if environ is None else environ
    index = int(index)
    if index < 0:
        raise ValueError('GPU index must be non-negative')
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(index)
    devices = [item.strip() for item in visible.split(',') if item.strip()]
    if index >= len(devices) or devices[index] == '-1':
        raise ValueError(f'GPU {index} is outside CUDA_VISIBLE_DEVICES={visible!r}')
    return devices[index]


def get_service_name(service=None):
    """返回当前服务实例名。

    实例名用于在同一个调度器作业中区分由不同 Python 虚拟环境启动的服务。
    未显式传入时读取 ``MLP_SERVICE``，最后退化为 ``default``。
    """
    value = service if service is not None else os.environ.get(SERVICE_ENV, DEFAULT_SERVICE)
    value = str(value).strip()
    if not value:
        return DEFAULT_SERVICE
    if len(value) > 128:
        raise ValueError('Service name must not exceed 128 characters')
    return value


def _safe_file_component(value):
    """把作业/服务标识转换为不可逃逸运行目录的稳定文件名片段。"""
    value = str(value)
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in value)
    while '..' in safe:
        safe = safe.replace('..', '__')
    if not safe:
        safe = 'unknown'
    if safe != value or len(safe) > 80:
        digest = hashlib.sha256(value.encode('utf-8')).hexdigest()[:10]
        safe = f'{safe[:64]}_{digest}'
    return safe


def _recv_exact(conn, size):
    """从 socket 精确读取 size 字节；对端提前关闭时抛出清晰异常。"""
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            received = size - remaining
            raise ConnectionError(
                f'Connection closed after {received} of {size} expected bytes'
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def get_job_id():
    """返回当前批处理作业的唯一标识，供端口文件命名使用。

    按常见调度器环境变量的优先级查找；本地运行时则退化为
    ``<用户名>_default``。这样同一节点上的不同作业通常不会误连彼此的服务。
    """
    for var in ['MLP_JOBID', 'PBS_JOBID', 'LSB_JOBID', 'SLURM_JOB_ID', 'PJM_JOBID']:
        jobid = os.environ.get(var)
        if jobid:
            return jobid
    user = (os.environ.get('USER') or os.environ.get('LOGNAME') or
            os.environ.get('USERNAME') or 'unknown')
    return f'{user}_default'


def send_pickle(conn, obj):
    """把一个 Python 对象完整写入已连接的 socket。

    TCP 是字节流，没有消息边界，所以先发送网络字节序的 4 字节无符号长度，
    再发送 pickle 数据。接收方据此知道一条消息何时结束。
    """
    data = pickle.dumps(obj)
    if len(data) > MAX_MESSAGE_SIZE:
        raise ValueError(
            f'Pickle payload is {len(data)} bytes; limit is {MAX_MESSAGE_SIZE}'
        )
    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)


def recv_pickle(conn):
    """从 socket 接收一条长度前缀消息并反序列化。

    ``recv`` 可能只返回部分数据，因此循环读取，直到累计到消息头声明的长度。
    对端在消息头之前关闭连接时返回 ``None``。
    """
    try:
        raw_len = _recv_exact(conn, 4)
    except ConnectionError as exc:
        if 'after 0 of 4' in str(exc):
            return None
        raise
    msg_len = struct.unpack("!I", raw_len)[0]
    if msg_len > MAX_MESSAGE_SIZE:
        raise ValueError(
            f'Refusing pickle payload of {msg_len} bytes; limit is {MAX_MESSAGE_SIZE}'
        )
    data = _recv_exact(conn, msg_len)
    return pickle.loads(data)


def get_port_file(service=None):
    """得到当前作业、当前命名服务实例对应的服务发现文件路径。

    ``default`` 实例沿用旧文件名以保持兼容；命名实例增加服务名后缀，因此同一
    作业可以同时运行来自多个虚拟环境的 MACE/UMA/Orb 服务。
    """
    job_id = _safe_file_component(get_job_id())
    service_name = get_service_name(service)
    runtime_dir = os.environ.get(RUNTIME_DIR_ENV) or tempfile.gettempdir()
    if service_name == DEFAULT_SERVICE:
        filename = f'mlps_port_{job_id}.json'
    else:
        filename = f'mlps_port_{job_id}_{_safe_file_component(service_name)}.json'
    return os.path.join(runtime_dir, filename)


def get_service_log_file(prefix, service=None, directory=None):
    """生成不会因服务名产生目录穿越的实例专属日志路径。"""
    service_name = get_service_name(service)
    suffix = '' if service_name == DEFAULT_SERVICE else f'_{_safe_file_component(service_name)}'
    return os.path.join(directory or os.getcwd(), f'{prefix}{suffix}.log')


def _load_port_record(service=None):
    """读取并校验服务发现记录。"""
    port_file = get_port_file(service)
    with open(port_file, encoding='utf-8') as f:
        record = json.load(f)
    port = record.get('port')
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f'Invalid port record in {port_file}: {port!r}')
    return record


def get_service_record(service=None):
    """返回命名服务的发现元数据副本。"""
    return dict(_load_port_record(service))


def get_port_number(service=None):
    """读取服务监听端口；端口文件不存在时打印提示并退出。"""
    port_file = get_port_file(service)
    if not os.path.isfile(port_file):
        print(f'Port file is not found.: {port_file}')
        sys.exit(1)
    return _load_port_record(service)['port']


def get_server_pid(service=None):
    """读取端口文件中记录的守护进程 PID；文件无效或字段缺失时返回 None。"""
    port_file = get_port_file(service)
    if not os.path.isfile(port_file):
        return None
    try:
        return _load_port_record(service).get('pid')
    except Exception:
        return None


def is_pid_alive(pid):
    """判断 PID 是否仍存在；无权限发送信号意味着进程存在。"""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def probe_service(service=None, timeout=1.0):
    """检查命名服务的监听端口是否可连接。

    不等待 PING 响应，因为单 worker 服务在长时间计算时无法立即消费控制消息，
    但监听端口仍能可靠表明实例存在。
    """
    try:
        port = _load_port_record(service)['port']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(('localhost', port))
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def prepare_service_start(service=None):
    """在启动前阻止覆盖活跃实例，并清理同名实例的陈旧发现文件。"""
    port_file = get_port_file(service)
    if not os.path.exists(port_file):
        return
    if probe_service(service):
        name = get_service_name(service)
        raise RuntimeError(
            f"Service {name!r} is already running; stop it first or choose another --service"
        )
    remove_port_file(service)


def save_port_file(port, pid=None, service=None, metadata=None):
    """发布服务端口和守护进程 PID，表示服务已经可以接受任务。"""
    port_file = get_port_file(service)
    if pid is None:
        pid = os.getpid()
    record = dict(metadata or {})
    record.update({
        'port': port,
        'pid': pid,
        'service': get_service_name(service),
        'python': sys.executable,
    })
    os.makedirs(os.path.dirname(port_file), exist_ok=True)
    temporary = f'{port_file}.{os.getpid()}.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, port_file)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def wait_for_server_down(port, timeout=30.0):
    """轮询等待服务不再接受连接。

    在超时前端口关闭则返回 True，否则返回 False。守护进程会在回收 worker 后
    关闭监听 socket，因此拒绝连接通常意味着 worker 已退出、GPU 显存也已释放。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(('localhost', port))
            # 仍能建立连接，说明监听 socket 尚未关闭。
            time.sleep(0.5)
        except (ConnectionRefusedError, socket.timeout, OSError):
            return True
    return False


def force_kill_server(pid):
    """向守护进程所在的整个进程组发送 SIGKILL（包括所有 worker）。

    这是正常退出超时后的最后兜底手段，仅适用于支持 POSIX 进程组的系统。
    """
    import signal
    if not pid:
        return False
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def wait_for_pid_gone(pid, timeout=10.0):
    """等待 PID 消失；进程已被回收返回 True，超时返回 False。"""
    if not pid:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.2)
    return False


def remove_port_file(service=None):
    """尽力删除服务发现文件；清理失败不影响退出流程。"""
    port_file = get_port_file(service)
    try:
        os.remove(port_file)
    except OSError:
        pass


def send_task(task, service=None, timeout=300):
    """等待服务就绪，发送一个计算任务，并同步返回计算结果。

    服务模型加载可能耗时较长，因此最多等待端口文件 300 秒。端口发布后，
    每次调用新建一个 localhost TCP 连接，完成一次请求/响应后自动关闭。
    """
    if timeout <= 0:
        raise ValueError('timeout must be greater than zero')
    port_file = get_port_file(service)

    # 端口文件只有在所有 worker 完成模型加载后才会生成，所以它也是“就绪标志”。
    interval = 0.5
    deadline = time.monotonic() + timeout
    last_error = None
    connected_socket = None
    while time.monotonic() < deadline:
        if not os.path.isfile(port_file):
            time.sleep(interval)
            continue
        candidate = None
        try:
            record = _load_port_record(service)
            port = record['port']
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate.settimeout(min(5.0, max(0.1, deadline - time.monotonic())))
            candidate.connect(('localhost', port))
            connected_socket = candidate
            break
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if candidate is not None:
                candidate.close()
            last_error = exc
            try:
                stale_pid = _load_port_record(service).get('pid')
            except Exception:
                stale_pid = None
            if stale_pid is not None and not is_pid_alive(stale_pid):
                remove_port_file(service)
            time.sleep(interval)

    if connected_socket is not None:
        with connected_socket as s:
            # 推理/Hessian 可能远超服务发现超时，连接成功后不限制计算时间。
            s.settimeout(None)
            send_pickle(s, task)
            result = recv_pickle(s)
            if result is None:
                raise ConnectionError('Service closed the connection without a response')
            return result

    detail = f' Last error: {last_error}' if last_error else ''
    raise TimeoutError(
        f'Timeout waiting {timeout:g}s for service {get_service_name(service)!r} '
        f'using {port_file}.{detail}'
    )


def read_int_list_string(list_string):
    """把 ``0,2-4;6`` 一类写法展开为整数列表 ``[0, 2, 3, 4, 6]``。"""
    blocks = [x.strip() for x in list_string.strip().replace(',', ' ').replace(';', ' ').replace(':', ' ').split()]
    index = []
    for block in blocks:
        if '-' in block:
            start, end = [int(x) for x in block.split('-')]
            for i in range(start, end + 1):
                index.append(i)
        else:
            index.append(int(block))
    return index


def get_device_list(num_workers, gpu_number_list):
    """按 worker 顺序分配指定 GPU；GPU 不够时，多出的 worker 使用 CPU。"""
    device_list = []
    for i in range(num_workers):
        if i < len(gpu_number_list):
            device_list.append(f'cuda:{gpu_number_list[i]}')
        else:
            device_list.append('cpu')
    return device_list


def daemonize(error_log_file: Optional[str] = None):
    """通过经典的两次 fork 把当前进程转为 POSIX 守护进程。

    第一次 fork 后调用 ``setsid`` 脱离终端并创建新会话；第二次 fork 防止进程
    再次取得控制终端。最后把标准输入/输出指向 ``/dev/null``，标准错误按参数
    指向日志或 ``/dev/null``。该实现面向 Linux/WSL/HPC，原生 Windows 无 ``fork``。
    """
    # 父进程立即退出，让子进程在后台继续执行。
    if os.fork() > 0:
        sys.exit(0)

    os.setsid()  # 新建会话与进程组，便于退出时整体终止 daemon + workers。

    # 第二次 fork 后，最终子进程不再是 session leader。
    if os.fork() > 0:
        sys.exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    if error_log_file is None:
        error_redirect = os.devnull
    else:
        error_redirect = error_log_file

    # 替换三个标准文件描述符，避免后台进程继续占用启动它的终端。
    with open(os.devnull, 'r') as devnull_in, \
         open(os.devnull, 'a') as devnull_out, \
         open(error_redirect, 'a') as errfile:
        os.dup2(devnull_in.fileno(), sys.stdin.fileno())
        os.dup2(devnull_out.fileno(), sys.stdout.fileno())
        os.dup2(errfile.fileno(), sys.stderr.fileno()) 
