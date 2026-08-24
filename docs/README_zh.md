# QMint

[English](../README.md) | 中文

QMint（Quantum Machine-Learning Interface）是一个面向量子化学程序的本地模型路由器：它把 ASE 兼容的机器学习势函数包装为一个多 worker 服务，并通过轻量适配器向 Gaussian 和 ORCA 提供能量、梯度及 Hessian。服务核心与程序适配器解耦，后续可以在 `qmint/interfaces/` 中增加 VASP 等接口。

作者：Kun Tang  · 版本：0.2.1  · 协议：[MIT](../LICENSE)

## 主要能力

- 单一多 worker 服务：`server` 是兼容别名，不再维护重复的 `server-multi` 源码。
- 直接运行 `qmint` 会进入引导式 TUI，可配置模型、worker 数、CPU/单卡/多卡、GPU ID、Hessian 和 debug。
- 首次初始化时才询问是否下载 MACE-OMol、MACE-POLAR-M/L 和 OrbMol-v2；UMA 受权限限制，需要手动下载。
- 终端模型切换：`qmint models`、`qmint use` / `qmint switch`、`qmint start`。
- 无第三方 TUI 依赖：界面显示 QMint 字符画、作者 Kun Tang 和引用方式。
- 后端：Fairchem/UMA、MACE、OrbMol-v2；每个后端仍建议使用隔离的 Conda 环境。
- Gaussian External、ORCA ExtOpt、ORCA Hessian standalone 接口。
- 本地服务状态文件权限为 `0600`，每次启动生成随机会话令牌，避免同机其他进程伪造任务。

## 安装

```bash
git clone https://github.com/Senppoa/QMint.git
cd QMint
python -m pip install -e .
```

基础安装提供协议、CLI 和适配器；实际推理后端按需安装：

```bash
python -m pip install fairchem-core
python -m pip install mace-torch
python -m pip install "git+https://github.com/orbital-materials/orb-models.git"
python -m pip install "git+https://github.com/Senppoa/orb-hessian.git"
```

三种后端的 PyTorch/e3nn 依赖可能冲突，请分别创建环境。模型文件放到 `MLP_MODEL_DIR`，或通过 `qmint config set model-dir /path/to/models` 指定目录。

## 从终端切换模型

```bash
qmint models
qmint use uma-m
qmint switch mace-omol
qmint model add my-mace /data/models/my.model --backend mace --description "fine-tuned MACE"
qmint use my-mace
qmint start --gpu 0,1 --workers 2
qmint status
qmint stop
```

`qmint start` 的模型、worker、GPU 和 Hessian 参数都可以临时覆盖持久化配置：

```bash
qmint start -m orbmol-v2 -b orb -g --hessian analytic
```

GPU 参数：不写 `--gpu` 表示 CPU；`--gpu` 自动使用所有可见 GPU；`--gpu 0,2` 使用指定卡。`server start ...` 与 `server exit` 仍可用于旧脚本。

## TUI

```bash
qmint        # 默认进入 TUI
qmint tui    # 显式调用，效果相同
```

TUI 中可逐项配置与 `qmint start` 一致的模型、worker 数、CPU/单卡/多卡、GPU ID、Hessian 模式和 debug 参数。Enter 启动服务，`s` 停止服务。通过 `q`、Esc、`Ctrl-C` 或异常退出时，TUI 都会停止所有模型 worker，卸载模型并释放 CPU/GPU 显存。

## Gaussian

将 `mlpint` 配置为 Gaussian External 脚本，在 Route Section 使用：

```text
# opt external='mlpint'
```

先在对应环境启动 QMint：

```bash
qmint use uma-s
qmint start --gpu
g16 molecule.gjf
qmint stop
```

Gaussian 传入的 Bohr 坐标会由适配器转换为 ASE 使用的 Å；梯度和下三角 Hessian 按 Gaussian External 格式写回。线程数读取 `MLP_THREADS`，其次读取 `OMP_NUM_THREADS`。

## ORCA

`mlpint-orca` 用于 ORCA `ExtOpt` 能量+梯度：

```text
! ExtOpt
%method
  ProgExt "/absolute/path/to/mlpint-orca"
end
```

`mlpint-orca-hessian` 同时写 `.engrad` 和 `.hess`，也支持独立计算：

```bash
mlpint-orca-hessian --xyz structure.xyz --charge 0 --mult 1 --threads 4 -o structure.hess
```

解析 Hessian 需要后端支持；OrbMol-v2 还需要 Kun Tang 的 [`orb-hessian`](https://github.com/Senppoa/orb-hessian) 补丁，并用 `qmint start --hessian analytic` 启动。

## 配置与环境变量

| 项目 | 默认值 |
| --- | --- |
| 配置文件 | `~/.config/qmint/config.json`（可用 `QMINT_CONFIG_HOME` 覆盖） |
| 模型目录 | `~/.local/share/qmint/models`（`MLP_MODEL_DIR` 优先） |
| 服务状态 | `/tmp/qmint_<job-id>.json` |
| 日志 | `~/.local/state/qmint/server.log` |
| 线程 | `MLP_THREADS` > `OMP_NUM_THREADS` > `1` |

服务只监听 `127.0.0.1`。不要把状态文件或模型权重提交到版本库。

## 开发与验证

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m compileall -q qmint
```

当前测试覆盖配置/模型注册、认证帧协议、Gaussian/ORCA 文件格式和 ASE 计算任务。真实 Fairchem/MACE/OrbMol 推理需要对应权重和后端环境。

## 项目结构

```text
qmint/
  calculator.py       ASE 任务、后端加载、Hessian
  cli.py              qmint/server 兼容命令与模型切换
  config.py           持久化用户配置
  models.py           内置/自定义模型注册表
  protocol.py         本地认证 Socket 协议
  server.py           唯一多 worker 服务实现
  tui.py              curses TUI
  interfaces/         Gaussian、ORCA 适配器（未来可扩展 VASP）
tests/                无模型权重的快速回归测试
```

## 许可证

QMint 按 MIT License 开源，详见 [LICENSE](../LICENSE)。第三方模型、后端和补丁包分别受其自身许可证约束。
