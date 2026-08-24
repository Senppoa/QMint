# QMint

[English](../README.md) | 中文

QMint 将机器学习原子间势接入 Gaussian 和 ORCA。它在本地运行 ASE 兼容模型，并通过命令行适配器提供能量、梯度和 Hessian 计算。

支持的模型后端：

- Fairchem / UMA
- MACE
- OrbMol-v2

## 安装

```bash
git clone https://github.com/Senppoa/QMint.git
cd QMint
python -m pip install -e .
```

根据使用的模型安装后端：

```bash
python -m pip install fairchem-core
python -m pip install mace-torch
python -m pip install "git+https://github.com/orbital-materials/orb-models.git"
```

OrbMol 解析 Hessian 还需要 [`orb-hessian`](https://github.com/Senppoa/orb-hessian)：

```bash
python -m pip install "git+https://github.com/Senppoa/orb-hessian.git"
```

## 配置模型

设置模型权重目录：

```bash
qmint config set model-dir /path/to/models
```

也可以设置环境变量 `MLP_MODEL_DIR`。如果模型文件位于其他目录，可将它注册到 QMint：

```bash
qmint model add my-mace /data/models/my.model \
  --backend mace --description "fine-tuned MACE"
```

查看模型并选择默认模型：

```bash
qmint models
qmint use mace-omol
```

首次打开 TUI 时，可以下载公开的 MACE-OMol、MACE-POLAR-M/L 和 OrbMol-v2 权重。UMA checkpoint 需要从 [Fairchem 仓库](https://github.com/facebookresearch/fairchem)下载后放入模型目录。

## 运行 QMint

打开终端界面：

```bash
qmint
```

TUI 可设置模型、worker 数、CPU 或 GPU、GPU ID、Hessian 模式和 debug 日志。退出时，它会停止本次启动的 worker。

脚本中可以直接使用 CLI：

```bash
qmint start --model mace-omol --gpu 0,1 --workers 2
qmint status
qmint stop
```

`--gpu` 使用全部可见 GPU，`--gpu 0,2` 指定设备；不传 `--gpu` 时使用 CPU。启动参数只覆盖本次运行的持久化设置：

```bash
qmint start --model orbmol-v2 --backend orb --gpu --hessian analytic
```

命令简写为：`-m` 对应 `--model`，`-b` 对应 `--backend`，`-n` 对应 `--workers`，`-g` 对应 `--gpu`，`-d` 对应 `--debug`。

## Gaussian

安装 QMint 后，`qmint-gaussian` 会自动加入当前 Python 环境。从该环境启动 Gaussian，并在 Route Section 中直接使用这个命令：

```text
# opt external='qmint-gaussian'
```

运行 Gaussian 前启动 QMint：

```bash
qmint use uma-s
qmint start --gpu
g16 molecule.gjf
qmint stop
```

QMint 按 Gaussian External 格式写回能量、梯度、电学属性占位值和下三角 Hessian。计算线程数由 `MLP_THREADS` 或 `OMP_NUM_THREADS` 设置。

## ORCA

安装 QMint 后，`qmint-orca` 和 `qmint-orca-hessian` 也会加入当前环境。ORCA 通过 `EXTOPTEXE` 环境变量查找外部优化器，因此需要在激活环境后、启动 ORCA 前设置一次：

```bash
conda activate your-environment
export EXTOPTEXE="$(command -v qmint-orca)"
```

该设置只对当前 shell 生效。每次打开新终端时需要重新设置；使用作业调度脚本时，也要把同一条 `export` 命令写进脚本。ORCA 输入文件只需要：

```text
! ExtOpt
```

需要 Hessian 接口时，改为：

```bash
export EXTOPTEXE="$(command -v qmint-orca-hessian)"
```

`qmint-orca-hessian` 会写入 `.engrad` 和 `.hess` 文件，也可以直接读取 XYZ 文件：

```bash
qmint-orca-hessian --xyz structure.xyz --charge 0 --mult 1 \
  --threads 4 --output structure.hess
```

所选计算器支持解析 Hessian 时，使用 `qmint start --hessian analytic`；其他情况使用默认的数值模式。

## 文件位置

| 数据 | 默认路径 |
| --- | --- |
| 配置 | `~/.config/qmint/config.json` |
| 模型 | `~/.local/share/qmint/models` |
| 运行状态 | `/tmp/qmint_<job-id>.json` |
| 日志 | `~/.local/state/qmint/server.log` |

`QMINT_CONFIG_HOME` 可修改配置目录，`MLP_MODEL_DIR` 可修改模型目录。QMint 只监听 `127.0.0.1`，并使用随机令牌保护本地会话。

## 开发

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m compileall -q qmint
```

内部设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 引用与许可

引用信息见 [CITATION.cff](../CITATION.cff)。QMint 使用 [MIT License](../LICENSE)，模型权重和后端包使用各自的许可证。
