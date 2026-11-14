# 多 GPU 批量 Gaussian 计算

先激活模型所需的环境，然后在输入文件所在目录运行：

```bash
conda activate fairchem
cd /你的计算目录

/home/asus/program/gmintQAQ/1/gmint-g16 \
  -b fairchem \
  -m /你的模型目录/uma-s-1p1.pt \
  -g 0,1 \
  --all
```

`--all` 按自然文件名顺序扫描当前目录的 `.gjf` 和 `.com` 文件，不递归子目录。
GPU 0 先处理第一个文件，GPU 1 先处理第二个文件；哪张卡先完成，就领取下一个
尚未开始的文件。每张卡同一时刻只运行一个 Gaussian 文件，并始终连接自己的
模型服务。模型在该卡上只加载一次，整批结束后统一停止服务。

也可以选择全部可见 GPU，或明确列出输入文件：

```bash
gmint-g16 -b fairchem -m /data/models/uma.pt -g auto --all
gmint-g16 -b fairchem -m /data/models/uma.pt -g 0,2 1.gjf 2.gjf 3.gjf
```

使用 MACE 时激活 MACE 环境，替换后端和模型路径：

```bash
conda activate mace
gmint-g16 -b mace -m /data/models/MACE.model -g 0,1 --all
```

## 参数和设备编号

- 默认并发数等于选中 GPU 的数量；不需要额外指定 `-n`。
- `-n` / `--workers` 在这个批处理入口中表示最多同时运行的文件数。例如
  `-g 0,1,2,3 -n 2` 只使用前两张卡。不能超过选中卡的数量。
- `server-multi` 独立命令中的 `-n` 仍表示它自己的 worker 数量。
- 只指定一张 GPU 时，多个文件依次计算；不指定 `-g` 默认使用一个 CPU 服务。
- `-g` 使用当前环境可见的逻辑编号。例如 `CUDA_VISIBLE_DEVICES=2,3` 时，
  `-g 0,1` 对应原来的设备 2、3。脚本保留这个映射，也支持 GPU UUID 标识。
- 每张卡都需要容纳一份完整模型；单个文件的 Hessian 不会拆分到其他卡。

## 模型位置

`gmint-g16` 第 2 行的 `GMINT_MODEL_DIR` 仍可调整。留空时使用已有的
`MLP_MODEL_DIR`；`-m` 传绝对路径时直接加载该文件。都未指定目录且使用别名时，
服务到代码目录查找。`s` 别名仍指向 `uma-s-1p1.pt`，其他版本请传实际文件路径。

## 输入文件

Gaussian 的路由段应包含 `External="mlpint"`，或指向同版本 `mlpint` 的绝对路径。
不要在 External 命令里写死 `--service`；调度器会给每张卡设置独立的
`MLP_SERVICE`。不用手动设置 JOBID，默认会为本批自动生成。

并发文件的 `%chk` 必须使用不同路径；脚本会拒绝多个文件共同写同一个 checkpoint。
其他用户指定的写入文件（例如 `%rwf`）也应独立。各个 Gaussian 作业的
`GAUSS_SCRDIR` 由调度器自动隔离，并在该文件结束后清理。
每个输入的 `%mem` 和 `%nprocshared` 是单个 Gaussian 进程的资源配置，并发时会累加。

## 日志、结果和停止

每个输入的 Gaussian 输出写入同目录同名 `.log`。详细运行记录位于第一个输入目录：

```text
.gmint/<本批JOBID>/
├── summary.tsv
├── gpu-0/server.log
└── gpu-1/server.log
```

`summary.tsv` 记录每个输入的 GPU、CUDA 设备标识、服务名、结果、退出码和日志路径。
成功要求 Gaussian 退出码为 0，并且日志末尾有 `Normal termination of Gaussian`。
单个计算失败会继续处理其余文件；只要有失败或取消，整批返回非零退出码。

使用 `Ctrl+C` 或向调度器发送 `SIGTERM`，会停止本批的 Gaussian 进程、子进程和模型服务。
模型加载阶段取消也会清理尚未发布端口的服务。不要使用 `kill -9`，因为任何程序都无法
捕获该信号进行清理。

`--keep-server` 可以在正常完成调度后保留服务，控制台会打印各服务的停止命令。
模型日志会显示实际加载的权重路径和 CUDA 映射。

## 迁移和验证

同步整个更新后的代码目录，至少包含：

```text
gmint-g16  gmint_batch.py  server-multi  mlpint  utils.py
```

然后执行：

```bash
chmod +x gmint-g16 server-multi mlpint
bash -n gmint-g16
python -m unittest discover -s tests -v
```

集成测试使用真实子进程和模拟计算器，验证并发、GPU 路由、动态队列、失败继续、
信号取消和服务清理，不需要多张物理 GPU。可选的真实 UMA/Gaussian 冒烟测试：

```bash
# 先激活 fairchem 并加载 Gaussian 环境
GMINT_REAL_MODEL=/模型绝对路径/uma.pt \
  python -m unittest discover -s tests -p test_batch.py -v
```

该测试在 GPU 0 上运行两个水分子的力计算，并检查服务复用和退出清理。

本机已用 `fairchem-core 2.12.0`、`uma-s-1p1.pt` 和 Gaussian 16 通过这项真实测试，
完整测试共 21 项通过。双 GPU 的分卡并发由子进程集成测试验证；本机只有一张
物理 GPU，尚未进行双物理 GPU 实测。当前目录的 `uma-s-1p2p1.pt` 在本机现有
fairchem 环境中加载会报 `HydraModel.__init__() got an unexpected keyword argument
'model_id'`，需要使用与该权重匹配的后端版本；调度器不会修改虚拟环境。
