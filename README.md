# QMint

English | [中文](docs/README_zh.md)

QMint (Quantum Machine-Learning Interface) is a local model router for quantum-chemistry software. It exposes ASE-compatible machine-learning potentials through one multi-worker service and lightweight adapters for Gaussian and ORCA. The server is independent of the client program, so integrations such as VASP can be added without changing model execution.

Author: Kun Tang · Version: 0.2.1 · License: [MIT](LICENSE)

## Highlights

- One multi-worker service replaces the former duplicate `server` and `server-multi` implementations.
- Running `qmint` with no subcommand opens a guided TUI. It configures model, workers, CPU/single-GPU/multi-GPU execution, GPU IDs, Hessian mode, and debug logging.
- On first initialization only, QMint offers optional downloads for MACE-OMol, MACE-POLAR-M/L, and OrbMol-v2. UMA is access-gated and must be downloaded manually.
- Persistent model selection from the terminal with `qmint models`, `qmint use`, and `qmint switch`.
- Fairchem/UMA, MACE, and OrbMol-v2 backends.
- Gaussian External, ORCA ExtOpt, and standalone ORCA Hessian adapters.
- A loopback-only authenticated protocol with a random token and `0600` state files.
- Numeric and backend-provided analytic Hessian paths.

## Installation

```bash
git clone https://github.com/Senppoa/QMint.git
cd QMint
python -m pip install -e .
```

Install only the backend required by the model environment:

```bash
python -m pip install fairchem-core
python -m pip install mace-torch
python -m pip install "git+https://github.com/orbital-materials/orb-models.git"
python -m pip install "git+https://github.com/Senppoa/orb-hessian.git"  # optional analytic OrbMol Hessian support
```

Fairchem, MACE, and OrbMol may require incompatible PyTorch/e3nn versions. Separate Conda environments are recommended. Place weights in `MLP_MODEL_DIR`, or configure a default directory:

```bash
qmint config set model-dir /path/to/models
```

## Model Switching

```bash
qmint models
qmint use uma-m
qmint switch mace-omol

qmint model add my-mace /data/models/my.model \
  --backend mace --description "fine-tuned MACE"
qmint use my-mace

qmint start --gpu 0,1 --workers 2
qmint status
qmint stop
```

Arguments passed to `qmint start` override persistent settings for that invocation:

```bash
qmint start -m orbmol-v2 -b orb -g --hessian analytic
```

Omit `--gpu` for CPU execution, use `--gpu` without a value for all visible GPUs, or pass a list such as `--gpu 0,2`. The historical `server start ...` and `server exit` commands remain available as compatibility entry points.

## Terminal UI

```bash
qmint        # default: open the TUI
qmint tui    # explicit equivalent
```

The first run shows a download checklist for models with public URLs. The setup screen displays an ASCII QMint logo, author `Kun Tang`, and the citation `Tang, K. (2026). QMint: Quantum Machine-Learning Interface.` See [CITATION.cff](CITATION.cff) for the machine-readable citation.

Use the guided fields to choose a model, worker count, CPU/single-GPU/multi-GPU mode, GPU IDs (for example `0,1` or `auto`), Hessian mode, and debug logging. These are the same parameters accepted by `qmint start`. Enter starts the service and `s` stops it. Exiting with `q`, Esc, `Ctrl-C`, or an error always stops all TUI-owned model workers so their CPU/GPU memory is released. Downloads are never retried automatically after the initial configuration; missing files are reported with their expected paths.

## Gaussian

Install QMint so that the `mlpint` console entry point is visible to Gaussian, then use it as an External program:

```text
# opt external='mlpint'
```

```bash
qmint use uma-s
qmint start --gpu
g16 molecule.gjf
qmint stop
```

The adapter converts Gaussian coordinates from Bohr to angstrom and writes energy, gradient, dummy electric properties, and the packed lower-triangular Hessian in Gaussian External format. Worker threads are selected from `MLP_THREADS`, then `OMP_NUM_THREADS`, and default to one.

## ORCA

Use `mlpint-orca` as an ORCA ExtOpt program for energy and gradients:

```text
! ExtOpt
%method
  ProgExt "/absolute/path/to/mlpint-orca"
end
```

`mlpint-orca-hessian` writes both `.engrad` and `.hess` files and also supports standalone execution:

```bash
mlpint-orca-hessian --xyz structure.xyz --charge 0 --mult 1 \
  --threads 4 --output structure.hess
```

Analytic Hessians require calculator support. OrbMol-v2 additionally requires Kun Tang's [`orb-hessian`](https://github.com/Senppoa/orb-hessian) patch and a server started with `--hessian analytic`.

## Configuration

| Setting | Default |
| --- | --- |
| Configuration | `~/.config/qmint/config.json` (`QMINT_CONFIG_HOME` overrides it) |
| Model directory | `~/.local/share/qmint/models` (`MLP_MODEL_DIR` takes precedence) |
| Server state | `/tmp/qmint_<job-id>.json` |
| Server log | `~/.local/state/qmint/server.log` |
| Threads | `MLP_THREADS` > `OMP_NUM_THREADS` > `1` |

The server listens only on `127.0.0.1`. Do not commit model weights or runtime state files.

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m compileall -q qmint
```

The fast test suite covers configuration and model registration, authenticated framing, Gaussian/ORCA file handling, unit conversion, and ASE task execution. Real backend tests require the corresponding weights and environment.

## Repository Layout

```text
qmint/
  calculator.py       Backend loading, ASE tasks, and Hessians
  cli.py              QMint CLI and compatibility commands
  config.py           Persistent user configuration
  models.py           Built-in and custom model registry
  protocol.py         Authenticated local socket protocol
  server.py           The single multi-worker service
  tui.py              Curses terminal UI
  interfaces/         Gaussian and ORCA adapters; extension point for VASP
tests/                Fast regression tests and calculation inputs
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component boundaries.

## Model Downloads

The first-run TUI offers these public model URLs:

| Model | URL |
| --- | --- |
| MACE-OMol extra-large | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model) |
| MACE-POLAR-M | [direct download](https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model) |
| MACE-POLAR-L | [direct download](https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-L.model) |
| OrbMol-v2 | [Orbital Materials public bucket](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orbmol-v2-teqabfhg-20260523.ckpt) |

UMA checkpoints are access-gated. Follow the [Fairchem UMA documentation](https://github.com/facebookresearch/fairchem) to download them manually, then place the files in the configured model directory.

## License

QMint is released under the [MIT License](LICENSE). Model weights, calculator backends, and optional patches retain their respective third-party licenses.
