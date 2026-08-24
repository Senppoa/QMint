# QMint

English | [中文](docs/README_zh.md)

QMint connects machine-learning interatomic potentials to Gaussian and ORCA. It runs an ASE-compatible model as a local service and provides command-line adapters for energy, gradient, and Hessian calculations.

Supported model backends:

- Fairchem / UMA
- MACE
- OrbMol-v2

## Install

```bash
git clone https://github.com/Senppoa/QMint.git
cd QMint
python -m pip install -e .
```

Install the package required by your model:

```bash
python -m pip install fairchem-core
python -m pip install mace-torch
python -m pip install "git+https://github.com/orbital-materials/orb-models.git"
```

OrbMol analytic Hessians also require [`orb-hessian`](https://github.com/Senppoa/orb-hessian):

```bash
python -m pip install "git+https://github.com/Senppoa/orb-hessian.git"
```

## Configure models

Set the directory that contains model weights:

```bash
qmint config set model-dir /path/to/models
```

You can also set `MLP_MODEL_DIR`. To register a model stored elsewhere:

```bash
qmint model add my-mace /data/models/my.model \
  --backend mace --description "fine-tuned MACE"
```

List available models and select the default:

```bash
qmint models
qmint use mace-omol
```

The first TUI run can download the public MACE-OMol, MACE-POLAR-M/L, and OrbMol-v2 weights. Download UMA checkpoints from the [Fairchem repository](https://github.com/facebookresearch/fairchem) and place them in the model directory.

## Run QMint

Open the terminal interface:

```bash
qmint
```

The TUI controls the model, worker count, CPU or GPU execution, GPU IDs, Hessian mode, and debug logging. It stops the workers it started when you exit.

For scripts, use the CLI directly:

```bash
qmint start --model mace-omol --gpu 0,1 --workers 2
qmint status
qmint stop
```

`--gpu` uses all visible GPUs, while `--gpu 0,2` selects specific devices. Omit it for CPU execution. Command-line options override saved settings for the current start:

```bash
qmint start --model orbmol-v2 --backend orb --gpu --hessian analytic
```

The short options are `-m` for `--model`, `-b` for `--backend`, `-n` for `--workers`, `-g` for `--gpu`, and `-d` for `--debug`.

## Gaussian

Use `qmint-gaussian` as the Gaussian External program:

```text
# opt external='qmint-gaussian'
```

Start QMint before running Gaussian:

```bash
qmint use uma-s
qmint start --gpu
g16 molecule.gjf
qmint stop
```

QMint writes the energy, gradient, electric-property placeholders, and packed lower-triangular Hessian in Gaussian External format. Set calculator threads with `MLP_THREADS` or `OMP_NUM_THREADS`.

## ORCA

Use `qmint-orca` for ORCA ExtOpt energy and gradient calculations:

```text
! ExtOpt
%method
  ProgExt "/absolute/path/to/qmint-orca"
end
```

`qmint-orca-hessian` writes `.engrad` and `.hess` files. It can also run on an XYZ file:

```bash
qmint-orca-hessian --xyz structure.xyz --charge 0 --mult 1 \
  --threads 4 --output structure.hess
```

Use `qmint start --hessian analytic` when the selected calculator provides analytic Hessians. Otherwise, use the default numeric mode.

## Paths

| Data | Default path |
| --- | --- |
| Configuration | `~/.config/qmint/config.json` |
| Models | `~/.local/share/qmint/models` |
| Runtime state | `/tmp/qmint_<job-id>.json` |
| Log | `~/.local/state/qmint/server.log` |

`QMINT_CONFIG_HOME` changes the configuration directory. `MLP_MODEL_DIR` changes the model directory. QMint listens on `127.0.0.1` and protects each local session with a random token.

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m compileall -q qmint
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the internal design.

## Citation and license

Citation metadata is available in [CITATION.cff](CITATION.cff). QMint is released under the [MIT License](LICENSE). Model weights and backend packages keep their own licenses.
