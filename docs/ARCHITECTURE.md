# QMint architecture

QMint has three stable layers:

1. `qmint.server` owns the local listener, worker lifecycle, GPU assignment and graceful shutdown. It is the only server implementation.
2. `qmint.calculator` translates a framed task into an ASE `Atoms` object, loads one optional backend per worker, and returns energy/gradient/Hessian in atomic units.
3. `qmint.interfaces` contains thin adapters for external programs. Gaussian and ORCA know their own file formats but never import Fairchem/MACE/OrbMol directly.

The server state file contains a random one-process token and is mode `0600`. Adapters read the token and send a length-framed request over loopback. This prevents accidental cross-job requests while keeping the protocol independent of the model backend.

To add VASP, implement a module under `qmint/interfaces/` that converts VASP's calculator callback or socket contract into the existing task dictionary. The model registry and server do not need to change.

Running `qmint` without a subcommand opens the curses TUI. On first configuration only, the TUI offers verified public model downloads; subsequent sessions never retry them automatically. Both the TUI and CLI call the same `qmint.server.start` function with model, worker, GPU, Hessian and debug settings. Workers started by the TUI are stopped in a `finally` block when the interface exits so model memory is released.
