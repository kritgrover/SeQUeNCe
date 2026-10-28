# Path-oblivious reproduction

This folder ports the **implemented author simulations** at
[`vlmutolo/quantum-network@7c57be5`](https://github.com/vlmutolo/quantum-network/tree/7c57be549c275509b1674aaab3f7341c955041db/simulation-final)
to SeQUeNCe. Each original loop iteration runs as a real `Timeline` event through
`Event` and `Process`. The Bell-pair inventory abstraction is unchanged; one
simulator tick represents one author iteration, not a calibrated physical time.

## Files

- `topology.py`: generation graphs and weighted consumers.
- `model.py`: inventory dynamics and the original stopping rules.
- `sweeps.py`: original sweep values and CSV filenames.
- `run.py`: execution, CSV output, seeds, and input manifests.
- `plot.py`: the original graph quantities from new or archived CSV data.
- `test_parity.py`: direct comparisons with the pinned author functions.

## Setup

Run commands from the **SeQUeNCe repository root**. An environment with SeQUeNCe's
dependencies already installed can be used directly. The smaller tested set for
this example, including the dependencies imported by the actual Timeline, is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r example/path_oblivious/requirements.txt
```

## Run the original sweeps

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious.run --sweep distillation --seed 271
.\.venv\Scripts\python.exe -m example.path_oblivious.run --sweep nodes --seed 271
.\.venv\Scripts\python.exe -m example.path_oblivious.run --sweep nodes-grid --seed 271
```

Each command writes into `example/path_oblivious/results/<sweep>/`. Use an empty
`--output` directory for another run; existing results are never silently replaced.
There are no shortened defaults: each case retains the author's **5,000,000-step
ceiling and early-stopping rule**.

| Preset | Cases | Source settings |
|---|---:|---|
| `distillation` | 5 | `infinite-swapping.py`: 25-node grid, D=1..5 |
| `nodes` | 6 | `simulation-vary-nodes.py`: cycle, N=20,30,...,70, D=1 |
| `nodes-grid` | 24 | `simulation-vary-nodes_wrap_grid.py`: cycle/grid, N=20,30,...,70, D=1,2 |
| `finite-distillation` | 14 | Active sweep in `simulation-swap-rate.py`: cycle/grid, N=25, D=1..7 |
| `swap-rate` | 12 | Commented-out source sweep: cycle/grid, N=25, D=1, rates=150,200,...,400 |
| `distillation-both` | 10 | Repeat the original distillation settings for both plotted topologies |

All cases use initial capacity 20 and source generation-rate argument 40. The
distillation and finite profiles select 35 consumer pairs; node sweeps use
`floor(0.1*N*(N-1)/2)`. Source generation-batch sizes, consumer weights, ordering,
graph sharing, denominator recurrence, swap behavior, and stopping thresholds are
preserved. Scientific corrections are intentionally deferred.

The finite profiles make one execution accommodation: the source's undefined
lowercase `false` is interpreted as Python `False`. The source's ignored
`infinite_swapping_enabled` flag remains ignored. Removing GUI topology previews,
using local RNG objects, and writing results after each completed case do not
change simulation transitions.

List exact cases without running, or make an explicitly shortened check:

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious.run --sweep nodes-grid --list
.\.venv\Scripts\python.exe -m example.path_oblivious.run --sweep nodes-grid --max-steps 20 --seed 271 --output example/path_oblivious/results/smoke-nodes
```

`--case 0` selects a zero-based case; repeat the option for more cases. The default
runs the entire selected preset. `--max-steps` is recorded in the manifest, and a
short run is not a completed reproduction of the published curve.

## Plot new results

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious.plot --kind distillation --data example/path_oblivious/results/distillation --output example/path_oblivious/results/distillation/plots
.\.venv\Scripts\python.exe -m example.path_oblivious.plot --kind nodes --data example/path_oblivious/results/nodes-grid --output example/path_oblivious/results/nodes-grid/plots
```

Use `--kind swap-rate` for that sweep. Each command writes five PNGs: swap overhead
(linear and logarithmic), swaps per completed request, remaining Bell pairs, and
completed requests. The data columns and calculations match the author plotters;
node plots include both D curves when both files exist. Plot appearance is kept
simple; PNG encoding and fonts need not be byte-identical to the author's PDFs.

For the exact archived plotted points, read the tracked CSVs directly from the
local author clone without changing its checked-out branch:

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious.plot --kind distillation --reference-repo ../quantum-network --output example/path_oblivious/results/reference-distillation
.\.venv\Scripts\python.exe -m example.path_oblivious.plot --kind nodes --reference-repo ../quantum-network --output example/path_oblivious/results/reference-nodes
.\.venv\Scripts\python.exe -m example.path_oblivious.plot --kind swap-rate --reference-repo ../quantum-network --output example/path_oblivious/results/reference-swap-rate
```

These outputs are marked as **archived author data**, not newly simulated results.
Their distillation data has D=1..4, while the source driver's active sweep has
D=1..5. Both are preserved in their respective modes. These are the repository's
older experiments, not an invented reconstruction of the revised HotNets plots.

## Verify equivalence

```powershell
.\.venv\Scripts\python.exe -m pytest example/path_oblivious -q
```

The tests load original function bodies using `git show` at the pinned commit in
the sibling `quantum-network` clone (`AUTHOR_REPO` can select another clone).
They compare every observed iteration's aggregate metrics, inventory, queue state,
early stopping, and real SeQUeNCe event counters, plus graph construction and
sweep values. The original source is used as the oracle, including its behavior
that may be revised in a later stage.

The authors did not record RNG seeds for their published runs. New simulations
therefore reproduce their implementation and distributions; reproducing those
exact historical samples requires their stored CSVs. New runs record a master
seed, per-topology and per-case seeds, graph and consumer inputs, package versions,
executed iterations, and stopping reasons in `inputs.json` and `manifest.json`.
Keep the same Python/package versions for exact seeded trajectory comparisons.
