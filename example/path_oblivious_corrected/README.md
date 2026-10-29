# Corrected path-oblivious simulations

This folder contains a separate corrected implementation and comparison runners.
The original `example/path_oblivious` folder remains the author-code baseline.
Both implementations use the real SeQUeNCe event kernel. The algorithm is a
**local balancing heuristic**; local termination does not establish global
max-min optimality, and a plateau in a cumulative ratio does not establish
steady state. These workloads do not claim exact final HotNets figure reproduction.

Run commands from `E:\thesis\SeQUeNCe` using the existing environment:

```powershell
Set-Location E:\thesis\SeQUeNCe
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.compare --help
```

## Corrections and model convention

A corrected node visit performs at most one currently preferable swap, selecting
the smallest output-buffer count. It requires
`C_yz + 1 <= min(C_xy - D, C_xz - D)` at action time. Ties retain the first
minimum in the original neighbor-set combination order. Later visits recompute
candidates; shuffled node passes and swapping to quiescence remain supported.

`D` is an input-pair resource multiplier. A swap consumes `D` raw pairs from each
input and creates one raw output; completing a request consumes `D` raw pairs.
The implementation checks nonnegative inventory and
`P = P_initial + generated - (2*D - 1)*swaps - D*completed` during execution.
Unlimited inventory, perfect swaps, no decoherence, instantaneous buffer knowledge,
and blocking global FIFO consumption remain model assumptions.

Every trajectory produces both overhead definitions:

- Legacy: `swaps / sum(f[hops-1])`, retaining the original recurrence and indexing.
- Corrected: `swaps / sum(s_D(hops))`, where `s_D(1)=0` and
  `s_D(h)=D*(1+s_D(h//2)+s_D((h+1)//2))` for `h>1`.

Tables are sized to the needed hop lengths. Completed-request hop histograms
permit both denominators to be recomputed. Zero denominators are JSON `null`
and blank CSV fields, including positive-swaps/zero-denominator cases.
Metrics are cumulative from initialization, with no discarded warm-up; swaps
invested in remaining inventory stay in the numerator.

## Controlled comparisons

This pilot command runs both dynamics and calculates both metrics on each run:

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.compare --steps 1000 --distillations 1 2 3 --replicates 5 --seed 271 --output ../tmp/po-comparison
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.plot --data ../tmp/po-comparison
```

Use an empty output directory. Choose the main evaluation horizon, replication
count, and completion threshold after pilots, then hold them fixed. A 1,000-step
pilot is not evidence of adequate precision or stationarity.

Within each replicate, both algorithms and every requested D share the actual
graph, consumers, weights, generation inputs, and FIFO request sequence. Generation
and finite-event arrays are generated before running either algorithm. Requests
are cached by FIFO index, and node permutations by `(iteration, pass)`: additional
passes or requests cannot shift unrelated inputs. All realized input values are
saved. This deliberate stream reorganization differs from literal author sampling;
matching does not rely on giving the algorithms the same integer seed.

The default workload is a 25-node 5x5 wrapped-grid overlay, 35 consumer pairs,
weights sampled uniformly from integers 1 through 6, and Poisson generation batches
with mean 10 independent of D. The grid constructor shuffles wrapped-grid edges
and adds them only until connected; it does not create a full torus. Supported
choices include:

- `--graph cycle --nodes 25`, or `--graph wrap_grid_overlay --nodes 30 --rows 5 --columns 6`.
  Explicit grid dimensions must multiply to the node count; dimensions are saved.
- `--consumer-policy fixed-35` (default), `--consumer-policy fraction-10-percent`
  for the labeled historical `floor(0.1*N*(N-1)/2)` count, or `--consumers 20`
  for a labeled custom count.
- `--batch-mean 10`, held constant across D.
- `--minimum-completed 100` and `--checkpoint-interval 100`. The threshold flags
  sparse completion data; meeting it does not certify precision.

Each unordered physical grid edge appears once in the shuffled candidate list,
including grids with two rows or two columns.

Scheduling is explicit:

- `--scheduling generation-quiescent-fifo` (default): one physical-edge generation
  batch, swap to quiescence, then drain the FIFO. Rate arguments are rejected.
- `--scheduling finite-event-fifo --generation-rate 40 --swap-rate 200`: choose
  one generation event or one node visit with relative weights `E*generation_rate`
  and `N*swap_rate`, then drain the FIFO. Omitted finite weights default to 40 and 200.

Both dynamics always execute the same fixed observation horizon. One tick means
one iteration; neither scheduling mode supplies calibrated physical seconds.

Replay the saved values with the same horizon and D list into a new directory:

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.compare --steps 1000 --distillations 1 2 3 --replay ../tmp/po-comparison/inputs --output ../tmp/po-replay
```

Replay uses saved workload settings and the number of saved replicates. It performs
no resampling; missing request or visitation entries fail explicitly. Adding new
D values can require unsaved visits, so create a fresh controlled experiment for
a changed D sweep or observation horizon.

## Numerical outputs and plots

Comparison outputs include:

- `manifest.json` and `inputs/input-*.json`: experiment choices, completion status,
  graph/consumer values, and explicit replay inputs.
- `results.csv` and `runs/*.json`: both overheads, both denominator totals, swaps,
  generated pairs, completed requests, initial/remaining inventory, actual horizon,
  stopping reason, hop histograms, per-consumer completions including zeros, and
  FIFO head. Failed runs retain available partial totals and error status.
- `summary.csv`: means and two-sided 95% Student-t intervals across independent
  replicates, with finished/defined/minimum-completion counts.
- `paired_differences.csv` and `paired_summary.csv`: corrected-minus-original
  differences for matched replicates and their intervals.
- `checkpoints.csv` and `horizon_summary.csv`: cumulative totals for inventory drift
  and horizon sensitivity. Checkpoints within one trajectory are correlated and
  are never counted as additional independent replicates.

If any group member is unfinished or has an undefined metric, that metric's group
mean and interval remain undefined. Low-completion runs remain included and are
flagged. A single replicate has no confidence interval. Plots show these counts,
uncertainty, and missing estimates, including horizon sensitivity and inventory
drift. `--output` can choose the plot directory; its default is `DATA/plots`.
Dense horizon plots retain every checkpoint in their curves, shaded confidence
bands, and missing-value markers, while limiting count tables and axis labels to
ten checkpoints. The complete numerical data remain in `horizon_summary.csv`.

## Historical workload presets and metric-only runs

The copied sweep definitions remain available, retaining their D-dependent batch
means and original topology/traffic grouping. The corrected runner labels these
as historical workloads; they are not controlled across D or changed dynamics.

```powershell
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.run --sweep distillation --list
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.run --sweep distillation --case 0 --max-steps 1000 --seed 271 --output ../tmp/po-corrected-sweep
.\.venv\Scripts\python.exe -m example.path_oblivious_corrected.run --sweep distillation --case 0 --max-steps 1000 --seed 271 --dynamics original --stop-mode author --output ../tmp/po-metric-only
```

The runner defaults to corrected dynamics and fixed stopping. Without `--max-steps`,
each preset retains its 5,000,000-iteration horizon. `--case` is repeatable;
`--checkpoint-interval` controls saved checkpoints. Other presets are `distillation-both`,
`nodes`, `nodes-grid`, `finite-distillation`, and `swap-rate`.

`--dynamics original --stop-mode author` keeps the original transitions, RNG draw
order, and legacy-ratio stopping check while recording the corrected metric from
the same trajectory. Historical outputs retain original CSV columns and add both
overheads, corrected denominator, raw counts, per-case histograms and checkpoints.
Use the comparison runner for plots that compare algorithm behavior on matching
inputs; its plotter consumes comparison summaries.

For Python callers, `model.run_simulation(..., dynamics='corrected',
stop_mode='fixed', time_steps=1000)` returns both metrics and raw accounting.
Pass `dynamics='original', stop_mode='author'` for metric-only author behavior.

## Independent LP reference

`lp.solve_lp` minimizes generation for fixed completed-request rates. It includes
all unordered pair buffers and all distinct pivot/endpoint triples, allows virtual
outputs, and bounds generation on physical edges. Generation is throttled and
unused supply is permitted. The LP checks numerical balance and bound residuals;
it is a rate resource bound, not a queue-stability or heuristic-optimality result.

Here `capacity` is a generation **rate**, not the simulator's initial inventory:

```python
import networkx as nx
from example.path_oblivious_corrected.lp import solve_lp

graph = nx.path_graph(3)
nx.set_edge_attributes(graph, 1.0, "capacity")
result = solve_lp(graph, {(0, 2): 0.1}, distillation_pairs=2)
assert result.success
print(result.objective)  # 0.8: 0.4 generation on each link, 0.2 swaps
```

`build_lp` exposes sparse coefficients; `solve_problem` solves an assembled model.
The analytical simulation denominator remains separate from this LP.

## Correctness checks

```powershell
.\.venv\Scripts\python.exe -m pytest example/path_oblivious example/path_oblivious_corrected -q --basetemp E:/thesis/tmp/po-correctness-tests
```

Use a dedicated disposable `--basetemp` directory: pytest replaces its contents.
The tests preserve executable parity checks against the pinned author code in the
sibling `quantum-network` clone (`AUTHOR_REPO` may select another clone). They also
cover metric/RNG/stopping preservation, denominator examples and independent tree
enumeration, current-state swap preference, inventory conservation, the executable
global-max-min counterexample, matched replay, output statistics, and known LP
solutions. No existing results are needed or copied.
