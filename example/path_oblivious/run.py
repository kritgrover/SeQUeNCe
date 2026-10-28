"""Run the author's sweep presets through SeQUeNCe and write original CSV columns."""

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time

from .sweeps import SOURCE_COMMIT, SOURCE_URL, SWEEP_NAMES, get_cases
from .topology import create_graph


METRIC_COLUMNS = (
    "total swaps", "optimal direct path sum", "worst direct path sum",
    "total pairs consumed", "total bell pairs",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_sweep(name, output, *, seed=None, max_steps=None, case_indices=None):
    """Run complete source cases; an explicit max_steps is a shortened check."""
    from .model import run_simulation

    cases = get_cases(name)
    indices = list(range(len(cases))) if case_indices is None else case_indices
    if not indices or any(i < 0 or i >= len(cases) for i in indices):
        raise ValueError(f"Case indices must lie in 0..{len(cases) - 1}")
    if len(set(indices)) != len(indices):
        raise ValueError("Case indices must be unique")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    master_seed = random.SystemRandom().randrange(2**32) if seed is None else seed
    rng = random.Random(master_seed)
    inputs = {}
    graphs = {}
    for case in cases:
        if case.topology_group in graphs:
            continue
        graph_seed = rng.randrange(2**32)
        graph, consumers = create_graph(
            case.nodes, case.graph_type, case.consumer_count, seed=graph_seed)
        graphs[case.topology_group] = graph, consumers
        inputs[case.topology_group] = {
            "seed": graph_seed,
            "nodes": list(graph.nodes()),
            "edges": [[u, v, graph[u][v]["capacity"]] for u, v in graph.edges()],
            "consumers": [[u, v, weight] for (u, v), weight in consumers.items()],
        }
    simulation_seeds = [rng.randrange(2**32) for _ in cases]
    manifest = {
        "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
        "sweep": name, "seed": master_seed,
        "max_steps_override": max_steps, "selected_case_indices": indices,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "networkx", "matplotlib", "scipy", "qutip", "qutip-qip", "stim")},
        "time_unit": "one author loop iteration per SeQUeNCe tick; no physical calibration",
        "runtime_accommodation": "finite profile interprets source lowercase false as False",
        "cases": [],
    }
    write_json(output / "inputs.json", inputs)
    write_json(output / "manifest.json", manifest)
    rows_by_file = {}
    for index in indices:
        case = cases[index]
        steps = case.time_steps if max_steps is None else max_steps
        graph, consumers = graphs[case.topology_group]
        print(f"[{index + 1}/{len(cases)}] {case.graph_type} N={case.nodes} "
              f"D={case.distillation_pairs} swap_rate={case.swap_rate}", flush=True)
        started = time.perf_counter()
        result = run_simulation(
            graph, consumers, profile=case.profile,
            distillation_pairs=case.distillation_pairs, time_steps=steps,
            generation_rate=case.generation_rate, swap_rate=case.swap_rate,
            seed=simulation_seeds[index])
        row = {case.x_column: case.x_value, **dict(zip(METRIC_COLUMNS, result.author_metrics))}
        rows = rows_by_file.setdefault(case.output_csv, [])
        rows.append(row)
        with (output / case.output_csv).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[case.x_column, *METRIC_COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        manifest["cases"].append({
            "index": index, **asdict(case), "source_script": case.source_script,
            "simulation_seed": simulation_seeds[index], "requested_steps": steps,
            "executed_steps": result.steps, "stop_reason": result.stop_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "metrics": row,
        })
        write_json(output / "manifest.json", manifest)
        print(f"  {result.steps} iterations; {result.stop_reason}; "
              f"{result.total_swaps} swaps; {result.total_pairs_consumed} consumed", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=SWEEP_NAMES, default="distillation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Recorded seed; omitted means fresh random seed")
    parser.add_argument("--max-steps", type=int, help="Explicit short-check override; default is original 5,000,000")
    parser.add_argument("--case", type=int, action="append", help="Run a zero-based case index; repeat to select several")
    parser.add_argument("--list", action="store_true", help="Print the exact cases without running")
    args = parser.parse_args()
    if args.list:
        for i, case in enumerate(get_cases(args.sweep)):
            print(json.dumps({"index": i, **asdict(case)}))
        return
    output = args.output or Path(__file__).parent / "results" / args.sweep
    run_sweep(args.sweep, output, seed=args.seed, max_steps=args.max_steps, case_indices=args.case)


if __name__ == "__main__":
    main()
