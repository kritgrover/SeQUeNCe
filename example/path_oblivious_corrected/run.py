"""Run corrected or original dynamics on labeled historical sweep workloads.

These presets retain the source's D-dependent batch means, consumer counts,
topology sharing, and combined RNG draw stream. Use ``compare`` for controlled
input replay across algorithms or D values. Both denominator definitions are
computed from each run, and the default stopping mode is a fixed horizon.
"""

import argparse
import csv
from dataclasses import asdict
import json
from numbers import Integral
from pathlib import Path
import random

from .sweeps import SWEEP_NAMES, get_cases
from .topology import GRID_DIMENSIONS, create_graph


METRIC_COLUMNS = (
    "total swaps", "optimal direct path sum", "worst direct path sum",
    "total pairs consumed", "total bell pairs",
)
EXTRA_COLUMNS = (
    "corrected direct path sum", "legacy overhead", "corrected overhead",
    "total generated", "initial inventory", "remaining inventory",
    "executed steps", "stop reason", "completion status", "dynamics",
)
METRIC_DEFINITIONS = {
    "legacy": "swaps / sum(f[hops-1]); f[0]=0, f[1]=D, f[k]=D*(f[k//2]+f[(k+1)//2])",
    "corrected": "swaps / sum(s_D(hops)); s_D(1)=0, s_D(h)=D*(1+s_D(h//2)+s_D((h+1)//2))",
    "undefined": "Zero denominator is null in JSON and empty in CSV, never zero overhead",
    "window": "Cumulative from initial state; no warm-up removed; includes work retained as inventory",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _batch_mean(case):
    d = case.distillation_pairs
    if case.profile == "finite":
        return 3 if d <= 2 else (10 if d <= 5 else 20)
    if d == 1:
        return 10 if case.profile == "distillation" else 5
    return {2: 30, 3: 200, 4: 650, 5: 850}[d]


def _case_settings(case):
    finite = case.profile == "finite"
    return {
        "scheduling": "finite_event_fifo" if finite else "generation_quiescent_fifo",
        "scheduling_definition": (
            "One generation or one node visit, weighted by E*generation_rate and N*swap_rate; then drain blocking FIFO"
            if finite else "One uniform physical edge and Poisson batch, shuffled node passes to quiescence, then drain blocking FIFO"
        ),
        "generation_batch": {"distribution": "Poisson", "mean": _batch_mean(case)},
        "rate_arguments_applicable": finite,
        "rate_argument_note": (
            "Relative event-selection weights; no physical clock calibration"
            if finite else "generation_rate and swap_rate are historical unused arguments"
        ),
        "traffic_count_specification": (
            "floor(0.1*N*(N-1)/2) historical baseline"
            if case.profile in ("nodes", "nodes-grid") else "fixed 35 historical/paper traffic count"
        ),
        "consumer_weight_distribution": "Independent discrete uniform integers 1..6",
        "grid_dimensions": (
            {"rows": GRID_DIMENSIONS[case.nodes][0], "columns": GRID_DIMENSIONS[case.nodes][1]}
            if case.graph_type == "wrap_grid_overlay" else None
        ),
    }


def _completion_status(result, requested_steps):
    if result.steps == requested_steps:
        return "horizon_complete"
    if result.stop_reason == "author_stabilization":
        return "author_stopped_early"
    return "unfinished"


def run_sweep(name, output, *, seed=None, max_steps=None, case_indices=None,
              dynamics="corrected", stop_mode="fixed", checkpoint_interval=100):
    """Retain historical workload presets, with explicit dynamics and stopping.

    Selecting a subset preserves the full sweep's graph and simulation seed
    allocation. This runner intentionally does not reorganize the author's RNG
    streams; equal seeds across changed dynamics do not imply matching inputs.
    """
    from .model import TIE_BREAKING, run_simulation

    cases = get_cases(name)
    indices = list(range(len(cases))) if case_indices is None else list(case_indices)
    if not indices or any(not isinstance(i, Integral) or isinstance(i, bool)
                          or i < 0 or i >= len(cases) for i in indices):
        raise ValueError(f"Case indices must lie in 0..{len(cases) - 1}")
    if len(set(indices)) != len(indices):
        raise ValueError("Case indices must be unique")
    if max_steps is not None and (not isinstance(max_steps, Integral)
                                  or isinstance(max_steps, bool) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if seed is not None and (not isinstance(seed, Integral) or isinstance(seed, bool)
                             or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in 0..2**32-1")
    if dynamics not in ("original", "corrected"):
        raise ValueError("dynamics must be 'original' or 'corrected'")
    if stop_mode not in ("fixed", "author"):
        raise ValueError("stop_mode must be 'fixed' or 'author'")
    if (not isinstance(checkpoint_interval, Integral) or isinstance(checkpoint_interval, bool)
            or checkpoint_interval < 1):
        raise ValueError("checkpoint_interval must be a positive integer")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {output}")
    master_seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
    rng = random.Random(master_seed)
    inputs, graphs = {}, {}
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
            "grid_dimensions": _case_settings(case)["grid_dimensions"],
        }
    simulation_seeds = [rng.randrange(2**32) for _ in cases]
    manifest = {
        "implementation": f"path_oblivious_corrected/{dynamics}",
        "experiment_specification": "historical_workload_presets",
        "source_script_settings": "Known-source arXiv-era workloads; not an exact final HotNets figure reproduction",
        "sweep": name, "seed": master_seed, "dynamics": dynamics, "stop_mode": stop_mode,
        "max_steps_override": max_steps, "selected_case_indices": indices,
        "checkpoint_interval": checkpoint_interval,
        "metric_definitions": METRIC_DEFINITIONS,
        "time_unit": "One loop iteration per SeQUeNCe tick; no physical calibration",
        "input_streams": "Author-style combined Python RNG for requests, generation-edge selection and node visitation; independent NumPy Poisson RNG",
        "controlled_across_dynamics_or_D": False,
        "comparison_note": "Use example.path_oblivious_corrected.compare for matched replay inputs and common horizons",
        "tie_breaking": TIE_BREAKING if dynamics == "corrected" else "Original stable candidate sort and stale candidate batching",
        "stopping_note": "Fixed mode executes the requested horizon; author mode checks only legacy ratios and does not establish precision or stationarity",
        "cases": [],
    }
    for index in indices:
        case = cases[index]
        manifest["cases"].append({
            "index": index, **asdict(case), "source_script": case.source_script,
            **_case_settings(case),
            "simulation_seed": simulation_seeds[index],
            "requested_steps": case.time_steps if max_steps is None else max_steps,
            "status": "pending", "result_file": f"case-{index:04d}.json",
        })
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs.json", inputs)
    write_json(output / "manifest.json", manifest)
    rows_by_file = {}
    for entry in manifest["cases"]:
        index = entry["index"]
        case = cases[index]
        steps = entry["requested_steps"]
        graph, consumers = graphs[case.topology_group]
        print(f"[{index + 1}/{len(cases)}] {case.graph_type} N={case.nodes} "
              f"D={case.distillation_pairs} dynamics={dynamics}", flush=True)
        entry["status"] = "running"
        write_json(output / "manifest.json", manifest)
        try:
            result = run_simulation(
                graph, consumers, profile=case.profile,
                distillation_pairs=case.distillation_pairs, time_steps=steps,
                generation_rate=case.generation_rate, swap_rate=case.swap_rate,
                seed=simulation_seeds[index], dynamics=dynamics, stop_mode=stop_mode,
                checkpoint_interval=checkpoint_interval)
        except Exception as error:
            entry.update(status="failed", error_type=type(error).__name__, error=str(error))
            write_json(output / "manifest.json", manifest)
            raise
        completion = _completion_status(result, steps)
        row = {
            case.x_column: case.x_value,
            **dict(zip(METRIC_COLUMNS, result.author_metrics)),
            "corrected direct path sum": result.corrected_denominator,
            "legacy overhead": result.legacy_overhead,
            "corrected overhead": result.corrected_overhead,
            "total generated": result.total_generated,
            "initial inventory": result.initial_inventory,
            "remaining inventory": result.remaining_inventory,
            "executed steps": result.steps, "stop reason": result.stop_reason,
            "completion status": completion, "dynamics": dynamics,
        }
        detail = {
            "case_index": index, "implementation": manifest["implementation"],
            "experiment_specification": manifest["experiment_specification"],
            "metric_definitions": METRIC_DEFINITIONS,
            "requested_steps": steps, "executed_steps": result.steps,
            "stop_reason": result.stop_reason, "completion_status": completion,
            "zero_completions": result.total_pairs_consumed == 0,
            "metrics": row,
            "completed_hops": result.completed_hops,
            "completed_by_pair": [[u, v, count] for (u, v), count in result.completed_by_pair.items()],
            "checkpoints": result.checkpoints,
        }
        write_json(output / entry["result_file"], detail)
        rows = rows_by_file.setdefault(case.output_csv, [])
        rows.append(row)
        with (output / case.output_csv).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[case.x_column, *METRIC_COLUMNS, *EXTRA_COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        entry.update(status="complete", executed_steps=result.steps,
                     stop_reason=result.stop_reason, completion_status=completion,
                     zero_completions=result.total_pairs_consumed == 0, metrics=row)
        write_json(output / "manifest.json", manifest)
        print(f"  {result.steps} iterations; {result.stop_reason}; "
              f"{result.total_swaps} swaps; {result.total_pairs_consumed} consumed", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=SWEEP_NAMES, default="distillation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Recorded seed; omitted means a fresh random seed")
    parser.add_argument("--max-steps", type=int, help="Observation-horizon override; default is historical 5,000,000")
    parser.add_argument("--case", type=int, action="append", help="Run a zero-based case index; repeat for several")
    parser.add_argument("--dynamics", choices=("corrected", "original"), default="corrected")
    parser.add_argument("--stop-mode", choices=("fixed", "author"), default="fixed")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--list", action="store_true", help="Print exact cases without running")
    args = parser.parse_args()
    if args.list:
        for i, case in enumerate(get_cases(args.sweep)):
            print(json.dumps({"index": i, **asdict(case), **_case_settings(case)}))
        return
    output = args.output or Path(__file__).parent / "results" / args.sweep
    run_sweep(args.sweep, output, seed=args.seed, max_steps=args.max_steps,
              case_indices=args.case, dynamics=args.dynamics, stop_mode=args.stop_mode,
              checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    main()
