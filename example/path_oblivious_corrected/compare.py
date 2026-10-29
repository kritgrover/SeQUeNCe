"""Compare original and corrected local dynamics on explicit common input tapes."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics

from scipy.stats import t as student_t

from .model import InventorySimulation
from .replay import ReplayInputs, SCHEDULES
from .topology import create_graph, GRID_DIMENSIONS


GROUP_FIELDS = ("graph_type", "nodes", "rows", "columns", "scheduling",
                "distillation_pairs", "horizon")
METRICS = ("legacy_overhead", "corrected_overhead", "total_swaps",
           "total_pairs_consumed", "total_generated", "remaining_inventory")
RAW_FIELDS = ("initial_inventory", "legacy_denominator", "corrected_denominator",
              *METRICS, "steps", "stop_reason", "finished", "minimum_completed_met", "status")
SUMMARY_FIELDS = (*GROUP_FIELDS, "dynamics", "metric", "n_total", "n_finished",
                  "n_defined", "n_minimum_completed", "mean", "ci_low", "ci_high", "status")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """95% Student-t intervals across replicates; never silently drop failures.

    No pooled ratio or conditional mean is substituted when any member of a
    group has an undefined metric. Low-completion observations remain included
    and are flagged; meeting the count threshold is not a precision claim.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in (*GROUP_FIELDS, "dynamics"))].append(row)
    summaries = []
    for key, members in groups.items():
        for metric in METRICS:
            values = [r[metric] for r in members
                      if r.get(metric) is not None and math.isfinite(r[metric])]
            finished = sum(r["finished"] for r in members)
            enough = sum(r["minimum_completed_met"] for r in members)
            mean = low = high = None
            status = []
            if finished < len(members):
                status.append("unfinished_runs")
            if len(values) < len(members):
                status.append("undefined_observations")
            if enough < len(members):
                status.append("insufficient_completions")
            if len(values) == len(members) and finished == len(members):
                mean = statistics.mean(values)
                if len(values) >= 2:
                    margin = float(student_t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
                    low, high = mean - margin, mean + margin
                else:
                    status.append("single_replication_no_ci")
            summaries.append({
                **dict(zip((*GROUP_FIELDS, "dynamics"), key)), "metric": metric,
                "n_total": len(members), "n_finished": finished, "n_defined": len(values),
                "n_minimum_completed": enough, "mean": mean, "ci_low": low, "ci_high": high,
                "status": ";".join(status) or "fixed_horizon_estimate",
            })
    return summaries


def paired_differences(rows):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[tuple(row[k] for k in (*GROUP_FIELDS, "replicate", "input_id"))][row["dynamics"]] = row
    deltas = []
    for key, pair in pairs.items():
        if set(pair) != {"original", "corrected"}:
            continue
        original, corrected = pair["original"], pair["corrected"]
        row = dict(zip((*GROUP_FIELDS, "replicate", "input_id"), key))
        row.update(dynamics="corrected-minus-original",
                   finished=original["finished"] and corrected["finished"],
                   minimum_completed_met=original["minimum_completed_met"] and corrected["minimum_completed_met"])
        for metric in METRICS:
            a, b = original.get(metric), corrected.get(metric)
            row[metric] = None if a is None or b is None else b - a
        deltas.append(row)
    return deltas


def run_comparison(output, *, horizon, distillations=(1, 2), replicates=5,
                   seed=271, graph_type="wrap_grid_overlay", nodes=25,
                   rows=None, columns=None, consumer_count=35, consumer_policy="fixed-35",
                   batch_mean=10, scheduling=SCHEDULES[0], generation_rate=None,
                   swap_rate=None, minimum_completed=100, checkpoint_interval=None,
                   replay=None):
    if horizon < 1 or replicates < 1 or minimum_completed < 0:
        raise ValueError("Positive horizon/replicates and nonnegative minimum_completed required")
    distillations = tuple(distillations)
    if (not distillations or len(set(distillations)) != len(distillations)
            or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in distillations)):
        raise ValueError("Distillation multipliers must be unique positive integers")
    checkpoint_interval = max(1, horizon // 10) if checkpoint_interval is None else checkpoint_interval
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if scheduling == SCHEDULES[0] and (generation_rate is not None or swap_rate is not None):
        raise ValueError("Rate arguments apply only to finite-event-fifo")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {output}")

    tapes = []
    if replay is not None:
        paths = sorted(Path(replay).glob("input-*.json"))
        if not paths:
            raise ValueError("Replay directory contains no input-*.json files")
        tapes = [ReplayInputs.load(path) for path in paths]
        if any(tape.horizon != horizon for tape in tapes):
            raise ValueError("Replay horizon must equal the requested observation horizon")
        def specification(tape):
            return (tape.scheduling, tape.batch_mean, tape.generation_rate, tape.swap_rate,
                    *(tape.metadata.get(k) for k in
                      ("graph_type", "nodes", "rows", "columns", "consumer_policy", "consumer_count")))
        if len({specification(tape) for tape in tapes}) != 1:
            raise ValueError("Replay directory must contain replicates of one workload specification")
        replicates = len(tapes)
    else:
        if consumer_policy == "fixed-35":
            consumer_count = 35
        elif consumer_policy == "fraction-10-percent":
            consumer_count = math.floor(0.1 * nodes * (nodes - 1) / 2)
        elif consumer_policy != "custom":
            raise ValueError("Unknown consumer policy")
        if graph_type == "wrap_grid_overlay" and rows is None and columns is None:
            if nodes not in GRID_DIMENSIONS:
                raise ValueError("Specify rows and columns for this grid size")
            rows, columns = GRID_DIMENSIONS[nodes]
        master = random.Random(seed)
        for replicate in range(replicates):
            graph_seed, input_seed = master.randrange(2**32), master.randrange(2**32)
            graph, consumers = create_graph(nodes, graph_type, consumer_count,
                                            seed=graph_seed, rows=rows, columns=columns)
            metadata = {"graph_type": graph_type, "nodes": nodes, "rows": rows,
                        "columns": columns, "consumer_policy": consumer_policy,
                        "consumer_count": consumer_count, "graph_seed": graph_seed,
                        "replicate": replicate}
            tapes.append(ReplayInputs(graph, consumers, horizon=horizon, batch_mean=batch_mean,
                                      seed=input_seed, scheduling=scheduling,
                                      generation_rate=generation_rate, swap_rate=swap_rate,
                                      metadata=metadata))

    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir()
    (output / "runs").mkdir()
    manifest = {
        "experiment": "controlled-local-rule-comparison-v1",
        "status": "running", "seed": seed if replay is None else None,
        "replay_directory": str(replay) if replay is not None else None,
        "distillation_pairs": distillations, "replicates": replicates,
        "horizon": horizon, "checkpoint_interval": checkpoint_interval,
        "minimum_completed_requests": minimum_completed,
        "warm_up": "none; cumulative swaps include work retained in final inventory",
        "time_unit": "iteration; not calibrated seconds",
        "stopping": "fixed common horizon; author ratio stabilization disabled for both dynamics",
        "confidence_interval": "two-sided 95% Student t across independent replicates; paired differences also saved",
        "undefined_policy": "JSON null / CSV blank; group mean and CI undefined if any member undefined or unfinished",
        "low_completion_policy": "retain observations and flag insufficient completions; threshold is not precision certification",
        "streams": "recorded generation/event arrays and shared FIFO request tape; node passes keyed by iteration/pass so divergent passes cannot shift inputs",
        "tie_breaking": "first minimum in existing set-neighbor combination order; original unchanged routine retains its original batching",
        "metrics": {"legacy_overhead": "executed swaps / sum(f[hops-1])",
                    "corrected_overhead": "executed swaps / sum(s_D(hops)); s_D(1)=0; s_D(h)=D*(1+s_D(floor(h/2))+s_D(ceil(h/2)))"},
        "interpretation": "local balancing heuristic, no global max-min or steady-state claim",
        "workloads": [{**tape.metadata, "scheduling": tape.scheduling,
                       "generation_distribution": "Poisson, independent of D",
                       "batch_mean": tape.batch_mean, "generation_rate": tape.generation_rate,
                       "swap_rate": tape.swap_rate, "consumer_weights": "uniform integers 1..6; realized weights saved",
                       "input_file": f"inputs/input-{i:03d}.json"} for i, tape in enumerate(tapes)],
        "runs": [],
    }
    write_json(output / "manifest.json", manifest)
    result_rows, checkpoints = [], []
    failures = 0
    for replicate, tape in enumerate(tapes):
        input_id = f"input-{replicate:03d}"
        for d in distillations:
            for dynamics in ("original", "corrected"):
                run_id = f"r{replicate:03d}-d{d}-{dynamics}"
                group = {k: tape.metadata.get(k) for k in GROUP_FIELDS}
                group.update(scheduling=tape.scheduling, distillation_pairs=d, horizon=horizon)
                row = {**group, "replicate": replicate, "input_id": input_id, "dynamics": dynamics,
                       **dict.fromkeys(RAW_FIELDS), "finished": False, "minimum_completed_met": False}
                print(f"{run_id}: {horizon} iterations on {input_id}", flush=True)
                simulation = None
                try:
                    simulation = InventorySimulation(
                        tape.graph, tape.consumers,
                        profile="finite" if tape.scheduling == SCHEDULES[1] else "distillation",
                        distillation_pairs=d, time_steps=horizon,
                        dynamics=dynamics, stop_mode="fixed", input_source=tape,
                        checkpoint_interval=checkpoint_interval)
                    result = simulation.run()
                    row.update({field: getattr(result, field) for field in
                                ("initial_inventory", "legacy_denominator", "corrected_denominator", *METRICS,
                                 "steps", "stop_reason")})
                    row["finished"] = result.steps == horizon and result.stop_reason == "fixed_horizon"
                    row["minimum_completed_met"] = result.total_pairs_consumed >= minimum_completed
                    row["status"] = ("unfinished" if not row["finished"] else
                                     "completed_horizon" if row["minimum_completed_met"]
                                     else "insufficient_completions")
                    if not row["finished"]:
                        failures += 1
                    detail = {"run_id": run_id, **row,
                              "completed_hops": result.completed_hops,
                              "completed_by_pair": [[u, v, count] for (u, v), count in result.completed_by_pair.items()],
                              "queue_head": result.checkpoints[-1]["queue_head"],
                              "requests_loaded": simulation.requests_loaded,
                              "final_inventory": [[u, v, int(count)] for (u, v), count in result.edge_states.items()]}
                    for point in result.checkpoints:
                        checkpoints.append({**group, "replicate": replicate, "dynamics": dynamics,
                                            "input_id": input_id, **point})
                except Exception as error:
                    failures += 1
                    row.update(status="failed", stop_reason="error",
                               steps=simulation.steps if simulation is not None else 0)
                    if simulation is not None:
                        # Keep actual partial totals while finished=False prevents
                        # them from masquerading as full-horizon observations.
                        row.update(initial_inventory=simulation.initial_inventory,
                                   total_swaps=int(simulation.total_swaps),
                                   total_generated=int(simulation.total_generated),
                                   total_pairs_consumed=int(simulation.total_pairs_consumed),
                                   legacy_denominator=int(simulation.optimal_direct_path_sum),
                                   corrected_denominator=int(simulation.corrected_direct_path_sum),
                                   remaining_inventory=int(sum(simulation.edge_states.values())))
                        for point in simulation.checkpoints:
                            checkpoints.append({**group, "replicate": replicate, "dynamics": dynamics,
                                                "input_id": input_id, **point})
                    detail = {"run_id": run_id, **row,
                              "error": f"{type(error).__name__}: {error}"}
                    if simulation is not None:
                        detail.update(completed_hops=dict(simulation.completed_hops),
                                      completed_by_pair=[[u, v, count] for (u, v), count
                                                         in simulation.completed_by_pair.items()],
                                      queue_head=simulation.consumption_queue[0]
                                      if simulation.consumption_queue else None)
                    print(f"  FAILED: {detail['error']}", flush=True)
                result_rows.append(row)
                write_json(output / "runs" / f"{run_id}.json", detail)
                manifest["runs"].append({"run_id": run_id, "file": f"runs/{run_id}.json",
                                         "status": row["status"]})
                tape.save(output / "inputs" / f"{input_id}.json")
                write_csv(output / "results.csv", result_rows,
                          (*GROUP_FIELDS, "replicate", "input_id", "dynamics", *RAW_FIELDS))
                write_json(output / "manifest.json", manifest)
    write_csv(output / "checkpoints.csv", checkpoints,
              (*GROUP_FIELDS, "replicate", "input_id", "dynamics", "step", "total_swaps",
               "total_generated", "total_pairs_consumed", "legacy_denominator", "corrected_denominator",
               "legacy_overhead", "corrected_overhead", "remaining_inventory", "queue_head"))
    write_csv(output / "summary.csv", summarize(result_rows), SUMMARY_FIELDS)
    # Each checkpoint horizon has the same replicate population as the final
    # horizon. Missing checkpoints from failures are explicit unfinished rows.
    by_run_step = {(p["replicate"], p["distillation_pairs"], p["dynamics"], p["step"]): p
                   for p in checkpoints}
    horizons = sorted(set(range(checkpoint_interval, horizon + 1, checkpoint_interval)) | {horizon})
    horizon_rows = []
    for row in result_rows:
        for step in horizons:
            point = by_run_step.get((row["replicate"], row["distillation_pairs"], row["dynamics"], step))
            horizon_rows.append({**{k: row[k] for k in (*GROUP_FIELDS, "dynamics")},
                                 "horizon": step,
                                 **{metric: point.get(metric) if point else None for metric in METRICS},
                                 "finished": point is not None,
                                 "minimum_completed_met": bool(point and point["total_pairs_consumed"] >= minimum_completed)})
    write_csv(output / "horizon_summary.csv", summarize(horizon_rows), SUMMARY_FIELDS)
    deltas = paired_differences(result_rows)
    write_csv(output / "paired_differences.csv", deltas,
              (*GROUP_FIELDS, "replicate", "input_id", "dynamics", *METRICS,
               "finished", "minimum_completed_met"))
    write_csv(output / "paired_summary.csv", summarize(deltas), SUMMARY_FIELDS)
    manifest["status"] = "completed_with_failures" if failures else "completed"
    manifest["failed_runs"] = failures
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="Prespecified common observation horizon")
    parser.add_argument("--distillations", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--graph", choices=("cycle", "wrap_grid_overlay"), default="wrap_grid_overlay")
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    traffic = parser.add_mutually_exclusive_group()
    traffic.add_argument("--consumer-policy", choices=("fixed-35", "fraction-10-percent"), default="fixed-35")
    traffic.add_argument("--consumers", type=int, help="Explicitly labeled custom traffic count")
    parser.add_argument("--batch-mean", type=float, default=10)
    parser.add_argument("--scheduling", choices=SCHEDULES, default=SCHEDULES[0])
    parser.add_argument("--generation-rate", type=float)
    parser.add_argument("--swap-rate", type=float)
    parser.add_argument("--minimum-completed", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--replay", type=Path, help="Use a saved inputs directory; its workload settings take precedence")
    args = parser.parse_args()
    try:
        result = run_comparison(
            args.output, horizon=args.steps, distillations=args.distillations,
            replicates=args.replicates, seed=args.seed, graph_type=args.graph, nodes=args.nodes,
            rows=args.rows, columns=args.columns, consumer_count=args.consumers,
            consumer_policy="custom" if args.consumers is not None else args.consumer_policy,
            batch_mean=args.batch_mean, scheduling=args.scheduling,
            generation_rate=args.generation_rate, swap_rate=args.swap_rate,
            minimum_completed=args.minimum_completed, checkpoint_interval=args.checkpoint_interval,
            replay=args.replay)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    if result["failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
