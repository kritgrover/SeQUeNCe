"""Plot the author's CSV metrics, including exact replotting of archived data."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sweeps import SOURCE_COMMIT


PLOTS = (
    ("swap_overhead", "Swap Overhead", False),
    ("swap_overhead_log", "Swap Overhead", True),
    ("swap_efficiency", "Swap Efficiency", False),
    ("total_bell_pairs", "Total Bell Pairs", False),
    ("total_pairs_consumed", "Total Pairs Consumed", False),
)
KINDS = {
    "distillation": ("distillation", "distillation pairs", "Distillation Pairs Count"),
    "nodes": ("node", "nodes", "Nodes"),
    "swap-rate": ("swap_rate", "swap rate", "Swap Rate"),
}


def export_reference(repo, output, kind):
    """Read the pinned author's tracked CSVs; never run or modify its checkout."""
    repo, output = Path(repo).resolve(), Path(output)
    suffix = {
        "distillation": "distillation_metrics_output_infinite_swapping.csv",
        "nodes": "node_metrics_output-distillation-1.csv",
        "swap-rate": "swap_rate_metrics_pool_output.csv",
    }[kind]
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty reference output directory: {output}")
    for graph_type in ("cycle", "wrap_grid_overlay"):
        name = f"{graph_type}_{suffix}"
        raw = subprocess.run([
            "git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
            "show", f"{SOURCE_COMMIT}:simulation-final/{name}",
        ], check=True, capture_output=True).stdout
        (output / name).write_bytes(raw)
    (output / "provenance.json").write_text(json.dumps({
        "kind": "archived author data, not a new simulation",
        "source_commit": SOURCE_COMMIT,
    }, indent=2) + "\n", encoding="utf-8")


def read_curves(data, kind):
    file_part, x_column, _ = KINDS[kind]
    curves = []
    for path in sorted(Path(data).glob(f"*_{file_part}_metrics_*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        graph_type = "wrap_grid_overlay" if path.name.startswith("wrap_grid_overlay_") else "cycle"
        label = graph_type
        if kind == "nodes" and "-distillation-" in path.stem:
            label += f" (distillation={path.stem.rsplit('-', 1)[1]})"
        if any(curve["label"] == label for curve in curves):
            raise ValueError(f"Multiple datasets for {label}; plot each sweep directory separately")
        def values(key):
            return np.array([float(row[key]) for row in rows])
        swaps = values("total swaps")
        denominator = values("optimal direct path sum")
        consumed = values("total pairs consumed")
        with np.errstate(divide="ignore", invalid="ignore"):
            overhead = swaps / denominator
            efficiency = swaps / consumed
        curves.append({
            "label": label, "graph_type": graph_type, "x": values(x_column),
            "swap_overhead": overhead, "swap_overhead_log": overhead,
            "swap_efficiency": efficiency, "total_bell_pairs": values("total bell pairs"),
            "total_pairs_consumed": consumed,
        })
    if not curves:
        raise FileNotFoundError(f"No {kind} CSVs found in {data}")
    return curves


def plot_results(data, output, kind):
    curves = read_curves(data, kind)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _, _, xlabel = KINDS[kind]
    for key, ylabel, log_scale in PLOTS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for curve in curves:
            ax.plot(curve["x"], curve[key], marker="o", linewidth=2.5,
                    linestyle="--" if curve["graph_type"] == "wrap_grid_overlay" else "-",
                    label=curve["label"])
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if log_scale or (kind == "distillation" and key != "swap_overhead") or (kind == "swap-rate" and key == "total_bell_pairs"):
            if any(np.any(np.isfinite(c[key]) & (c[key] > 0)) for c in curves):
                ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=160)
        plt.close(fig)
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Directory of a single sweep's output CSVs")
    source.add_argument("--reference-repo", type=Path, help="Replot tracked data from the pinned quantum-network commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data
    if args.reference_repo:
        data = args.output / "reference_data"
        export_reference(args.reference_repo, data, args.kind)
    plot_results(data, args.output, args.kind)
    print(f"Wrote five plots to {args.output}")


if __name__ == "__main__":
    main()
