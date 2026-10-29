"""Plot paired comparison output, including incomplete or undefined groups.

Means and Student-t intervals are supplied by compare.py from independent
replicates. This module does not turn checkpoints into additional replicates,
drop undefined ratios, or replace unavailable intervals with zero-width bars.
"""

import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_FIELDS = ("graph_type", "nodes", "rows", "columns", "scheduling", "horizon")
MAX_HORIZON_COLUMNS = 10
RAW_METRICS = (
    ("total_swaps", "Executed swaps"),
    ("total_pairs_consumed", "Completed requests"),
    ("total_generated", "Generated pairs"),
    ("remaining_inventory", "Remaining inventory"),
)
SERIES = (
    ("original", "legacy_overhead", "Original dynamics / legacy metric", "#386cb0", "-"),
    ("original", "corrected_overhead", "Original dynamics / corrected metric", "#386cb0", "--"),
    ("corrected", "legacy_overhead", "Corrected dynamics / legacy metric", "#e67e22", "-"),
    ("corrected", "corrected_overhead", "Corrected dynamics / corrected metric", "#e67e22", "--"),
)


def _number(value):
    """CSV blanks are absent values, never numerical zero."""
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def read_summary(data, filename="summary.csv", group_fields=GROUP_FIELDS):
    path = Path(data) / filename
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {*GROUP_FIELDS, "distillation_pairs", "dynamics", "metric",
                    "n_total", "n_finished", "n_defined", "n_minimum_completed",
                    "mean", "ci_low", "ci_high", "status"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing summary columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No comparison rows in {path}")
    seen = set()
    groups = {}
    for row in rows:
        group = tuple(row[field] for field in group_fields)
        row["distillation_pairs"] = int(row["distillation_pairs"])
        row["horizon"] = int(row["horizon"])
        for key in ("n_total", "n_finished", "n_defined", "n_minimum_completed"):
            row[key] = int(row[key])
        for key in ("mean", "ci_low", "ci_high"):
            row[key] = _number(row[key])
        identity = (*group, row["distillation_pairs"], row["horizon"],
                    row["dynamics"], row["metric"])
        if identity in seen:
            raise ValueError(f"Duplicate summary row: {identity}")
        seen.add(identity)
        groups.setdefault(group, []).append(row)
    return groups


def _count_text(row):
    if row is None:
        return "missing group"
    flags = []
    if row["n_finished"] != row["n_total"]:
        flags.append("unfinished")
    if row["n_defined"] != row["n_total"]:
        flags.append("undefined")
    if row["n_minimum_completed"] != row["n_total"]:
        flags.append("low Q")
    if row["mean"] is not None and (row["ci_low"] is None or row["ci_high"] is None):
        flags.append("no CI")
    return (f"{row['n_defined']}/{row['n_total']} defined; "
            f"{row['n_finished']}/{row['n_total']} finished\n"
            f"{row['n_minimum_completed']}/{row['n_total']} meet min Q\n"
            + (", ".join(flags) if flags else "complete"))


def _isolated_indices(valid):
    valid = np.asarray(valid, dtype=bool)
    return np.flatnonzero(valid & ~np.r_[False, valid[:-1]] & ~np.r_[valid[1:], False])


def _draw_panel(ax, table_ax, rows, series, ylabel, x_field="distillation_pairs",
                max_columns=None):
    x_values = sorted({row[x_field] for row in rows})
    dense = max_columns is not None and len(x_values) > max_columns
    display_indices = (np.linspace(0, len(x_values) - 1, max_columns, dtype=int).tolist()
                       if dense else list(range(len(x_values))))
    display_values = [x_values[index] for index in display_indices]
    lookup = {(row[x_field], row["dynamics"], row["metric"]): row
              for row in rows}
    positions = np.asarray(x_values, dtype=float)
    spacing = min(np.diff(positions)) if len(positions) > 1 else max(1, abs(positions[0]) * 0.1)
    table_cells = []
    series_labels = []
    for series_index, (dynamics, metric, label, color, linestyle) in enumerate(series):
        selected = [lookup.get((value, dynamics, metric)) for value in x_values]
        means = [row["mean"] if row is not None and row["mean"] is not None
                 else math.nan for row in selected]
        # The small positional offset distinguishes coincident points and bars.
        offset = (series_index - (len(series) - 1) / 2) * 0.055 * spacing
        x = positions + offset
        marker_indices = display_indices
        if dense:
            # A singleton between missing means has no connecting line segment.
            marker_indices = sorted(set(display_indices).union(_isolated_indices(np.isfinite(means))))
        ax.plot(x, means, marker="o", color=color, linestyle=linestyle,
                linewidth=1.7, markersize=5, markevery=marker_indices, label=label)
        if dense:
            # Keep every observation and uncertainty interval, but use vectorized
            # bands/markers instead of one artist and table column per checkpoint.
            has_ci = [row is not None and row["mean"] is not None
                      and row["ci_low"] is not None and row["ci_high"] is not None
                      for row in selected]
            lower = [row["ci_low"] if valid else math.nan
                     for row, valid in zip(selected, has_ci)]
            upper = [row["ci_high"] if valid else math.nan
                     for row, valid in zip(selected, has_ci)]
            ax.fill_between(x, lower, upper, color=color, alpha=0.16)
            # An isolated interval has no shaded area, so draw its full extent.
            isolated_ci = _isolated_indices(has_ci)
            if len(isolated_ci):
                ax.vlines(x[isolated_ci], np.asarray(lower)[isolated_ci],
                          np.asarray(upper)[isolated_ci], color=color, linewidth=1.5)
            missing_x = x[~np.isfinite(means)]
            if len(missing_x):
                ax.scatter(missing_x, np.full(len(missing_x), 0.025 + 0.045 * series_index),
                           marker="x", color=color, s=35,
                           transform=ax.get_xaxis_transform())
        else:
            for xi, row in zip(x, selected):
                if row is None or row["mean"] is None:
                    ax.text(xi, 0.025 + 0.045 * series_index, "×", color=color,
                            transform=ax.get_xaxis_transform(), ha="center", fontsize=13)
                elif row["ci_low"] is not None and row["ci_high"] is not None:
                    lower = max(0.0, row["mean"] - row["ci_low"])
                    upper = max(0.0, row["ci_high"] - row["mean"])
                    ax.errorbar([xi], [row["mean"]], yerr=[[lower], [upper]],
                                color=color, fmt="none", capsize=3)
        table_cells.append([_count_text(selected[index]) for index in display_indices])
        series_labels.append(label)
    ax.set_xticks(positions[display_indices], [str(value) for value in display_values])
    ax.set(xlabel=("Observation horizon (iterations)" if x_field == "horizon"
                   else "Input-pair resource multiplier D"), ylabel=ylabel)
    ax.set_xlim(positions[0] - 0.4 * spacing, positions[-1] + 0.4 * spacing)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    table_ax.axis("off")
    table = table_ax.table(cellText=table_cells, rowLabels=series_labels,
                           colLabels=[f"{'t' if x_field == 'horizon' else 'D'}={value}"
                                      for value in display_values],
                           cellLoc="center", rowLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.5)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_index == 0:
            cell.set_facecolor("#eeeeee")
        elif column_index == -1:
            cell.set_text_props(color=series[row_index - 1][3])


def _title(group):
    values = dict(zip(GROUP_FIELDS, group))
    grid = (f" ({values['rows']}×{values['columns']})"
            if values["rows"] and values["columns"] else "")
    return (f"{values['graph_type']}{grid}, N={values['nodes']}; "
            f"{values['scheduling']}; horizon={values['horizon']} iterations")


def _filename(group):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(group)).strip("_")


def plot_results(data, output=None):
    """Write overhead and raw-count plots for every workload/horizon group."""
    data = Path(data)
    groups = read_summary(data)
    output = Path(output) if output is not None else data / "plots"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, rows in groups.items():
        n_d = len({row["distillation_pairs"] for row in rows})
        # Separate figures keep the sample-count tables legible for every metric.
        panels = [("overhead", "Executed swaps / denominator", SERIES)]
        for metric, label in RAW_METRICS:
            panels.append((metric, label, (
                ("original", metric, "Original dynamics", "#386cb0", "-"),
                ("corrected", metric, "Corrected dynamics", "#e67e22", "-"),
            )))
        for name, ylabel, series in panels:
            fig = plt.figure(figsize=(max(11, 5 + 1.35 * n_d), 8 if name == "overhead" else 7))
            grid = fig.add_gridspec(2, 1, height_ratios=[3.8, 1.5 if name == "overhead" else 0.9])
            ax = fig.add_subplot(grid[0])
            table_ax = fig.add_subplot(grid[1])
            _draw_panel(ax, table_ax, rows, series, ylabel)
            fig.suptitle(_title(group), fontsize=11, y=0.98)
            fig.text(0.5, 0.935,
                     "95% Student-t intervals across independent replicates; no warm-up discarded",
                     ha="center", fontsize=9)
            fig.text(0.5, 0.02,
                     "× = unavailable group mean (see counts); no CI = uncertainty unavailable. "
                     "Low-completion runs remain included.\n"
                     "Finite observation horizon; these plots do not establish steady state or global optimality.",
                     ha="center", fontsize=8)
            # Leave room for table row labels outside the plotting axes.
            fig.subplots_adjust(left=0.31 if name == "overhead" else 0.22,
                                right=0.97, top=0.89, bottom=0.10, hspace=0.35)
            path = output / f"{_filename(group)}_{name}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    if (data / "horizon_summary.csv").exists():
        paths.extend(plot_horizon_sensitivity(data, output))
    return paths


def plot_horizon_sensitivity(data, output):
    """Use cross-replicate summaries at each correlated checkpoint."""
    fields = (*GROUP_FIELDS[:-1], "distillation_pairs")
    groups = read_summary(data, "horizon_summary.csv", fields)
    paths = []
    for group, rows in groups.items():
        values = dict(zip(fields, group))
        n_horizons = len({row["horizon"] for row in rows})
        for metric, label in (("corrected_overhead", "Executed swaps / corrected denominator"),
                              ("remaining_inventory", "Remaining inventory")):
            series = (
                ("original", metric, "Original dynamics", "#386cb0", "-"),
                ("corrected", metric, "Corrected dynamics", "#e67e22", "-"),
            )
            fig = plt.figure(figsize=(max(11, 4 + 1.6 * min(n_horizons, MAX_HORIZON_COLUMNS)), 7))
            grid = fig.add_gridspec(2, 1, height_ratios=[3.8, 0.9])
            ax = fig.add_subplot(grid[0])
            table_ax = fig.add_subplot(grid[1])
            _draw_panel(ax, table_ax, rows, series, label, x_field="horizon",
                        max_columns=MAX_HORIZON_COLUMNS)
            shape = (f" ({values['rows']}×{values['columns']})"
                     if values["rows"] and values["columns"] else "")
            fig.suptitle(f"Horizon sensitivity: {values['graph_type']}{shape}, "
                         f"N={values['nodes']}, D={values['distillation_pairs']}; "
                         f"{values['scheduling']}", fontsize=11, y=0.98)
            uncertainty_note = "95% Student-t intervals across independent replicates at each observation horizon"
            if n_horizons > MAX_HORIZON_COLUMNS:
                uncertainty_note += (f"\nAll {n_horizons:,} checkpoints shown; shaded bands show intervals. "
                                     f"Counts/labels at {MAX_HORIZON_COLUMNS} checkpoints; full data in horizon_summary.csv.")
            fig.text(0.5, 0.935, uncertainty_note, ha="center", va="top", fontsize=9)
            fig.text(0.5, 0.02,
                     "Checkpoints along each run are correlated; they are not extra independent replicates. "
                     "No warm-up discarded.\n"
                     "× = unavailable group mean; no CI = uncertainty unavailable; low-completion runs remain included.",
                     ha="center", fontsize=8)
            fig.subplots_adjust(left=0.22, right=0.97, top=0.89, bottom=0.10, hspace=0.35)
            path = Path(output) / f"horizon_{_filename(group)}_{metric}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Comparison directory containing summary.csv")
    parser.add_argument("--output", type=Path, help="Defaults to DATA/plots")
    args = parser.parse_args()
    paths = plot_results(args.data, args.output)
    print(f"Wrote {len(paths)} plots to {args.output or args.data / 'plots'}")


if __name__ == "__main__":
    main()
