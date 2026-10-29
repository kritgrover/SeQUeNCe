"""Reporting checks: undefined groups and unavailable uncertainty stay visible."""

import csv

import matplotlib.pyplot as plt
import pytest

from .plot import RAW_METRICS, SERIES, _draw_panel, plot_results, read_summary


def make_summary(path):
    rows = []
    for d in (1, 2):
        for dynamics in ("original", "corrected"):
            for metric in ("legacy_overhead", "corrected_overhead", *(m for m, _ in RAW_METRICS)):
                undefined = d == 2 and "overhead" in metric
                rows.append({
                    "graph_type": "cycle", "nodes": 6, "rows": "", "columns": "",
                    "scheduling": "quiescent", "horizon": 100,
                    "distillation_pairs": d, "dynamics": dynamics, "metric": metric,
                    "n_total": 2, "n_finished": 2,
                    "n_defined": 0 if undefined else 2,
                    "n_minimum_completed": 0 if undefined else 2,
                    "mean": "" if undefined else 4,
                    "ci_low": "" if undefined else 1,
                    "ci_high": "" if undefined else 7,
                    "status": "undefined" if undefined else "complete",
                })
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_undefined_groups_remain_in_plot_and_count_table(tmp_path):
    make_summary(tmp_path)
    groups = read_summary(tmp_path)
    assert len(groups) == 1
    rows = next(iter(groups.values()))
    assert len(rows) == 24
    assert all(row["mean"] is None for row in rows
               if row["distillation_pairs"] == 2 and "overhead" in row["metric"])
    fig, (ax, table_ax) = plt.subplots(2)
    _draw_panel(ax, table_ax, rows, SERIES, "Overhead")
    assert len(ax.get_legend_handles_labels()[1]) == 4
    assert sum(text.get_text() == "×" for text in ax.texts) == 4
    texts = [cell.get_text().get_text() for table in table_ax.tables
             for cell in table.get_celld().values()]
    assert sum("0/2 defined" in text and "undefined" in text for text in texts) == 4
    plt.close(fig)
    paths = plot_results(tmp_path)
    assert len(paths) == 5
    assert all(path.stat().st_size > 1000 for path in paths)


def test_missing_required_columns_are_rejected(tmp_path):
    (tmp_path / "summary.csv").write_text("metric,mean\nlegacy_overhead,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing summary columns"):
        read_summary(tmp_path)


def test_horizon_plots_use_cross_replicate_checkpoint_summaries(tmp_path):
    rows = make_summary(tmp_path)
    checkpoint_rows = []
    for horizon in (10, 100):
        checkpoint_rows.extend({**row, "horizon": horizon} for row in rows)
    with (tmp_path / "horizon_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    paths = plot_results(tmp_path)
    assert len(paths) == 9  # five primary plots plus two horizon metrics for each D
    assert sum(path.name.startswith("horizon_") for path in paths) == 4
