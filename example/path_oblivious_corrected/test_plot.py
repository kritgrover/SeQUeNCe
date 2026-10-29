"""Reporting checks: undefined groups and unavailable uncertainty stay visible."""

import csv

from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pytest

from . import plot
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


def test_dense_horizon_plots_keep_all_observations_in_bounded_images(tmp_path, monkeypatch):
    # Five million iterations with checkpoints every 100 steps previously
    # requested a 12.8-million-pixel-wide image. Exercise actual PNG rendering.
    count = 50_000
    group = ("cycle", "25", "", "", "generation-quiescent-fifo", "2")
    missing_index, spike_index = 1234, 2345  # neither is a displayed table column
    rows = []
    for index in range(count):
        for dynamics in ("original", "corrected"):
            for metric in ("corrected_overhead", "remaining_inventory"):
                missing = index == missing_index
                mean = 40 if index == spike_index else 4
                rows.append({
                    "horizon": 100 * (index + 1), "dynamics": dynamics, "metric": metric,
                    "mean": None if missing else mean,
                    "ci_low": None if missing else mean - 1,
                    "ci_high": None if missing else mean + 1,
                    "n_total": 2, "n_finished": 2, "n_defined": 0 if missing else 2,
                    "n_minimum_completed": 2,
                })
    monkeypatch.setattr(plot, "read_summary", lambda *args: {group: rows})
    original_savefig = Figure.savefig
    rendered = []

    def check_and_save(fig, path, **kwargs):
        ax, table_ax = fig.axes
        assert fig.get_figwidth() * kwargs["dpi"] <= 3200
        assert len(ax.lines) == 2
        for line in ax.lines:
            assert len(line.get_xdata()) == count
            assert np.isnan(line.get_ydata()[missing_index])
            assert line.get_ydata()[spike_index] == 40
        ticks = ax.get_xticks()
        assert len(ticks) <= 10
        assert (ticks[0], ticks[-1]) == (100, 5_000_000)
        assert 100 * (missing_index + 1) not in ticks
        assert 100 * (spike_index + 1) not in ticks
        cells = next(iter(table_ax.tables)).get_celld()
        assert len({column for _, column in cells if column >= 0}) <= 10
        # Missing off-table observations still have markers; their confidence
        # bands also keep gaps and the off-table peak rather than being sampled.
        bands = [item for item in ax.collections if hasattr(item, "get_paths")
                 and len(item.get_paths()) == 2]
        assert len(bands) == 2
        for band in bands:
            assert max(path.vertices[:, 1].max() for path in band.get_paths()) == 41
        missing_markers = [item for item in ax.collections
                           if item.get_offsets().shape == (1, 2)
                           and abs(float(item.get_offsets()[0, 0]) - 123500) < 10]
        assert len(missing_markers) == 2
        assert any("All 50,000 checkpoints shown" in text.get_text() for text in fig.texts)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title = next(text for text in fig.texts if text.get_text().startswith("Horizon sensitivity:"))
        note = next(text for text in fig.texts if "All 50,000 checkpoints shown" in text.get_text())
        assert not title.get_window_extent(renderer).overlaps(note.get_window_extent(renderer))
        original_savefig(fig, path, **kwargs)
        rendered.append(path)

    monkeypatch.setattr(Figure, "savefig", check_and_save)
    paths = plot.plot_horizon_sensitivity(tmp_path, tmp_path)
    assert paths == rendered and len(paths) == 2
    assert all(path.stat().st_size > 1000 for path in paths)


@pytest.mark.parametrize("isolated_mean", [False, True])
def test_dense_panel_preserves_isolated_observations_and_intervals(isolated_mean):
    rows = []
    for index in range(20):
        rows.append({
            "horizon": index + 1, "dynamics": "original", "metric": "corrected_overhead",
            "mean": 4 if index == 5 or not isolated_mean else None,
            "ci_low": 3 if index == 5 else None, "ci_high": 5 if index == 5 else None,
            "n_total": 2, "n_finished": 2, "n_defined": 2 if index == 5 else 0,
            "n_minimum_completed": 2,
        })
    fig, (ax, table_ax) = plt.subplots(2)
    try:
        _draw_panel(ax, table_ax, rows, (SERIES[1],), "Overhead", x_field="horizon", max_columns=10)
        assert 6 not in ax.get_xticks()  # the lone CI is deliberately off-table
        if isolated_mean:
            assert 5 in ax.lines[0].get_markevery()
        intervals = [item for item in ax.collections if isinstance(item, LineCollection)]
        assert len(intervals) == 1
        np.testing.assert_array_equal(intervals[0].get_segments(), [[[6, 3], [6, 5]]])
    finally:
        plt.close(fig)
