"""Controlled-input, numerical-output, and common-horizon integration checks."""

import csv
import json
import math

import pytest

from .compare import run_comparison, summarize
from .model import InventorySimulation, generate_corrected_array, generate_recursive_array
from .replay import ReplayInputs
from .topology import create_graph


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("scheduling", ["generation-quiescent-fifo", "finite-event-fifo"])
def test_common_inputs_metrics_and_saved_replay(tmp_path, scheduling):
    output = tmp_path / "comparison"
    manifest = run_comparison(output, horizon=24, distillations=(1, 2), replicates=2,
                              nodes=6, graph_type="cycle", consumer_count=4,
                              consumer_policy="custom", scheduling=scheduling,
                              minimum_completed=3, checkpoint_interval=6)
    assert manifest["status"] == "completed"
    results = read_csv(output / "results.csv")
    assert len(results) == 8
    for replicate in range(2):
        tape = ReplayInputs.load(output / "inputs" / f"input-{replicate:03d}.json")
        assert len(tape.requests) >= 49999
        expected_generated = sum(batch for step, (_, batch) in enumerate(tape.generations)
                                 if not tape.events or tape.events[step] == "generate")
        for row in [r for r in results if int(r["replicate"]) == replicate]:
            d = int(row["distillation_pairs"])
            assert row["stop_reason"] == "fixed_horizon"
            assert int(row["steps"]) == 24
            assert int(row["total_generated"]) == expected_generated
            assert int(row["remaining_inventory"]) == (
                int(row["initial_inventory"]) + expected_generated
                - (2*d - 1)*int(row["total_swaps"]) - d*int(row["total_pairs_consumed"]))
            details = json.loads((output / "runs" / f"r{replicate:03d}-d{d}-{row['dynamics']}.json").read_text())
            histogram = {int(h): count for h, count in details["completed_hops"].items()}
            max_hops = max(histogram, default=1)
            legacy = generate_recursive_array(d, max_hops)
            corrected = generate_corrected_array(d, max_hops)
            assert int(row["legacy_denominator"]) == sum(legacy[h-1]*n for h, n in histogram.items())
            assert int(row["corrected_denominator"]) == sum(corrected[h]*n for h, n in histogram.items())
            assert sum(histogram.values()) == int(row["total_pairs_consumed"])
            assert sum(item[2] for item in details["completed_by_pair"]) == int(row["total_pairs_consumed"])
            assert len(details["completed_by_pair"]) == 4
    replay_output = tmp_path / "replayed"
    run_comparison(replay_output, horizon=24, distillations=(1, 2), seed=999999,
                   replay=output / "inputs", minimum_completed=3, checkpoint_interval=6)
    assert read_csv(replay_output / "results.csv") == results
    assert read_csv(replay_output / "checkpoints.csv") == read_csv(output / "checkpoints.csv")
    assert read_csv(replay_output / "summary.csv") == read_csv(output / "summary.csv")


def test_saved_grid_labels_and_missing_inputs_fail_instead_of_resampling(tmp_path):
    graph, consumers = create_graph(6, "wrap_grid_overlay", 3, seed=4, rows=2, columns=3)
    tape = ReplayInputs(graph, consumers, horizon=2, batch_mean=10, seed=7)
    requests = [tape.request(i) for i in range(9)]
    order = tape.node_order(1, 0, list(graph.nodes()))
    tape.save(tmp_path / "inputs.json")
    loaded = ReplayInputs.load(tmp_path / "inputs.json")
    assert list(loaded.graph.nodes()) == list(graph.nodes())
    assert list(loaded.graph.edges(data=True)) == list(graph.edges(data=True))
    assert loaded.consumers == consumers
    assert [loaded.request(i) for i in range(9)] == requests
    assert loaded.node_order(1, 0, list(graph.nodes())) == order
    with pytest.raises(ValueError, match="request tape exhausted"):
        loaded.request(9)
    with pytest.raises(ValueError, match="visitation tape exhausted"):
        loaded.node_order(1, 1, list(graph.nodes()))


def test_divergent_visits_and_request_reads_cannot_shift_generation():
    graph, consumers = create_graph(6, "cycle", 3, seed=4)
    tape = ReplayInputs(graph, consumers, horizon=4, batch_mean=10, seed=7)
    before = [tape.generation(i) for i in range(1, 5)]
    first = tape.node_order(2, 0, list(graph.nodes()))
    for index in range(100):
        tape.request(index)
        tape.node_order(1, index, list(graph.nodes()))
    assert [tape.generation(i) for i in range(1, 5)] == before
    assert tape.node_order(2, 0, list(graph.nodes())) == first


def test_fixed_horizon_ignores_author_plateau(monkeypatch):
    from . import model
    monkeypatch.setattr(model, "is_stabilized", lambda *args: True)
    graph, consumers = create_graph(4, "cycle", 6, seed=2)
    tape = ReplayInputs(graph, consumers, horizon=120, batch_mean=10, seed=10)
    for dynamics in ("original", "corrected"):
        simulation = InventorySimulation(graph, consumers, profile="distillation",
                                         distillation_pairs=1, time_steps=120,
                                         input_source=tape, dynamics=dynamics, stop_mode="fixed")
        result = simulation.run()
        assert result.steps == 120
        assert result.legacy_denominator > 0
        assert result.stop_reason == "fixed_horizon"


def test_undefined_metrics_and_low_counts_are_not_zero_or_filtered(tmp_path):
    output = tmp_path / "direct"
    run_comparison(output, horizon=2, distillations=(1,), replicates=2,
                   graph_type="cycle", nodes=2, consumer_policy="custom", consumer_count=1,
                   minimum_completed=1000)
    rows = read_csv(output / "results.csv")
    assert all(r["legacy_overhead"] == r["corrected_overhead"] == "" for r in rows)
    assert all(r["status"] == "insufficient_completions" for r in rows)
    for row in read_csv(output / "summary.csv"):
        assert row["n_total"] == "2"
        assert row["n_minimum_completed"] == "0"
        if "overhead" in row["metric"]:
            assert row["mean"] == row["ci_low"] == row["ci_high"] == ""
            assert row["n_defined"] == "0"
        else:
            assert row["mean"] != ""


def test_failed_runs_remain_in_output_and_invalidate_summary(tmp_path, monkeypatch):
    original_run = InventorySimulation.run
    calls = 0

    def fail_once(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            self.step()
            raise RuntimeError("deliberate test failure")
        return original_run(self)

    monkeypatch.setattr(InventorySimulation, "run", fail_once)
    output = tmp_path / "failed"
    manifest = run_comparison(output, horizon=2, distillations=(1,), replicates=2,
                              graph_type="cycle", nodes=4, consumer_policy="custom", consumer_count=2)
    assert manifest["status"] == "completed_with_failures"
    assert manifest["failed_runs"] == 1
    results = read_csv(output / "results.csv")
    assert len(results) == 4
    assert results[0]["status"] == "failed"
    assert results[0]["steps"] == "1"
    assert int(results[0]["total_generated"]) > 0
    assert results[0]["remaining_inventory"] != ""
    for row in read_csv(output / "summary.csv"):
        if row["dynamics"] == "original":
            assert row["n_total"] == "2"
            assert row["n_finished"] == "1"
            assert row["mean"] == ""
    for row in read_csv(output / "paired_summary.csv"):
        assert row["n_total"] == "2"
        assert row["mean"] == ""


def test_confidence_intervals_use_replicates_not_checkpoints():
    common = dict(graph_type="cycle", nodes=4, rows=None, columns=None,
                  scheduling="generation-quiescent-fifo", distillation_pairs=1,
                  horizon=100, dynamics="original", finished=True, minimum_completed_met=True)
    result = summarize([{**common, "legacy_overhead": value} for value in (1, 2, 3)])
    row = next(r for r in result if r["metric"] == "legacy_overhead")
    assert row["n_total"] == row["n_defined"] == 3
    assert row["mean"] == 2
    assert row["ci_low"] == pytest.approx(2 - 4.302652729696142 / math.sqrt(3))


def test_rate_controls_and_invalid_grid_rejected(tmp_path):
    with pytest.raises(ValueError, match="Rate arguments"):
        run_comparison(tmp_path / "rates", horizon=2, generation_rate=40)
    with pytest.raises(ValueError, match="rows.*columns"):
        create_graph(12, "wrap_grid_overlay", 3, rows=3, columns=5)
    with pytest.raises(ValueError, match="both rows"):
        create_graph(12, "wrap_grid_overlay", 3, rows=3)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        run_comparison(output, horizon=2)
    assert (output / "existing.txt").read_text() == "preserve"


def test_heterogeneous_replay_workloads_cannot_be_pooled(tmp_path):
    graph, consumers = create_graph(6, "cycle", 3, seed=4)
    saved = tmp_path / "inputs"
    saved.mkdir()
    for index, batch_mean in enumerate((10, 20)):
        tape = ReplayInputs(graph, consumers, horizon=2, batch_mean=batch_mean, seed=7)
        tape.save(saved / f"input-{index:03d}.json")
    with pytest.raises(ValueError, match="one workload specification"):
        run_comparison(tmp_path / "out", horizon=2, replay=saved)
