"""Historical-workload runner persistence and stopping checks."""

import csv
from dataclasses import replace
import json

import networkx as nx
import pytest

from . import model, run
from .sweeps import SweepCase


@pytest.fixture
def direct_cases(monkeypatch):
    case = SweepCase("distillation", "cycle", 2, 1, 1, "shared", "metrics.csv", "D", 1)
    cases = (case, replace(case, distillation_pairs=2, x_value=2))
    monkeypatch.setattr(run, "get_cases", lambda name: cases)

    def create_graph(*args, **kwargs):
        graph = nx.path_graph(2)
        nx.set_edge_attributes(graph, 20, "capacity")
        return graph, {(0, 1): 3}

    monkeypatch.setattr(run, "create_graph", create_graph)
    return cases


def test_runner_persists_undefined_metrics_and_full_horizon(tmp_path, direct_cases):
    output = tmp_path / "run"
    manifest = run.run_sweep("test", output, seed=17, max_steps=3)
    assert manifest["dynamics"] == "corrected"
    assert manifest["stop_mode"] == "fixed"
    assert manifest["controlled_across_dynamics_or_D"] is False
    assert "packages" not in manifest and "python" not in manifest
    rows = list(csv.DictReader((output / "metrics.csv").open(newline="")))
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert row["executed steps"] == "3"
        assert row["stop reason"] == "fixed_horizon"
        assert row["legacy overhead"] == row["corrected overhead"] == ""
        detail = json.loads((output / f"case-{index:04d}.json").read_text())
        assert detail["metrics"]["corrected overhead"] is None
        assert detail["completed_hops"] == {"1": detail["metrics"]["total pairs consumed"]}
        assert sum(item[2] for item in detail["completed_by_pair"]) == detail["metrics"]["total pairs consumed"]
        assert detail["checkpoints"][0]["step"] == 0
        assert detail["checkpoints"][-1]["step"] == 3
        d = index + 1
        assert int(row["remaining inventory"]) == (
            int(row["initial inventory"]) + int(row["total generated"])
            - (2 * d - 1) * int(row["total swaps"]) - d * int(row["total pairs consumed"]))


def test_subset_preserves_seed_allocation_and_actual_inputs(tmp_path, direct_cases):
    full = run.run_sweep("test", tmp_path / "full", seed=31, max_steps=1)
    selected = run.run_sweep("test", tmp_path / "selected", seed=31, max_steps=1, case_indices=[1])
    assert selected["cases"][0]["simulation_seed"] == full["cases"][1]["simulation_seed"]
    assert selected["cases"][0]["metrics"] == full["cases"][1]["metrics"]
    inputs = json.loads((tmp_path / "selected" / "inputs.json").read_text())
    assert inputs["shared"]["nodes"] == [0, 1]
    assert inputs["shared"]["edges"] == [[0, 1, 20]]
    assert inputs["shared"]["consumers"] == [[0, 1, 3]]


@pytest.mark.parametrize("kwargs", [
    {"max_steps": 0}, {"case_indices": []}, {"case_indices": [0, 0]},
    {"case_indices": [2]}, {"dynamics": "invalid"}, {"stop_mode": "invalid"},
    {"seed": -1}, {"checkpoint_interval": 0},
])
def test_invalid_settings_do_not_create_output(tmp_path, direct_cases, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run.run_sweep("test", output, **kwargs)
    assert not output.exists()


def test_nonempty_output_is_never_replaced(tmp_path, direct_cases):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run.run_sweep("test", tmp_path, max_steps=1)
    assert marker.read_text() == "keep"


def test_failed_run_remains_visible_in_manifest(tmp_path, direct_cases, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    monkeypatch.setattr(model, "run_simulation", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        run.run_sweep("test", tmp_path, seed=1, max_steps=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cases"][0]["status"] == "failed"
    assert manifest["cases"][1]["status"] == "pending"
