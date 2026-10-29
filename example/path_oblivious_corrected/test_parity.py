"""Compare the port against executable, pinned author code, not a second model.

Run from the SeQUeNCe checkout with ``python -m pytest
example/path_oblivious/test_parity.py -q``. The sibling ``quantum-network``
clone supplies the oracle; AUTHOR_REPO can point to another local clone.
Only the finite-rate source's undefined ``false`` is bound to False, matching
the documented runtime spelling accommodation in the port.
"""

import ast
from collections import deque
from functools import lru_cache, partial
import itertools
import math
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from example.path_oblivious_corrected.model import InventorySimulation as _Simulation, run_simulation as _run
from example.path_oblivious_corrected.sweeps import SOURCE_COMMIT, get_cases
from example.path_oblivious_corrected.topology import create_graph

# The added accounting must preserve the author's entire trajectory and stop rule.
InventorySimulation = partial(_Simulation, dynamics="original", stop_mode="author")
run_simulation = partial(_run, dynamics="original", stop_mode="author")


SCRIPTS = {
    "distillation": "infinite-swapping.py",
    "nodes": "simulation-vary-nodes.py",
    "nodes-grid": "simulation-vary-nodes_wrap_grid.py",
    "finite": "simulation-swap-rate.py",
}


@lru_cache(maxsize=None)
def author_source(profile):
    default = Path(__file__).resolve().parents[3] / "quantum-network"
    repo = Path(os.environ.get("AUTHOR_REPO", default)).resolve()
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
         "show", f"{SOURCE_COMMIT}:simulation-final/{SCRIPTS[profile]}"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout


def load_author(profile, seed, *, record=None, consumer_count=35):
    """Load original function bodies, excluding GUI, imports, and main()."""
    tree = ast.parse(author_source(profile))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if record is not None:
        simulation = next(node for node in functions
                          if node.name == "simulate_consumption_sequence")
        loop = next(node for node in simulation.body
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == "t")
        # Record after consumption, before the unchanged stopping check. This
        # observes every iteration, including the iteration that stops early.
        observer = ast.parse(
            "record(t, edge_states, (total_swap_count, optimal_swap_count, "
            "worst_optimal_swap_count, succeeded, total_bell_pairs), consumption_queue)"
        ).body[0]
        loop.body.insert(len(loop.body) - 1, observer)
    namespace = {
        "nx": nx, "itertools": itertools, "deque": deque, "math": math,
        "random": random.Random(seed),
        "np": SimpleNamespace(random=np.random.RandomState(seed),
                              abs=np.abs, diff=np.diff, max=np.max),
        "initial_capacity": 20,
        "edges_with_consumption_non_zero_rate": consumer_count,
        "false": False,
        "record": record,
        "print": lambda *args, **kwargs: None,
        "plt": SimpleNamespace(**{name: lambda *args, **kwargs: None
                                  for name in ("figure", "title", "axis", "tight_layout", "show")}),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 SCRIPTS[profile], "exec"), namespace)
    return namespace


def compare_trajectory(profile, graph, consumers, distillation, steps, seed=23):
    observed = []

    def record(t, state, metrics, queue):
        observed.append({
            "step": t, "author_metrics": metrics, "edge_states": state.copy(),
            "queue_length": len(queue), "queue_head": queue[0] if queue else None,
        })

    author = load_author(profile, seed, record=record)
    args = [graph.copy(), consumers.copy(), steps, 40, 200, distillation]
    if profile == "finite":
        args.append(True)  # Original function ignores this flag.
    expected = author["simulate_consumption_sequence"](*args)

    simulation = InventorySimulation(
        graph, consumers, profile=profile, distillation_pairs=distillation,
        time_steps=steps, seed=seed, trace=True,
    )
    result = simulation.run()
    assert result.author_metrics == expected
    assert result.trace == observed
    assert result.steps == len(observed)
    assert result.edge_states == observed[-1]["edge_states"]
    # Exercise the actual SeQUeNCe scheduler; no substitute event loop is used.
    assert simulation.timeline.run_counter == result.steps
    assert simulation.timeline.schedule_counter == result.steps
    return result


@pytest.mark.parametrize("profile", SCRIPTS)
@pytest.mark.parametrize("distillation", [1, 2, 4])
def test_original_and_sequence_match_every_iteration(profile, distillation):
    graph = nx.cycle_graph(6)
    nx.set_edge_attributes(graph, 20, "capacity")
    consumers = {(0, 3): 2, (1, 4): 3, (2, 5): 1, (0, 1): 4}
    compare_trajectory(profile, graph, consumers, distillation, steps=50)


@pytest.mark.parametrize("profile,distillation", [("distillation", 2), ("nodes-grid", 1)])
def test_tuple_node_grid_trajectory(profile, distillation):
    graph, consumers = create_graph(25, "wrap_grid_overlay", 35, seed=7)
    compare_trajectory(profile, graph, consumers, distillation, steps=20)


@pytest.mark.parametrize("profile", ["distillation", "nodes", "nodes-grid"])
def test_original_early_stopping_iteration_is_preserved(profile):
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 20, "capacity")
    result = compare_trajectory(profile, graph, {(0, 3): 1}, 1, steps=2500)
    assert result.steps < 2500
    assert result.stop_reason == "author_stabilization"


def test_fifo_refill_matches_original():
    # A large initial buffer reaches the original 49,999-request refill boundary
    # in one iteration. Preserve even the author's fixed initial-total accounting.
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 60000, "capacity")
    result = compare_trajectory("distillation", graph, {(0, 1): 1}, 1, steps=1)
    assert result.total_pairs_consumed > 49999


@pytest.mark.parametrize("profile,n,graph_type,consumer_count", [
    ("distillation", 25, "cycle", 35),
    ("distillation", 25, "wrap_grid_overlay", 35),
    ("nodes", 20, "cycle", 19),
    ("nodes-grid", 20, "wrap_grid_overlay", 19),
    ("nodes-grid", 70, "wrap_grid_overlay", 241),
    ("finite", 25, "wrap_grid_overlay", 35),
])
def test_topology_and_consumer_draw_order(profile, n, graph_type, consumer_count):
    author = load_author(profile, 7, consumer_count=consumer_count)
    args = [n, 20, graph_type]
    if profile in ("nodes", "nodes-grid"):
        args.append(consumer_count)
    with patch.object(nx, "draw"):
        expected_graph, expected_consumers = author["create_graph"](*args)
    graph, consumers = create_graph(n, graph_type, consumer_count, seed=7)
    assert list(graph.nodes()) == list(expected_graph.nodes())
    assert list(graph.edges(data=True)) == list(expected_graph.edges(data=True))
    assert list(consumers.items()) == list(expected_consumers.items())


def test_sweeps_preserve_original_active_and_commented_values():
    distillation = get_cases("distillation")
    assert [(c.graph_type, c.nodes, c.distillation_pairs, c.consumer_count)
            for c in distillation] == [("wrap_grid_overlay", 25, d, 35) for d in range(1, 6)]
    nodes = get_cases("nodes")
    assert [(c.graph_type, c.nodes, c.distillation_pairs) for c in nodes] == [
        ("cycle", n, 1) for n in range(20, 71, 10)]
    grid = get_cases("nodes-grid")
    assert [(c.graph_type, c.nodes, c.distillation_pairs) for c in grid] == [
        (g, n, d) for d in (1, 2) for g in ("cycle", "wrap_grid_overlay")
        for n in range(20, 71, 10)]
    for case in (*nodes, *grid):
        assert case.consumer_count == math.floor(0.1 * case.nodes * (case.nodes - 1) / 2)
    rates = get_cases("swap-rate")
    assert [(c.graph_type, c.swap_rate, c.distillation_pairs) for c in rates] == [
        (g, rate, 1) for g in ("cycle", "wrap_grid_overlay") for rate in range(150, 410, 50)]
    finite = get_cases("finite-distillation")
    assert [(c.graph_type, c.distillation_pairs) for c in finite] == [
        (g, d) for g in ("cycle", "wrap_grid_overlay") for d in range(1, 8)]
    for case in (*distillation, *nodes, *grid, *rates, *finite):
        assert case.time_steps == 5_000_000
        assert case.generation_rate == 40


def test_zero_steps_does_not_schedule_an_event():
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 20, "capacity")
    result = run_simulation(graph, {(0, 1): 1}, profile="distillation",
                            distillation_pairs=1, time_steps=0, seed=3)
    assert result.steps == 0
    assert result.author_metrics == (0, 0, 0, 0, 20)
