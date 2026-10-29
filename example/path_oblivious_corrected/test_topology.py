"""Uniform physical-edge sampling and historical topology compatibility."""

import random

import networkx as nx
import pytest

from example.path_oblivious.topology import create_graph as create_original_graph

from .topology import GRID_DIMENSIONS, create_graph


@pytest.mark.parametrize("rows,columns", [(2, 3), (3, 2), (2, 2), (2, 25), (25, 2)])
def test_two_wide_grids_shuffle_each_physical_edge_once(monkeypatch, rows, columns):
    candidates = []
    original_shuffle = random.Random.shuffle

    def capture_shuffle(rng, values):
        if not candidates:
            candidates.extend(values)
        original_shuffle(rng, values)

    monkeypatch.setattr(random.Random, "shuffle", capture_shuffle)
    graph, consumers = create_graph(rows * columns, "wrap_grid_overlay", 1,
                                    rows=rows, columns=columns, seed=7)

    # NetworkX supplies the distinct physical links independently of our
    # candidate-building loop. Sampling a permutation of this set is uniform.
    physical = {frozenset(edge) for edge in
                nx.grid_2d_graph(rows, columns, periodic=True).edges()}
    assert {frozenset(edge) for edge in candidates} == physical
    assert len(candidates) == len(physical)
    assert nx.is_connected(graph)
    assert {frozenset(edge) for edge in graph.edges()} <= physical
    assert all(data["capacity"] == 20 for _, _, data in graph.edges(data=True))
    assert len(consumers) == 1


@pytest.mark.parametrize("seed", [7, 271])
@pytest.mark.parametrize("nodes", sorted(GRID_DIMENSIONS))
@pytest.mark.parametrize("graph_type", ["cycle", "wrap_grid_overlay"])
def test_historical_topology_and_consumers_keep_seeded_order(nodes, graph_type, seed):
    expected_graph, expected_consumers = create_original_graph(
        nodes, graph_type, 35, seed=seed)
    graph, consumers = create_graph(nodes, graph_type, 35, seed=seed)
    assert list(graph.nodes()) == list(expected_graph.nodes())
    assert list(graph.edges(data=True)) == list(expected_graph.edges(data=True))
    assert list(consumers.items()) == list(expected_consumers.items())
