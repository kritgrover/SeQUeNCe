"""Independent small-network resource balances for the corrected LP reference."""

from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from .lp import build_lp, solve_lp, solve_problem
from . import lp


def chain(capacity=1):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, capacity, "capacity")
    return graph


def check_solution(graph, demands, d, result):
    """Reconstruct each balance independently of the sparse constraint matrix."""
    assert result.success
    assert result.max_constraint_violation <= 1e-8
    assert result.max_bound_violation <= 1e-8
    for (u, v), generated in result.generation.items():
        capacity = graph[u][v]["capacity"] if graph.has_edge(u, v) else 0
        assert -1e-8 <= generated <= capacity + 1e-8
        produced = sum(value for (pivot, y, z), value in result.swaps.items()
                       if {y, z} == {u, v})
        used = sum(value for (pivot, y, z), value in result.swaps.items()
                   if {pivot, y} == {u, v} or {pivot, z} == {u, v})
        demand = demands.get((u, v), demands.get((v, u), 0))
        expected_slack = generated + produced - d * (demand + used)
        assert expected_slack >= -1e-8
        assert result.balance_slack[u, v] == pytest.approx(expected_slack, abs=1e-8)


@pytest.mark.parametrize("d,expected", [(1, 0.1), (2, 0.2), (3, 0.3)])
def test_direct_link_throttles_to_exact_required_rate(d, expected):
    graph = nx.Graph()
    graph.add_edge("a", "b", capacity=1)
    result = solve_lp(graph, {("a", "b"): 0.1}, distillation_pairs=d)
    check_solution(graph, {("a", "b"): 0.1}, d, result)
    assert result.objective == pytest.approx(expected)
    assert result.generation["a", "b"] == pytest.approx(expected)
    assert result.swaps == {}


@pytest.mark.parametrize("d,expected_generation,expected_swaps", [(1, 0.2, 0.1),
                                                                (2, 0.8, 0.2)])
def test_two_link_chain_uses_virtual_output(d, expected_generation, expected_swaps):
    graph = chain()
    result = solve_lp(graph, {(0, 2): 0.1}, distillation_pairs=d)
    check_solution(graph, {(0, 2): 0.1}, d, result)
    assert result.objective == pytest.approx(expected_generation)
    assert result.swaps[1, 0, 2] == pytest.approx(expected_swaps)
    assert result.generation[0, 2] == 0
    assert result.generation[0, 1] == pytest.approx(expected_generation / 2)
    assert result.generation[1, 2] == pytest.approx(expected_generation / 2)


def test_coefficients_include_all_pivots_and_pair_input_costs():
    problem = build_lp(chain(), {(0, 2): 0.1}, distillation_pairs=3)
    assert problem.pairs == ((0, 1), (0, 2), (1, 2))
    assert len(problem.swaps) == 3  # All possible pivots, not only physical wedges.
    matrix = problem.matrix.toarray()
    np.testing.assert_array_equal(matrix[:, :3], -np.eye(3))
    np.testing.assert_array_equal(matrix[:, problem.swap_indices[1, 0, 2]], [3, -1, 3])
    np.testing.assert_array_equal(matrix[:, problem.swap_indices[0, 1, 2]], [3, 3, -1])
    np.testing.assert_array_equal(matrix[:, problem.swap_indices[2, 0, 1]], [-1, 3, 3])
    np.testing.assert_allclose(problem.rhs, [0, -0.3, 0])
    assert problem.bounds[problem.generation_indices[0, 2]] == (0, 0)


def test_disconnected_demand_is_infeasible():
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=1)
    graph.add_node(2)
    result = solve_lp(graph, {(0, 2): 0.1})
    assert not result.success
    assert result.status == "infeasible"
    assert result.objective is None
    assert result.generation == {}


@pytest.mark.parametrize("d,demand,feasible", [(1, 1.0, True), (1, 1.01, False),
                                             (2, 0.25, True), (2, 0.26, False)])
def test_chain_known_capacity_boundary(d, demand, feasible):
    graph = chain()
    result = solve_lp(graph, {(0, 2): demand}, distillation_pairs=d)
    assert result.success is feasible
    if feasible:
        check_solution(graph, {(0, 2): demand}, d, result)
        assert result.objective == pytest.approx(2)
    else:
        assert result.status == "infeasible"


def test_simultaneous_demands_share_input_capacity():
    graph = chain()
    demands = {(0, 1): 0.4, (0, 2): 0.6}
    result = solve_lp(graph, demands)
    check_solution(graph, demands, 1, result)
    assert result.objective == pytest.approx(1.6)
    assert not solve_lp(graph, {(0, 1): 0.4, (0, 2): 0.61}).success


def test_zero_demand_and_mixed_node_labels():
    graph = nx.Graph()
    graph.add_edge("left", (0, 1), capacity=1)
    result = solve_lp(graph, {((0, 1), "left"): 0.0})
    assert result.success
    assert result.objective == 0


@pytest.mark.parametrize("demands", [{(0, 1): -0.1}, {(0, 2): float("nan")},
                                     {(0, 0): 0.1}, {(0, 3): 0.1},
                                     {(0, 1): 0.1, (1, 0): 0.1}])
def test_invalid_demands_fail_before_solver(demands):
    with pytest.raises(ValueError):
        build_lp(chain(), demands)


@pytest.mark.parametrize("capacity", [-1, float("nan"), float("inf")])
def test_invalid_generation_capacity(capacity):
    with pytest.raises(ValueError):
        build_lp(chain(capacity), {})


def test_successful_solver_status_does_not_hide_bad_residuals():
    problem = build_lp(chain(), {(0, 2): 0.1})
    fake_result = SimpleNamespace(success=True, status=0, message="claimed optimal",
                                  x=np.zeros(len(problem.objective)))
    with patch.object(lp, "linprog", return_value=fake_result):
        result = solve_problem(problem)
    assert not result.success
    assert result.status == "numerical_failure"
    assert result.max_constraint_violation == pytest.approx(0.1)


def test_successful_solver_status_does_not_hide_capacity_violation():
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=1)
    problem = build_lp(graph, {(0, 1): 0.1})
    fake_result = SimpleNamespace(success=True, status=0, message="claimed optimal",
                                  x=np.array([2.0]))
    with patch.object(lp, "linprog", return_value=fake_result):
        result = solve_problem(problem)
    assert not result.success
    assert result.max_bound_violation == pytest.approx(1)
