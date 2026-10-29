"""Mathematical regressions and trajectory checks for the isolated corrections."""

from functools import lru_cache
import itertools
import random

import networkx as nx
import numpy as np
import pytest

from example.path_oblivious.model import InventorySimulation as OriginalSimulation
from .model import (
    InventorySimulation, attempt_original_swap, attempt_preferable_swap,
    generate_corrected_array, generate_recursive_array, get_neighbours,
    get_preferable_swaps, norm_edge, overhead, run_simulation,
)


def path_graph(nodes, capacity=20):
    graph = nx.path_graph(nodes)
    nx.set_edge_attributes(graph, capacity, "capacity")
    return graph


@pytest.mark.parametrize("D,hops,legacy,corrected", [
    (1, 4, 3, 3), (2, 2, 2, 2), (2, 3, 8, 6),
    (2, 4, 20, 10), (2, 8, 104, 42), (3, 4, 63, 21),
])
def test_published_cost_regressions(D, hops, legacy, corrected):
    assert generate_recursive_array(D, hops)[hops - 1] == legacy
    assert generate_corrected_array(D, hops)[hops] == corrected


def test_d_one_cost_is_number_of_internal_swaps():
    assert generate_corrected_array(1, 150)[1:] == list(range(150))


@pytest.mark.parametrize("D", [1, 2, 3, 5])
def test_cost_matches_enumeration_of_all_binary_splits(D):
    # Independently enumerate the possible costs for every split, including
    # unbalanced trees; do not assume the recurrence's balanced split.
    @lru_cache(maxsize=None)
    def tree_costs(hops):
        if hops == 1:
            return {0}
        return {D * (1 + left + right)
                for split in range(1, hops)
                for left in tree_costs(split)
                for right in tree_costs(hops - split)}

    table = generate_corrected_array(D, 10)
    assert table[1:] == [min(tree_costs(hops)) for hops in range(1, 11)]


def test_stale_three_spoke_regression():
    before = {(0, 1): 2, (0, 2): 2, (0, 3): 2}
    original = before.copy()
    corrected = before.copy()
    assert attempt_original_swap(0, [1, 2, 3], original, 1, 0, 6) == (True, 3, 3)
    assert attempt_preferable_swap(0, [1, 2, 3], corrected, 1, 0, 6) == (True, 1, 5)
    assert attempt_preferable_swap(0, [1, 2, 3], corrected, 1, 1, 5) == (False, 1, 5)


@pytest.mark.parametrize("D", [1, 2, 3])
def test_every_action_is_currently_preferred_and_minimum(D):
    rng = random.Random(231)
    pairs = list(itertools.combinations(range(5), 2))
    for _ in range(60):
        state = {edge: rng.randrange(15) for edge in pairs}
        total = sum(state.values())
        count = 0
        for pivot in [0, 1, 2, 3, 4] * 3:
            before = state.copy()
            neighbors = get_neighbours(state, pivot)
            candidates = get_preferable_swaps(neighbors, before, pivot, D)
            changed, next_count, remaining = attempt_preferable_swap(
                pivot, neighbors, state, D, count, total)
            assert changed == bool(candidates)
            assert next_count - count == int(changed)
            assert min(state.values()) >= 0
            assert remaining == sum(state.values())
            assert total - remaining == (2 * D - 1) * int(changed)
            if changed:
                output = next(edge for edge in state if state[edge] == before.get(edge, 0) + 1)
                assert before.get(output, 0) == min(value for _, value in candidates)
                y, z = output
                assert before.get(output, 0) + 1 <= min(
                    before[norm_edge(pivot, y)] - D,
                    before[norm_edge(pivot, z)] - D)
                first_minimum = min(candidates, key=lambda candidate: candidate[1])[0]
                assert output == norm_edge(*first_minimum)
            count, total = next_count, remaining


def test_local_quiescence_does_not_imply_global_max_min_optimality():
    # A=0, B=1, C=2, D=3; neither feasible witness action is preferred.
    state = {(0, 1): 0, (0, 2): 1, (0, 3): 2,
             (1, 2): 2, (1, 3): 1, (2, 3): 2}
    for pivot in range(4):
        assert not get_preferable_swaps(get_neighbours(state, pivot), state, pivot, 1)
    assert min(state.values()) == 0
    for pivot, y, z in [(2, 0, 1), (3, 0, 2)]:
        xy, xz, yz = norm_edge(pivot, y), norm_edge(pivot, z), norm_edge(y, z)
        assert state[xy] >= 1 and state[xz] >= 1  # physical feasibility only
        state[xy] -= 1
        state[xz] -= 1
        state[yz] += 1
    assert list(state.values()) == [1] * 6


def compare_original_metric_only(profile, D, steps=40, graph=None, consumers=None):
    graph = path_graph(6) if graph is None else graph
    consumers = {(0, 5): 3, (0, 2): 1, (1, 4): 2} if consumers is None else consumers
    kwargs = dict(profile=profile, distillation_pairs=D, time_steps=steps,
                  seed=37, trace=True)
    original = OriginalSimulation(graph, consumers, **kwargs)
    metric_only = InventorySimulation(graph, consumers, dynamics="original", **kwargs)
    expected, actual = original.run(), metric_only.run()
    assert actual.author_metrics == expected.author_metrics
    assert actual.trace == expected.trace
    assert actual.edge_states == expected.edge_states
    assert (actual.steps, actual.stop_reason) == (expected.steps, expected.stop_reason)
    assert metric_only.random.getstate() == original.random.getstate()
    left, right = metric_only.numpy.get_state(), original.numpy.get_state()
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]
    assert metric_only.consumption_queue == original.consumption_queue
    assert metric_only.overlay == original.overlay
    assert metric_only.efficiency == original.efficiency
    assert sum(actual.completed_hops.values()) == actual.total_pairs_consumed
    assert sum(actual.completed_by_pair.values()) == actual.total_pairs_consumed
    assert actual.corrected_denominator == sum(
        metric_only.corrected_costs[hops] * count
        for hops, count in actual.completed_hops.items())
    assert actual.legacy_denominator == sum(
        metric_only.optimal_costs[hops - 1] * count
        for hops, count in actual.completed_hops.items())
    return actual


@pytest.mark.parametrize("profile", ["distillation", "nodes", "nodes-grid", "finite"])
@pytest.mark.parametrize("D", [1, 2, 4])
def test_metric_only_preserves_states_requests_rng_and_stopping(profile, D):
    compare_original_metric_only(profile, D)


@pytest.mark.parametrize("profile", ["distillation", "nodes", "nodes-grid"])
def test_metric_only_preserves_early_stop(profile):
    result = compare_original_metric_only(profile, 1, steps=2500,
                                          graph=path_graph(4), consumers={(0, 3): 1})
    assert result.stop_reason == "author_stabilization"
    assert result.steps < 2500


def test_metric_only_preserves_refill_and_historical_inventory_quirk():
    result = compare_original_metric_only(
        "distillation", 1, steps=1, graph=path_graph(2, 60000), consumers={(1, 0): 1})
    assert result.total_pairs_consumed > 49999
    assert result.remaining_inventory - result.total_bell_pairs == 60000 - 20


def test_tuple_nodes_metric_only():
    graph = nx.grid_2d_graph(3, 4)
    nx.set_edge_attributes(graph, 20, "capacity")
    compare_original_metric_only("nodes-grid", 2, graph=graph,
                                  consumers={((0, 0), (2, 3)): 1, ((1, 0), (0, 3)): 4})


@pytest.mark.parametrize("D", [1, 2, 4])
@pytest.mark.parametrize("profile", ["distillation", "finite"])
def test_run_inventory_conservation_and_checkpoints(D, profile):
    result = run_simulation(path_graph(5, 7), {(0, 4): 1, (0, 1): 2},
                            profile=profile, distillation_pairs=D, time_steps=43,
                            stop_mode="fixed", seed=51, checkpoint_interval=10)
    assert result.stop_reason == "fixed_horizon"
    assert result.steps == 43
    assert [point["step"] for point in result.checkpoints] == [0, 10, 20, 30, 40, 43]
    for point in result.checkpoints:
        assert point["remaining_inventory"] == (
            result.initial_inventory + point["total_generated"]
            - (2 * D - 1) * point["total_swaps"] - D * point["total_pairs_consumed"])
    assert result.remaining_inventory == result.total_bell_pairs
    assert result.initial_inventory == 28
    assert min(result.edge_states.values()) >= 0


def test_fixed_horizon_continues_past_author_stopping():
    graph = path_graph(4)
    original = run_simulation(graph, {(0, 3): 1}, profile="distillation",
                              distillation_pairs=1, dynamics="original", seed=37,
                              time_steps=2500)
    fixed = run_simulation(graph, {(0, 3): 1}, profile="distillation",
                           distillation_pairs=1, dynamics="original", seed=37,
                           stop_mode="fixed", time_steps=2500)
    assert original.steps < fixed.steps == 2500
    assert fixed.stop_reason == "fixed_horizon"


def test_cost_tables_cover_long_physical_paths():
    simulation = InventorySimulation(path_graph(62), {(0, 61): 1},
                                     profile="distillation", distillation_pairs=2,
                                     time_steps=0, seed=4)
    assert len(simulation.optimal_costs) == len(simulation.worst_costs) == 61
    assert len(simulation.corrected_costs) == 62
    assert simulation.corrected_costs[61] == 2 * (
        1 + simulation.corrected_costs[30] + simulation.corrected_costs[31])
    # Seed a virtual buffer in this unit fixture so consumption actually indexes
    # the long-path table; this is not a physical-generation transition.
    simulation.edge_states[(0, 61)] = 2
    simulation.initial_inventory += 2
    simulation.total_bell_pairs += 2
    simulation._consume()
    assert simulation.total_pairs_consumed == 1
    assert simulation.completed_hops == {61: 1}
    assert simulation.corrected_direct_path_sum == simulation.corrected_costs[61]


@pytest.mark.parametrize("steps", [0, 2])
def test_undefined_ratios_keep_raw_totals(steps):
    result = run_simulation(path_graph(2), {(0, 1): 1}, profile="distillation",
                            distillation_pairs=1, time_steps=steps,
                            stop_mode="fixed", seed=1)
    assert result.legacy_denominator == result.corrected_denominator == 0
    assert result.legacy_overhead is None and result.corrected_overhead is None
    assert result.total_pairs_consumed == 0 if steps == 0 else result.total_pairs_consumed > 0
    assert result.stop_reason == "fixed_horizon"
    assert overhead(10, 0) is None
    assert overhead(0, 0) is None


def test_invalid_consumer_and_capacity_rejected():
    with pytest.raises(ValueError, match="distinct"):
        InventorySimulation(path_graph(2), {(0, 0): 1}, profile="distillation",
                            distillation_pairs=1)
    with pytest.raises(ValueError, match="capacities"):
        InventorySimulation(path_graph(2, -1), {(0, 1): 1}, profile="distillation",
                            distillation_pairs=1)
