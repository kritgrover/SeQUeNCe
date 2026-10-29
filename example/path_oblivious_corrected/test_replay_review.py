"""Replay regression checks with real dynamics and distinct FIFO refill counts."""

import networkx as nx
import pytest

from .model import InventorySimulation
from .replay import ReplayInputs, SCHEDULES
from .topology import create_graph


def execute(tape, dynamics, D):
    simulation = InventorySimulation(
        tape.graph, tape.consumers,
        profile="finite" if tape.scheduling == SCHEDULES[1] else "distillation",
        distillation_pairs=D, time_steps=tape.horizon, stop_mode="fixed",
        dynamics=dynamics, input_source=tape, checkpoint_interval=1, trace=True)
    result = simulation.run()
    return simulation, result


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_real_tuple_grid_trace_replays_in_reverse_run_order(tmp_path, schedule):
    graph, consumers = create_graph(12, "wrap_grid_overlay", 9,
                                     rows=3, columns=4, seed=71)
    tape = ReplayInputs(graph, consumers, horizon=12, batch_mean=6,
                        seed=711, scheduling=schedule)
    expected = {}
    for D in (1, 2):
        for dynamics in ("original", "corrected"):
            _, expected[D, dynamics] = execute(tape, dynamics, D)
    tape.save(tmp_path / "grid.json")
    replay = ReplayInputs.load(tmp_path / "grid.json")
    assert list(replay.graph.nodes()) == list(graph.nodes())
    assert list(replay.graph.edges(data=True)) == list(graph.edges(data=True))
    assert list(replay.consumers.items()) == list(consumers.items())
    for D, dynamics in reversed(expected):
        _, actual = execute(replay, dynamics, D)
        assert actual == expected[D, dynamics]
    assert replay.frozen


def test_different_fifo_refills_replay_exactly_without_shifting_generation(tmp_path):
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 60000, "capacity")
    # Reverse orientation also exercises original refill's raw pair ordering.
    tape = ReplayInputs(graph, {(1, 0): 1}, horizon=2, batch_mean=4, seed=118)
    first_simulation, first = execute(tape, "original", 1)
    second_simulation, second = execute(tape, "corrected", 3)
    assert first_simulation.requests_loaded == 99999
    assert second_simulation.requests_loaded == 49999
    assert first.total_pairs_consumed > 49999 > second.total_pairs_consumed
    assert first.total_generated == second.total_generated == sum(batch for _, batch in tape.generations)
    tape.save(tmp_path / "refill.json")
    replay = ReplayInputs.load(tmp_path / "refill.json")
    # Replay the smaller prefix first; the large run must still receive the
    # entire saved request sequence and identical generation observations.
    replay_second_simulation, replay_second = execute(replay, "corrected", 3)
    replay_first_simulation, replay_first = execute(replay, "original", 1)
    assert replay_first == first
    assert replay_second == second
    assert replay_first_simulation.consumption_queue == first_simulation.consumption_queue
    assert replay_second_simulation.consumption_queue == second_simulation.consumption_queue
