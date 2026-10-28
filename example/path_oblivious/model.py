"""Author-code inventory dynamics scheduled by SeQUeNCe's real event kernel.

Source: vlmutolo/quantum-network, commit
7c57be549c275509b1674aaab3f7341c955041db, simulation-final/.
The recurrence, candidate batching, queue ordering, and stopping rules below
deliberately preserve that implementation. One Timeline tick is one author
iteration; it is not a calibrated physical duration.
"""

from collections import deque
from dataclasses import dataclass, field
import itertools
import random

import networkx as nx
import numpy as np

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline


PROFILES = ("distillation", "nodes", "nodes-grid", "finite")
INITIAL_CAPACITY = 20


def norm_edge(u, v):
    return (u, v) if u <= v else (v, u)


def generate_recursive_array(D, N=50):
    """Original denominator recurrence, including its original indexing."""
    f = [0] * N
    if N > 1:
        f[1] = D
    for n in range(2, N):
        f[n] = D * (f[n // 2] + f[(n + 1) // 2])
    return f


def generate_worst_recursive_array(D, N=50):
    f = [0] * N
    if N > 1:
        f[1] = D
    for n in range(2, N):
        f[n] = D * (f[n - 1] + f[1])
    return f


def get_neighbours(edge_states, x):
    # Keep the author's set iteration order, including zero-valued dictionary keys.
    neighbors = set()
    for u, v in edge_states:
        if u == x:
            if edge_states[u, v] != 0:
                neighbors.add(v)
        elif v == x:
            if edge_states[u, v] != 0:
                neighbors.add(u)
    return neighbors


def get_preferable_swaps(neighbors, edge_states, x, distillation_pairs):
    swaps = []
    for y, z in itertools.combinations(neighbors, 2):
        c_xy = edge_states.get(norm_edge(x, y), 0)
        c_xz = edge_states.get(norm_edge(x, z), 0)
        c_yz = edge_states.get(norm_edge(y, z), 0)
        if (c_xy >= distillation_pairs and c_xz >= distillation_pairs
                and c_xy > c_yz + distillation_pairs
                and c_xz > c_yz + distillation_pairs):
            swaps.append(((y, z), c_yz))
    return swaps


def attempt_preferable_swap(x, neighbors, edge_states, distillation_pairs,
                           swap_count, total_bell_pairs):
    swaps = get_preferable_swaps(neighbors, edge_states, x, distillation_pairs)
    if not swaps:
        return False, swap_count, total_bell_pairs
    swaps_sorted = sorted(swaps, key=lambda item: item[1])
    min_c_yz = swaps_sorted[0][1]
    changed = False
    delta = 2
    swap_made = 0
    for (y, z), c_yz in swaps_sorted:
        if c_yz > min_c_yz + delta:
            break
        if swap_made >= 1:
            break
        edge_xy = norm_edge(x, y)
        edge_xz = norm_edge(x, z)
        edge_yz = norm_edge(y, z)
        if (edge_states[edge_xy] >= distillation_pairs
                and edge_states[edge_xz] >= distillation_pairs):
            edge_states[edge_xy] -= distillation_pairs
            edge_states[edge_xz] -= distillation_pairs
            edge_states[edge_yz] = edge_states.get(edge_yz, 0) + 1
            total_bell_pairs -= 2 * distillation_pairs
            total_bell_pairs += 1
            swap_count += 1
            changed = True
            # Intentionally no swap_made increment or preference recomputation.
    return changed, swap_count, total_bell_pairs


def _stable(values, window, tolerance):
    if len(values) < max(2, window):
        return False
    return bool(np.max(np.abs(np.diff(values[-window:]))) < tolerance)


def is_stabilized(profile, overlay, efficiency, distillation_pairs, nodes):
    """The separate original drivers use different convergence thresholds."""
    if profile == "finite":
        return _stable(overlay, 20, 0.04) and _stable(efficiency, 20, 0.04)
    if profile == "distillation":
        overlay_window = efficiency_window = 12
        overlay_tolerance = efficiency_tolerance = 0.5
        if distillation_pairs == 1:
            overlay_tolerance = efficiency_tolerance = 0.1
        elif distillation_pairs <= 2:
            overlay_window, overlay_tolerance = 8, 0.7
            efficiency_tolerance = 3
        elif distillation_pairs <= 3:
            overlay_window, overlay_tolerance = 8, 1
            efficiency_window, efficiency_tolerance = 7, 150
        elif distillation_pairs <= 4:
            overlay_window, overlay_tolerance = 8, 1
            efficiency_window, efficiency_tolerance = 7, 1800
        elif distillation_pairs <= 5:
            overlay_window, overlay_tolerance = 8, 1
            efficiency_window, efficiency_tolerance = 7, 18000
    elif profile == "nodes":
        overlay_window = efficiency_window = 9
        overlay_tolerance = efficiency_tolerance = 0.5
        if nodes <= 20:
            overlay_window = efficiency_window = 15
            overlay_tolerance = efficiency_tolerance = 0.1
        # These are intentionally separate if clauses, as in the source.
        if nodes <= 40:
            overlay_window = efficiency_window = 10
            overlay_tolerance = efficiency_tolerance = 0.2
        elif nodes <= 60:
            overlay_window, overlay_tolerance = 8, 0.5
            efficiency_window, efficiency_tolerance = 7, 1
        elif nodes <= 80:
            overlay_window, overlay_tolerance = 8, 0.5
            efficiency_window, efficiency_tolerance = 7, 1.5
    else:  # nodes-grid
        overlay_window = efficiency_window = 9
        overlay_tolerance = efficiency_tolerance = 0.5
        if nodes <= 30:
            overlay_window = efficiency_window = 15
            overlay_tolerance, efficiency_tolerance = 0.05, 0.1
        elif nodes <= 50:
            overlay_window = efficiency_window = 10
            overlay_tolerance, efficiency_tolerance = 0.1, 0.3
        elif nodes <= 80:
            overlay_window = efficiency_window = 10
            overlay_tolerance, efficiency_tolerance = 0.2, 1
    return (_stable(overlay, overlay_window, overlay_tolerance)
            and _stable(efficiency, efficiency_window, efficiency_tolerance))


@dataclass
class SimulationResult:
    total_swaps: int
    optimal_direct_path_sum: int
    worst_direct_path_sum: int
    total_pairs_consumed: int
    total_bell_pairs: int
    steps: int
    stop_reason: str
    edge_states: dict
    trace: list = field(default_factory=list)

    @property
    def author_metrics(self):
        return (self.total_swaps, self.optimal_direct_path_sum,
                self.worst_direct_path_sum, self.total_pairs_consumed,
                self.total_bell_pairs)


class InventorySimulation:
    """A whole original loop iteration is executed as one SeQUeNCe event."""

    def __init__(self, graph, consumption_rates, *, profile, distillation_pairs,
                 time_steps=5_000_000, generation_rate=40, swap_rate=200,
                 seed=None, trace=False):
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        if time_steps < 0:
            raise ValueError("time_steps must be nonnegative")
        if distillation_pairs < 1 or (profile != "finite" and distillation_pairs > 5):
            raise ValueError("Unsupported distillation factor for this author profile")
        self.profile = profile
        self.distillation_pairs = distillation_pairs
        self.time_steps = time_steps
        self.generation_rate = generation_rate
        self.swap_rate = swap_rate
        self.random = random.Random(seed)
        # RandomState uses the same Poisson algorithm as np.random.poisson.
        self.numpy = np.random.RandomState(seed)
        self.edge_states = {norm_edge(u, v): graph[u][v]["capacity"]
                            for u, v in graph.edges()}
        self.edges = list(graph.edges())
        self.nodes = list(graph.nodes())
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
        self.consumption_edges = list(consumption_rates.keys())
        self.consumption_weights = list(consumption_rates.values())
        self.optimal_costs = generate_recursive_array(distillation_pairs)
        self.worst_costs = generate_worst_recursive_array(distillation_pairs)
        self.total_swaps = self.optimal_direct_path_sum = 0
        self.worst_direct_path_sum = self.total_pairs_consumed = 0
        self.total_bell_pairs = INITIAL_CAPACITY * graph.number_of_edges()
        self.steps = 0
        self.stop_reason = "step_limit"
        self.overlay = []
        self.efficiency = []
        self.trace_enabled = trace
        self.trace = []
        self.consumption_queue = deque()
        for _ in range(1, 50000):
            u, v = self.random.choices(self.consumption_edges,
                                       weights=self.consumption_weights)[0]
            self.consumption_queue.append(norm_edge(u, v))
        # stop_time is exclusive; the final original iteration must execute.
        self.timeline = Timeline(stop_time=time_steps + 1)

    def _generate(self):
        edge = norm_edge(*self.random.choice(self.edges))
        d = self.distillation_pairs
        if self.profile == "finite":
            mean = 3 if d <= 2 else (10 if d <= 5 else 20)
        elif d == 1:
            mean = 10 if self.profile == "distillation" else 5
        else:
            mean = {2: 30, 3: 200, 4: 650, 5: 850}[d]
        generated = self.numpy.poisson(mean)
        self.edge_states[edge] += generated
        self.total_bell_pairs += generated

    def _swap_at(self, node):
        neighbors = get_neighbours(self.edge_states, node)
        changed, self.total_swaps, self.total_bell_pairs = attempt_preferable_swap(
            node, neighbors, self.edge_states, self.distillation_pairs,
            self.total_swaps, self.total_bell_pairs)
        return changed

    def _swap_to_quiescence(self):
        while True:
            changed = False
            for node in self.random.sample(self.nodes, len(self.nodes)):
                if self._swap_at(node):
                    changed = True
            if not changed:
                return

    def _consume(self):
        while self.consumption_queue:
            if len(self.consumption_queue) == 1:
                for _ in range(50000):
                    u, v = self.random.choices(self.consumption_edges,
                                               weights=self.consumption_weights)[0]
                    self.consumption_queue.append((u, v))
            u, v = self.consumption_queue[0]
            edge = norm_edge(u, v)
            if self.edge_states.get(edge, 0) < self.distillation_pairs:
                return
            self.consumption_queue.popleft()
            self.edge_states[edge] -= self.distillation_pairs
            path_length = self.shortest_paths[u][v]
            self.optimal_direct_path_sum += self.optimal_costs[path_length - 1]
            self.worst_direct_path_sum += self.worst_costs[path_length - 1]
            self.total_pairs_consumed += 1
            self.total_bell_pairs -= self.distillation_pairs

    def step(self):
        self.steps += 1
        if self.profile == "finite":
            event = self.random.choices(
                ["generate", "swap"],
                weights=[len(self.edges) * self.generation_rate,
                         self.swap_rate * len(self.nodes)])[0]
            if event == "generate":
                self._generate()
            else:
                # Source says `infinite_swapping = false`; interpreting the
                # spelling as False is the only runtime repair in this profile.
                self._swap_at(self.random.choice(self.nodes))
        else:
            self._generate()
            self._swap_to_quiescence()
        self._consume()

        interval = 20000 if self.profile == "finite" else 100
        if (self.steps % interval == 0 and self.optimal_direct_path_sum > 0
                and self.total_pairs_consumed > 0):
            self.overlay.append(self.total_swaps / self.optimal_direct_path_sum)
            self.efficiency.append(self.total_swaps / self.total_pairs_consumed)
            if is_stabilized(self.profile, self.overlay, self.efficiency,
                             self.distillation_pairs, len(self.nodes)):
                self.stop_reason = "author_stabilization"
        if self.trace_enabled:
            self.trace.append({
                "step": self.steps,
                "author_metrics": self._metrics(),
                "edge_states": self.edge_states.copy(),
                "queue_length": len(self.consumption_queue),
                "queue_head": self.consumption_queue[0] if self.consumption_queue else None,
            })
        if self.stop_reason == "step_limit" and self.steps < self.time_steps:
            self.timeline.schedule(Event(self.steps + 1, Process(self, "step", []), priority=0))

    def _metrics(self):
        return tuple(int(value) for value in (
            self.total_swaps, self.optimal_direct_path_sum,
            self.worst_direct_path_sum, self.total_pairs_consumed,
            self.total_bell_pairs))

    def run(self):
        if self.time_steps:
            self.timeline.schedule(Event(1, Process(self, "step", []), priority=0))
        self.timeline.init()
        self.timeline.run()
        return SimulationResult(*self._metrics(), self.steps, self.stop_reason,
                                self.edge_states.copy(), self.trace)


def run_simulation(graph, consumption_rates, *, profile, distillation_pairs,
                   time_steps=5_000_000, generation_rate=40, swap_rate=200,
                   seed=None, trace=False):
    """Run an author profile; seed=None retains the original unseeded sampling."""
    simulation = InventorySimulation(
        graph, consumption_rates, profile=profile,
        distillation_pairs=distillation_pairs, time_steps=time_steps,
        generation_rate=generation_rate, swap_rate=swap_rate,
        seed=seed, trace=trace)
    return simulation.run()
