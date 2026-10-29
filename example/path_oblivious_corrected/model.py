"""Local inventory balancing, with both denominator definitions on each trace.

``dynamics='original'`` preserves the copied author's transitions, draw order,
and optional stopping check. The default corrected rule performs one currently
preferred minimum-output swap per node visit. Both modes use SeQUeNCe's event
kernel; ticks are loop iterations, not calibrated physical time. D is an input
resource multiplier: a swap consumes D of each input and creates one raw pair;
a completed request consumes D raw pairs.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import itertools
import random

import networkx as nx
import numpy as np

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from example.path_oblivious.model import attempt_preferable_swap as attempt_original_swap


PROFILES = ("distillation", "nodes", "nodes-grid", "finite")
INITIAL_CAPACITY = 20
TIE_BREAKING = "first minimum in existing neighbor-set combination order"


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


def generate_corrected_array(D, max_hops):
    """Return s_D indexed by physical hop count, with s_D(1)=0.

    Only the lengths actually required by the graph are allocated. Index zero
    is a harmless zero sentinel; consumer pairs must have distinct endpoints.
    """
    if D < 1 or int(D) != D or max_hops < 0:
        raise ValueError("D must be a positive integer and max_hops nonnegative")
    costs = [0] * (max_hops + 1)
    for hops in range(2, max_hops + 1):
        costs[hops] = D * (1 + costs[hops // 2] + costs[(hops + 1) // 2])
    return costs


def overhead(swaps, denominator):
    """A zero denominator is undefined, including a positive numerator / 0."""
    return None if denominator == 0 else swaps / denominator


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
    """Execute at most one minimum-output preferred action in the current state.

    Ties retain the first candidate from the author's neighbor-set combination
    order. The outer shuffled node passes are unchanged, and every later visit
    rebuilds its candidates. This local heuristic makes no global-optimality
    claim.
    """
    candidates = get_preferable_swaps(neighbors, edge_states, x, distillation_pairs)
    if not candidates:
        return False, swap_count, total_bell_pairs
    (y, z), output = min(candidates, key=lambda item: item[1])
    xy, xz, yz = norm_edge(x, y), norm_edge(x, z), norm_edge(y, z)
    if output + 1 > min(edge_states[xy] - distillation_pairs,
                        edge_states[xz] - distillation_pairs):
        raise AssertionError("Selected swap is not currently preferable")
    edge_states[xy] -= distillation_pairs
    edge_states[xz] -= distillation_pairs
    edge_states[yz] = output + 1
    return True, swap_count + 1, total_bell_pairs - (2 * distillation_pairs - 1)


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
    corrected_direct_path_sum: int = 0
    completed_hops: dict = field(default_factory=dict)
    completed_by_pair: dict = field(default_factory=dict)
    total_generated: int = 0
    initial_inventory: int = 0
    checkpoints: list = field(default_factory=list)

    @property
    def author_metrics(self):
        return (self.total_swaps, self.optimal_direct_path_sum,
                self.worst_direct_path_sum, self.total_pairs_consumed,
                self.total_bell_pairs)

    @property
    def legacy_denominator(self):
        return self.optimal_direct_path_sum

    @property
    def corrected_denominator(self):
        return self.corrected_direct_path_sum

    @property
    def legacy_overhead(self):
        return overhead(self.total_swaps, self.legacy_denominator)

    @property
    def corrected_overhead(self):
        return overhead(self.total_swaps, self.corrected_denominator)

    @property
    def remaining_inventory(self):
        return int(sum(self.edge_states.values()))


class InventorySimulation:
    """One loop iteration per event, with optional explicit exogenous inputs.

    An ``input_source`` supplies ``generation(step)``, ``request(index)``,
    ``node_order(step, pass_index, nodes)``, ``finite_event(step)``, and
    ``finite_node(step, nodes)``. Steps start at 1; request and pass indices at 0.
    Scheduling is explicit in ``profile``: finite chooses a single generation
    or node-visit event, other profiles generate then swap to quiescence. All
    profiles drain the same blocking FIFO after that event.
    """

    def __init__(self, graph, consumption_rates, *, profile, distillation_pairs,
                 time_steps=5_000_000, generation_rate=40, swap_rate=200,
                 seed=None, trace=False, dynamics="corrected", stop_mode="author",
                 input_source=None, checkpoint_interval=100):
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        if time_steps < 0 or int(time_steps) != time_steps:
            raise ValueError("time_steps must be nonnegative")
        if (distillation_pairs < 1 or int(distillation_pairs) != distillation_pairs
                or (input_source is None and profile != "finite" and distillation_pairs > 5)):
            raise ValueError("Unsupported distillation factor for this author profile")
        if dynamics not in ("original", "corrected"):
            raise ValueError("dynamics must be 'original' or 'corrected'")
        if stop_mode not in ("author", "fixed"):
            raise ValueError("stop_mode must be 'author' or 'fixed'")
        if checkpoint_interval is not None and (checkpoint_interval < 1
                                               or int(checkpoint_interval) != checkpoint_interval):
            raise ValueError("checkpoint_interval must be a positive integer or None")
        if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
            raise ValueError("The physical graph must be simple and undirected")
        if graph.number_of_edges() == 0:
            raise ValueError("The physical graph must have at least one generation edge")
        if not consumption_rates or any(weight <= 0 for weight in consumption_rates.values()):
            raise ValueError("Consumers must have positive weights")
        if profile == "finite" and (generation_rate < 0 or swap_rate < 0
                                    or generation_rate + swap_rate == 0):
            raise ValueError("Finite event weights must be nonnegative with positive total")
        self.profile = profile
        self.dynamics = dynamics
        self.stop_mode = stop_mode
        self.input_source = input_source
        self.checkpoint_interval = checkpoint_interval
        self.distillation_pairs = distillation_pairs
        self.time_steps = time_steps
        self.generation_rate = generation_rate
        self.swap_rate = swap_rate
        self.random = random.Random(seed)
        # RandomState uses the same Poisson algorithm as np.random.poisson.
        self.numpy = np.random.RandomState(seed)
        self.edge_states = {norm_edge(u, v): graph[u][v]["capacity"]
                            for u, v in graph.edges()}
        if any(value < 0 or int(value) != value for value in self.edge_states.values()):
            raise ValueError("Initial capacities must be nonnegative integers")
        self.edges = list(graph.edges())
        self.nodes = list(graph.nodes())
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
        for u, v in consumption_rates:
            if u == v or v not in self.shortest_paths.get(u, {}):
                raise ValueError("Each consumer must connect distinct, physically connected nodes")
        self.consumption_edges = list(consumption_rates.keys())
        self.consumption_weights = list(consumption_rates.values())
        max_hops = max(self.shortest_paths[u][v] for u, v in self.consumption_edges)
        self.optimal_costs = generate_recursive_array(distillation_pairs, max_hops)
        self.worst_costs = generate_worst_recursive_array(distillation_pairs, max_hops)
        self.corrected_costs = generate_corrected_array(distillation_pairs, max_hops)
        self.total_swaps = self.optimal_direct_path_sum = 0
        self.worst_direct_path_sum = self.total_pairs_consumed = 0
        self.corrected_direct_path_sum = self.total_generated = 0
        self.initial_inventory = int(sum(self.edge_states.values()))
        # The historical aggregate assumes capacity 20, even for custom inputs.
        # Retain that quirk only for original metrics; conservation always uses
        # actual inventory, so arbitrary-capacity tests can check both meanings.
        self.total_bell_pairs = (INITIAL_CAPACITY * graph.number_of_edges()
                                 if dynamics == "original" else self.initial_inventory)
        self.completed_hops = Counter()
        self.completed_by_pair = {norm_edge(*edge): 0 for edge in self.consumption_edges}
        self.steps = 0
        self.stop_reason = "step_limit"
        self.overlay = []
        self.efficiency = []
        self.trace_enabled = trace
        self.trace = []
        self.checkpoints = []
        self.consumption_queue = deque()
        self.requests_loaded = 0
        for _ in range(1, 50000):
            u, v = self._next_request()
            self.consumption_queue.append(norm_edge(u, v))
        # stop_time is exclusive; the final original iteration must execute.
        self.timeline = Timeline(stop_time=time_steps + 1)
        self._check_inventory()
        self._checkpoint()

    def _next_request(self):
        if self.input_source is None:
            request = self.random.choices(self.consumption_edges,
                                          weights=self.consumption_weights)[0]
        else:
            request = self.input_source.request(self.requests_loaded)
            if norm_edge(*request) not in self.completed_by_pair:
                raise ValueError("Replay request is not a configured consumer")
        self.requests_loaded += 1
        return request

    def _check_inventory(self):
        expected = (self.initial_inventory + self.total_generated
                    - (2 * self.distillation_pairs - 1) * self.total_swaps
                    - self.distillation_pairs * self.total_pairs_consumed)
        actual = sum(self.edge_states.values())
        if actual != expected or any(value < 0 for value in self.edge_states.values()):
            raise AssertionError("Inventory conservation or nonnegativity violated")
        return int(actual)

    def _generate(self):
        if self.input_source is not None:
            edge, generated = self.input_source.generation(self.steps)
            edge = norm_edge(*edge)
            if edge not in self.edges and (edge[1], edge[0]) not in self.edges:
                raise ValueError("Replay generation must use a physical edge")
            if generated < 0 or int(generated) != generated:
                raise ValueError("Replay generation batch must be a nonnegative integer")
        else:
            edge = norm_edge(*self.random.choice(self.edges))
            d = self.distillation_pairs
            if self.profile == "finite":
                mean = 3 if d <= 2 else (10 if d <= 5 else 20)
            elif d == 1:
                mean = 10 if self.profile == "distillation" else 5
            else:
                mean = {2: 30, 3: 200, 4: 650, 5: 850}[d]
            generated = int(self.numpy.poisson(mean))
        self.edge_states[edge] += generated
        self.total_bell_pairs += generated
        self.total_generated += generated
        self._check_inventory()

    def _swap_at(self, node):
        neighbors = get_neighbours(self.edge_states, node)
        swap = attempt_original_swap if self.dynamics == "original" else attempt_preferable_swap
        changed, self.total_swaps, self.total_bell_pairs = swap(
            node, neighbors, self.edge_states, self.distillation_pairs,
            self.total_swaps, self.total_bell_pairs)
        self._check_inventory()
        return changed

    def _swap_to_quiescence(self):
        pass_index = 0
        while True:
            changed = False
            if self.input_source is None:
                order = self.random.sample(self.nodes, len(self.nodes))
            else:
                order = self.input_source.node_order(self.steps, pass_index, self.nodes)
                if len(order) != len(self.nodes) or set(order) != set(self.nodes):
                    raise ValueError("Replay node order must be a permutation of the nodes")
            for node in order:
                if self._swap_at(node):
                    changed = True
            if not changed:
                return
            pass_index += 1

    def _consume(self):
        while self.consumption_queue:
            if len(self.consumption_queue) == 1:
                for _ in range(50000):
                    u, v = self._next_request()
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
            self.corrected_direct_path_sum += self.corrected_costs[path_length]
            self.completed_hops[path_length] += 1
            self.completed_by_pair[edge] += 1
            self.total_pairs_consumed += 1
            self.total_bell_pairs -= self.distillation_pairs
            self._check_inventory()

    def step(self):
        self.steps += 1
        if self.profile == "finite":
            if self.input_source is None:
                event = self.random.choices(
                    ["generate", "swap"],
                    weights=[len(self.edges) * self.generation_rate,
                             self.swap_rate * len(self.nodes)])[0]
            else:
                event = self.input_source.finite_event(self.steps)
                if event not in ("generate", "swap"):
                    raise ValueError("Finite replay event must be 'generate' or 'swap'")
            if event == "generate":
                self._generate()
            else:
                node = (self.random.choice(self.nodes) if self.input_source is None
                        else self.input_source.finite_node(self.steps, self.nodes))
                if node not in self.nodes:
                    raise ValueError("Finite replay pivot is not a graph node")
                self._swap_at(node)
        else:
            self._generate()
            self._swap_to_quiescence()
        self._consume()

        interval = 20000 if self.profile == "finite" else 100
        if (self.steps % interval == 0 and self.optimal_direct_path_sum > 0
                and self.total_pairs_consumed > 0):
            self.overlay.append(self.total_swaps / self.optimal_direct_path_sum)
            self.efficiency.append(self.total_swaps / self.total_pairs_consumed)
            if (self.stop_mode == "author"
                    and is_stabilized(self.profile, self.overlay, self.efficiency,
                                      self.distillation_pairs, len(self.nodes))):
                self.stop_reason = "author_stabilization"
        if self.trace_enabled:
            self.trace.append({
                "step": self.steps,
                "author_metrics": self._metrics(),
                "edge_states": self.edge_states.copy(),
                "queue_length": len(self.consumption_queue),
                "queue_head": self.consumption_queue[0] if self.consumption_queue else None,
            })
        if self.checkpoint_interval is not None and self.steps % self.checkpoint_interval == 0:
            self._checkpoint()
        if self.stop_reason == "step_limit" and self.steps < self.time_steps:
            self.timeline.schedule(Event(self.steps + 1, Process(self, "step", []), priority=0))

    def _checkpoint(self):
        self.checkpoints.append({
            "step": self.steps,
            "total_swaps": int(self.total_swaps),
            "total_generated": int(self.total_generated),
            "total_pairs_consumed": int(self.total_pairs_consumed),
            "legacy_denominator": int(self.optimal_direct_path_sum),
            "corrected_denominator": int(self.corrected_direct_path_sum),
            "legacy_overhead": overhead(self.total_swaps, self.optimal_direct_path_sum),
            "corrected_overhead": overhead(self.total_swaps, self.corrected_direct_path_sum),
            "remaining_inventory": self._check_inventory(),
            "queue_head": self.consumption_queue[0] if self.consumption_queue else None,
        })

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
        if self.stop_mode == "fixed":
            self.stop_reason = "fixed_horizon"
        if self.checkpoints[-1]["step"] != self.steps:
            self._checkpoint()
        return SimulationResult(*self._metrics(), self.steps, self.stop_reason,
                                self.edge_states.copy(), self.trace,
                                int(self.corrected_direct_path_sum),
                                dict(self.completed_hops), self.completed_by_pair.copy(),
                                int(self.total_generated), self.initial_inventory,
                                self.checkpoints)


def run_simulation(graph, consumption_rates, *, profile, distillation_pairs,
                   time_steps=5_000_000, generation_rate=40, swap_rate=200,
                   seed=None, trace=False, dynamics="corrected", stop_mode="author",
                   input_source=None, checkpoint_interval=100):
    """Run one trajectory and compute both metrics without separate experiments."""
    simulation = InventorySimulation(
        graph, consumption_rates, profile=profile,
        distillation_pairs=distillation_pairs, time_steps=time_steps,
        generation_rate=generation_rate, swap_rate=swap_rate,
        seed=seed, trace=trace, dynamics=dynamics, stop_mode=stop_mode,
        input_source=input_source, checkpoint_interval=checkpoint_interval)
    return simulation.run()
