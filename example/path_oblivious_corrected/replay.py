"""Shared, serializable exogenous inputs for controlled comparisons.

Generation and finite events are recorded before running either algorithm.
Requests are a shared cached sequence, indexed by FIFO position, and visitation
orders are cached by (iteration, pass). Extra passes cannot shift future inputs.
This stream organization is deliberately distinct from literal author sampling.
"""

import json
import math
from pathlib import Path
import random

import networkx as nx
import numpy as np

from .model import norm_edge


SCHEDULES = ("generation-quiescent-fifo", "finite-event-fifo")


def _node(value):
    return tuple(_node(x) for x in value) if isinstance(value, list) else value


class ReplayInputs:
    def __init__(self, graph, consumers, *, horizon, batch_mean, seed,
                 scheduling="generation-quiescent-fifo", generation_rate=None,
                 swap_rate=None, metadata=None):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not math.isfinite(batch_mean) or batch_mean < 0:
            raise ValueError("batch_mean must be finite and nonnegative")
        if scheduling not in SCHEDULES:
            raise ValueError("Unsupported scheduling mode")
        if scheduling == SCHEDULES[0] and (generation_rate is not None or swap_rate is not None):
            raise ValueError("Rate controls do not apply to generation-quiescent-fifo")
        if not graph.number_of_edges() or not consumers:
            raise ValueError("Need generation edges and consumers")
        self.graph = graph.copy()
        self.consumers = dict(consumers)
        self.nodes = list(graph.nodes())
        self.edges = [norm_edge(u, v) for u, v in graph.edges()]
        self.consumer_edges = list(consumers)
        self.consumer_weights = list(consumers.values())
        if any(not math.isfinite(w) or w <= 0 for w in self.consumer_weights):
            raise ValueError("Consumer weights must be positive and finite")
        self.horizon = horizon
        self.scheduling = scheduling
        self.batch_mean = batch_mean
        self.seed = seed
        self.metadata = metadata or {}
        self.generation_rate = generation_rate
        self.swap_rate = swap_rate
        master = random.Random(seed)
        self.stream_seeds = {key: master.randrange(2**32) for key in
                             ("edges", "batches", "requests", "events", "finite_nodes", "visits")}
        edge_rng = random.Random(self.stream_seeds["edges"])
        batches = np.random.RandomState(self.stream_seeds["batches"]).poisson(batch_mean, horizon)
        self.generations = [(edge_rng.randrange(len(self.edges)), int(batch)) for batch in batches]
        if scheduling == SCHEDULES[1]:
            generation_rate = 40 if generation_rate is None else generation_rate
            swap_rate = 200 if swap_rate is None else swap_rate
            if (not all(math.isfinite(v) and v >= 0 for v in (generation_rate, swap_rate))
                    or generation_rate + swap_rate <= 0):
                raise ValueError("Finite rates must be nonnegative with positive total")
            self.generation_rate, self.swap_rate = generation_rate, swap_rate
            event_rng = random.Random(self.stream_seeds["events"])
            self.events = event_rng.choices(
                ["generate", "swap"], weights=[len(self.edges) * generation_rate,
                                                len(self.nodes) * swap_rate], k=horizon)
            finite_rng = random.Random(self.stream_seeds["finite_nodes"])
            self.finite_nodes = [finite_rng.randrange(len(self.nodes)) for _ in range(horizon)]
        else:
            self.events, self.finite_nodes = [], []
        self.requests = []
        self.visit_orders = {}
        self.request_rng = random.Random(self.stream_seeds["requests"])
        self.frozen = False

    def generation(self, step):
        edge, batch = self.generations[step - 1]
        return self.edges[edge], batch

    def request(self, index):
        if index < 0:
            raise IndexError(index)
        if self.frozen and index >= len(self.requests):
            raise ValueError("Saved request tape exhausted; rerun the original specification")
        while index >= len(self.requests):
            self.requests.append(self.request_rng.choices(
                range(len(self.consumer_edges)), weights=self.consumer_weights)[0])
        return self.consumer_edges[self.requests[index]]

    def node_order(self, step, pass_index, nodes):
        if list(nodes) != self.nodes:
            raise ValueError("Replay node order does not match graph")
        key = f"{step}:{pass_index}"
        if key not in self.visit_orders:
            if self.frozen:
                raise ValueError(f"Saved visitation tape exhausted at {key}")
            rng = random.Random(f"{self.stream_seeds['visits']}:{key}")
            self.visit_orders[key] = rng.sample(range(len(self.nodes)), len(self.nodes))
        return [self.nodes[i] for i in self.visit_orders[key]]

    def finite_event(self, step):
        return self.events[step - 1]

    def finite_node(self, step, nodes):
        if list(nodes) != self.nodes:
            raise ValueError("Replay node order does not match graph")
        return self.nodes[self.finite_nodes[step - 1]]

    def to_dict(self):
        rank = {node: i for i, node in enumerate(self.nodes)}
        return {
            "format": "path-oblivious-inputs-v1", "horizon": self.horizon,
            "scheduling": self.scheduling, "batch_mean": self.batch_mean,
            "generation_rate": self.generation_rate, "swap_rate": self.swap_rate,
            "seed": self.seed, "stream_seeds": self.stream_seeds, "metadata": self.metadata,
            "nodes": self.nodes,
            "edges": [[rank[u], rank[v], int(self.graph[u][v]["capacity"])] for u, v in self.edges],
            "consumers": [[rank[u], rank[v], weight] for (u, v), weight in self.consumers.items()],
            "generations": self.generations, "requests": self.requests,
            "events": self.events, "finite_nodes": self.finite_nodes,
            "visit_orders": self.visit_orders,
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False) + "\n",
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Load recorded values only. Missing draws fail instead of resampling."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["format"] != "path-oblivious-inputs-v1":
            raise ValueError("Unsupported input tape format")
        obj = cls.__new__(cls)
        obj.nodes = [_node(n) for n in data["nodes"]]
        obj.graph = nx.Graph()
        obj.graph.add_nodes_from(obj.nodes)
        obj.edges = []
        for u, v, capacity in data["edges"]:
            edge = norm_edge(obj.nodes[u], obj.nodes[v])
            obj.edges.append(edge)
            obj.graph.add_edge(*edge, capacity=capacity)
        obj.consumers = {(obj.nodes[u], obj.nodes[v]): w for u, v, w in data["consumers"]}
        obj.consumer_edges = list(obj.consumers)
        obj.consumer_weights = list(obj.consumers.values())
        for key in ("horizon", "scheduling", "batch_mean", "generation_rate", "swap_rate",
                    "seed", "stream_seeds", "metadata", "generations", "requests", "events",
                    "finite_nodes", "visit_orders"):
            setattr(obj, key, data[key])
        obj.request_rng = None
        obj.frozen = True
        return obj
