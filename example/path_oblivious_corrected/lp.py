"""Minimum-generation reference for the constant-D pair-buffer abstraction.

For every unordered pair p, generation g[p] is bounded by the physical edge's
generation capacity (zero on nonedges). Each distinct pivot x and endpoint pair
(y,z) has a nonnegative swap rate sigma[x,y,z], including virtual output pairs.
A swap produces one raw output and consumes D raw pairs from each input. Fixed
completed-request rates c[p] consume D raw pairs. The constraints are

    D * (c[p] + sum(swaps using p)) <= g[p] + sum(swaps producing p).

The objective is min sum(g). Generation is controllable/throttled, and positive
balance slack is permitted as unused supply. This is a rate feasibility bound,
not a finite-buffer stationary process or a claim that the heuristic attains it.
The graph's ``capacity`` attribute means *generation rate* in this module; it
must not be confused with initial inventory in the simulation model.
"""

from dataclasses import dataclass
from itertools import combinations
import math
from numbers import Integral

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class PairBalanceLP:
    pairs: tuple
    swaps: tuple
    generation_indices: dict
    swap_indices: dict
    objective: np.ndarray
    matrix: object
    rhs: np.ndarray
    bounds: tuple
    demands: dict
    distillation_pairs: int


@dataclass(frozen=True)
class LPResult:
    success: bool
    status: str
    message: str
    objective: float | None
    generation: dict
    swaps: dict
    balance_slack: dict
    max_constraint_violation: float | None
    max_bound_violation: float | None


def _nonnegative_finite(value, label):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def build_lp(graph, demands, *, distillation_pairs=1,
             capacity_attribute="capacity"):
    """Build inspectable ``A x <= b`` coefficients for fixed demand.

    Nodes need only be hashable: canonical pairs follow graph node insertion
    order rather than requiring mutually sortable labels. Reversed demand keys
    are accepted, but specifying the same unordered pair twice is rejected.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Use a simple undirected generation graph")
    if any(u == v for u, v in graph.edges()):
        raise ValueError("Generation self-loops are unsupported")
    if (isinstance(distillation_pairs, bool)
            or not isinstance(distillation_pairs, Integral)
            or distillation_pairs < 1):
        raise ValueError("distillation_pairs must be a positive integer")
    nodes = tuple(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("At least two nodes are required")
    ranks = {node: index for index, node in enumerate(nodes)}
    pairs = tuple(combinations(nodes, 2))
    pair_rows = {pair: index for index, pair in enumerate(pairs)}

    def canonical(pair):
        if len(pair) != 2:
            raise ValueError("Demand keys must contain two endpoints")
        u, v = pair
        if u not in ranks or v not in ranks or u == v:
            raise ValueError(f"Invalid demand pair: {pair!r}")
        return (u, v) if ranks[u] < ranks[v] else (v, u)

    fixed_demands = {}
    for pair, demand in demands.items():
        pair = canonical(pair)
        if pair in fixed_demands:
            raise ValueError(f"Duplicate unordered demand pair: {pair!r}")
        fixed_demands[pair] = _nonnegative_finite(demand, "Demand")

    bounds = []
    for u, v in pairs:
        if graph.has_edge(u, v):
            if capacity_attribute not in graph[u][v]:
                raise ValueError(f"Missing {capacity_attribute!r} on edge {(u, v)!r}")
            capacity = _nonnegative_finite(graph[u][v][capacity_attribute], "Capacity")
        else:
            capacity = 0.0
        bounds.append((0.0, capacity))

    swaps = tuple((pivot, y, z) for pivot in nodes
                  for y, z in pairs if pivot != y and pivot != z)
    generation_indices = {pair: i for i, pair in enumerate(pairs)}
    swap_indices = {swap: len(pairs) + i for i, swap in enumerate(swaps)}
    objective = np.zeros(len(pairs) + len(swaps))
    objective[:len(pairs)] = 1
    rows, columns, coefficients = [], [], []
    for pair, column in generation_indices.items():
        rows.append(pair_rows[pair])
        columns.append(column)
        coefficients.append(-1.0)
    for (pivot, y, z), column in swap_indices.items():
        for pair, coefficient in (((y, z), -1.0),
                                  (canonical((pivot, y)), float(distillation_pairs)),
                                  (canonical((pivot, z)), float(distillation_pairs))):
            rows.append(pair_rows[pair])
            columns.append(column)
            coefficients.append(coefficient)
        bounds.append((0.0, None))
    matrix = coo_matrix((coefficients, (rows, columns)),
                        shape=(len(pairs), len(objective))).tocsr()
    rhs = -distillation_pairs * np.array([fixed_demands.get(pair, 0.0) for pair in pairs])
    return PairBalanceLP(pairs, swaps, generation_indices, swap_indices,
                         objective, matrix, rhs, tuple(bounds), fixed_demands,
                         int(distillation_pairs))


def solve_problem(problem, *, tolerance=1e-7):
    """Solve an assembled problem and independently check balances and bounds.

    ``success`` requires both an optimal solver status and violations no larger
    than the stated absolute tolerance. Infeasible/failed solves return no
    invented objective or variable values.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    result = linprog(problem.objective, A_ub=problem.matrix, b_ub=problem.rhs,
                     bounds=problem.bounds, method="highs")
    if not result.success:
        status = {1: "iteration_limit", 2: "infeasible", 3: "unbounded",
                  4: "solver_failure"}.get(result.status, "solver_failure")
        return LPResult(False, status, str(result.message), None, {}, {}, {}, None, None)
    values = np.asarray(result.x)
    slack = problem.rhs - problem.matrix @ values
    constraint_violation = max(0.0, float(-np.min(slack)))
    bound_violation = 0.0
    for value, (lower, upper) in zip(values, problem.bounds):
        bound_violation = max(bound_violation, lower - float(value))
        if upper is not None:
            bound_violation = max(bound_violation, float(value) - upper)
    valid = (np.all(np.isfinite(values)) and np.all(np.isfinite(slack))
             and constraint_violation <= tolerance and bound_violation <= tolerance)
    return LPResult(
        bool(valid), "optimal" if valid else "numerical_failure", str(result.message),
        float(problem.objective @ values),
        {pair: float(values[index]) for pair, index in problem.generation_indices.items()},
        {swap: float(values[index]) for swap, index in problem.swap_indices.items()},
        {pair: float(slack[index]) for index, pair in enumerate(problem.pairs)},
        constraint_violation, bound_violation,
    )


def solve_lp(graph, demands, *, distillation_pairs=1,
             capacity_attribute="capacity", tolerance=1e-7):
    """Minimize total generated raw-pair rate for fixed completed-request rates."""
    return solve_problem(build_lp(graph, demands, distillation_pairs=distillation_pairs,
                                  capacity_attribute=capacity_attribute), tolerance=tolerance)
