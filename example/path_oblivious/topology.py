"""Generation graphs and weighted consumers from the author experiment drivers."""

import itertools
import random

import networkx as nx


INITIAL_CAPACITY = 20
GRID_DIMENSIONS = {
    20: (4, 5),
    25: (5, 5),
    30: (5, 6),
    40: (5, 8),
    50: (5, 10),
    60: (6, 10),
    70: (7, 10),
    80: (8, 10),
}


def create_graph(n, graph_type, consumer_count, *, seed=None):
    """Return the physical graph and insertion-ordered consumer weights.

    The source's plotting calls are omitted. Edge insertion, random draws,
    consumer selection, and the otherwise-unused remaining-edge shuffle retain
    the author implementation's order. The 25-node grid is exactly its 5x5 grid.
    """
    rng = random.Random(seed)
    graph = nx.Graph()

    if graph_type == "cycle":
        for i in range(n):
            graph.add_edge(i, (i + 1) % n, capacity=INITIAL_CAPACITY)
        nodes = list(range(n))
    elif graph_type == "wrap_grid_overlay":
        rows, cols = GRID_DIMENSIONS[n]
        nodes = [(i, j) for i in range(rows) for j in range(cols)]
        graph.add_nodes_from(nodes)
        edges = []
        for i in range(rows):
            for j in range(cols):
                current = (i, j)
                edges.append((current, (i, (j + 1) % cols)))
                edges.append((current, ((i + 1) % rows, j)))
        rng.shuffle(edges)
        for u, v in edges:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, capacity=INITIAL_CAPACITY)
            if nx.is_connected(graph):
                break

        # The author adds zero extra edges, but still shuffles this list.
        remaining_edges = [edge for edge in edges if not graph.has_edge(*edge)]
        rng.shuffle(remaining_edges)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    all_pairs = [
        (u, v) if u <= v else (v, u)
        for u, v in itertools.combinations(nodes, 2)
    ]
    selected = rng.sample(all_pairs, consumer_count)
    consumption_rates = {edge: rng.randint(1, 6) for edge in selected}
    return graph, consumption_rates
