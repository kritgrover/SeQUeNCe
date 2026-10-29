"""Unchanged sweep values from quantum-network commit 7c57be5."""

from dataclasses import dataclass
import math


SOURCE_COMMIT = "7c57be549c275509b1674aaab3f7341c955041db"
SOURCE_URL = f"https://github.com/vlmutolo/quantum-network/tree/{SOURCE_COMMIT}/simulation-final"
SWEEP_NAMES = (
    "distillation",
    "distillation-both",
    "nodes",
    "nodes-grid",
    "finite-distillation",
    "swap-rate",
)


@dataclass(frozen=True)
class SweepCase:
    profile: str
    graph_type: str
    nodes: int
    distillation_pairs: int
    consumer_count: int
    topology_group: str
    output_csv: str
    x_column: str
    x_value: int
    generation_rate: int = 40
    swap_rate: int = 200
    time_steps: int = 5_000_000

    @property
    def source_script(self):
        return {
            "distillation": "infinite-swapping.py",
            "nodes": "simulation-vary-nodes.py",
            "nodes-grid": "simulation-vary-nodes_wrap_grid.py",
            "finite": "simulation-swap-rate.py",
        }[self.profile]


def get_cases(name):
    """Return cases in source-driver order, including shared-topology groups.

    ``distillation-both`` repeats the original distillation sweep for the two
    topologies used by its comparison plots. ``swap-rate`` exposes the source's
    commented-out sweep. The other names mirror the four drivers' active mains.
    """
    cases = []
    if name in ("distillation", "distillation-both", "finite-distillation"):
        finite = name == "finite-distillation"
        graph_types = ("wrap_grid_overlay",) if name == "distillation" else ("cycle", "wrap_grid_overlay")
        profile = "finite" if finite else "distillation"
        suffix = "pool_output" if finite else "output-1"
        for graph_type in graph_types:
            for distillation in range(1, 8 if finite else 6):
                cases.append(SweepCase(
                    profile=profile,
                    graph_type=graph_type,
                    nodes=25,
                    distillation_pairs=distillation,
                    consumer_count=35,
                    topology_group=f"{name}:{graph_type}",
                    output_csv=f"{graph_type}_distillation_metrics_{suffix}.csv",
                    x_column="distillation pairs",
                    x_value=distillation,
                ))
    elif name in ("nodes", "nodes-grid"):
        graph_types = ("cycle",) if name == "nodes" else ("cycle", "wrap_grid_overlay")
        distillations = (1,) if name == "nodes" else (1, 2)
        for distillation in distillations:
            for graph_type in graph_types:
                suffix = "output-1" if name == "nodes" else f"output-distillation-{distillation}"
                for nodes in range(20, 71, 10):
                    cases.append(SweepCase(
                        profile=name,
                        graph_type=graph_type,
                        nodes=nodes,
                        distillation_pairs=distillation,
                        consumer_count=math.floor(0.1 * nodes * (nodes - 1) / 2),
                        topology_group=f"{name}:{distillation}:{graph_type}:{nodes}",
                        output_csv=f"{graph_type}_node_metrics_{suffix}.csv",
                        x_column="nodes",
                        x_value=nodes,
                    ))
    elif name == "swap-rate":
        for graph_type in ("cycle", "wrap_grid_overlay"):
            for rate in range(150, 410, 50):
                cases.append(SweepCase(
                    profile="finite",
                    graph_type=graph_type,
                    nodes=25,
                    distillation_pairs=1,
                    consumer_count=35,
                    topology_group=f"{name}:{graph_type}",
                    output_csv=f"{graph_type}_swap_rate_metrics_pool_output.csv",
                    x_column="swap rate",
                    x_value=rate,
                    swap_rate=rate,
                ))
    else:
        raise ValueError(f"Unknown sweep {name!r}; choose from {SWEEP_NAMES}")
    return tuple(cases)
