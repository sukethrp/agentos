from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
import networkx as nx


class WorkflowDAG:
    def __init__(self, nodes: list[dict], edges: list[dict]):
        self._nodes = {n["id"]: n for n in nodes}
        self._edges = edges
        self._graph = nx.DiGraph()
        for n in nodes:
            self._graph.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
        for e in edges:
            attrs = {}
            if "condition_expr" in e:
                attrs["condition_expr"] = e["condition_expr"]
            self._graph.add_edge(e["source"], e["target"], **attrs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowDAG":
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        return cls(nodes=nodes, edges=edges)

    def nodes(self) -> list[str]:
        return list(self._graph.nodes())

    def node_data(self, node_id: str) -> dict:
        return dict(self._graph.nodes[node_id])

    def predecessors(self, node_id: str) -> list[str]:
        return list(self._graph.predecessors(node_id))

    def successors(self, node_id: str) -> list[str]:
        return list(self._graph.successors(node_id))

    def edge_data(self, u: str, v: str) -> dict:
        return dict(self._graph.edges[u, v])

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._graph))

    def roots(self) -> list[str]:
        return [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]
