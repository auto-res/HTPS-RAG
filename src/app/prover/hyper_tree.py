"""Definitions of the search tree used by the prover.
"""
import math
from enum import Enum
from loguru import logger
from lean_dojo import (
    LeanError,
    TimeoutError,
    ProofGivenUp,
    ProofFinished,
    TacticResult,
    Goal,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Union
from src.app.constants import PUCT_C
from collections import deque

class HTPSCrashError(Exception):
    pass

class Status(Enum):
    """Status of a node or a proof search."""

    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.


class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def verifiability(self) -> float:
        "The probability of this node being provable."
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


@dataclass
class ProofFinishedNode(Node):
    inner: ProofFinished
    status = Status.PROVED
    verifiability = 1.0
    is_terminal = True

    def to_dict(self) -> dict:
        return {
            "is_terminal": self.is_terminal,
            "status": self.status.value
            }


@dataclass
class ErrorNode(Node):
    inner: Union[LeanError, TimeoutError, ProofGivenUp]
    status = Status.FAILED
    verifiability = 0.0
    is_terminal = True

    def to_dict(self) -> dict:
        return {
            "is_terminal": self.is_terminal,
            "status": self.status.value
            }


@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.

    Nodes are sorted by _inverse_ priority, for compatibility with the `heapq` library.
    That is, node_a < node_b is true if node_a has _higher_
    ority than node_b.
    """
    goal: Goal = field(compare=True, init=True, repr=True)

    # All edges known to lead to this node.
    # May change at any time as other nodes are explored.
    in_edges: Optional[List["Edge"]] = field(default_factory=list, init=False, compare=False, repr=False)

    # All edges out of this node that we've considered, or None for unexplored nodes.
    # When a node is explored, this list is populated, and must not change after that.
    _out_edges: Optional[List["Edge"]] = field(
        default=None, init=False, compare=False, repr=False
    )

    is_terminal = False  # type: ignore[override]

    @property
    def verifiability(self) -> float:
        if self.status == Status.PROVED:
            return 1.0
        elif self.status == Status.FAILED:
            return 0.0
        if not self.is_explored:
            return 0.5
        return max(edge.action_value for edge in self.out_edges)

    @property
    def key(self) -> str:
        return self.goal.pp

    @property
    def out_edges(self) -> Optional[list["Edge"]]:
        return self._out_edges

    # This setter implements exploring this node
    @out_edges.setter
    def out_edges(self, out_edges: Iterable["Edge"]):
        if self.is_explored:
            raise RuntimeError("Node is already explored.")

        self._out_edges = out_edges

    # A node is considered explored if we've evaluated the actor in the node to generate
    # a list of candidate children. Explored nodes are never re-searched.
    @property
    def is_explored(self) -> bool:
        return self.out_edges is not None

    _status = Status.OPEN

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    @property
    def visit_count(self) -> int:
        if not self.is_explored:
            return 0

        return sum(edge.visit_count for edge in self.out_edges)

    def to_dict(self) -> dict:
        d = {}
        t = []
        if self.is_explored:
            for edge in self.out_edges:
                edge_d = edge.to_dict()
                t.append(edge_d)

        d["out_edges"] = t
        return d


@dataclass
class Edge:
    """An edge in the search tree, representing a tactic."""

    tactic: str
    logprob: float
    dsts: list[Node]
    visit_count: int = field(default=0, init=False, compare=False, repr=False)
    total_value: float = field(default=0.0, init=False, compare=False, repr=False)

    @property
    def action_value(self) -> float:

        return self.total_value / self.visit_count if self.visit_count > 0 else 0.5

    def puct(self, parent_visit_count: int) -> float:
        return self.action_value + PUCT_C * math.exp(self.logprob) * math.sqrt(parent_visit_count) / (1 + self.visit_count)

    def update(self, value: float, proving: bool=False, invalid: bool=False) -> None:
        self.visit_count += 1
        if proving:
            self.total_value = self.visit_count
        elif invalid:
            self.total_value = 0.0
        else:
            self.total_value += value

        logger.debug(f"total_value: {self.total_value}, visit_count: {self.visit_count}")

    def to_dict(self) -> dict:
        d = {}
        d["tactic"] = self.tactic
        d["logprob"] = self.logprob
        d["visit_count"] = self.visit_count
        return d

class HyperGraph:
    def __init__(self, root: InternalNode):
        self.__nodes: dict[str, InternalNode] = {root.goal.pp: root}
        self.__root: InternalNode = root

    def __getitem__(self, key: str) -> InternalNode:
        return self.__nodes[key]

    def __setitem__(self, key, value):
        self.__nodes[key] = value

    def __len__(self) -> int:
        return len(self.__nodes)

    def __contains__(self, key: str) -> bool:
        return key in self.__nodes

    def add_node(self, node: InternalNode) -> None:
        self.__nodes[node.key] = node

    @property
    def nodes(self) -> list[InternalNode]:
        return list(self.__nodes.values())

    def render(self) -> str:
        """
        Render the search tree as a string with a form of DOT language .
        """
        internal_node_design = '{} [label="{}\nverifiability: {}", color=orange, style=filled];\n'
        def render_internal_node(node: InternalNode) -> tuple[str, str, list[Edge]]:
            """
            Args:
                node: InternalNode
            Returns:
                node_str: str
                arrow_str: str
                edges: list[Edge]
            """
            node_str = internal_node_design.format(id(node), node.goal.pp, node.verifiability)
            edges = []
            arrow_str = ""
            if node.out_edges is not None:
                for edge in node.out_edges:
                    arrow_str += f'{id(node)} -> {id(edge)};\n'
                    edges.append(edge)
            return node_str, arrow_str, edges

        def render_error_node(node: ErrorNode) -> str:
            return f'{id(node)} [label="error", color=red, style=filled];\n'

        def render_proved_node(node: ProofFinishedNode) -> str:
            return f'{id(node)} [label="proved", color=green, style=filled];\n'

        # - render_edge: edge → str, list[Node]
        edge_design = '{} [label="{}\nlogprob:{}\naction value:{}", color=lightblue, style=filled, shape=box];\n'
        def render_edge(edge: Edge) -> tuple[str, str, list[Node]]:
            edge_str = edge_design.format(id(edge), edge.tactic, edge.logprob, edge.action_value)
            arrow_str = ""

            # convert dsts
            def convert(node: Node) -> Node:
                if isinstance(node, InternalNode) and node.goal.pp in self.__nodes:
                        return self.__nodes[node.goal.pp]
                else:
                    return node

            dsts = [convert(node) for node in edge.dsts]

            for node in dsts:
                arrow_str += f'{id(edge)} -> {id(node)};\n'
            return edge_str, arrow_str, dsts
        rendered_node_ids = set()
        yet_rendered_nodes = deque()
        yet_rendered_nodes.append(self.__root)
        nodes_str = ""
        arrows_str = ""
        edges_str = ""
        while yet_rendered_nodes:
            this_node = yet_rendered_nodes.popleft()
            if isinstance(this_node, ProofFinishedNode):
                nodes_str += render_proved_node(this_node)
                continue
            elif isinstance(this_node, ErrorNode):
                nodes_str += render_error_node(this_node)
                continue

            node_str, arrow_str, edges = render_internal_node(this_node)
            nodes_str += node_str
            arrows_str += arrow_str
            rendered_node_ids.add(id(this_node))
            for edge in edges:
                edge_str, arrow_str, nodes = render_edge(edge)
                for node in nodes:
                    if not id(node) in rendered_node_ids:
                        yet_rendered_nodes.append(node)
                edges_str += edge_str
                arrows_str += arrow_str

        # concatinating
        return f"digraph Hypergraph {{\n{nodes_str}\n{edges_str}\n{arrows_str}}}"
