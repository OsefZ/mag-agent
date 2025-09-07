"""
Simple schema definitions for MAG Agent memory system.
Minimal implementation for RepoBench evaluation.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class NodeType(Enum):
    """Types of nodes in the graph."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    DESIGN_CONSTRAINT = "design_constraint"
    AGENT_ACTION = "agent_action"

class EdgeType(Enum):
    """Types of edges in the graph."""
    CONTAINS = "contains"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"
    USES = "uses"
    DESCRIBES = "describes"
    PRODUCES = "produces"
    CONTRADICTS = "contradicts"

@dataclass
class Node:
    """A node in the graph."""
    label: NodeType
    content: str
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Edge:
    """An edge in the graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}