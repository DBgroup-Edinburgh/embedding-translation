"""
Tree structure for hierarchical MoE.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
import numpy as np


@dataclass
class HierarchyNode:
    """Node in the hierarchy tree."""
    node_id: int
    level: int
    parent: Optional['HierarchyNode'] = None
    children: List['HierarchyNode'] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    expert: Optional[Any] = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None


class TreeNode:
    """Simplified tree node for bottom-up hierarchical construction."""
    
    def __init__(
        self,
        node_id: int,
        level: int,
        centroid: np.ndarray
    ):
        self.node_id = node_id
        self.level = level
        self.centroid = centroid  # np.ndarray (d_in,)
        self.parent_id: Optional[int] = None
        self.child_ids: List[int] = []
        self.data_indices: Optional[np.ndarray] = None  # np.ndarray[int] - global sample indices
        self.adapter: Optional[Any] = None  # LoRA adapter or expert


class HierarchyTree:
    """Hierarchy tree structure for multi-level MoE."""
    
    def __init__(self, num_levels: int, branch_factor: int):
        self.num_levels = num_levels
        self.branch_factor = branch_factor
        self.root: Optional[HierarchyNode] = None
        self.levels: Dict[int, List[HierarchyNode]] = {}
    
    def get_leaf_nodes(self) -> List[HierarchyNode]:
        """Get all leaf nodes."""
        return [node for nodes in self.levels.values() for node in nodes if node.is_leaf()]
    
    def get_nodes_at_level(self, level: int) -> List[HierarchyNode]:
        """Get all nodes at a specific level."""
        return self.levels.get(level, [])


class BottomUpHierarchyTree:
    """Bottom-up constructed hierarchy tree using list-based storage."""
    
    def __init__(self, num_levels: int, branch_factor: int):
        self.num_levels = num_levels
        self.branch_factor = branch_factor
        self.nodes: List[TreeNode] = []  # List[TreeNode], node_id = 下标
        self.level_nodes: List[List[int]] = [[] for _ in range(num_levels)]  # 每层有哪些 node_id
        self.root_id: Optional[int] = None
    
    def get_node(self, node_id: int) -> TreeNode:
        """Get node by ID."""
        return self.nodes[node_id]
    
    def get_nodes_at_level(self, level: int) -> List[TreeNode]:
        """Get all nodes at a specific level."""
        return [self.nodes[nid] for nid in self.level_nodes[level]]
    
    def get_leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes."""
        leaf_level = self.num_levels - 1
        return [self.nodes[nid] for nid in self.level_nodes[leaf_level]]

