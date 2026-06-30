import numpy as np

from embedding_translation.mapper.hmoe.tree_structure import BottomUpHierarchyTree, TreeNode
from embedding_translation.mapper.hmoe.hierarchical_lora.mapper import HierarchicalLoRAMoEMapper


def _toy_mapper(tau):
    """Hand-built 3-node tree: root(0) -> leaves L(1) at x=-1, R(2) at x=+1."""
    m = HierarchicalLoRAMoEMapper.__new__(HierarchicalLoRAMoEMapper)
    t = BottomUpHierarchyTree(num_levels=2, branch_factor=2)
    root = TreeNode(0, 0, np.array([0.0, 0.0], dtype=np.float32))
    left = TreeNode(1, 1, np.array([-1.0, 0.0], dtype=np.float32))
    right = TreeNode(2, 1, np.array([1.0, 0.0], dtype=np.float32))
    root.child_ids = [1, 2]
    t.nodes = [root, left, right]
    t.root_id = 0
    m.tree = t
    m.tau = tau
    m.distance_metric = "euclidean"
    return m


def test_confident_point_reaches_leaf():
    m = _toy_mapper(tau=0.8)
    # clearly nearer the left leaf (rho ~ 0.05 < tau) -> descend to leaf 1
    assert m._route_single(np.array([-0.9, 0.0], dtype=np.float32)) == 1


def test_ambiguous_point_stops_at_internal():
    m = _toy_mapper(tau=0.8)
    # equidistant between the two leaves (rho == 1.0 > tau) -> stop at root 0
    assert m._route_single(np.array([0.0, 0.5], dtype=np.float32)) == 0


def test_tau_one_always_reaches_leaf():
    m = _toy_mapper(tau=1.0)
    # tau=1.0 disables backoff -> even an ambiguous point descends to a leaf
    assert m._route_single(np.array([0.0, 0.5], dtype=np.float32)) in (1, 2)
