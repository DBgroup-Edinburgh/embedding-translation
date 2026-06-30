import numpy as np

from scripts.repro.harness import hmoe_config
from embedding_translation.mapper.hmoe import HMoEMapper


def _fit(train_internal_experts):
    rng = np.random.default_rng(0)
    src = rng.normal(size=(400, 16)).astype(np.float32)
    tgt = rng.normal(size=(400, 16)).astype(np.float32)
    cfg = hmoe_config(
        num_levels=2, branch_factor=2, base_epochs=2, lora_epochs=2, inner_epochs=2,
        train_internal_experts=train_internal_experts,
    )
    m = HMoEMapper(cfg)
    m.fit(src, tgt, np.arange(400))
    return m._inner


def test_internal_nodes_get_experts_when_enabled():
    inner = _fit(True)
    ids = set(inner.lora_adapters.keys())
    internal = {n.node_id for n in inner.tree.nodes if len(n.child_ids) > 0}
    assert internal & ids, "no internal-node experts trained despite flag=True"


def test_leaf_only_when_disabled():
    inner = _fit(False)
    ids = set(inner.lora_adapters.keys())
    internal = {n.node_id for n in inner.tree.nodes if len(n.child_ids) > 0}
    leaves = {n.node_id for n in inner.tree.nodes if len(n.child_ids) == 0}
    assert not (internal & ids), "internal experts trained despite flag=False"
    assert leaves & ids, "no leaf experts trained"
