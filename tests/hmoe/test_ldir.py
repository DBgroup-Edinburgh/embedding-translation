import torch

from scripts.repro.harness import hmoe_config
from embedding_translation.mapper.hmoe.hierarchical_lora.mapper import HierarchicalLoRAMoEMapper


def test_default_dir_norm_is_fixed():
    # The fixed-scale norm makes beta=0.7 (paper) well-behaved; it must be the
    # default so mixing/chaining use it without per-call overrides.
    cfg = hmoe_config()
    assert cfg.hmoe_config.dir_norm == "fixed"


def _bare_mapper(dir_norm, dir_scale):
    m = HierarchicalLoRAMoEMapper.__new__(HierarchicalLoRAMoEMapper)
    m.dir_norm = dir_norm
    m._dir_scale = dir_scale
    return m


def test_fixed_norm_matches_formula():
    m = _bare_mapper("fixed", 2.0)
    u = torch.tensor([1.0, 0.0])
    e = torch.tensor([[3.0, 4.0]])          # off-axis energy = 25 - 9 = 16
    val = m._off_axis_penalty(e, u)
    assert abs(val.item() - (16.0 / 2.0)) < 1e-5


def test_fixed_norm_monotone_in_offaxis():
    m = _bare_mapper("fixed", 1.0)
    u = torch.tensor([1.0, 0.0])
    small = m._off_axis_penalty(torch.tensor([[5.0, 1.0]]), u)   # off = 1
    large = m._off_axis_penalty(torch.tensor([[5.0, 3.0]]), u)   # off = 9
    assert large.item() > small.item()
