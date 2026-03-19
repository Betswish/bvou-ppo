from beippo.config import ExperimentConfig
from beippo.modes import apply_mode


def test_apply_mode_full():
    cfg = apply_mode(ExperimentConfig(), "full")
    assert cfg.train.full_tune is True
    assert cfg.lora.enabled is False
    assert cfg.selector.enabled is False


def test_apply_mode_bvou_lora():
    cfg = apply_mode(ExperimentConfig(), "bvou_lora")
    assert cfg.train.full_tune is False
    assert cfg.lora.enabled is True
    assert cfg.selector.enabled is True
