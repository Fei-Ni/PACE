"""
Checkpoint config helpers used by `baseframework.from_pretrained`.
"""

from omegaconf import OmegaConf

from pace.model.checkpoint_io import read_checkpoint_meta


def dict_to_namespace(d):
    """Create an OmegaConf DictConfig from a plain dictionary."""
    return OmegaConf.create(d)


def read_mode_config(pretrained_checkpoint):
    """Return (config dict, dataset statistics) for any checkpoint layout (see `pace.model.checkpoint_io`)."""
    cfg, norm_stats, _ = read_checkpoint_meta(pretrained_checkpoint, require_weights=False)
    return cfg, norm_stats
