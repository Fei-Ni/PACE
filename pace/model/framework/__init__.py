"""
Framework factory.

`build_framework(cfg)` instantiates the class registered under `cfg.framework.name`. PACE is registered as
"PACE".
"""

from pace.model.tools import FRAMEWORK_REGISTRY


def build_framework(cfg):
    """
    Build a framework model from config.

    Args:
        cfg: OmegaConf config containing `framework.name`.
    Returns:
        nn.Module: instantiated framework model.
    """
    # Importing the module registers its framework names.
    import pace.model.framework.pace  # noqa: F401

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY:
        raise NotImplementedError(
            f"Framework {framework_id!r} is not implemented; available: {sorted(FRAMEWORK_REGISTRY.list())}"
        )
    return FRAMEWORK_REGISTRY[framework_id](cfg)


__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
