def get_vlm_model(config):
    """Build the VLM interface. PACE uses Qwen3-VL (base model resolved by `pace.model.checkpoint_io`)."""
    from .QWen3 import _QWen3_VL_Interface

    return _QWen3_VL_Interface(config)
