"""
mixtures.py

Registry of dataset mixtures: {mixture_name: [(dataset_dir_name, sampling_weight, robot_type[, lerobot_version])]}.
`dataset_dir_name` is a sub-directory of `datasets.vla_data.data_root_dir`; `robot_type` selects the modality /
normalisation config in `data_config.ROBOT_TYPE_CONFIG_MAP`.
"""

DATASET_NAMED_MIXTURES = {
    # LIBERO (LeRobot conversions from the IPEC-COMMUNITY Hugging Face organisation); PACE trains on all four suites.
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # BridgeData V2 (OXE LeRobot conversion).
    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
}
