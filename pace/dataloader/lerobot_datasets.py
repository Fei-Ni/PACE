from pathlib import Path

from pace.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from pace.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from pace.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from pace.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    lerobot_version: str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset (sub-directory of data_root_dir).
    :param robot_type: The robot type config to use (key of ROBOT_TYPE_CONFIG_MAP).
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    video_backend = data_cfg.get("video_backend", "torchvision_av") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        lerobot_version=lerobot_version,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool = True,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    if hasattr(data_cfg, "get") and data_cfg.get("delete_pause_frame", None) is not None:
        delete_pause_frame = bool(data_cfg.get("delete_pause_frame"))
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for entry in mixture_spec:
        d_name, d_weight, robot_type = entry[0], entry[1], entry[2]
        d_version = entry[3] if len(entry) > 3 else None
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type, d_version))

    dataset_mixture = []
    for d_name, d_weight, robot_type, d_version in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            lerobot_version=d_version,
        ), d_weight))

    dataset = LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        max_retries=int(getattr(data_cfg, "sample_max_retries", 10)),
        strict_sample_errors=bool(getattr(data_cfg, "strict_sample_errors", False)),
        return_progress_anchors=bool(getattr(data_cfg, "return_progress_anchors", False)),
        sequential_step_sampling=bool(getattr(data_cfg, "sequential_step_sampling", False)),
        action_stats_mask_mode=str(getattr(data_cfg, "action_stats_mask_mode", "gripper_false")),
        **kwargs,
    )
    dataset.progress_future_offset = int(getattr(data_cfg, "progress_future_offset", 7))
    return dataset
