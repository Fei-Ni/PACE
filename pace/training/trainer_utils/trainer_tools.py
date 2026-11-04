"""
Training utilities: CLI dotlist parsing, per-module learning-rate groups, checkpoint warm start and module freezing.
"""

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from PIL import Image

from pace.training.dotlist import normalize_dotlist_args  # noqa: F401  (re-exported)

logger = get_logger(__name__)


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    used_params = set()
    param_groups = []

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            logger.warning(f"module path `{module_name}` from trainer.learning_rate not found in the model")

    # assign base learning rate to the remaining unused parameters
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in used_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        dist.barrier()  # synchronize when distributed training
        if dist.get_rank() == 0 and frozen:
            print(f"Frozen modules: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None, strict=True):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)
        - strict=False allows new parameters (e.g. the PACE progress heads)
          to be absent from the checkpoint without raising an error

        return:
            model with loaded parameters
        """
        if not checkpoint_path:
            return []
        if dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        def _summarize_names(label, names, limit=8):
            if dist.get_rank() != 0 or not names:
                return
            preview = names[:limit]
            suffix = " ..." if len(names) > limit else ""
            print(f"⚠️  {label} ({len(names)}): {preview}{suffix}")

        # Rows of freshly added special tokens (e.g. PACE query tokens) keep their
        # own initialisation when a vocabulary-mismatched embedding is row-merged.
        preserve_rows = [int(i) for i in (getattr(model, "progress_special_token_ids", None) or [])]

        def _row_merge_vocab(key, target_value, value):
            # Adding special tokens resizes the token embedding / LM head, so the
            # warm-start checkpoint no longer matches in dim 0. Skipping the whole
            # tensor would silently drop the finetuned vocabulary; instead copy the
            # overlapping rows and keep the new token rows as initialised.
            if not (key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")):
                return None
            if value.ndim != 2 or target_value.ndim != 2 or value.shape[1] != target_value.shape[1]:
                return None
            merged = target_value.detach().clone()
            n_rows = min(value.shape[0], target_value.shape[0])
            merged[:n_rows] = value[:n_rows].to(dtype=merged.dtype)
            for row in preserve_rows:
                if 0 <= row < merged.shape[0]:
                    merged[row] = target_value[row]
            return merged

        def _filter_compatible_state_dict(target_state_dict, source_state_dict):
            compatible = {}
            unexpected = []
            shape_mismatch = []
            for key, value in source_state_dict.items():
                if key not in target_state_dict:
                    unexpected.append(key)
                    continue
                target_value = target_state_dict[key]
                value_shape = tuple(value.shape) if hasattr(value, "shape") else None
                target_shape = tuple(target_value.shape) if hasattr(target_value, "shape") else None
                if value_shape != target_shape:
                    merged = _row_merge_vocab(key, target_value, value)
                    if merged is not None:
                        compatible[key] = merged
                        if dist.get_rank() == 0:
                            print(
                                f"🔁 row-merged {key}: ckpt{value_shape} -> model{target_shape} "
                                f"(kept {len(preserve_rows)} new token rows)"
                            )
                        continue
                    shape_mismatch.append(f"{key}: ckpt{value_shape} != model{target_shape}")
                    continue
                compatible[key] = value
            missing = [key for key in target_state_dict.keys() if key not in compatible]
            return compatible, missing, unexpected, shape_mismatch

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        if strict:
                            module.load_state_dict(sub_state_dict, strict=True)
                            if dist.get_rank() == 0:
                                print(f"✅ parameters loaded to module '{path}'")
                        else:
                            compatible, missing, unexpected, shape_mismatch = _filter_compatible_state_dict(
                                module.state_dict(), sub_state_dict
                            )
                            module.load_state_dict(compatible, strict=False)
                            if dist.get_rank() == 0:
                                print(
                                    f"✅ parameters compat-loaded to module '{path}' "
                                    f"({len(compatible)}/{len(module.state_dict())} tensors)"
                                )
                            _summarize_names(f"{path} missing after compat load", missing)
                            _summarize_names(f"{path} unexpected ckpt keys", unexpected)
                            _summarize_names(f"{path} shape-mismatch keys", shape_mismatch)
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                if strict:
                    missing, unexpected = model.load_state_dict(checkpoint, strict=True)
                else:
                    compatible, missing, unexpected, shape_mismatch = _filter_compatible_state_dict(
                        model.state_dict(), checkpoint
                    )
                    missing, unexpected_load = model.load_state_dict(compatible, strict=False)
                    unexpected = list(unexpected) + list(unexpected_load)
                    _summarize_names("shape-mismatch keys skipped", shape_mismatch)
                if dist.get_rank() == 0:
                    if missing:
                        print(f"⚠️  new keys not in checkpoint (zero-init kept): {missing}")
                    if unexpected:
                        print(f"⚠️  checkpoint keys not in model (ignored): {unexpected}")
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter
