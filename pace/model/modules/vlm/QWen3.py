# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modified for PACE: special-token helpers for the progress query tokens, base-VLM resolution.

import os
from typing import List, Optional

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from pace.model.checkpoint_io import resolve_base_vlm

logger = get_logger(__name__)


class _QWen3_VL_Interface(nn.Module):
    """
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration): model loading, chat-template packing and
    helpers to add / train the PACE special tokens.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """Load Qwen3-VL (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and its processor."""
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        # env BASE_VLM > existing local `base_vlm` directory > Hugging Face id Qwen/Qwen3-VL-4B-Instruct
        model_id = resolve_base_vlm(qwenvl_config.get("base_vlm", None))
        # PACE_ATTN_IMPLEMENTATION (e.g. "sdpa") overrides the configured kernel when flash-attn is unavailable.
        attn_implementation = os.environ.get("PACE_ATTN_IMPLEMENTATION") or qwenvl_config.get(
            "attn_implementation", "flash_attention_2"
        )
        device_map = qwenvl_config.get("device_map", "cuda")
        print(f"[PACE] base VLM: {model_id} (attn_implementation={attn_implementation})")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self._embed_tokens_hook_handle = None
        self._embed_tokens_hook_mask = None

        # Expose the text hidden size at the top level (used to size the progress heads and the DiT).
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass delegating to the underlying Qwen3-VL model."""

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Build model inputs (chat template) from images + instructions, following the official Qwen3-VL Instruct
        format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct. If `datasets.vla_data.CoT_prompt` is set, the
        instruction is inserted into that prompt template at `{instruction}`.
        """
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)

    def ensure_special_tokens(
        self,
        tokens: List[str],
        *,
        init_std: float = 0.01,
        init_token_id: Optional[int] = None,
    ) -> List[int]:
        """
        Add `tokens` as special tokens if missing (resizing the embeddings; new rows are initialised around the EOS
        embedding with std `init_std`) and return their ids in the given order.
        """
        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()
        missing_tokens = [token for token in tokens if token not in vocab]
        if missing_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            self.model.resize_token_embeddings(len(tokenizer))

            embed = self.model.get_input_embeddings().weight
            if init_token_id is None:
                init_token_id = tokenizer.eos_token_id
                if init_token_id is None:
                    init_token_id = tokenizer.pad_token_id
                if init_token_id is None:
                    init_token_id = 0

            with torch.no_grad():
                ref = embed[int(init_token_id)].detach().clone()
                for token in missing_tokens:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    embed[token_id].copy_(ref + init_std * torch.randn_like(ref))

        return [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]

    def set_trainable_token_rows(self, token_ids: List[int]):
        """Mask input-embedding gradients so only the rows of `token_ids` are updated (other layers unaffected)."""
        embed_tokens = self.model.get_input_embeddings()
        embed_tokens.weight.requires_grad = True

        if self._embed_tokens_hook_handle is not None:
            self._embed_tokens_hook_handle.remove()
            self._embed_tokens_hook_handle = None

        mask = torch.zeros(
            embed_tokens.weight.shape[0],
            device=embed_tokens.weight.device,
            dtype=embed_tokens.weight.dtype,
        )
        mask[token_ids] = 1.0
        self._embed_tokens_hook_mask = mask.view(-1, 1)

        def grad_hook(grad):
            if (
                self._embed_tokens_hook_mask.device != grad.device
                or self._embed_tokens_hook_mask.dtype != grad.dtype
            ):
                self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.to(
                    device=grad.device,
                    dtype=grad.dtype,
                )
            return grad * self._embed_tokens_hook_mask

        self._embed_tokens_hook_handle = embed_tokens.weight.register_hook(grad_hook)
