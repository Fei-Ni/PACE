"""Helpers for appending the PACE query / history tokens to a Qwen3-VL batch."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from transformers import BatchFeature


def append_query_token_ids(
    batch_inputs: BatchFeature,
    query_token_ids: Sequence[int],
) -> Tuple[BatchFeature, int]:
    """Append `query_token_ids` to every sequence; returns the new batch and the original sequence length."""
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    device = input_ids.device

    query_ids = torch.tensor(query_token_ids, device=device, dtype=input_ids.dtype)
    query_ids = query_ids.unsqueeze(0).expand(batch_size, -1)
    query_mask = torch.ones_like(query_ids, dtype=attention_mask.dtype)

    out = BatchFeature(dict(batch_inputs))
    base_seq_len = int(input_ids.shape[1])
    out["input_ids"] = torch.cat([input_ids, query_ids], dim=1)
    out["attention_mask"] = torch.cat([attention_mask, query_mask], dim=1)

    if "labels" in out:
        ignore_tail = torch.full(
            (batch_size, len(query_token_ids)),
            fill_value=-100,
            device=out["labels"].device,
            dtype=out["labels"].dtype,
        )
        out["labels"] = torch.cat([out["labels"], ignore_tail], dim=1)

    return out, base_seq_len
