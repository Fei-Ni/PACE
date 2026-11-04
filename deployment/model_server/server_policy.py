# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modified for PACE: checkpoint resolution (file / directory / Hugging Face repo id) and an EFFECTIVE summary line.
"""
WebSocket policy server for PACE (used by both the LIBERO and the SimplerEnv clients).

    python -m deployment.model_server.server_policy --ckpt_path <ckpt> --port 10093 --use_bf16

`--ckpt_path` accepts a `.pt` file, a checkpoint directory or a Hugging Face repo id (see pace.model.checkpoint_io).
"""

import argparse
import logging
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from pace.model.framework.base_framework import baseframework


def describe_policy(vla) -> str:
    cfg = vla.config
    progress = cfg.framework.get("progress", {}) or {}
    action = cfg.framework.action_model
    fields = {
        "framework": cfg.framework.name,
        "chunk": action.get("action_horizon"),
        "denoise_steps": action.get("num_inference_timesteps"),
        "delta": progress.get("progress_future_offset"),
        "dino_variant": getattr(vla, "dino_variant", progress.get("dino_variant")),
        "K_goal_queries": getattr(vla, "num_goal_queries", None),
        "goal_dit_token": getattr(vla, "use_goal_dit_token", None),
        "audit_stall_chunks": getattr(vla, "audit_stall_chunks", None),
        "audit_threshold": getattr(vla, "audit_threshold", None),
        "image_size": cfg.datasets.vla_data.get("image_size", None),
        "norm_stats": ",".join(sorted(getattr(vla, "norm_stats", {}) or {})),
    }
    return "EFFECTIVE: " + " ".join(f"{k}={v}" for k, v in fields.items())


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    print(describe_policy(vla), flush=True)

    logging.info("Creating server (host: %s, bind: %s, port: %d)", socket.gethostname(), args.host, args.port)
    server = WebsocketPolicyServer(
        policy=vla,
        host=args.host,
        port=args.port,
        metadata={"policy": "PACE"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="Serve a PACE checkpoint over WebSocket.")
    parser.add_argument("--ckpt_path", type=str, required=True, help=".pt file, checkpoint dir or HF repo id")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument(
        "--use_bf16", action="store_true", help="cast the model to bfloat16 (setting of the released evaluations)"
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
