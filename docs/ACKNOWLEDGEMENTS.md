# Acknowledgements

PACE builds on a number of open-source projects. Their licenses and the files derived from them are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Foundational code

- **[StarVLA](https://github.com/starVLA/starVLA)**: the modular VLA codebase this repository builds on: Qwen-VL interface, framework registry, LeRobot data loading, training loop, WebSocket policy server, LIBERO and SimplerEnv evaluation clients.
- **[Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)** (N1.5): the flow-matching DiT action head and the LeRobot dataset/transform pipeline adopted by StarVLA.
- **[CogACT](https://github.com/microsoft/CogACT)**: adaptive action ensembling for SimplerEnv.
- **[openpi](https://github.com/Physical-Intelligence/openpi)**: the msgpack-based WebSocket policy protocol.

## Prior work

- **[SemanticVLA](https://github.com/Fei-Ni/SemanticVLA_Offcial)**: our earlier VLA on the same StarVLA stack (Qwen3-VL-4B backbone, flow-matching action head, LIBERO / SimplerEnv evaluation).
- **[UniVLA](https://github.com/OpenDriveLab/UniVLA)**: task-centric latent actions for generalist policies.
- **[π<sub>0</sub> / openpi](https://github.com/Physical-Intelligence/openpi)**: the flow-matching VLA policy family and its open release.

## Models

- **[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)**: the vision-language backbone (`Qwen/Qwen3-VL-4B-Instruct`).
- **[DINOv2](https://github.com/facebookresearch/dinov2)**: the frozen ViT-B/14 encoder that defines progress; a minimal subset of its code is vendored in `third_party/dinov2`.

## Data and benchmarks

- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** and the LeRobot conversions by **[IPEC-COMMUNITY](https://huggingface.co/IPEC-COMMUNITY)**.
- **[BridgeData V2](https://rail-berkeley.github.io/bridgedata/)** via the **[Open X-Embodiment](https://robotics-transformer-x.github.io/)** release.
- **[SimplerEnv](https://github.com/simpler-env/SimplerEnv)** and **[ManiSkill3](https://github.com/haosulab/ManiSkill)** for WidowX evaluation.
- **[LeRobot](https://github.com/huggingface/lerobot)** for the dataset format.

Questions and issues are welcome on the GitHub issue tracker.
