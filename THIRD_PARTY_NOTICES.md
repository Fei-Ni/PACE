# Third-party notices

PACE is released under the MIT License (see [LICENSE](LICENSE)). It is built on the StarVLA codebase and includes
or adapts code from the projects listed below. Their licenses and copyright notices are reproduced or referenced here.
A copy of the Apache License, Version 2.0 is included in [third_party/dinov2/LICENSE](third_party/dinov2/LICENSE).

## StarVLA (MIT)

PACE is a derivative of StarVLA (https://github.com/starVLA/starVLA), commit e4541ca. The framework skeleton
(`pace/model/framework/base_framework.py`, `pace/model/framework/QwenGR00T.py`, `pace/model/modules/vlm/`,
`pace/model/tools.py`, `pace/dataloader/lerobot_datasets.py`, `pace/training/trainer_utils/trainer_tools.py`,
`deployment/model_server/`, `examples/LIBERO/eval_libero.py`, `examples/SimplerEnv/adaptive_ensemble.py` and the
evaluation clients) comes from StarVLA and was modified for PACE. StarVLA license text, verbatim:

```text
    MIT License

    Copyright (c) StarVLA Team.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    Rebases are allowed for forks and feature branches. When rebasing from upstream StarVLA, use descriptive commit messages, e.g., "chore: clone from StarVLA".
    Preserve attribution: keep at least the two latest upstream StarVLA commits as separate.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```

## NVIDIA Isaac GR00T N1.5 (Apache-2.0)

The flow-matching action head and the LeRobot data pipeline are derived from NVIDIA Isaac GR00T
(https://github.com/NVIDIA/Isaac-GR00T), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, licensed under the
Apache License, Version 2.0. The files keep their original copyright / SPDX headers; files changed by the StarVLA
authors or for PACE state so in their header:

- `pace/model/modules/action_model/GR00T_ActionHeader.py`
- `pace/model/modules/action_model/flow_matching_head/__init__.py`
- `pace/model/modules/action_model/flow_matching_head/action_encoder.py`
- `pace/model/modules/action_model/flow_matching_head/cross_attention_dit.py`
- `pace/dataloader/gr00t_lerobot/datasets.py`
- `pace/dataloader/gr00t_lerobot/data_config.py`
- `pace/dataloader/gr00t_lerobot/embodiment_tags.py`
- `pace/dataloader/gr00t_lerobot/schema.py`
- `pace/dataloader/gr00t_lerobot/video.py`
- `pace/dataloader/gr00t_lerobot/transform/__init__.py`
- `pace/dataloader/gr00t_lerobot/transform/base.py`
- `pace/dataloader/gr00t_lerobot/transform/state_action.py`

## openpi (Apache-2.0)

The WebSocket policy server / client and the msgpack / image helpers follow the openpi client
(https://github.com/Physical-Intelligence/openpi), Copyright 2024 Physical Intelligence, licensed under the Apache
License, Version 2.0:

- `deployment/model_server/tools/msgpack_numpy.py` (adapted from `openpi_client/msgpack_numpy.py`, which in turn
  adapts https://github.com/lebedov/msgpack-numpy, BSD-3-Clause)
- `deployment/model_server/tools/image_tools.py` (`convert_to_uint8` and `resize_with_pad` from
  `openpi_client/image_tools.py`)
- `deployment/model_server/tools/websocket_policy_server.py` and
  `deployment/model_server/tools/websocket_policy_client.py` (structure of the openpi websocket server / client)

The adapted msgpack-numpy code carries this notice:

```
Copyright (c) 2013-2022, Lev E. Givon.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.
* Neither the name of Lev E. Givon nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## DINOv2 (Apache-2.0)

PACE uses a frozen DINOv2 ViT-B/14 (https://github.com/facebookresearch/dinov2), Copyright (c) Meta Platforms, Inc.
and affiliates, as the progress encoder. `third_party/dinov2/` vendors only the files needed to build the backbone
(`dinov2/hub/backbones.py`, `dinov2/hub/utils.py`, `dinov2/models/`, `dinov2/layers/` and the package `__init__.py`
files), unmodified, together with the Apache-2.0 [LICENSE](third_party/dinov2/LICENSE); see
[third_party/dinov2/README.md](third_party/dinov2/README.md). None of the non-commercially licensed parts of the DINOv2
repository (Cell-DINO, X-Ray DINO) are included. The pretrained weights (`dinov2_vitb14_pretrain.pth`, Apache-2.0)
are downloaded from the official DINOv2 release on first use; the frozen encoder weights are also contained in the
released PACE checkpoints.

## Other code

- `examples/SimplerEnv/adaptive_ensemble.py`: adaptive action ensembling from CogACT
  (https://github.com/microsoft/CogACT), MIT, as shipped in StarVLA.
- `pace/training/trainer_utils/overwatch.py`: from OpenVLA / Prismatic (https://github.com/openvla/openvla), MIT.
- `RotationTransform` in `pace/dataloader/gr00t_lerobot/transform/state_action.py`: adapted from Diffusion Policy
  (https://github.com/real-stanford/diffusion_policy), MIT.
- `_quat2axisangle` in `examples/LIBERO/eval_libero.py`: copied from robosuite
  (https://github.com/ARISE-Initiative/robosuite), MIT.

## Models, benchmarks and data (not included in this repository)

- Qwen3-VL-4B-Instruct (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), Apache-2.0: base VLM; the released PACE
  checkpoints contain fine-tuned Qwen3-VL weights.
- StarVLA `Qwen3VL-GR00T-Bridge-RT-1` checkpoint (https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1):
  initialisation of the Bridge training stage.
- LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO), MIT: evaluation benchmark; training data from the
  LeRobot conversions published by IPEC-COMMUNITY on Hugging Face (`libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot`).
- BridgeData V2 (https://rail-berkeley.github.io/bridgedata/): training data, via the LeRobot conversion
  `IPEC-COMMUNITY/bridge_orig_lerobot`.
- SimplerEnv (https://github.com/simpler-env/SimplerEnv), MIT, and ManiSkill3 (https://github.com/haosulab/ManiSkill),
  Apache-2.0: SimplerEnv WidowX evaluation.
- LeRobot (https://github.com/huggingface/lerobot), Apache-2.0: dataset format.
