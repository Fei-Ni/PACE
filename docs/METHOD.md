# Method

This page describes PACE as implemented in this repository (`pace/model/framework/pace.py`). Symbols follow the paper.

## Notation

| Symbol | Meaning | In the code |
|---|---|---|
| f<sub>ψ</sub> | frozen DINOv2 ViT-B/14; CLS token, L2-normalized (d = 768) | `dino_model`, `_batched_dinov2_cls_features` |
| z<sub>i</sub> = f<sub>ψ</sub>(o<sub>i</sub>) | embedding of the primary camera image at step i | |
| z<sub>s</sub>, z<sub>g</sub> | embeddings of the start frame and the goal (last) frame of a demonstration | `start_image`, `goal_image` |
| ẑ<sub>g</sub> | **foresight anchor**: the predicted goal embedding | `z_goal_pred` |
| p(z; z<sub>s</sub>, z<sub>g</sub>) | progress of z along start→goal (below) | `compute_goal_projection_progress` |
| p<sub>cur</sub> | current progress read out by the policy | `predicted_progress` |
| Δp<sub>cur</sub> | forward commitment: progress the next chunk should add | `predicted_delta_progress` |
| Δp<sub>prev</sub> | historical pace: progress realized since the previous chunk | `observed_dp_prev` |
| Δp<sub>audit</sub> | realized − committed increment | `progress_audit_delta` |
| H | action chunk length (8 on LIBERO, 16 on SimplerEnv) | `action_horizon` |
| Δ | progress offset, H − 1 steps | `progress_future_offset` |

## Progress as a geometric projection

For an image embedding z, progress is its projection onto the start→goal direction:

```
p(z; z_s, z_g) = clamp( <z − z_s, z_g − z_s> / ||z_g − z_s||² , 0, 1 )
```

Training labels use the demonstration's first and last frames: p<sub>t</sub><sup>gt</sup> = p(z<sub>t</sub>; z<sub>s</sub>, z<sub>g</sub>). No human annotation is needed, and no reward model.

## Architecture

The backbone is Qwen3-VL-4B. Its input is the instruction plus two observation frames, the current one (t) and the one Δ steps earlier (t − Δ); each frame contains every camera view of the benchmark (LIBERO: third-person + wrist; SimplerEnv: third-person), resized to 224 × 224. After the text PACE appends:

1. a **Δp<sub>prev</sub> token**, whose embedding is the scalar Δp<sub>prev</sub> passed through a zero-initialized 2-layer MLP;
2. **K = 4 goal-query tokens** q<sub>goal</sub>;
3. one **progress token** q<sub>p</sub> and one **commitment token** q<sub>Δp</sub>.

The new tokens are added to the vocabulary, initialized as the end-of-sequence embedding plus small Gaussian noise (σ = 0.01), and only their rows of the embedding table are trained. The query tokens are read after transformer layer 18 (0-based, i.e. the 19th of 36 layers):

- the K goal-query states are mean-pooled and mapped by a 2-layer MLP (L2-normalized output) to ẑ<sub>g</sub>;
- q<sub>p</sub> → linear head → p<sub>cur</sub>; q<sub>Δp</sub> → linear head → Δp<sub>cur</sub>.

The **action decoder** is a GR00T-style flow-matching DiT (DiT-B, 16 layers) that cross-attends to the backbone's last-layer states over the image and instruction tokens (the appended Δp<sub>prev</sub> and query tokens are not part of that context; Δp<sub>prev</sub> reaches the actions through the readouts). Three conditioning tokens are **prepended** to its input sequence (not injected through AdaLN):

```
A_k ~ π(· | backbone states, φ_g(ẑ_g), φ_p(p_cur), φ_Δ(Δp_cur))
```

φ<sub>g</sub>, φ<sub>p</sub>, φ<sub>Δ</sub> are 2-layer MLPs whose last layer is zero-initialized, so the three conditioning tokens start at zero. Their inputs are detached: the action loss does not flow back into the progress heads.

## Training objective

With Δ-step-apart frames t − Δ, t, t + Δ of a demonstration:

| Loss | Target | Weight |
|---|---|---|
| L<sub>flow</sub> | flow-matching velocity of the action chunk | 1 |
| L<sub>goal</sub> = MSE(ẑ<sub>g</sub>, z<sub>g</sub>) (mean over the 768 dimensions) | goal-frame embedding | 0.1 |
| L<sub>progress</sub> = (p<sub>cur</sub> − p<sub>t</sub><sup>gt</sup>)² | projection label at t | 0.1 |
| L<sub>commit</sub> = (Δp<sub>cur</sub> − (p<sub>t+Δ</sub><sup>gt</sup> − p<sub>t</sub><sup>gt</sup>))² | realized next increment | 0.1 |
| L<sub>cons</sub> = (p<sub>cur</sub> − p(z<sub>t</sub>; z<sub>s</sub>, sg[ẑ<sub>g</sub>]))² | projection onto the *predicted* anchor | 0.1 |

During training the Δp<sub>prev</sub> token receives the ground-truth increment p<sub>t</sub><sup>gt</sup> − p<sub>t−Δ</sub><sup>gt</sup>. L<sub>cons</sub> ties the progress readout to the anchor the policy will actually have at test time (sg = stop-gradient).

## Inference loop

For each episode the policy keeps a small state that is reset on the first step:

1. **Start.** Store z<sub>s</sub> = f<sub>ψ</sub>(o<sub>0</sub>); Δp<sub>prev</sub> = 0.
2. **Forward.** Run the backbone on (history frame, o<sub>t</sub>, instruction, Δp<sub>prev</sub>) and read ẑ<sub>g</sub>, p<sub>cur</sub>, Δp<sub>cur</sub>. The history frame is the latest query at or before step t − Δ (t − 8 with the LIBERO client, which queries once per 8-step chunk; t − Δ exactly for a client that queries every step).
3. **Act.** Sample the action chunk with 4 flow-matching (Euler) steps, conditioned on ẑ<sub>g</sub>, p<sub>cur</sub>, Δp<sub>cur</sub>.
4. **Measure.** At the next query, the realized increment is measured with the frozen encoder against the anchor predicted at the previous query: Δp<sub>prev</sub> = p(z<sub>t</sub>; z<sub>s</sub>, ẑ<sub>g</sub>) − p(z<sub>t−Δ</sub>; z<sub>s</sub>, ẑ<sub>g</sub>). Both frames are looked up by environment step, so the window stays close to the training window whatever the query cadence.
5. **Audit.** Δp<sub>audit</sub> = Δp<sub>prev</sub> − (the Δp<sub>cur</sub> committed at the history query): negative when the policy delivered less than it promised. Two signals turn the residuals into a failure alarm:
   - `audit_score`, a shortfall CUSUM: S ← max(0, S + w · (−Δp<sub>audit</sub> − κ)), with w = environment steps since the previous query / H and κ = `audit_kappa` (0). Only shortfalls accumulate; over-delivery pays them back.
   - `audit_stall`: action chunks since p<sub>cur</sub> last rose by more than 0.02.

   `audit_flag` is raised when `audit_score > audit_threshold` or `audit_stall > audit_stall_chunks` (`audit_threshold` 0.4 on every benchmark; `audit_stall_chunks` 18 on LIBERO and 2.12 on SimplerEnv, calibrated on held-out checkpoints, see [PROGRESS.md](PROGRESS.md#knowing-when-it-fails)). Weighting by environment steps makes per-chunk and per-step clients see the same alarm. The audit never changes the actions.

The p<sub>cur</sub> that conditions the action head is always the learned readout; the geometric projection is only used for Δp<sub>prev</sub> and the audit.

## Outputs of `predict_action`

| Key | Meaning |
|---|---|
| `normalized_actions` | action chunk in normalized units (the clients unnormalize with `dataset_statistics.json`) |
| `predicted_progress` | p<sub>cur</sub> |
| `predicted_delta_progress` | Δp<sub>cur</sub> |
| `observed_dp_prev` | Δp<sub>prev</sub> fed to this query |
| `progress_audit_delta` | Δp<sub>audit</sub> (NaN until enough history exists) |
| `audit_score` | shortfall CUSUM of Δp<sub>audit</sub> (unmet commitments, in progress units) |
| `audit_stall` | action chunks since p<sub>cur</sub> last rose by more than 0.02 |
| `audit_flag` | failure alarm: `audit_score > audit_threshold` or `audit_stall > audit_stall_chunks` |
