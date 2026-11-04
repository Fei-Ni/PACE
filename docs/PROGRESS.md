# Progress readings and the failure alarm

PACE reads its own progress p<sub>cur</sub> at every query and checks every commitment it makes. This page measures how well the reading tracks hindsight progress and how well the failure alarm catches failed episodes on the released checkpoints, and explains how to reproduce the plots and videos with [tools/progress_viz](../tools/progress_viz).

## Definitions

- **Predicted progress**: the p<sub>cur</sub> the policy outputs online during the rollout (`predicted_progress`). SimplerEnv queries the policy every step; LIBERO queries once per 8-step chunk, and the value is held for the steps in between.
- **Hindsight ("true") progress**: the training-label definition applied after the rollout: frozen DINOv2 ViT-B/14 CLS embeddings (L2-normalized), projection of each frame onto start→goal, clamped to [0, 1]. The start is the first frame. The goal is the episode's own last frame for successes. For failures, whose last frame is not a goal, it is the last frame of a *successful* rollout from the same initial state (same LIBERO episode index / same SimplerEnv seed), taken from another evaluation run; failures without such a partner are left out of the failure-side numbers.

Hindsight progress is a visual quantity, not a clock: it can dip when the arm moves away or when a multi-stage task passes through states that look less like the goal. It is what the policy is trained to predict.

## Results on the released checkpoint

| | LIBERO |
|---|---|
| Checkpoint | PACE-LIBERO |
| Rollouts analyzed | 400 successes (100 per suite, random) + all 28 failures |
| **Successful episodes** (compared at the query steps) | |
| Spearman ρ, predicted vs hindsight (median over episodes) | 0.96 |
| Episodes with ρ > 0.8 | 96% |
| Mean absolute error | 0.055 |
| Final predicted progress (mean) | 0.90 |
| Query-to-query changes that are not drops of more than 0.02 | 77% |
| **Failed episodes** | |
| Final predicted / hindsight progress (mean) | 0.78 / 0.72 |
| **Separating successes from failures** | |
| AUROC of the final predicted progress | 0.82 |

Metrics are computed where the policy produces a reading, every 8th step on LIBERO. Comparing every environment step instead, with the LIBERO reading held between queries, gives ρ 0.91, MAE 0.074 and 97% non-dropping steps on LIBERO (the held values inflate the last number). Weakest group: LIBERO-Goal. The reading is not monotone, and neither is hindsight progress; no smoothing is applied anywhere in the model.

These numbers use our own evaluation set and automatic labels, so they are not directly comparable with the paper's Table 3 (MAE on LIBERO-Long rollouts) or Table 4 (failure detection on human-annotated rollouts).

## Knowing when it fails

The failure alarm (METHOD.md, step 5) is computed at every query from outputs the policy already produces:

- **Unmet commitments**, `audit_score`: a shortfall CUSUM over the audit residuals Δp<sub>audit</sub> = realized − committed increment, S ← max(0, S + w · (−Δp<sub>audit</sub> − κ)) with κ = 0 and w = environment steps since the previous query / H. Only shortfalls add up; delivering more than promised pays them back.
- **Stalled progress**, `audit_stall`: action chunks since p<sub>cur</sub> last rose by more than 0.02.

`audit_flag` is raised when `audit_score > audit_threshold` or `audit_stall > audit_stall_chunks`. Both signals count in action chunks, so a client that queries once per chunk (LIBERO) and one that queries every step (SimplerEnv with action ensembling) get the same alarm. The alarm never changes the actions.

### Calibration

The shortfall threshold is the same on every benchmark: `audit_threshold` = 0.4, an alarm once the commitments the policy did not meet add up to 0.4 of the task. Progress is normalized to [0, 1] for every task, so this reads the same on LIBERO and SimplerEnv. The stall limit depends on how a benchmark is timed (chunk length, query rate, episode length), so it is calibrated per benchmark with one fixed rule: on evaluation runs of held-out checkpoints of the same recipe (other training seeds and earlier steps; never the released checkpoints), it is the 97.5th percentile of the per-episode longest stall over the successful episodes, so on its own it flags 2.5% of them. This gives `audit_stall_chunks` 18 for LIBERO (H = 8) and 2.12 for SimplerEnv (H = 16). Both values are stored in each checkpoint's `config.yaml` under `framework.progress`. On the same held-out runs, the shared 0.4 behaves like a threshold calibrated per benchmark at a 2.5% false-alarm rate (those would be 0.417 on LIBERO and 0.388 on SimplerEnv).

### Results

Shipped thresholds, full evaluation of the released checkpoint (same protocol as Key Results). *Episode left* is 1 − alarm step / episode length for failed episodes, which run to the time limit (LIBERO: 220 / 280 / 300 / 520 steps for Spatial / Object / Goal / Long).

| Checkpoint | Failed episodes flagged | Successful episodes flagged | Episode left at the alarm (median) |
|---|---|---|---|
| **PACE-LIBERO** (2,000 episodes) | **92.9%** of 28 | 7.2% of 1,972 | 50% |

Long tasks draw most of the false alarms: on LIBERO-Long they land mostly on successful episodes that take much longer than usual. If your application stops or resets on the alarm, raise the thresholds (or calibrate at a lower false-alarm budget) to trade detection for fewer interruptions.

### Calibrating your own policy

Evaluate one or more checkpoints other than the one you will deploy with progress traces on (the LIBERO client always writes them; for SimplerEnv set `SIMPLER_RECORD_DIR`), then:

```bash
python -m tools.progress_viz.calibrate_audit --chunk_steps 8 --false_alarm 0.05 \
    --dev <eval_dir_a> <eval_dir_b> --test <eval_dir_deployed>
```

It prints the `framework.progress` block to put in the deployed checkpoint's `config.yaml` and the detection table for every directory. `--threshold` and `--stall_chunks` evaluate fixed values instead.

## Reproducing

1. **Record the rollouts.** The LIBERO client already writes, next to each `rollout_*.mp4`, a `progress_*.json` with the per-step readings. For SimplerEnv set `SIMPLER_RECORD_DIR=/path/to/rec` before `examples/SimplerEnv/eval_simpler.sh`; frames and per-step readings are saved per episode.
2. **Compute hindsight progress and the metrics.**
   ```bash
   python -m tools.progress_viz.libero_progress_study --eval_dir <libero_eval_out> --out ./progress_libero \
       [--oracle_dirs <other_eval_out> ...]
   python -m tools.progress_viz.progress_metrics ./progress_libero/traces.json
   ```
   `python -m tools.progress_viz.simpler_progress_study --rec_dir <rec> --out ./progress_simpler` does the same for SimplerEnv (run all of them from the repository root). `--oracle_dirs` lists other evaluation runs to borrow success goals from for the failed episodes.
3. **Render videos.** `pick_demos.py` picks, per task, the success whose ρ is closest to the median and the best-tracked failure; `render_progress_demo.py` draws the side-by-side videos (left: rollout, right: both curves growing online).
   ```bash
   python -m tools.progress_viz.pick_demos ./progress_libero/traces.json
   python -m tools.progress_viz.render_progress_demo --traces ./progress_libero/traces.json --out_dir ./demos --select <ids>
   ```
4. **Showreel, side-by-side wall and alarm comparison.** `render_showreel.py` draws the commit-realize loop: each chunk grows its forward commitment as a dashed segment, the realized anchored progress follows, and the gap at each check (exactly the alarm residual) drops into the meter of accumulated unmet commitments. It draws the animations in the README and on the project page from recorded episodes (LIBERO: `rollout_*.mp4` + `progress_*.json`; SimplerEnv: `seed*_frames.npz` + `seed*_progress.json`), with the alarm recomputed at the shipped thresholds. `.webp` output is an animated WebP (small, and it always plays on GitHub); `.gif` and `.mp4` work too.
   ```bash
   # one clip at a time, with task tabs that advance automatically
   python -m tools.progress_viz.render_showreel --layout player --out showreel.webp \
       --clip libero "LIBERO · Long" <rollout.mp4> <progress.json> \
       --clip simpler "SimplerEnv · Carrot" <seed3_success_frames.npz> <seed3_success_progress.json>
   # all clips side by side, in sync
   python -m tools.progress_viz.render_showreel --layout wall --cols 4 --out wall.webp --clip ... --clip ...
   # a success and a failure of the same task, pausing when the alarm fires
   python -m tools.progress_viz.render_showreel --layout versus --out alarm.webp \
       --clip libero Success <rollout_a.mp4> <progress_a.json> --clip libero Failure <rollout_b.mp4> <progress_b.json>
   ```
5. **Failure-alarm figures.** `plot_audit_mechanism.py` draws the per-signal breakdown of a few episodes and `plot_detection.py` the alarm-timing figure from a summary such as [docs/detection_libero_release.json](detection_libero_release.json), which holds the numbers measured above.
   ```bash
   python -m tools.progress_viz.plot_audit_mechanism --out audit_mechanism.png \
       --episode <progress_a.json> Success --episode <progress_b.json> Failure
   python -m tools.progress_viz.plot_detection --data docs/detection_libero_release.json --out alarm_timing.png
   ```
   The defaults are the LIBERO thresholds; pass `--threshold`, `--stall_chunks` and `--chunk_steps` for other policies.
