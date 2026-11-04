"""
Commitment audit: an online failure detector built from PACE's own progress outputs.

At every query the policy has committed, Delta steps earlier, to a progress increment (dp_cur), and the frozen encoder
measures the increment that was actually realised. Their difference is the audit residual

    residual = dp_realised - dp_committed            (negative = the policy delivered less than it promised)

Failing executions under-deliver persistently, while successful ones fluctuate around zero. The detector therefore
combines two signals, both measured in action chunks so that clients querying once per chunk (LIBERO) and once per
step (SimplerEnv with action ensembling) behave the same:

  * shortfall CUSUM:  S <- max(0, S + w * (-residual - kappa)),  w = env steps since the previous query / chunk_steps.
    Only shortfalls accumulate; over-delivery pulls S back towards zero.
  * stall:            action chunks since the p_cur readout last rose by more than `progress_eps`
                      (catches quiet stalls where neither the commitment nor the realised increment is large).

An alarm (`flag`) is raised when S > threshold or stall > stall_chunks. The detector only reports; it never changes
the actions. The shortfall threshold (0.4 of the task) is shared by all benchmarks; the stall limit is calibrated per
benchmark on held-out checkpoints (see docs/PROGRESS.md).

This module is pure Python so it can be unit-tested without torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

DEFAULT_AUDIT_KAPPA = 0.0
DEFAULT_AUDIT_THRESHOLD = 0.4
DEFAULT_AUDIT_STALL_CHUNKS = 18.0
DEFAULT_PROGRESS_EPS = 0.02


@dataclass(frozen=True)
class AuditState:
    score: float = 0.0  # shortfall CUSUM
    last_step: Optional[int] = None
    best_progress: float = -math.inf
    best_step: int = 0


def _finite(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def update_audit(
    state: Optional[AuditState],
    env_step: int,
    residual: Optional[float],
    progress: Optional[float],
    chunk_steps: int,
    kappa: float = DEFAULT_AUDIT_KAPPA,
    threshold: float = DEFAULT_AUDIT_THRESHOLD,
    stall_chunks: float = DEFAULT_AUDIT_STALL_CHUNKS,
    progress_eps: float = DEFAULT_PROGRESS_EPS,
) -> Tuple[AuditState, float, float, bool]:
    """
    Advance the detector by one query.

    Args:
        state: detector state of this episode (None at the first query).
        env_step: environment step of the query.
        residual: audit residual of this query (NaN / None before a commitment is available).
        progress: p_cur readout of this query.
        chunk_steps: environment steps per action chunk.
        kappa: slack subtracted from every shortfall.
        threshold: alarm threshold on the shortfall CUSUM.
        stall_chunks: alarm threshold on the stall length in chunks (<= 0 disables the stall signal).
        progress_eps: minimum rise of p_cur that counts as new progress.

    Returns:
        (new_state, score, stall, flag)
    """
    chunk_steps = int(chunk_steps)
    if chunk_steps < 1:
        raise ValueError(f"chunk_steps must be >= 1, got {chunk_steps}")
    env_step = int(env_step)
    if state is None:
        state = AuditState(last_step=env_step, best_step=env_step)
    weight = max(env_step - (state.last_step if state.last_step is not None else env_step), 0) / chunk_steps
    score = state.score
    if _finite(residual):
        score = max(0.0, score + weight * (-float(residual) - float(kappa)))
    best_progress, best_step = state.best_progress, state.best_step
    if _finite(progress) and float(progress) > best_progress + float(progress_eps):
        best_progress, best_step = float(progress), env_step
    stall = (env_step - best_step) / chunk_steps
    flag = score > float(threshold) or (float(stall_chunks) > 0 and stall > float(stall_chunks))
    new_state = replace(state, score=score, last_step=env_step, best_progress=best_progress, best_step=best_step)
    return new_state, score, stall, bool(flag)
