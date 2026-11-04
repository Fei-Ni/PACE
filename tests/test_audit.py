"""Commitment-audit failure detector (shortfall CUSUM + p_cur stall): pure-Python logic, no torch needed."""

import math

import pytest

from pace.model.progress.audit import (
    DEFAULT_AUDIT_KAPPA,
    DEFAULT_AUDIT_STALL_CHUNKS,
    DEFAULT_AUDIT_THRESHOLD,
    update_audit,
)


def run(seq, chunk_steps=8, stride=None, **kw):
    """seq: list of (residual, progress) per query; queries every `stride` env steps (default: once per chunk)."""
    stride = chunk_steps if stride is None else stride
    state, out = None, []
    for i, (res, prog) in enumerate(seq):
        state, score, stall, flag = update_audit(state, i * stride, res, prog, chunk_steps, **kw)
        out.append((score, stall, flag))
    return state, out


def test_defaults_are_the_libero_calibration():
    assert DEFAULT_AUDIT_KAPPA == 0.0
    assert DEFAULT_AUDIT_THRESHOLD == pytest.approx(0.4)
    assert DEFAULT_AUDIT_STALL_CHUNKS == pytest.approx(18.0)


def test_first_query_has_no_score():
    _, out = run([(float("nan"), 0.0)])
    assert out[0] == (0.0, 0.0, False)


def test_only_shortfalls_accumulate():
    # under-delivery of 0.1 per chunk accumulates; over-delivery pulls the score back towards zero
    _, out = run([(None, 0.0), (-0.1, 0.1), (-0.1, 0.2), (+0.15, 0.4), (+0.5, 0.6)], threshold=10)
    scores = [s for s, _, _ in out]
    assert scores == pytest.approx([0.0, 0.1, 0.2, 0.05, 0.0])


def test_kappa_is_a_slack():
    _, out = run([(None, 0.0), (-0.05, 0.1), (-0.05, 0.2)], kappa=0.04, threshold=10)
    assert out[-1][0] == pytest.approx(0.02)


def test_alarm_on_persistent_shortfall():
    seq = [(None, 0.0)] + [(-0.1, 0.05 * i) for i in range(1, 8)]
    _, out = run(seq, threshold=0.45, stall_chunks=0)
    flags = [f for _, _, f in out]
    assert flags.index(True) == 5  # 0.5 > 0.45 after five shortfalls of 0.1
    assert all(flags[5:])


def test_alarm_on_stall_without_large_residuals():
    # quiet stall: p_cur stuck at 0.3, residuals tiny -> only the stall signal fires
    seq = [(None, 0.0), (0.0, 0.3)] + [(0.0, 0.3)] * 25
    _, out = run(seq, threshold=0.4, stall_chunks=18)
    flags = [f for _, _, f in out]
    stalls = [st for _, st, _ in out]
    assert stalls[-1] == pytest.approx(25)
    assert flags.index(True) == 20  # stall of 19 chunks > 18
    assert max(s for s, _, _ in out) == 0.0


def test_progress_resets_the_stall_counter():
    seq = [(None, 0.0)] + [(0.0, 0.1)] * 10 + [(0.0, 0.5)] + [(0.0, 0.5)] * 3
    _, out = run(seq, stall_chunks=100)
    assert out[10][1] == pytest.approx(9)
    assert out[11][1] == 0.0
    assert out[-1][1] == pytest.approx(3)


def test_measured_in_chunks_not_calls():
    # same shortfall rate queried every step (chunk 16) or once per chunk gives the same score
    per_chunk = [(None, 0.0)] + [(-0.1, 0.0)] * 4
    _, a = run(per_chunk, chunk_steps=16)
    per_step = [(None, 0.0)] + [(-0.1, 0.0)] * 64
    _, b = run(per_step, chunk_steps=16, stride=1)
    assert a[-1][0] == pytest.approx(b[-1][0])
    assert a[-1][1] == pytest.approx(b[-1][1])


def test_state_is_immutable():
    s0, _, _, _ = update_audit(None, 0, None, 0.0, 8)
    s1, _, _, _ = update_audit(s0, 8, -0.2, 0.1, 8)
    assert s0.score == 0.0 and s1.score == pytest.approx(0.2)


def test_invalid_chunk_steps():
    with pytest.raises(ValueError):
        update_audit(None, 0, 0.0, 0.0, 0)


def test_projection_progress_matches_definition():
    torch = pytest.importorskip("torch")
    from pace.model.progress.progress_modules import compute_goal_projection_progress

    start = torch.zeros(3, 2)
    goal = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    current = torch.tensor([[0.25, 0.7], [2.0, 0.0], [0.3, 0.3]])
    progress = compute_goal_projection_progress(current, start, goal).squeeze(-1)
    assert progress.tolist() == pytest.approx([0.25, 1.0, 0.0])
