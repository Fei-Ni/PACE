"""Failure-alarm calibration tool: trace loading (LIBERO and SimplerEnv formats), threshold choice, evaluation."""

import json

import pytest

np = pytest.importorskip("numpy")

from tools.progress_viz.calibrate_audit import calibrate, evaluate, load_runs  # noqa: E402


def write_libero(path, success, residuals, progress, chunk_steps=8):
    T = len(residuals) * chunk_steps
    per_step_res = [r for r in residuals for _ in range(chunk_steps)]
    per_step_p = [p for p in progress for _ in range(chunk_steps)]
    json.dump(dict(success=success, query_steps=list(range(0, T, chunk_steps)), progress=per_step_p,
                   audit_delta=per_step_res), open(path, "w"))


def test_libero_and_simpler_formats_give_the_same_trace(tmp_path):
    res = [None, 0.0, -0.1, -0.1, 0.05]
    prog = [0.0, 0.1, 0.1, 0.1, 0.2]
    (tmp_path / "lib").mkdir()
    (tmp_path / "sim").mkdir()
    write_libero(tmp_path / "lib" / "progress_task_episode0_failure.json", False, res, prog, chunk_steps=4)
    # SimplerEnv recorder: every step is a query; same values held for 4 steps -> same chunk-weighted CUSUM
    json.dump(dict(success=0.0, steps=20, predicted_progress=[p for p in prog for _ in range(4)],
                   progress_audit_delta=[r for r in res for _ in range(4)]),
              open(tmp_path / "sim" / "seed0_failure_progress.json", "w"))
    lib = load_runs([str(tmp_path / "lib")], 4, 0.0)[str(tmp_path / "lib")][0]
    sim = load_runs([str(tmp_path / "sim")], 4, 0.0)[str(tmp_path / "sim")][0]
    assert lib["trace"][-1][1] == pytest.approx(0.15)
    assert sim["trace"][-1][1] == pytest.approx(lib["trace"][-1][1])


def test_calibrate_and_evaluate(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(40):  # successes: small residuals, steady progress
        write_libero(tmp_path / f"progress_t_episode{i}_success.json", True, list(rng.normal(0, 0.02, 20)),
                     list(np.linspace(0, 1, 20)))
    for i in range(10):  # failures: persistent shortfall, flat progress after chunk 5
        write_libero(tmp_path / f"progress_t_episode{40 + i}_failure.json", False, [0.0] * 5 + [-0.08] * 35,
                     list(np.linspace(0, 0.5, 5)) + [0.5] * 35)
    eps = load_runs([str(tmp_path)], 8, 0.0)[str(tmp_path)]
    threshold, stall, n = calibrate(eps, false_alarm=0.1)
    assert n == 40 and threshold > 0 and stall >= 0
    r = evaluate(eps, threshold, stall)
    assert r["n_succ"] == 40 and r["n_fail"] == 10
    assert r["false_alarm"] <= 0.1 + 1e-9
    assert r["detected"] == 1.0
    assert 0.0 < r["remaining"] < 1.0
