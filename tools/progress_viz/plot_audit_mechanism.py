"""
Plot how the commitment audit reaches its decision, one column per episode (e.g. a success next to a failure).

Rows, all against the environment step:
  progress    p_cur read out by the policy, and the progress measured by the frozen encoder
  check       audit residual of every chunk: realized increment - committed increment (below zero = shortfall)
  shortfall   the shortfall CUSUM (`audit_score`) against its threshold
  stall       action chunks since p_cur last rose (`audit_stall`) against its threshold

Each episode is a `progress_*.json` written by examples/LIBERO/eval_libero.sh. The detector is the one in
pace.model.progress.audit, with the thresholds of the LIBERO config by default.

    python -m tools.progress_viz.plot_audit_mechanism \\
        --episode <progress_a.json> "success" --episode <progress_b.json> "failure" --out assets/pace_audit_mechanism.png
"""

from __future__ import annotations

import argparse
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pace.model.progress.audit import update_audit  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3", "#fcfcfb"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c9c8c3"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"


def load(path, threshold, stall_chunks, chunk_steps):
    d = json.load(open(path))
    T = len(d["progress"])
    q = [s for s in d["query_steps"] if s < T]
    state, rows, alarm, cause = None, [], None, None
    for s in q:
        res = d["audit_delta"][s]
        res = np.nan if res is None else float(res)
        state, score, stall, flag = update_audit(state, s, res, d["progress"][s], chunk_steps,
                                                 threshold=threshold, stall_chunks=stall_chunks)
        rows.append((s, d["progress"][s], d["observed_progress"][s], res, score, stall))
        if flag and alarm is None:
            alarm = s
            cause = "unmet commitments" if score > threshold else "progress stalled"
    a = np.array([[np.nan if v is None else v for v in r] for r in rows], float)
    a[0, 2] = 0.0
    return dict(step=a[:, 0], p=a[:, 1], m=a[:, 2], res=a[:, 3], score=a[:, 4], stall=a[:, 5], T=T,
                success=bool(d["success"]), alarm=alarm, cause=cause, task=d.get("task", ""))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5, length=2)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", nargs=2, action="append", metavar=("JSON", "LABEL"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--stall_chunks", type=float, default=18)
    ap.add_argument("--chunk_steps", type=int, default=8)
    ap.add_argument("--time_limit", type=int, default=520)
    a = ap.parse_args()
    eps = [load(j, a.threshold, a.stall_chunks, a.chunk_steps) for j, _ in a.episode]
    n = len(eps)
    fig, axes = plt.subplots(4, n, figsize=(6.6 * n, 8.8), dpi=160, facecolor=SURFACE, sharex=True, squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.25, 1, 1, 1], hspace=0.38, wspace=0.07))
    res_lim = max(0.12, float(np.nanmax(np.abs(np.concatenate([e["res"] for e in eps])))) * 1.1)
    score_lim = max(a.threshold * 2.2, max(float(e["score"].max()) for e in eps) * 1.08)
    stall_lim = max(a.stall_chunks * 1.6, max(float(e["stall"].max()) for e in eps) * 1.08)
    for c, (e, (_, label)) in enumerate(zip(eps, a.episode)):
        axp, axr, axs, axt = axes[:, c]
        for ax in axes[:, c]:
            style(ax)
            ax.set_xlim(0, a.time_limit)
        x = e["step"]
        # progress
        axp.plot(x, e["p"], color=BLUE, lw=2, label=r"current progress $p_{\mathrm{cur}}$")
        axp.plot(x, e["m"], color=ORANGE, lw=2, ls=(0, (4, 2)), label=r"anchored progress $p^{\mathrm{anchor}}$ (frozen encoder)")
        axp.set_ylim(-0.05, 1.12)
        # commitment check
        res = np.nan_to_num(e["res"])
        width = a.chunk_steps * 0.8
        axr.bar(x[res < 0], res[res < 0], width=width, color=INK2, lw=0, label="unmet: realized below committed")
        axr.bar(x[res >= 0], res[res >= 0], width=width, color=NEUTRAL, lw=0, label="met or exceeded")
        axr.axhline(0, color=MUTED, lw=0.8)
        axr.set_ylim(-res_lim, res_lim)
        # shortfall CUSUM
        axs.plot(x, e["score"], color=INK, lw=2, drawstyle="steps-post")
        axs.axhline(a.threshold, color=CRITICAL, lw=1, ls=(0, (3, 3)))
        axs.text(a.time_limit, a.threshold * 1.06, f"threshold {a.threshold:g}", color=CRITICAL, fontsize=8,
                 ha="right", va="bottom")
        axs.set_ylim(0, score_lim)
        # stall
        axt.plot(x, e["stall"], color=INK, lw=2, drawstyle="steps-post")
        axt.axhline(a.stall_chunks, color=CRITICAL, lw=1, ls=(0, (3, 3)))
        axt.text(a.time_limit, a.stall_chunks * 1.04, f"threshold {a.stall_chunks:g} chunks", color=CRITICAL,
                 fontsize=8, ha="right", va="bottom")
        axt.set_ylim(0, stall_lim)
        axt.set_xlabel("environment step", fontsize=9, color=INK2)
        # outcome markers
        if e["alarm"] is not None:
            for ax in axes[:, c]:
                ax.axvline(e["alarm"], color=CRITICAL, lw=1.5, zorder=0.5)
                ax.axvspan(e["alarm"], a.time_limit, color=CRITICAL, alpha=0.05, lw=0, zorder=0)
            axp.text(e["alarm"] + 6, 1.08, f"ALARM · step {e['alarm']}\n({e['cause']})", color=CRITICAL, fontsize=9,
                     weight="bold", va="top")
        if e["success"]:
            axp.axvline(e["T"], color=GOOD, lw=1.5, zorder=0.5)
            axp.text(e["T"] + 6, 0.12, f"task done · step {e['T']}", color=GOOD, fontsize=9, weight="bold")
        elif e["T"] >= a.time_limit:
            axp.text(a.time_limit - 4, 0.12, "time limit, task not done", color=INK2, fontsize=9, ha="right")
        outcome = "success" if e["success"] else "failure"
        axp.set_title(f"{label}" if label.lower() == outcome else f"{label} ({outcome})", fontsize=11, color=INK,
                      loc="left", weight="bold")
        if c == 0:
            axp.set_ylabel("progress", fontsize=9, color=INK2)
            axr.set_ylabel("realized − committed\nper chunk", fontsize=9, color=INK2)
            axs.set_ylabel("unmet commitments\n(shortfall CUSUM)", fontsize=9, color=INK2)
            axt.set_ylabel("chunks since\nprogress rose", fontsize=9, color=INK2)
        else:
            for ax in axes[:, c]:
                ax.tick_params(labelleft=False)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=8.5, frameon=False, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.005))
    task = eps[0]["task"].replace("_", " ")
    head = "How the alarm decides" + (f": {task}" if task and all(e["task"] == eps[0]["task"] for e in eps) else "")
    fig.suptitle(head, x=0.08, y=0.99, ha="left", fontsize=12.5, color=INK, weight="bold")
    fig.text(0.08, 0.952, textwrap.fill(
        "Every chunk commits to a progress increment; the frozen encoder checks it once the chunk has run. "
        "The alarm fires when the unmet commitments add up past their threshold, or when progress has not risen "
        "for too many chunks.", 120), fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.095)
    fig.savefig(a.out, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
