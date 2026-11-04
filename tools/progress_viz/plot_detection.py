"""
Plot when the failure alarm fires, from a detection summary JSON (docs/detection_libero_release.json): on LIBERO-Long,
the steps at which the alarm fires in failed episodes against the steps at which successful episodes finish.

    python -m tools.progress_viz.plot_detection --data docs/detection_libero_release.json --out assets/pace_alarm_timing.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3", "#fcfcfb"
NEUTRAL, CRITICAL = "#c9c8c3", "#d03b3b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.data))
    tl = d["_timeline_libero_10"]
    limit = tl["time_limit"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=180, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=2)
    bins = np.arange(0, limit + 20, 20)
    ax.hist(tl["success_lengths"], bins=bins, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5,
            label="successful episodes: step at which the task is done")
    alarms = [s for s in tl["failure_alarm_steps"] if s is not None]
    top = ax.get_ylim()[1]
    yj = top * (1.04 + 0.05 * (np.arange(len(alarms)) % 3))
    ax.scatter(alarms, yj, s=46, color=CRITICAL, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="failed episodes: step at which the alarm fires")
    med_s, med_a = np.median(tl["success_lengths"]), np.median(alarms)
    ytop = top * 1.32
    for x, col, txt, ha in ((med_s, INK2, f"median success\nstep {med_s:.0f}", "right"),
                            (med_a, CRITICAL, f"median alarm\nstep {med_a:.0f}", "left"),
                            (limit, INK2, "time\nlimit", "right")):
        ax.axvline(x, color=col, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(x + (6 if ha == "left" else -6), ytop, txt, ha=ha, va="top", fontsize=9, color=col)
    ax.set_ylim(0, top * 1.36)
    ax.set_xlim(0, limit + 15)
    ax.set_xlabel("environment step (LIBERO-Long)", fontsize=10, color=INK2)
    ax.set_ylabel("successful episodes", fontsize=10, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2, fontsize=9, frameon=False, labelcolor=INK2)
    ax.set_title(f"The alarm fires around when a success would be done, halfway to the time limit\n"
                 f"({len(alarms)} of {len(tl['failure_alarm_steps'])} failed LIBERO-Long episodes flagged)",
                 fontsize=11, color=INK, loc="left")
    fig.savefig(a.out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
