"""
Render the progress animations of the README and the project page from recorded episodes.

What is drawn (the commit-realize loop of PACE, chunk by chunk):
  - orange: realized progress, the anchored progress p_anchor of the frames (frozen encoder, projected onto the
    predicted foresight anchor);
  - blue: committed progress. Each chunk first grows its forward commitment dp_cur out of the current point as a
    dashed segment to a hollow target; once the chunk has run, the target joins the solid committed line;
  - at each check the gap between the two lines (red = unmet, green = exceeded) drops into the meter of accumulated
    unmet commitments, which is the failure alarm's shortfall CUSUM (exactly the residual the alarm uses);
  - a fixed zoom window above the curve follows the latest chunks.

Each clip is one recorded episode:
  LIBERO       rollout_*.mp4 + progress_*.json written by examples/LIBERO/eval_libero.sh
  SimplerEnv   seed*_frames.npz + seed*_progress.json written by examples/SimplerEnv/eval_simpler.sh (SIMPLER_RECORD_DIR)

    # one clip at a time with task tabs (README showreel); --bare drops tabs for one-clip videos
    python -m tools.progress_viz.render_showreel --layout player --out assets/pace_showreel.webp \\
        --clip libero "LIBERO · Long" <rollout.mp4> <progress.json> \\
        --clip simpler "WidowX · Carrot" <seed4_success_frames.npz> <seed4_success_progress.json>
    # all clips side by side (--layout wall), or a success and a failure of the same task (--layout versus)

The alarm is recomputed with pace.model.progress.audit and the shipped thresholds (shortfall 0.4 on every benchmark; stall
limit 18 chunks of 8 steps on LIBERO, 2.12 chunks of 16 steps on SimplerEnv).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from pace.model.progress.audit import update_audit  # noqa: E402

BG, PANEL, EDGE, TRACK = "#0d1117", "#161b22", "#30363d", "#21262d"
TEXT, MUTED = "#e6edf3", "#8b949e"
BLUE, ORANGE, GREEN, AMBER, RED = "#58a6ff", "#f0883e", "#3fb950", "#d29922", "#f85149"
KINDS = {"libero": dict(threshold=0.4, stall_chunks=18.0, chunk_steps=8),
         "simpler": dict(threshold=0.4, stall_chunks=2.12, chunk_steps=16)}
plt.rcParams["font.family"] = ["Avenir Next", "Helvetica Neue", "DejaVu Sans"]


def _num(v):
    return np.nan if v is None else float(v)


def load_clip(kind, label, media, js):
    d = json.load(open(js))
    cfg = KINDS[kind]
    if kind == "libero":
        import av
        frames = [f.to_ndarray(format="rgb24") for f in av.open(media).decode(video=0)]
        T = min(len(frames), len(d["progress"]))
        p = np.array([_num(v) for v in d["progress"][:T]])
        dp = np.array([_num(v) for v in d["commitment"][:T]])
        m = np.array([_num(v) for v in d["observed_progress"][:T]])
        res = [_num(v) for v in d["audit_delta"][:T]]
        queries = [s for s in d["query_steps"] if s < T]
        text = d.get("task", "")
    else:
        frames = list(np.load(media)["frames"])
        T = min(len(frames), len(d["predicted_progress"]))
        p = np.array(d["predicted_progress"][:T], float)
        dp = np.array(d["predicted_delta_progress"][:T], float)
        m = np.array(d["observed_progress_dino"][:T], float)
        res = [_num(v) for v in d["progress_audit_delta"][:T]]
        queries = list(range(T))
        text = d.get("instruction", d.get("task", ""))
    state, score, cus, qi = None, np.zeros(T), np.zeros(T), 0
    r, cu = 0.0, 0.0
    for s in range(T):
        if qi < len(queries) and queries[qi] == s:
            state, cu, st, _ = update_audit(state, s, res[s], p[s], cfg["chunk_steps"], threshold=cfg["threshold"],
                                            stall_chunks=cfg["stall_chunks"])
            r = max(cu / cfg["threshold"], st / cfg["stall_chunks"])
            qi += 1
        score[s], cus[s] = r, cu
    m[0] = 0.0
    m = _ffill(m)
    # Commit-realize chunks, the way the alarm checks them: the commitment dp made at t0 against the anchored progress
    # realized by t1 (LIBERO: the next query, 8 steps later; per-step SimplerEnv: Delta = 15 steps later). The alarm's
    # residual at t1 decides whether the commitment was kept.
    res_arr = np.array(res, float)
    dpn = np.nan_to_num(dp)
    if kind == "libero":
        span, starts = cfg["chunk_steps"], queries
        qs = np.array(queries)
        rv = np.interp(np.arange(T), qs, m[qs])  # anchored progress is measured at the queries; drawn linearly between
    else:
        span = cfg["chunk_steps"] - 1
        starts, rv = list(range(0, T, span)), m.copy()
    chunks = []
    for t0 in starts:
        t1 = t0 + span
        ok = t1 < T and np.isfinite(res_arr[t1])
        chunks.append(dict(t0=t0, t1=t1, base=float(rv[t0]), commit=float(dpn[t0]),
                           res=float(res_arr[t1]) if ok else None))
    return dict(label=label, frames=frames[:T], p=p, dp=dpn, m=m, rv=rv, chunks=chunks, score=score, cus=cus,
                thr=cfg["threshold"], T=T,
                success=bool(float(d["success"]) > 0), text=text.strip().rstrip(".").capitalize(),
                delta=cfg["chunk_steps"] - 1, span=span)


def _style(ax, labelsize=8):
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(EDGE)
    ax.tick_params(colors=MUTED, labelsize=labelsize, length=0)
    ax.grid(True, color=TRACK, lw=0.8)


FLIGHT = 5  # frames a check chip takes to drop into the alarm meter


def _ease(u):
    return u * u  # accelerate like a falling object


def _checks(c):
    """Chunk ends where a commitment is checked: {t1: chunk index}."""
    return {ch["t1"]: i for i, ch in enumerate(c["chunks"]) if ch["res"] is not None}


def _timeline(clips, stride, flight, grow=3):
    """Frames as (t, {clip: (chunk, u)}, {clip: (chunk, g)}). At every check environment time stops while the gap
    drops into the meter (u: 0 -> 1); then the next commitment grows out of the current point (g: 0 -> 1)."""
    T = max(c["T"] for c in clips)
    checks = [_checks(c) for c in clips]
    starts = [{ch["t0"]: i for i, ch in enumerate(c["chunks"])} for c in clips]
    steps = sorted(set(range(0, T, stride)) | {t1 for ck in checks for t1 in ck if t1 < T}
                   | {t0 for st in starts for t0 in st if t0 < T} | {T - 1})
    out = []
    for t in steps:
        hit = {ci: ck[t] for ci, ck in enumerate(checks) if t in ck}
        new = {ci: st[t] for ci, st in enumerate(starts) if t in st and t < clips[ci]["T"] - 1}
        if hit:
            for f in range(flight + 1):
                out.append((t, {ci: (k, f / flight) for ci, k in hit.items()}, {ci: (k, 0.0) for ci, k in new.items()}))
        if new:
            for g in range(1, grow + 1):
                out.append((t, {}, {ci: (k, g / grow) for ci, k in new.items()}))
        if not hit and not new:
            out.append((t, {}, {}))
    return out


def _state(c, t, fl):
    """(unmet commitments S, alarm score) shown at this frame: the check's update lands when its chip lands."""
    if fl is not None and fl[1] < 1.0 and t > 0:
        return float(c["cus"][t - 1]), float(c["score"][t - 1])
    return float(c["cus"][t]), float(c["score"][t])


def _anim_curve(ax, c, t, fl, small=False, gr=None):
    """Two solid lines: committed (blue, the target each chunk promised to reach) and realized (orange, anchored
    progress). Each chunk first grows its commitment out of the current point as a dashed arrow to a hollow target;
    once the chunk is checked the target joins the solid committed line, and the gap to the realized line at that
    moment (red = unmet, green = exceeded) drops into the alarm meter."""
    rv, T = c["rv"], c["T"]
    sc = 0.6 if small else 1.0
    lw = 2.2 * sc
    cx, cy = [], []  # vertices of the solid committed line
    for i, ch in enumerate(c["chunks"]):
        t0, t1, base = ch["t0"], ch["t1"], ch["base"]
        if t0 > t:
            break
        tip = base + ch["commit"]
        if not cx:
            cx.append(t0); cy.append(base)
        live = fl is not None and fl[0] == i
        pending = t < t1 or (live and fl[1] < 0.35)
        if pending:  # the commitment for the running chunk, drawn ahead of execution
            g = gr[1] if (gr is not None and gr[0] == i) else 1.0
            if g <= 0:
                continue
            xe, ye = t0 + g * (t1 - t0), base + g * (tip - base)
            ax.plot([t0, xe], [base, ye], color=BLUE, lw=8 * sc, alpha=0.16, zorder=6, solid_capstyle="round")
            ax.plot([t0, xe], [base, ye], color="#8cc4ff", lw=2.6 * sc, ls=(0, (1.6, 1.3)), zorder=6)
            if g >= 1.0:
                ax.scatter([t1], [tip], s=70 * sc, facecolor=PANEL, edgecolor="#8cc4ff", linewidth=2.2 * sc, zorder=8)
            if t < t1:
                continue
        else:
            cx.append(t1); cy.append(tip)
        if ch["res"] is None:
            continue
        e = float(rv[min(t1, T - 1)])
        col = RED if ch["res"] < 0 else GREEN
        ax.plot([t1, t1], [e, tip], color=col, lw=(4.2 if live else 1.8) * sc, alpha=1.0 if live else 0.75,
                zorder=7, solid_capstyle="butt")
    ax.plot(cx, cy, color=BLUE, lw=lw, zorder=6, solid_joinstyle="round")
    xs = np.arange(t + 1)
    ax.plot(xs, rv[xs], color=ORANGE, lw=lw, zorder=5, solid_joinstyle="round")
    ax.scatter([t], [rv[t]], s=34 * sc, color=ORANGE, edgecolor=TEXT, linewidth=0.9, zorder=9)


ZOOM_TOP = 1.8  # the main progress axis runs to this height; the band above 1.15 holds the fixed zoom window
ZOOM_MIN_STEPS = 80  # short episodes (SimplerEnv, about 30 steps) are readable without a zoom window


def _wants_zoom(c):
    return c["T"] + c["span"] > ZOOM_MIN_STEPS


def _zoom_inset(ax, c, t, fl, gr, rect=(0.5, 0.665, 0.495, 0.33)):
    """Picture-in-picture in a fixed spot above the curve: the latest chunks, magnified at a fixed scale (5 chunks
    wide, 0.3 progress high on LIBERO) and panning with time; its region is outlined on the main plot."""
    rv, T, span = c["rv"], c["T"], c["span"]
    W = min(5 * span, max(24, int(0.5 * (T + span))))
    Hy = 0.3 if span <= 8 else 0.5
    x0 = max(0, t - int(0.7 * W))
    # Frame what matters now: the current point, the live commitment and the chunk just checked. The window keeps
    # its fixed height unless those points alone need more; older history slides out like a following camera.
    need = [float(rv[t])]
    for k, ch in enumerate(c["chunks"]):
        live = ch["t0"] <= t < ch["t1"]
        just_checked = ch["t1"] <= t and ch["t1"] > t - span
        if live or just_checked or (fl is not None and fl[0] == k):
            need += [ch["base"], ch["base"] + ch["commit"]]
            if ch["t1"] <= t:
                need.append(float(rv[min(ch["t1"], T - 1)]))
    lo, hi = min(need), max(need)
    Hy = max(Hy, hi - lo + 0.08)
    y0 = (lo + hi) / 2 - Hy / 2
    y0 = min(max(y0, -0.06), 1.16 - Hy) if Hy < 1.22 else -0.06
    axi = ax.inset_axes(list(rect), zorder=12)
    axi.set_facecolor("#0f141b")
    for sp in axi.spines.values():
        sp.set_color("#8cc4ff"); sp.set_alpha(0.55); sp.set_linewidth(1.0)
    axi.tick_params(colors=MUTED, labelsize=6.5, length=0)
    axi.grid(True, color=TRACK, lw=0.6)
    _anim_curve(axi, c, t, fl, gr=gr)
    axi.set_xlim(x0, x0 + W); axi.set_ylim(y0, y0 + Hy)
    axi.text(0.015, 0.95, "zoom", transform=axi.transAxes, fontsize=7, color="#8cc4ff", ha="left", va="top")
    ax.add_patch(Rectangle((x0, y0), W, Hy, fill=False, ec="#8cc4ff", lw=0.9, ls=(0, (2, 2)), alpha=0.75, zorder=11))
    return axi


def _main_axes(ax, xmax, zoom=True):
    if not zoom:
        ax.set_xlim(0, xmax); ax.set_ylim(-0.04, 1.12)
        return
    ax.set_xlim(0, xmax); ax.set_ylim(-0.04, ZOOM_TOP)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(False)
    for yv in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        ax.axhline(yv, color=TRACK, lw=0.8, zorder=0)


def _chip(fig, ax, c, fl, meter, unmet_before):
    """The gap of the chunk being checked, dropping from the curve into the alarm meter."""
    if fl is None:
        return
    k, u = fl
    ch = c["chunks"][k]
    tip = ch["base"] + ch["commit"]
    e = float(c["rv"][min(ch["t1"], c["T"] - 1)])
    x0, y0 = fig.transFigure.inverted().transform(ax.transData.transform((ch["t1"], (tip + e) / 2)))
    mx, my, mw, mh, scale = meter
    w, h = 0.05, 0.032
    x1, y1 = min(mx + mw * min(unmet_before / scale, 1.0), mx + mw - w / 2), my + mh / 2
    v = _ease(u)
    x, y = x0 + (x1 - x0) * v, y0 + (y1 - y0) * v
    short = ch["res"] < 0
    col = RED if short else GREEN
    rounded(fig, x - w / 2, y - h / 2, w, h, col, r=0.008, z=10)
    fig.text(x, y, f"{-ch['res']:+.2f}", fontsize=8.5, color=BG if not short else TEXT, weight="bold", ha="center",
             va="center", zorder=11)


def _meter(fig, c, unmet, meter, label=True, fontsize=10):
    mx, my, mw, mh, scale = meter
    thr = c["thr"]
    frac = unmet / scale
    col = GREEN if unmet < 0.6 * thr else (AMBER if unmet <= thr else RED)
    rounded(fig, mx, my, mw, mh, TRACK, r=0.008, z=-1)
    rounded(fig, mx, my, mw * min(frac, 1.0), mh, col, r=0.008, z=1)
    xt = mx + mw * thr / scale
    fig.lines.append(plt.Line2D([xt] * 2, [my - 0.01, my + mh + 0.01], transform=fig.transFigure, color=TEXT, lw=1.2))
    if label:
        fig.text(mx, my + mh + 0.022, "unmet commitments, accumulated", fontsize=fontsize, color=MUTED, va="center")
        fig.text(mx + mw, my + mh + 0.022, f"{unmet:0.2f}  /  alarm at {thr:g}", fontsize=fontsize - 1, color=col,
                 ha="right", va="center")


def _anim_legend(ax, zoom, fontsize=8):
    """Symbol legend: in the empty band left of the zoom window, or in the empty lower right without one."""
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, lw=2.2, label="committed"),
          Line2D([], [], color="#8cc4ff", lw=2.2, ls=(0, (1.6, 1.3)), marker="o", markersize=6.5, markevery=[1],
                 markerfacecolor=PANEL, markeredgecolor="#8cc4ff", markeredgewidth=1.6,
                 label=r"next commitment $\Delta p_{\mathrm{cur}}$"),
          Line2D([], [], color=ORANGE, lw=2.2, label=r"realized $p^{\mathrm{anchor}}$"),
          Line2D([], [], color=RED, ls="None", marker="|", markersize=11, markeredgewidth=3.2, label="gap: unmet"),
          Line2D([], [], color=GREEN, ls="None", marker="|", markersize=11, markeredgewidth=3.2, label="gap: exceeded")]
    kw = dict(handles=hs, fontsize=fontsize, frameon=False, labelcolor=MUTED, handlelength=2.0, handletextpad=0.6,
              labelspacing=0.32, borderaxespad=0.25)
    if zoom:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), **kw)
    else:
        ax.legend(loc="lower right", **kw)


def _ffill(x):
    out, last = x.copy(), 0.0
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    return out


def rounded(fig, x, y, w, h, color, edge=None, r=0.012, lw=1.0, z=-2):
    if w <= 0 or h <= 0:
        return
    r = min(r, w / 2, h / 2)
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=fig.transFigure,
                                      fc=color, ec=edge or color, lw=lw, zorder=z))


def draw(clips, k, t, frac, done, args, path, fl=None, gr=None):
    c = clips[k]
    fig = plt.figure(figsize=(12.8, 6.6), dpi=100, facecolor=BG)
    if getattr(args, "bare", False):  # no header, tabs or footer (the page around the video provides them)
        clips = [c]
    # header + tabs
    if not getattr(args, "bare", False):
        _header_tabs(fig, clips, k, frac)
    _player_body(fig, c, t, done, args, fl, gr)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def _header_tabs(fig, clips, k, frac):
    fig.text(0.03, 0.935, "PACE", fontsize=22, color=TEXT, weight="bold", va="center")
    fig.text(0.105, 0.935, "reads its own progress, commits to the next step, and checks itself, online",
             fontsize=12.5, color=MUTED, va="center")
    n = len(clips)
    x0, gap = 0.03, 0.006
    w = (0.94 - gap * (n - 1)) / n
    for i, cl in enumerate(clips):
        x = x0 + i * (w + gap)
        active = i == k
        rounded(fig, x, 0.835, w, 0.05, BLUE if active else PANEL, EDGE if not active else BLUE, r=0.02, z=0)
        fig.text(x + w / 2, 0.86, cl["label"], ha="center", va="center", fontsize=9.5,
                 color=BG if active else MUTED, weight="bold" if active else "normal")
        if active:
            rounded(fig, x, 0.822, w * frac, 0.005, BLUE, r=0.002, z=0)


def _player_body(fig, c, t, done, args, fl=None, gr=None):
    unmet, sc = _state(c, t, fl)
    # video panel
    rounded(fig, 0.03, 0.1, 0.46, 0.7, PANEL, EDGE)
    axv = fig.add_axes([0.04, 0.17, 0.44, 0.6], zorder=2)
    axv.imshow(c["frames"][t])
    axv.set_xticks([]); axv.set_yticks([])
    for sp in axv.spines.values():
        sp.set_visible(False)
    axv.set_facecolor(PANEL)
    fig.text(0.045, 0.74, f"step {t}", fontsize=10, color=TEXT, va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#000000aa", ec="none"))
    if done:
        badge, col = ("✓ SUCCESS", GREEN) if c["success"] else ("✗ FAILED", RED)
    elif sc > 1.0:
        badge, col = ("⚠ FAILURE ALARM", RED)
    else:
        badge, col = ("● RUNNING", BLUE)
    fig.text(0.475, 0.74, badge, fontsize=10.5, color=BG, weight="bold", ha="right", va="center",
             bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none"))
    fig.text(0.26, 0.13, f"“{c['text']}”", fontsize=11, color=TEXT, ha="center", va="center", style="italic",
             wrap=True)
    # right panel
    rounded(fig, 0.51, 0.1, 0.46, 0.7, PANEL, EDGE)
    p, dp = float(np.nan_to_num(c["p"][t])), float(c["dp"][t])
    fig.text(0.53, 0.755, r"current progress  $p_{\mathrm{cur}}$", fontsize=12, color=MUTED, va="center")
    fig.text(0.95, 0.745, f"{p:0.2f}", fontsize=26, color=TEXT, weight="bold", ha="right", va="center")
    fig.text(0.53, 0.715, rf"forward commitment  $\Delta p_{{\mathrm{{cur}}}}$ = {(round(dp, 2) + 0.0):+0.2f}", fontsize=10,
             color=BLUE, va="center")
    axp = fig.add_axes([0.555, 0.29, 0.395, 0.375], zorder=2)
    _style(axp, 7.5)
    _anim_curve(axp, c, t, fl, gr=gr)
    zoom = _wants_zoom(c)
    _main_axes(axp, c["T"] + c["span"], zoom)
    axz = _zoom_inset(axp, c, t, fl, gr) if zoom else axp
    _anim_legend(axp, zoom, 8)
    meter = (0.555, 0.155, 0.395, 0.022, 1.5 * c["thr"])
    _meter(fig, c, unmet, meter, fontsize=9.5)
    _chip(fig, axz, c, fl, meter, unmet)
    if not getattr(args, "bare", False):
        fig.text(0.97, 0.04, args.footer, fontsize=8.5, color=MUTED, ha="right", va="center")


def _square(img):
    h, w = img.shape[:2]
    if w > h:
        x = (w - h) // 2
        return img[:, x:x + h]
    y = (h - w) // 2
    return img[y:y + w]


def draw_wall(clips, u, done, args, path):
    """All clips at once, each advanced to the same fraction u of its own episode."""
    n = len(clips)
    cols = args.cols
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(12.8, 1.3 + 3.55 * rows), dpi=100, facecolor=BG)
    H = 1.3 + 3.55 * rows
    fig.text(0.02, 1 - 0.42 / H, "PACE", fontsize=20, color=TEXT, weight="bold", va="center")
    fig.text(0.088, 1 - 0.42 / H, args.wall_title, fontsize=12, color=MUTED, va="center")
    cw, gap = (0.96 - 0.012 * (cols - 1)) / cols, 0.012
    ch = 3.4 / H
    for i, c in enumerate(clips):
        r, k = divmod(i, cols)
        x = 0.02 + k * (cw + gap)
        y = 1 - (0.8 + (r + 1) * 3.55) / H + 0.08 / H
        rounded(fig, x, y, cw, ch, PANEL, EDGE, r=0.01)
        t = min(int(round(u * (c["T"] - 1))), c["T"] - 1)
        vid_h = ch * 0.70
        axv = fig.add_axes([x + 0.006, y + ch - vid_h - 0.035 * 3.55 / H * 1.6, cw - 0.012, vid_h], zorder=2)
        axv.imshow(_square(c["frames"][t]))
        axv.set_xticks([]); axv.set_yticks([])
        for sp in axv.spines.values():
            sp.set_visible(False)
        fig.text(x + 0.008, y + ch - 0.025 * 3.55 / H * 1.2, c["label"], fontsize=9.5, color=TEXT, weight="bold",
                 va="center")
        alarm = c["score"][t] > 1.0
        if done or t >= c["T"] - 1:
            badge, col = ("✓", GREEN) if c["success"] else ("✗", RED)
        elif alarm:
            badge, col = ("⚠", RED)
        else:
            badge, col = None, None
        if badge:
            fig.text(x + cw - 0.008, y + ch - 0.025 * 3.55 / H * 1.2, badge, fontsize=11, color=col, weight="bold",
                     ha="right", va="center")
        axc = fig.add_axes([x + 0.008, y + 0.012, cw * 0.64, ch * 0.17], zorder=2)
        axc.set_facecolor(PANEL)
        for sp in axc.spines.values():
            sp.set_visible(False)
        axc.set_xticks([]); axc.set_yticks([])
        axc.axhline(1.0, color=TRACK, lw=1)
        axc.axhline(0.0, color=TRACK, lw=1)
        _anim_curve(axc, c, t, None, small=True)
        axc.set_xlim(0, c["T"] - 1 + c["span"]); axc.set_ylim(-0.08, 1.12)
        fig.text(x + cw - 0.008, y + 0.012 + ch * 0.085, f"{float(np.nan_to_num(c['p'][t])):0.2f}", fontsize=15,
                 color=TEXT, weight="bold", ha="right", va="center")
        fig.text(x + cw - 0.008, y + 0.012 + ch * 0.085 + 0.034 * 3.55 / H * 1.4, "progress", fontsize=7.5,
                 color=MUTED, ha="right", va="center")
    fig.text(0.98, 0.2 / H, args.footer, fontsize=8.5, color=MUTED, ha="right", va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def draw_versus(clips, t, args, path, fls=None, grs=None):
    """Two episodes side by side (e.g. a success and a failure of the same task), advanced in environment steps."""
    fls, grs = fls or {}, grs or {}
    fig = plt.figure(figsize=(12.8, 7.0), dpi=100, facecolor=BG)
    H = 7.0
    fig.text(0.02, 1 - 0.38 / H, "PACE", fontsize=20, color=TEXT, weight="bold", va="center")
    fig.text(0.088, 1 - 0.38 / H, args.wall_title, fontsize=12, color=MUTED, va="center")
    if clips[0]["text"]:
        fig.text(0.02, 1 - 0.78 / H, f"Task: “{clips[0]['text']}”", fontsize=10.5, color=TEXT, va="center",
                 style="italic")
    T = max(c["T"] for c in clips)
    for i, c in enumerate(clips):
        x, w = 0.02 + i * 0.485, 0.475
        s = min(t, c["T"] - 1)
        fl = fls.get(i) if t <= c["T"] - 1 else None
        unmet, sc = _state(c, s, fl)
        alarm_step = next((q for q in range(c["T"]) if c["score"][q] > 1.0), None)
        alarmed = sc > 1.0 or (alarm_step is not None and s > alarm_step)
        finished = t >= c["T"] - 1
        edge, lw = (RED, 3) if alarmed else ((GREEN, 3) if finished and c["success"] else (EDGE, 1))
        rounded(fig, x, 0.03, w, 0.83, PANEL, edge, r=0.01, lw=lw)
        fig.text(x + 0.012, 0.83, c["label"], fontsize=11, color=TEXT, weight="bold", va="center")
        axv = fig.add_axes([x + 0.012, 0.515, w * 0.4, 0.295], zorder=2)
        axv.imshow(_square(c["frames"][s]))
        axv.set_xticks([]); axv.set_yticks([])
        for sp in axv.spines.values():
            sp.set_visible(False)
        fig.text(x + 0.02, 0.785, f"step {s}", fontsize=9, color=TEXT, va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#000000aa", ec="none"))
        if alarmed:
            fig.text(x + 0.012 + w * 0.2, 0.54, f"⚠ FAILURE ALARM · step {alarm_step}", fontsize=10, color=TEXT,
                     weight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc=RED, ec="none"))
        elif finished:
            txt, col = ("✓ SUCCESS", GREEN) if c["success"] else ("TIME LIMIT", MUTED)
            fig.text(x + 0.012 + w * 0.2, 0.54, f"{txt} · step {c['T']}", fontsize=10, color=BG, weight="bold",
                     ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none"))
        p, dp = float(np.nan_to_num(c["p"][s])), float(c["dp"][s])
        fig.text(x + w - 0.012, 0.79, f"{p:0.2f}", fontsize=24, color=TEXT, weight="bold", ha="right", va="center")
        fig.text(x + w - 0.012, 0.745, r"current progress $p_{\mathrm{cur}}$", fontsize=9.5, color=MUTED, ha="right",
                 va="center")
        fig.text(x + w - 0.012, 0.69, rf"forward commitment $\Delta p_{{\mathrm{{cur}}}}$ = {(round(dp, 2) + 0.0):+0.2f}",
                 fontsize=9.5, color=BLUE, ha="right", va="center")
        mcol = GREEN if sc < 0.6 else (AMBER if sc <= 1.0 else RED)
        fig.text(x + w - 0.012, 0.625, "ALARM" if alarmed else "quiet", fontsize=18, color=RED if alarmed else mcol,
                 weight="bold", ha="right", va="center")
        fig.text(x + w - 0.012, 0.585, "failure alarm", fontsize=9.5, color=MUTED, ha="right", va="center")
        axp = fig.add_axes([x + 0.04, 0.155, w - 0.055, 0.32], zorder=2)
        _style(axp, 7.5)
        _anim_curve(axp, c, s, fl, gr=grs.get(i) if t <= c["T"] - 1 else None)
        zoom = _wants_zoom(c)
        _main_axes(axp, T, zoom)
        if alarm_step is not None and s >= alarm_step and alarmed:
            axp.axvline(alarm_step, color=RED, lw=1.2, alpha=0.8, zorder=3,
                        ymax=1.16 / (ZOOM_TOP + 0.04) if zoom else 1.0)
        _anim_legend(axp, zoom, 7.5)
        axz = (_zoom_inset(axp, c, s, fl, grs.get(i) if t <= c["T"] - 1 else None, rect=(0.45, 0.655, 0.545, 0.34))
               if zoom else axp)
        meter = (x + 0.04, 0.065, w - 0.055, 0.02, 1.5 * c["thr"])
        _meter(fig, c, unmet, meter, fontsize=8.5)
        _chip(fig, axz, c, fl, meter, unmet)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs=4, action="append", metavar=("KIND", "LABEL", "MEDIA", "JSON"), required=True)
    ap.add_argument("--out", required=True, help=".webp, .gif or .mp4")
    ap.add_argument("--quality", type=int, default=80, help="WebP quality")
    ap.add_argument("--frames_per_clip", type=int, default=36)
    ap.add_argument("--hold", type=int, default=10, help="frames to hold the end of each clip")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--footer", default="released checkpoints · raw readings, no smoothing")
    ap.add_argument("--layout", choices=["player", "wall", "versus"], default="player",
                    help="player: one clip at a time with tabs; wall: all clips side by side, in sync; versus: two "
                         "episodes side by side in environment steps (pauses when an alarm fires)")
    ap.add_argument("--stride", type=int, default=4, help="player / versus: environment steps per frame")
    ap.add_argument("--flight", type=int, default=FLIGHT, help="frames for a checked gap to drop into the alarm meter")
    ap.add_argument("--grow", type=int, default=3, help="frames for a new commitment to grow out of the current point")
    ap.add_argument("--cols", type=int, default=4, help="wall: clips per row")
    ap.add_argument("--bare", action="store_true", help="player: drop the header, tabs and footer (one clip per file)")
    ap.add_argument("--wall_title", default="one policy reading its own progress, live, on every task")
    args = ap.parse_args()
    clips = [load_clip(k, lab, m, j) for k, lab, m, j in args.clip]
    tmp = tempfile.mkdtemp()
    try:
        idx = 0
        if args.layout == "wall":
            us = list(np.linspace(0, 1, args.frames_per_clip)) + [1.0] * args.hold
            for j, u in enumerate(us):
                draw_wall(clips, float(u), j >= args.frames_per_clip, args, os.path.join(tmp, f"f{idx:05d}.png"))
                idx += 1
            clips = []
        if args.layout == "versus":
            T = max(c["T"] for c in clips)
            frames = _timeline(clips, args.stride, args.flight, args.grow) + [(T - 1, {}, {})] * args.hold
            for t, fls, grs in frames:
                draw_versus(clips, t, args, os.path.join(tmp, f"f{idx:05d}.png"), fls, grs)
                idx += 1
            clips = []
        for k, c in enumerate(clips):
            stride = args.stride if c["T"] >= 15 * args.stride else 1  # short episodes play every step
            frames = _timeline([c], stride, args.flight, args.grow)
            for j, (t, fls, grs) in enumerate(frames):
                draw(clips, k, t, (j + 1) / len(frames), False, args, os.path.join(tmp, f"f{idx:05d}.png"), fls.get(0),
                     grs.get(0))
                idx += 1
            for _ in range(args.hold):
                draw(clips, k, c["T"] - 1, 1.0, True, args, os.path.join(tmp, f"f{idx:05d}.png"))
                idx += 1
        pattern = os.path.join(tmp, "f%05d.png")
        scale = f"scale={args.width}:-2:flags=lanczos"
        if args.bare and args.layout == "player":  # keep only the two panels
            scale = f"crop=iw:trunc(ih*0.74/2)*2:0:trunc(ih*0.185),{scale}"
        if args.out.endswith(".webp"):  # animated WebP: smaller than GIF and always plays on GitHub
            from PIL import Image
            frames = [Image.open(os.path.join(tmp, f"f{i:05d}.png")).convert("RGB") for i in range(idx)]
            if args.bare and args.layout == "player":
                frames = [f.crop((0, int(f.height * 0.185), f.width, int(f.height * 0.925))) for f in frames]
            h = int(round(frames[0].height * args.width / frames[0].width / 2) * 2)
            frames = [f.resize((args.width, h), Image.LANCZOS) for f in frames]
            frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0,
                           quality=args.quality, method=6)
        elif args.out.endswith(".gif"):
            pal = os.path.join(tmp, "palette.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern, "-vf",
                            f"{scale},palettegen=max_colors=256:stats_mode=full", pal], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern, "-i", pal,
                            "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=sierra2_4a", args.out], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern, "-vf",
                            scale, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out], check=True)
    finally:
        shutil.rmtree(tmp)
    print("wrote", args.out, f"({os.path.getsize(args.out) / 1e6:.1f} MB, {idx} frames)")


if __name__ == "__main__":
    main()
