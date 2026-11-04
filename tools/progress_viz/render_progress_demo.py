"""
Render side-by-side progress demos: left = rollout frames, right = online predicted progress vs true progress.

    python -m tools.progress_viz.render_progress_demo --traces <out>/traces.json --out_dir <DIR> \
        [--select i j ...] [--fps 20] [--height 360]

Frames come from the LIBERO rollout video (`video`) or the recorded SimplerEnv frames (`frames_npz`) referenced in
traces.json. Output: <out_dir>/<index>_<group>_<episode>_<outcome>.mp4 (H.264).
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import numpy as np


def demo_filename(idx: int, row: dict) -> str:
    outcome = "success" if row["success"] else "failure"
    return f"{idx:03d}_{(row.get('suite') or 'simpler')}_{row.get('ep', row.get('seed', 0))}_{outcome}.mp4"


def load_frames(row):
    if row.get("frames_npz"):
        return list(np.load(row["frames_npz"])["frames"])
    import av

    with av.open(row["video"]) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def resize_h(img, h):
    from PIL import Image

    w = int(round(img.shape[1] * h / img.shape[0]))
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def render(row, idx: int, out_dir: str, fps: int, height: int) -> str:
    import av
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = load_frames(row)
    pred, true = np.asarray(row["pred"], float), np.asarray(row["true"], float)
    n = min(len(frames), len(pred), len(true))
    frames, pred, true = frames[:n], pred[:n], true[:n]
    fig, ax = plt.subplots(figsize=(height * 1.25 / 100, height / 100), dpi=100)
    xs = np.arange(n)
    (lp,) = ax.plot([], [], color="#1f6fd1", lw=2.2, label="predicted progress")
    (lt,) = ax.plot([], [], color="#e07b1a", lw=2.2, ls="--", label="true progress")
    vline = ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("env step")
    ax.set_ylabel("progress")
    outcome = "SUCCESS" if row["success"] else "FAILURE"
    title = (row.get("instruction") or row.get("task") or "").replace("_", " ")
    title = "\n".join(textwrap.wrap(title, 52)[:2])
    ax.set_title(f"{title}  [{outcome}]", fontsize=7.5, color="#2a7d2a" if row["success"] else "#c0392b")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    name = demo_filename(idx, row)
    out = av.open(os.path.join(out_dir, name), mode="w")
    stream = None
    for t in range(n):
        lp.set_data(xs[: t + 1], pred[: t + 1])
        if not np.isnan(true).all():
            lt.set_data(xs[: t + 1], true[: t + 1])
        vline.set_xdata([t, t])
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        left = resize_h(np.asarray(frames[t], dtype=np.uint8), height)
        plot = resize_h(plot, height)
        canvas = np.concatenate([left, plot], axis=1)
        if canvas.shape[1] % 2:
            canvas = canvas[:, :-1]
        if stream is None:
            stream = out.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = canvas.shape[1], canvas.shape[0], "yuv420p"
            stream.options = {"crf": "20"}
        for pkt in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
            out.mux(pkt)
    if stream is not None:
        for pkt in stream.encode():
            out.mux(pkt)
    out.close()
    plt.close(fig)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--select", type=int, nargs="*", default=None, help="row indices (default: all rows)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    with open(args.traces) as f:
        rows = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    for idx in args.select if args.select is not None else range(len(rows)):
        name = render(rows[idx], idx, args.out_dir, args.fps, args.height)
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
