# Checked by viewing frames and comparing with labels.csv.
"""Render one held-out real run: what the model predicted beside what the robot's sensors said.

Left: the model's prediction on a run it never saw in training or tuning. Right: the label from shaking and
roll. Bottom: the sensor signal along the line, with both as coloured bands.

    python3 -m presentation.render_real_run --run s1c_r006 --data DATA --bags BAGS --scores SCORES.npz --out OUT.mp4
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from learning.real_data import load_rows
from scripts.episode_io import load_episode, project_base_point
from scripts.frame_labels_v2 import JACKAL_REAL, Settings, episode_signals
from scripts.rosbag_to_episode import decode_image, read_bag

W, H, PANEL_W, PANEL_H = 1920, 1080, 960, 720
GREEN, RED, GRAY = (40, 170, 70), (200, 40, 40), (150, 150, 150)
HALF_WIDTH_M = 0.24
FONT = "/System/Library/Fonts/Helvetica.ttc"


def signal_panel(signals, thresholds, zones, frames, prediction, label, length):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(W / 100, (H - PANEL_H) / 100), dpi=100)
    ax = fig.add_axes([0.07, 0.15, 0.91, 0.56])
    bands = fig.add_axes([0.07, 0.74, 0.91, 0.22], sharex=ax)
    p, valid = signals["progress"], signals["valid"]
    severity = np.maximum(signals["vib"] / (2 * thresholds["vib_mps2"]), signals["roll"] / (2 * thresholds["roll_rps"]))
    for z in zones:
        if z["unacceptable"]:
            ax.axvspan(z["lo"], z["hi"], color="#c82828", alpha=0.16, lw=0)
    ax.plot(p[valid], severity[valid], color="#336699", lw=1.5)
    ax.axhline(1.0, color="#9a0000", ls="--", lw=1.3)
    ax.set_xlim(0, length)
    ax.set_ylim(0, max(1.5, float(severity[valid].max()) * 1.08))
    ax.set_ylabel("shake or roll\n(1 = limit)", fontsize=14)
    ax.set_xlabel("distance driven (m)", fontsize=14)
    ax.tick_params(labelsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colour = {1: "#28aa46", 0: "#c82828", None: "#bbbbbb"}
    step = 0.05
    for row, values in ((1, prediction), (0, label)):
        for s, value in zip(frames, values):
            bands.barh(row, step, left=s, height=0.8, color=colour[value], lw=0)
    bands.set_yticks([0, 1], ["label", "model"], fontsize=14)
    bands.tick_params(axis="x", labelbottom=False, length=0)
    for side in ("top", "right", "bottom", "left"):
        bands.spines[side].set_visible(False)
    fig.canvas.draw()
    box_a, box_b = ax.get_window_extent(), bands.get_window_extent()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    image = Image.open(buffer).convert("RGB").resize((W, H - PANEL_H))
    to_px = lambda m: box_a.x0 + m / length * (box_a.x1 - box_a.x0)
    height = H - PANEL_H
    return image, to_px, (height - box_b.y1, height - box_a.y0)


def camera_panel(photo, camera, near, far, colour, title, status):
    panel = photo.resize((PANEL_W, PANEL_H), Image.BILINEAR).convert("RGBA")
    corners = [project_base_point(camera, f, l) for f, l in ((near, HALF_WIDTH_M), (far, HALF_WIDTH_M), (far, -HALF_WIDTH_M), (near, -HALF_WIDTH_M))]
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if all(c is not None for c in corners):
        draw.polygon(corners, fill=colour + (70,), outline=colour + (255,), width=4)
    panel = Image.alpha_composite(panel, overlay).convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=colour, width=10)
    big, small = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 25)
    draw.rectangle([10, 10, 560, 106], fill=(255, 255, 255))
    draw.text((22, 15), title, fill=(20, 20, 20), font=small)
    draw.text((22, 56), status, fill=colour, font=big)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bags", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_rows(args.data)
    flag = np.load(args.scores)["flag"]
    prediction, label, reason = {}, {}, {}
    for r, f in zip(rows, flag):
        if r["run"] == args.run:
            k = int(r["frame_idx"])
            assert not np.isnan(f), "this run has no held-out prediction"
            prediction[k], label[k], reason[k] = int(f == 0), r["y"], r["reason"]
    known = [k for k in label if label[k] is not None]
    agree = sum(prediction[k] == label[k] for k in known)
    print(f"{args.run}: {len(prediction)} frames; agrees with the label on {agree}/{len(known)} labelled frames")

    episode = load_episode(args.data / "episodes" / args.run)
    settings = Settings(**JACKAL_REAL)
    signals = episode_signals(episode, settings)
    summary = json.loads((args.data / "labels.summary.json").read_text())
    thresholds = summary["thresholds_from_clean"]
    zones = next(e["zones"] for e in json.loads((args.data / "labels.zones.json").read_text()) if e["episode_id"] == args.run)
    route = signals["route"]
    frames = episode["frames"]
    progress = np.array([route.progress_and_offset(f["x_m"], f["y_m"])[0] for f in frames])
    times = np.array([f["t_s"] for f in frames])
    used = sorted(prediction)
    panel, to_px, (top, bottom) = signal_panel(signals, thresholds, zones, progress[used], [prediction[k] for k in used],
                                               [label[k] for k in used], float(signals["progress"].max()) + 0.1)

    meta = episode["meta"]
    bag = next(args.bags.glob(f"{args.run}_*"))
    images = read_bag(bag, {"image": meta["source"]["topics"]["image"]})["image"]
    t0 = meta["source"]["start_stamp_s"]
    camera = dict(meta["camera"])
    scale = PANEL_W / camera["width"]
    camera.update(width=PANEL_W, height=PANEL_H, fx=camera["fx"] * scale, fy=camera["fy"] * scale, cx=camera["cx"] * scale, cy=camera["cy"] * scale)
    start, stop = times[used[0]] - 1.0, times[used[-1]] + 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    ffmpeg = subprocess.Popen(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
                               "-movflags", "+faststart", str(args.out)], stdin=subprocess.PIPE)
    stamps = np.array([m[0] - t0 for m in images])
    for vt in np.arange(start, stop, args.speed / fps):
        i = int(np.clip(np.searchsorted(stamps, vt), 0, len(images) - 1))
        photo = decode_image(images[i][1], images[i][2]).rotate(180)
        k = int(np.clip(np.searchsorted(times, stamps[i], side="right") - 1, 0, len(times) - 1))
        if k in prediction:
            model_colour, model_text = (GREEN, "TRAVERSABLE") if prediction[k] else (RED, "NOT TRAVERSABLE")
            y = label[k]
            label_colour, label_text = (GRAY, "UNSURE") if y is None else (GREEN, "TRAVERSABLE") if y else (RED, "NOT TRAVERSABLE")
        else:
            model_colour, model_text, label_colour, label_text = GRAY, "not moving", GRAY, "not moving"
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(camera_panel(photo, camera, settings.strip_near_m, settings.strip_far_m, model_colour, "Model (never saw this run)", model_text), (0, 0))
        canvas.paste(camera_panel(photo, camera, settings.strip_near_m, settings.strip_far_m, label_colour, "Robot's sensors (the label)", label_text), (PANEL_W, 0))
        band = panel.copy()
        x = to_px(float(max(progress[k], 0.0)))
        ImageDraw.Draw(band).line([(x, top), (x, bottom)], fill=(20, 20, 20), width=3)
        canvas.paste(band, (0, PANEL_H))
        ffmpeg.stdin.write(canvas.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(args.out)


if __name__ == "__main__":
    main()
