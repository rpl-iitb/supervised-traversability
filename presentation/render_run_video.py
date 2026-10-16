"""Render a presentation video of one recorded proxy-sim drive.

The drive is REPLAYED from its logged 100 Hz state (pose, pitch, roll), so the
video shows exactly the motion the labeller saw. Layout (1920x1080):

    robot camera + look-ahead strip coloured by label | chase camera
    vertical shaking along the route, with the 0.3 m/s^2 limit and a cursor

    python3 -m presentation.render_run_video --episode data/episodes_v2/warehouse-0112-hazard \\
        --labels data/labels_v2/labels.csv --out presentation/assets/run_0112.mp4
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.episode_io import load_episode, project_base_point
from scripts.frame_labels_v2 import Settings, episode_signals
from scripts.proxy_sim import WHEEL_RADIUS_M, _appearance, build_mjcf

W, H = 1920, 1080
PANEL_W, PANEL_H = 960, 720
COLOURS = {"1": (40, 170, 70), "0": (200, 40, 40), "": (150, 150, 150)}


def quat_from_euler(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(roll / 2), math.sin(roll / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def signal_panel(signals, settings: Settings, zones):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(W / 100, (H - PANEL_H) / 100), dpi=100)
    ax = fig.add_axes([0.04, 0.16, 0.94, 0.78])
    p, vib, valid = signals["progress"], signals["vib"], signals["valid"]
    for z in zones:
        ax.axvspan(z["lo"], z["hi"], color="#c82828" if z["unacceptable"] else "#999999", alpha=0.18, lw=0)
    ax.plot(p[valid], vib[valid], color="#336699", lw=1.6)
    ax.axhline(settings.unacceptable_vib_mps2, color="#9a0000", ls="--", lw=1.4)
    ax.set_xlim(0, max(p.max(), 1))
    ax.set_ylim(0, max(1.0, float(np.nanmax(vib[valid])) * 1.1))
    ax.tick_params(labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    bbox = ax.get_window_extent()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    img = Image.open(buf).convert("RGB").resize((W, H - PANEL_H))
    to_px = lambda m: bbox.x0 + (m - x0) / (x1 - x0) * (bbox.x1 - bbox.x0)
    top, bottom = (H - PANEL_H) - bbox.y1, (H - PANEL_H) - bbox.y0
    return img, to_px, (top, bottom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--zones", type=Path, default=None, help="default: <labels>.zones.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=0.5, help="playback speed (0.5 = half speed)")
    args = parser.parse_args()

    import json

    import mujoco

    episode = load_episode(args.episode)
    scenario, meta, state = episode["scenario"], episode["meta"], episode["state"]
    settings = Settings()
    signals = episode_signals(episode, settings)
    zones_path = args.zones or args.labels.with_suffix(".zones.json")
    zones = next(e["zones"] for e in json.loads(zones_path.read_text()) if e["episode_id"] == meta["episode_id"])
    with args.labels.open(newline="") as f:
        labels = {int(r["frame_idx"]): r for r in csv.DictReader(f) if r["episode_id"] == meta["episode_id"]}
    frame_t = np.array([fr["t_s"] for fr in episode["frames"]])

    model = mujoco.MjModel.from_xml_string(build_mjcf(scenario, _appearance(scenario["seed"])))
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, PANEL_H, PANEL_W)
    root = model.jnt_qposadr[model.joint("robot_root").id]
    wheels = [model.jnt_qposadr[model.joint(n).id] for n in ("wheel_left", "wheel_right")]
    chase = mujoco.MjvCamera()
    chase.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.distance, chase.elevation = 2.2, -12.0

    col = lambda k: np.array([r[k] for r in state], dtype=float)
    t, x, y, z = col("t_s"), col("x_m"), col("y_m"), col("z_m")
    yaw, pitch, roll, wheel_v = col("yaw_rad"), col("pitch_rad"), col("roll_rad"), col("wheel_v_mps")
    wheel_angle = np.cumsum(np.nan_to_num(wheel_v)) * (t[1] - t[0]) / WHEEL_RADIUS_M
    kernel = np.ones(101) / 101  # 1 s smoothing so the chase camera does not shake with the robot
    smooth = lambda v: np.convolve(np.pad(v, 50, mode="edge"), kernel, mode="valid")
    xs, ys, yaws = smooth(x), smooth(y), smooth(np.unwrap(yaw))

    cam = dict(meta["camera"])
    scale = PANEL_W / cam["width"]
    cam.update(width=PANEL_W, height=PANEL_H, fx=cam["fx"] * scale, fy=cam["fy"] * scale, cx=cam["cx"] * scale, cy=cam["cy"] * scale)
    panel, to_px, (plot_top, plot_bottom) = signal_panel(signals, settings, zones)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                               "-movflags", "+faststart", str(args.out)], stdin=subprocess.PIPE)
    for vt in np.arange(0.0, t[-1], args.speed / args.fps):
        i = int(np.clip(np.searchsorted(t, vt), 0, len(t) - 1))
        data.qpos[root:root + 3] = [x[i], y[i], z[i]]
        data.qpos[root + 3:root + 7] = quat_from_euler(yaw[i], pitch[i], roll[i])
        for adr in wheels:
            data.qpos[adr] = wheel_angle[i]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, "front")
        fpv = Image.fromarray(renderer.render())
        k = int(np.clip(np.searchsorted(frame_t, vt, side="right") - 1, 0, len(frame_t) - 1))
        row = labels.get(k)
        if row is not None and row["reason"] != "beyond_route":
            colour = COLOURS[row["y"]]
            hw = settings.strip_half_width_m
            corners = [project_base_point(cam, f, l) for f, l in ((settings.strip_near_m, hw), (settings.strip_far_m, hw),
                                                                  (settings.strip_far_m, -hw), (settings.strip_near_m, -hw))]
            if all(c is not None for c in corners):
                overlay = Image.new("RGBA", fpv.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                draw.polygon(corners, fill=colour + (70,), outline=colour + (255,), width=4)
                fpv = Image.alpha_composite(fpv.convert("RGBA"), overlay).convert("RGB")
            ImageDraw.Draw(fpv).rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=colour, width=10)

        chase.lookat[:] = [xs[i], ys[i], 0.25]
        chase.azimuth = math.degrees(yaws[i]) + 65.0  # behind-right, mostly side-on so bumps show
        renderer.update_scene(data, chase)
        side = Image.fromarray(renderer.render())

        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(fpv, (0, 0))
        canvas.paste(side, (PANEL_W, 0))
        bottom = panel.copy()
        cx = to_px(float(signals["progress"][i]))
        ImageDraw.Draw(bottom).line([(cx, plot_top), (cx, plot_bottom)], fill=(20, 20, 20), width=3)
        canvas.paste(bottom, (0, PANEL_H))
        ffmpeg.stdin.write(canvas.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(args.out)


if __name__ == "__main__":
    main()
