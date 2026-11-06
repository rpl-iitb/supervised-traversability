# Checked: figures and tables of report Section 5.10. Builds the live go/no-go figures and tables for the report from the decision
# logs copied off the robot (data/gonogo_logs/, not in the repo) on 5-6 Nov 2026.
"""Live go/no-go report material.

    python3 hardware/gonogo/report_live_figures.py

Writes report/figures/gen/live_frames.jpg, report/figures/gen/live_scores.pdf, report/tables/live_results.tex and
report/tables/live_latency.tex, and prints the numbers quoted in the text. Event times are the STOP transitions in
console_gate.txt that match the trials in live_trials.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data" / "gonogo_logs"
GEN = ROOT / "report" / "figures" / "gen"
TABLES = ROOT / "report" / "tables"
MORNING = LOGS / "gonogo_gate_20261106_071536.csv"
AFTERNOON = LOGS / "gonogo_gate_20261106_093625.csv"
MARGIN = -1.0

# (label, group, csv, start, end): decisions from 0.3 s before each STOP to 2 s after it (robot standing
# still at the object), and the whole clean drive (wall time, s; STOP times from console_gate.txt).
WINDOWS = [
    ("Clean corridor (50 m)", "clean", MORNING, 1793950563.3, 1793950686.0),
    ("Lid", "hard", MORNING, 1793949518.76 - 0.3, 1793949518.76 + 2.0),
    ("Slipper (unseen)", "hard", AFTERNOON, 1793957949.03 - 0.3, 1793957949.03 + 2.0),
    ("White paper, run 1", "flat", AFTERNOON, 1793957876.13 - 0.3, 1793957876.13 + 2.0),
    ("White paper, run 2", "flat", AFTERNOON, 1793957890.33 - 0.3, 1793957890.33 + 2.0),
    ("White paper, run 3", "flat", AFTERNOON, 1793957903.23 - 0.3, 1793957903.23 + 2.0),
]
COLOURS = {"clean": "#8C8C8C", "hard": "#336699", "flat": "#D9822B"}
GROUP_NAMES = {"clean": "No object", "hard": "Hard object (should stop)", "flat": "Flat decoy (should not stop)"}


def load(path: Path) -> list[dict]:
    """Rows of a decision log. The morning log ends in NUL bytes (the robot rebooted before the file was flushed)."""
    text = path.read_bytes().replace(b"\x00", b"").decode()
    rows = []
    for r in csv.DictReader(text.splitlines()):
        try:
            rows.append({"t": float(r["wall_time"]), "z": float(r["z"]), "ms": float(r["infer_ms"]),
                         "age": float(r["frame_age_ms"]), "state": r["state"]})
        except (TypeError, ValueError):
            continue
    return rows


def window(rows, t0, t1):
    return np.array([r["z"] for r in rows if t0 <= r["t"] <= t1 and r["z"] == r["z"]])


def scores_figure(cache) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    rng = np.random.default_rng(0)
    for i, (label, group, path, t0, t1) in enumerate(WINDOWS):
        z = window(cache[path], t0, t1)
        y = len(WINDOWS) - 1 - i + rng.uniform(-0.18, 0.18, len(z))
        ax.scatter(z, y, s=6, color=COLOURS[group], alpha=0.55, linewidths=0)
        ax.plot([np.median(z)] * 2, [y.mean() - 0.3, y.mean() + 0.3], color="black", lw=1.5)
    ax.axvline(MARGIN, color="#444444", ls="--", lw=1)
    ax.text(MARGIN + 0.15, len(WINDOWS) - 0.45, "stop line ($z=-1$)", fontsize=7.5, color="#444444", va="top")
    ax.set_yticks(range(len(WINDOWS)), [w[0] for w in reversed(WINDOWS)], fontsize=8)
    ax.set_xlabel("ensemble score $z$ (higher = traversable)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4, label=GROUP_NAMES[g])
               for g, c in COLOURS.items()]
    handles.append(plt.Line2D([], [], color="black", lw=1.5, label="median"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=4)
    fig.tight_layout()
    fig.savefig(GEN / "live_scores.pdf")
    plt.close(fig)


def frames_figure() -> None:
    panels = [
        (ROOT / "data/gonogo_corridor.jpg", "(a) clean corridor: GO"),
        (ROOT / "data/gonogo_drive2_stop.jpg", "(b) lid: STOP"),
        (LOGS / "stop_frames/stop_093910.jpg", "(c) slipper, unseen: STOP"),
        (LOGS / "stop_frames/stop_093813.jpg", "(d) white paper: false STOP"),
        (ROOT / "data/gonogo_labmat.jpg", "(e) printed lab mat: false STOP"),
        (ROOT / "data/gonogo_model_input.png", "(f) model input, lab mat"),
    ]
    w, h, cap = 640, 480, 44
    sheet = Image.new("RGB", (3 * w, 2 * (h + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, (path, text) in enumerate(panels):
        im = Image.open(path).convert("RGB")
        if im.size != (w, h):  # the model input is a wide strip: letterbox it
            im.thumbnail((w, h))
            box = Image.new("RGB", (w, h), (127, 127, 127))
            box.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
            im = box
        x, y = (i % 3) * w, (i // 3) * (h + cap)
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + h + 6), text, fill="black", font=font)
    sheet.save(GEN / "live_frames.jpg", quality=90)


def tables(cache) -> dict:
    ms = np.array([r["ms"] for p in (MORNING, AFTERNOON) for r in cache[p] if r["ms"] > 0])
    age = np.array([r["age"] for p in (MORNING, AFTERNOON) for r in cache[p] if r["age"] == r["age"]])
    clean = window(cache[MORNING], *WINDOWS[0][3:])
    nums = {"infer_med": np.median(ms), "infer_p95": np.percentile(ms, 95), "age_med": np.median(age),
            "age_p95": np.percentile(age, 95), "clean_n": len(clean), "clean_flagged": int((clean < MARGIN).sum()),
            "clean_min": clean.min(), "clean_med": np.median(clean)}
    for label, group, path, t0, t1 in WINDOWS[1:]:
        z = window(cache[path], t0, t1)
        nums[label] = (np.median(z), z.min())

    def zcell(label):
        med, lo = nums[label]
        return f"${med:.1f}$ (${lo:.1f}$)"

    paper = [nums[f"White paper, run {k}"] for k in (1, 2, 3)]
    rows = [
        ("Clean corridor", "--", "about 50\\,m", "no stop", "--", f"${nums['clean_med']:.1f}$"),
        ("Lid", "yes", "corridor", "stop", "0.90", zcell("Lid")),
        ("Lid", "yes", "mosaic tiles", "stop", "$\\approx$0.9$^*$", "--"),
        ("Red cable", "yes", "corridor", "stop", "--", "--"),
        ("Pouch", "yes", "corridor", "stop", "--", "--"),
        ("Slipper", "\\textbf{no}", "corridor", "stop", "--", zcell("Slipper (unseen)")),
        ("Mosaic tiles, no object", "\\textbf{no}", "mosaic tiles", "no stop", "--", "--"),
        ("Desk pad (flat)", "yes", "corridor", "\\textit{stop}", "--", "--"),
        ("White paper ($\\times$3)", "\\textbf{no}", "corridor", "\\textit{stop}", "--",
         f"${max(p[0] for p in paper):.1f}$ to ${min(p[0] for p in paper):.1f}$"),
        ("Printed lab mats", "\\textbf{no}", "lab", "\\textit{stop}", "--", "$-1.3$ to $-2.2^\\dagger$"),
    ]
    lines = ["\\begin{tabular}{llllcc}", "\\toprule",
             "Object & In training? & Floor & Result & Gap (m) & $z$: median (min) \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "live_results.tex").write_text("\n".join(lines) + "\n")

    lat = [
        ("Camera frame reaches the node (median / 95th pct.)", f"{nums['age_med']:.0f} / {nums['age_p95']:.0f}"),
        ("Wait for the next 10\\,Hz decision (worst case)", "100"),
        ("Ensemble of five ResNet-18s on the robot CPU (median / 95th pct.)",
         f"{nums['infer_med']:.0f} / {nums['infer_p95']:.0f}"),
        ("Two more flagged frames (3 in a row)", "200"),
        ("Motor controller keeps the last command (\\texttt{cmd\\_vel\\_timeout})", "500"),
        ("\\textbf{Total before braking}", "\\textbf{about 900}"),
    ]
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Step & Time (ms) \\\\", "\\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in lat]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "live_latency.tex").write_text("\n".join(lines) + "\n")
    return nums


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    cache = {p: load(p) for p in (MORNING, AFTERNOON)}
    scores_figure(cache)
    frames_figure()
    for k, v in tables(cache).items():
        print(k, v)


if __name__ == "__main__":
    main()
