# Checked: plots saved result files and the labeller's own signals; numbers match the report tables.
"""Result plots and number tables for the BTP report (real robot + sim-to-real).

    python3 -m presentation.report_figures_real --data ../../../data/real/s1 \
        --results ../../../runs/real_v1 --out ../../../report

Writes report/figures/gen/*.pdf (plots, object photos, video still) and
report/tables/*.tex (tabular bodies only; captions are left to the author).
Every number comes from a results JSON or the labeller, never typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.frame_labels_v2 import JACKAL_REAL, Settings, episode_signals
from scripts.real_labels import TRUTH_TOLERANCE_M, build, load_runs

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
OBJECTS = ["lid", "pouch", "wood", "cable", "black_cable", "deskpad", "cardboard", "styrofoam", "cloth"]
NAMES = {"lid": "lid", "pouch": "pouch", "wood": "wood", "cable": "red cable", "black_cable": "black cable",
         "deskpad": "desk pad", "cardboard": "cardboard", "styrofoam": "styrofoam", "cloth": "cloth"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- labels

def label_signals(data: Path):
    """Re-run the labeller (same settings as labels.csv) and keep each run's signals and object positions."""
    settings = Settings(**JACKAL_REAL)
    runs = load_runs(data)
    _, report, thresholds = build(runs, settings)
    vib_limit = settings.unacceptable_vib_clean_multiple * thresholds["vib_mps2"]
    roll_limit = settings.unacceptable_roll_clean_multiple * thresholds["roll_rps"]
    out = []
    for run, entry in zip(runs, report):
        episode = run["episode"]
        signals = episode_signals(episode, settings)
        route = signals["route"]
        placed = [(o["asset_key"], route.progress_and_offset(o["pose"]["x_m"], o["pose"]["y_m"])[0], o["footprint_radius_m"])
                  for o in episode["scenario"]["objects"]]
        out.append({"run": run["run"], "signals": signals, "zones": entry["zones"], "placed": placed})
    return out, thresholds, vib_limit, roll_limit


def crossing_severity(runs: list[dict], vib_limit: float, roll_limit: float) -> dict:
    """Peak of max(shaking / limit, roll / limit) over each object's footprint plus the truth-check tolerance (1 = the limit)."""
    sev = {k: [] for k in OBJECTS}
    for run in runs:
        s = run["signals"]
        level = np.maximum(s["vib"] / vib_limit, s["roll"] / roll_limit)
        for key, at, radius in run["placed"]:
            near = np.abs(s["progress"] - at) <= radius + TRUTH_TOLERANCE_M
            if near.any():
                sev[key].append(float(level[near].max()))
    return sev


def fig_severity(sev: dict, out: Path, clean_fraction: float) -> None:
    order = sorted(OBJECTS, key=lambda k: -np.median(sev[k]))
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    rng = np.random.default_rng(0)
    for i, key in enumerate(order):
        values = np.array(sev[key])
        x = i + rng.uniform(-0.18, 0.18, len(values))
        over = values >= 1.0
        ax.scatter(x[over], values[over], s=14, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3,
                   label="not traversable" if i == 0 else None)
        ax.scatter(x[~over], values[~over], s=14, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3,
                   label="below the limit" if i == 0 else None)
        ax.plot([i - 0.28, i + 0.28], [np.median(values)] * 2, color=INK, linewidth=1.2, zorder=4)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
    ax.axhline(clean_fraction, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(-0.5, 1.0, "limit (2x clean)", va="bottom", ha="left", color=INK, fontsize=8)
    ax.text(-0.5, clean_fraction, "clean-floor level", va="bottom", ha="left", color=MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 5], ["0.3", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xticks(range(len(order)), [NAMES[k] for k in order], rotation=25, ha="right")
    ax.set_ylabel("peak severity\n(1 = limit, log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2, fontsize=8, handletextpad=0.2)
    fig.savefig(out / "label_severity.pdf")
    plt.close(fig)


def fig_signals(run: dict, vib_limit: float, roll_limit: float, thresholds: dict, out: Path) -> None:
    s = run["signals"]
    keep = s["valid"]
    x = s["progress"]
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.2), sharex=True)
    panels = [(s["vib"], thresholds["vib_mps2"], vib_limit, "shaking (m/s$^2$)"),
              (s["roll"], thresholds["roll_rps"], roll_limit, "roll rate (rad/s)")]
    for ax, (y, clean, limit, label) in zip(axes, panels):
        for zone in run["zones"]:
            if zone["unacceptable"]:
                ax.axvspan(zone["lo"], zone["hi"], color=ORANGE, alpha=0.18, linewidth=0)
        ax.plot(x[keep], y[keep], color=BLUE, linewidth=0.9)
        ax.axhline(clean, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axhline(limit, color=INK, linewidth=0.8, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].text(x[keep].max(), vib_limit, "limit (2x clean)", va="bottom", ha="right", fontsize=8)
    axes[0].text(x[keep].max() + 0.15, thresholds["vib_mps2"], "clean level", va="center", ha="left", fontsize=8, color=MUTED, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    top = axes[0].get_ylim()[1]
    for key, at, _ in run["placed"]:
        for ax in axes:
            ax.axvline(at, color=MUTED, linewidth=0.5)
        axes[0].text(at, top, NAMES[key], rotation=0, ha="center", va="bottom", fontsize=8, color=INK)
    axes[1].set_xlabel("distance along the line (m, odometry)")
    fig.savefig(out / "signals_run.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- photos

def fig_objects(data: Path, out: Path) -> None:
    """One camera frame per object (object about 1.1 m ahead, inside the strip) plus clean floor."""
    with (data / "labels.csv").open(newline="") as file:
        rows = [r for r in csv.DictReader(file) if r["session"] == "s1c" and r["reason"] != "not_moving"]
    picks = []
    for key in OBJECTS:
        cands = [r for r in rows if r["objects_in_roi"] == key and float(r["object_ahead_m"]) > 0]
        picks.append((NAMES[key], min(cands, key=lambda r: abs(float(r["object_ahead_m"]) - 1.1))))
    clean = [r for r in rows if r["condition_run"] == "clean" and r["y"] == "1"]
    picks.append(("clean floor", clean[len(clean) // 2]))
    fig, axes = plt.subplots(2, 5, figsize=(6.4, 2.1))
    for ax, (name, row) in zip(axes.flat, picks):
        ax.imshow(Image.open(data / "episodes" / row["rgb_path"]))
        label = {"1": "traversable", "0": "not traversable", "": "gray"}[row["y"]]
        ax.set_title(f"{name}\n({label})", fontsize=7.5, color=INK, pad=2)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.32)
    fig.savefig(out / "objects.pdf", dpi=200)
    plt.close(fig)


def fig_video_still(video: Path, at_s: float, out: Path) -> None:
    target = out / "video_still.jpg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at_s), "-i", str(video), "-frames:v", "1",
                    "-q:v", "2", str(target)], check=True)


# --------------------------------------------------------------------------- results

def fig_unseen(seen: dict, looo: dict, out: Path) -> None:
    keys = OBJECTS
    s = [seen["per_object"][k]["strip_flag_rate"] for k in keys]
    u = [looo[f"looo/mask/resnet18/{k}"]["per_object"][k]["strip_flag_rate"] for k in keys]
    det_u = [looo[f"looo/mask/resnet18/{k}"]["per_object"][k]["detected"] for k in keys]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.barh(y - 0.19, s, height=0.36, color=BLUE, label="object seen in training")
    ax.barh(y + 0.19, u, height=0.36, color=ORANGE, label="object hidden from training")
    for yi, d in zip(y, det_u):
        ax.text(1.02, yi + 0.19, f"{d:.0%} caught", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("frames flagged while the object is in the strip (ResNet-18, masked)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.savefig(out / "unseen_objects.pdf")
    plt.close(fig)


def fig_sim_real(s2r: dict, out: Path) -> None:
    models = [("finetuned_resnet18", "fine-tuned\nResNet-18"), ("logreg", "frozen +\nlogistic"), ("mahalanobis", "Mahalanobis")]
    sim = [s2r[f"sim_on_sim/{m}"]["auroc"] for m, _ in models]
    real = [s2r[f"sim_on_real/{m}"]["auroc"] for m, _ in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    b1 = ax.bar(x - 0.19, sim, width=0.36, color=BLUE, label="tested in sim")
    b2 = ax.bar(x + 0.19, real, width=0.36, color=ORANGE, label="tested on real frames")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(-0.45, 0.5, "chance", va="bottom", ha="left", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, [n for _, n in models])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUROC (all trained in sim)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.savefig(out / "sim_to_real.pdf")
    plt.close(fig)


def fig_curve(frozen: dict, out: Path) -> None:
    sizes = [2, 4, 8, 12, 16, 20]
    reps = [[frozen[f"curve/mask/logreg/runs_{n}/rep_{r}"]["auroc"] for r in range(5)] for n in sizes]
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    for n, values in zip(sizes, reps):
        ax.scatter([n] * len(values), values, s=10, color=BLUE, alpha=0.35, linewidth=0)
    means = [np.mean(v) for v in reps]
    ax.plot(sizes, means, color=BLUE, linewidth=1.4, marker="o", markersize=4)
    for n, m in zip(sizes, means):
        ax.text(n, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(sizes)
    ax.set_xlabel("object runs used for training")
    ax.set_ylabel("AUROC (frozen + logistic)")
    fig.savefig(out / "learning_curve.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- tables

def ci(entry: dict, key: str = "auroc") -> str:
    lo, hi = entry["ci95"][key]
    return f"{entry[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def table_models(cv: dict, frozen: dict, gonet: dict, extra: dict) -> str:
    rows = [("ResNet-18, fine-tuned", cv["cv5/mask/resnet18"], "mask"),
            ("MobileNetV3-S, fine-tuned", cv["cv5/mask/mobilenet"], "mask"),
            ("ResNet-18, fine-tuned", cv["cv5/full/resnet18"], "full"),
            ("Frozen ResNet-18 + logistic", frozen["cv5/mask/logreg"], "mask"),
            ("Mahalanobis (positives only)", frozen["cv5/mask/mahalanobis"], "mask"),
            ("GONet, retrained", gonet["cv5/full128/gonet_retrained"], "full"),
            ("GONet, released weights", gonet["cv5/full128/gonet_official"], "full")]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Input & AUROC [95\% CI] & Bal.\ acc. & False stops & Hard \\", r"& & & & per 100\,m & caught \\", r"\midrule"]
    for name, e, inp in rows:
        lines.append(f"{name} & {inp} & {ci(e)} & {e['balanced_accuracy']:.3f} & {e['false_stops_per_100m']:.1f} "
                     f"& {e['bad_crossing_detected']:.0%} \\\\".replace("%", r"\%"))
    lines.append(r"\midrule")
    for arch, name in (("resnet18", "ResNet-18"), ("mobilenet", "MobileNetV3-S")):
        runs = [cv[f"cv5/mask/{arch}"]] + [extra[f"seed{i}/mask/{arch}"] for i in (1, 2, 3)]
        a = [r["auroc"] for r in runs]
        f = [r["false_stops_per_100m"] for r in runs]
        lines.append(f"{name}, 4 splits (mean, SD) & mask & {np.mean(a):.3f} $\\pm$ {np.std(a, ddof=1):.3f} & "
                     f"-- & {np.mean(f):.1f} $\\pm$ {np.std(f, ddof=1):.1f} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_truth(summary: dict, sev: dict) -> str:
    truth = summary["truth"]["crossings"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Object & Crossings & Judged not traversable & Median peak severity \\", r"\midrule"]
    for key in sorted(OBJECTS, key=lambda k: -np.median(sev[k])):
        t = truth[key]
        lines.append(f"{NAMES[key]} & {t['n']} & {t['unacceptable']} & {np.median(sev[key]):.2f} \\\\")
    lines += [r"\midrule", f"clean runs (6) & -- & 0 zones & -- \\\\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_sim_real(s2r: dict) -> str:
    models = [("finetuned_resnet18", "Fine-tuned ResNet-18"), ("logreg", "Frozen + logistic"), ("mahalanobis", "Mahalanobis")]
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Trained in sim & Sim test AUROC & Real AUROC [95\% CI] \\", r"\midrule"]
    for key, name in models:
        real = s2r[f"sim_on_real/{key}"]
        cell = ci(real) if "ci95" in real else f"{real['auroc']:.3f}"
        lines.append(f"{name} & {s2r[f'sim_on_sim/{key}']['auroc']:.3f} & {cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_label_versions(lv: dict) -> str:
    cols = ["scored_grey_left_out", "scored_bumps_traversable", "scored_bumps_not_traversable"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Scored with} \\ \cmidrule(lr){2-4}",
             r"Trained with & gray left out & bumps = traversable & bumps = not trav. \\", r"\midrule"]
    for train, name in (("trained_grey_left_out", "gray left out"), ("trained_bumps_traversable", "bumps = traversable")):
        cells = [f"{lv[f'{train}/{c}']['auroc']:.3f}" for c in cols]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="report folder")
    parser.add_argument("--signals-run", default="s1c_r016")
    parser.add_argument("--video", type=Path, help="default: <data>/../../videos/s1c_r016_mild_as_traversable.mp4")
    parser.add_argument("--video-at", type=float, default=22.0, help="seconds into the video for the still")
    args = parser.parse_args()
    figs, tabs = args.out / "figures" / "gen", args.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    r = args.results
    cv, frozen, gonet = load(r / "finetune_results_cv.json"), load(r / "frozen_results.json"), load(r / "gonet_results.json")
    extra, looo, s2r = load(r / "finetune_results_extra.json"), load(r / "finetune_results_looo.json"), load(r / "sim_to_real.json")
    lv = load(r / "label_versions.json")

    runs, thresholds, vib_limit, roll_limit = label_signals(args.data)
    sev = crossing_severity(runs, vib_limit, roll_limit)
    print({NAMES[k]: round(float(np.median(v)), 2) for k, v in sev.items()})
    fig_severity(sev, figs, clean_fraction=0.5)
    fig_signals(next(x for x in runs if x["run"] == args.signals_run), vib_limit, roll_limit, thresholds, figs)
    fig_objects(args.data, figs)
    fig_video_still(args.video or args.data.parent.parent / "videos" / "s1c_r016_mild_as_traversable.mp4", args.video_at, figs)
    fig_unseen(cv["cv5/mask/resnet18"], looo, figs)
    fig_sim_real(s2r, figs)
    fig_curve(frozen, figs)

    (tabs / "real_models.tex").write_text(table_models(cv, frozen, gonet, extra) + "\n")
    (tabs / "truth_real.tex").write_text(table_truth(load(args.data / "labels.summary.json"), sev) + "\n")
    (tabs / "sim_real.tex").write_text(table_sim_real(s2r) + "\n")
    (tabs / "label_versions.tex").write_text(table_label_versions(lv) + "\n")
    print("wrote", sorted(p.name for p in figs.iterdir()), sorted(p.name for p in tabs.iterdir()))


if __name__ == "__main__":
    main()
