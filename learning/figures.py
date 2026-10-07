"""Static figures: label mechanism, examples, and baseline results.

    python3 -m learning.figures --episodes data/episodes --labels data/labels/labels.csv \
        --runs runs/baselines --output results/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scripts.episode_io import load_episode  # noqa: E402
from scripts.frame_labels import LabelConfig, hazard_encounters  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-3, status colours, chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#c3c2b7"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CLASS_ORDER = ["clean_floor", "oil_spill", "plastic_wrapper", "cable_hose", "small_lip"]
CLASS_LABEL = {"clean_floor": "Clean floor", "oil_spill": "Oil spill", "plastic_wrapper": "Plastic wrapper",
               "cable_hose": "Cable", "small_lip": "4 cm lip"}
MODEL_LABEL = {"frozen_logreg": "Frozen ResNet-18 + log. reg.", "finetuned_resnet18": "Fine-tuned ResNet-18",
               "positive_only_mahalanobis": "Positive-only (Mahalanobis)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def _labels_by_episode(labels_path: Path) -> dict:
    grouped: dict = {}
    with labels_path.open(newline="") as file:
        for row in csv.DictReader(file):
            grouped.setdefault(row["episode_id"], []).append(row)
    return grouped


def episode_timeline(episode_dir: Path, labels: list[dict], output: Path) -> None:
    episode = load_episode(episode_dir)
    state = episode["state"]
    t = np.array([r["t_s"] for r in state])
    az = np.array([r["imu_az"] for r in state])
    config = LabelConfig(vibration_soft=0.05, vibration_hard=0.3)
    encounters = hazard_encounters(episode, config)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2, 1.1]})

    axes[0].plot(t, [r["cmd_v_mps"] for r in state], color=MUTED, lw=2, ls="--", label="commanded")
    axes[0].plot(t, [r["v_fwd_mps"] for r in state], color=SERIES[0], lw=2, label="actual")
    axes[0].set_ylabel("forward speed (m/s)")
    axes[0].legend(loc="lower right", ncol=2)
    axes[0].set_title(f"How one episode becomes labels  ·  {episode['meta']['episode_id']}")

    window = 10
    rms = np.sqrt(np.convolve((az - np.median(az)) ** 2, np.ones(window) / window, mode="same"))
    axes[1].plot(t, rms, color=SERIES[1], lw=2)
    axes[1].axhline(0.3, color=CRITICAL, lw=1, ls=":")
    axes[1].text(t[-1], 0.3, "  jolt limit 0.3", color=INK_2, va="bottom", ha="right", fontsize=9)
    axes[1].set_ylabel("vertical vibration\nRMS (m/s²)")

    axes[2].plot(t, [r["hazard_contact_n"] for r in state], color=SERIES[2], lw=2)
    axes[2].set_ylabel("hazard contact (N)")

    for encounter in encounters:
        if "t_enter_s" not in encounter:
            continue
        span = [r["t_s"] for r in state if r["contact_object"] == encounter["instance_id"]] or [encounter["t_enter_s"]]
        for ax in axes[:3]:
            ax.axvspan(min(span) - 0.3, max(span) + 0.3, color=NEUTRAL, alpha=0.25, lw=0)
        axes[0].text((min(span) + max(span)) / 2, 0.02, CLASS_LABEL.get(encounter["asset_key"], encounter["asset_key"]),
                     ha="center", va="bottom", color=INK, fontsize=9, fontweight="bold")

    ax = axes[3]
    for row in labels:
        colour = {"1": GOOD, "0": CRITICAL}.get(row["y"], NEUTRAL)
        if row.get("reason") == "beyond_route":
            colour = MUTED  # excluded by design, not "unclear"
        ax.add_patch(Rectangle((float(row["t_s"]), 0), 0.1, 1, color=colour, lw=0))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frame label", rotation=0, ha="right", va="center")
    legend = [(0.0, "■ traversable", GOOD), (0.2, "■ not traversable", CRITICAL), (0.43, "■ gray (unclear / not reached)", NEUTRAL)]
    if any(row.get("reason") == "beyond_route" for row in labels):
        legend.append((0.72, "■ beyond route end (not used)", MUTED))
    for x, text, colour in legend:
        ax.text(x, -0.9, text, color=colour, transform=ax.transAxes, fontsize=9)
    fig.text(0.01, 0.005, "Each photo is labelled by what the robot later experienced on the floor it shows (0.6–1.8 m ahead): "
             "shaking, slow-down, slip or a stall.", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def frame_examples(episodes: Path, labels_path: Path, output: Path) -> None:
    with labels_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    picks = []
    for patch_class in CLASS_ORDER:
        candidates = [r for r in rows if r["patch_class"] == patch_class and r["reason"] not in
                      {"patch_not_covered", "window_contaminated_by_other_hazard", "hazard_not_visible"}
                      and (r["hazard_forward_m"] == "" or 1.1 < float(r["hazard_forward_m"]) < 1.5)]
        if candidates:
            picks.append(candidates[len(candidates) // 3])
    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.3))
    for ax, row in zip(np.atleast_1d(axes), picks):
        ax.imshow(plt.imread(episodes / row["rgb_path"]))
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row["bbox"]:
            u0, v0, u1, v1 = (float(v) for v in row["bbox"].split())
            ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec="#fab219", lw=1.6))
        verdict = {"1": ("traversable", GOOD), "0": ("not traversable", CRITICAL)}.get(row["y"], ("ignored: " + row["reason"].replace("_", " "), MUTED))
        ax.set_title(CLASS_LABEL[row["patch_class"]], fontsize=10)
        ax.text(0.0, -0.08, verdict[0], transform=ax.transAxes, color=verdict[1], fontsize=9, va="top", fontweight="bold")
        if row["tracking"]:
            ax.text(0.0, -0.2, f"speed ratio {float(row['tracking']):.2f} · vib {float(row['vibration_rms']):.2f} m/s²",
                    transform=ax.transAxes, color=INK_2, fontsize=8, va="top")
    fig.suptitle("Robot camera view with the known hazard footprint projected (yellow box)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def roc_curves(runs: Path, output: Path) -> None:
    from sklearn.metrics import roc_curve

    with (runs / "predictions_test.csv").open(newline="") as file:
        rows = [r for r in csv.DictReader(file) if r["y"] != ""]
    metrics = {m["model"]: m for m in json.loads((runs / "metrics.json").read_text())["results"]}
    y = np.array([int(r["y"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for colour, model in zip(SERIES, [m for m in MODEL_LABEL if m in rows[0]]):
        fpr, tpr, _ = roc_curve(1 - y, -np.array([float(r[model]) for r in rows]))
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{MODEL_LABEL[model]}  AUROC {metrics[model]['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("false alarms on traversable frames")
    ax.set_ylabel("non-traversable frames detected")
    ax.set_title("Held-out test seeds 100–119")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def flag_rates(runs: Path, output: Path) -> None:
    results = json.loads((runs / "metrics.json").read_text())["results"]
    by_name = {r["model"]: r for r in results}
    panels = [
        ("Trained on all hazard classes", [m for m in MODEL_LABEL if m in by_name]),
        ("Cable removed from training (unseen hazard)", [m for m in ("frozen_logreg__no_cable_hose_in_training",
                                                                     "positive_only_mahalanobis__no_cable_hose_in_training") if m in by_name]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (title, models) in zip(axes, panels):
        width = 0.8 / max(1, len(models))
        for index, model in enumerate(models):
            base = model.split("__")[0]
            colour = SERIES[list(MODEL_LABEL).index(base)]
            values = [by_name[model]["flag_rate_by_class"].get(c, {}).get("flagged_non_traversable", np.nan) for c in CLASS_ORDER]
            xs = np.arange(len(CLASS_ORDER)) + (index - (len(models) - 1) / 2) * width
            ax.bar(xs, values, width=width - 0.03, color=colour, label=MODEL_LABEL[base])
            for x, v in zip(xs, values):
                ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(CLASS_ORDER)), [CLASS_LABEL[c] for c in CLASS_ORDER])
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=8.5)
    axes[0].set_ylabel("test frames flagged non-traversable")
    fig.text(0.01, 0.005, "Clean floor and oil should be low; cable and lip high. Wrapper frames are unlabelled "
             "(ambiguous outcome), so their bar shows what the model does with an object it was never told about.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def appearance_shift(runs: Path, output: Path) -> None:
    """Same test scenes, original vs re-rendered look: hazards caught and false alarms."""
    by_name = {r["model"]: r for r in json.loads((runs / "metrics.json").read_text())["results"]}
    models = [m for m in MODEL_LABEL if m in by_name and f"{m}__shifted_appearance" in by_name]

    def rates(result):
        flags = result["flag_rate_by_class"]
        hazard = [flags[c] for c in ("cable_hose", "small_lip") if c in flags]
        caught = sum(h["flagged_non_traversable"] * h["frames"] for h in hazard) / sum(h["frames"] for h in hazard)
        return caught, flags["clean_floor"]["flagged_non_traversable"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = (("Hazards caught (cable + lip)", 0), ("False alarms on clean floor", 1))
    for ax, (title, index) in zip(axes, panels):
        xs = np.arange(len(models))
        for offset, (suffix, label, colour) in zip((-0.2, 0.2), (("", "Original look", SERIES[0]),
                                                                 ("__shifted_appearance", "New look (no retraining)", SERIES[1]))):
            values = [rates(by_name[m + suffix])[index] for m in models]
            ax.bar(xs + offset, values, width=0.37, color=colour, label=label)
            for x, v in zip(xs + offset, values):
                ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK_2)
        ax.set_xticks(xs, [MODEL_LABEL[m].replace(" + log. reg.", "\n+ log. reg.").replace(" (", "\n(") for m in models])
        ax.set_ylim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of test frames")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Same 20 test scenes, re-rendered with darker light, brown floor, coloured cables",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Hazards are still detected, but false alarms on clean floor rise: the decision threshold "
             "does not survive a change of appearance.", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--timeline-episode", default="warehouse-0001-hazard")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grouped = _labels_by_episode(args.labels)
    episode_timeline(args.episodes / args.timeline_episode, grouped[args.timeline_episode], args.output / "episode_timeline.png")
    frame_examples(args.episodes, args.labels, args.output / "frame_examples.png")
    if args.runs and (args.runs / "metrics.json").exists():
        roc_curves(args.runs, args.output / "roc_test.png")
        flag_rates(args.runs, args.output / "flag_rates.png")
        appearance_shift(args.runs, args.output / "appearance_shift.png")
    print(f"figures in {args.output}")


if __name__ == "__main__":
    main()
