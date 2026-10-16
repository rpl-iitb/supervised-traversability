"""Result figures for the update deck, drawn from the saved result files only.

    python3 -m presentation.results_figures
Reads runs/ci/ci.json (scene-bootstrap intervals) and
data/labels_gonet_rule/labels.summary.json (GONet's rule vs our labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("presentation/assets")
BLUE, RED, GRAY, LIGHT = "#336699", "#9A0000", "#8C8C8C", "#C9D6E3"
MODELS = [  # (key in ci.json, label, colour)
    ("gonet_official", "GONet (released)", "#BBBBBB"),
    ("gonet_own_labels", "GONet + motion labels", GRAY),
    ("positive_only_mahalanobis", "Positive-only + our labels", LIGHT),
    ("frozen_logreg", "Frozen ResNet + our labels", "#7FA3C6"),
    ("gonet_retrained", "GONet + our labels", BLUE),
    ("finetuned_resnet18", "Fine-tuned ResNet + our labels", RED),
]
HAZARDS = [("cable_hose", "Cable"), ("oil_spill", "Oil"), ("small_lip", "Lip"), ("plastic_wrapper", "Wrapper"),
           ("clean_floor", "Clean floor")]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=15)


def auroc(ci):
    fig, ax = plt.subplots(figsize=(13.33, 6.4), dpi=150)
    for row, (key, label, colour) in enumerate(MODELS):
        m = ci["models"][key]["auroc"]
        lo, hi = m["ci95"]
        ax.barh(row, m["value"], color=colour, height=0.62)
        ax.errorbar(m["value"], row, xerr=[[m["value"] - lo], [hi - m["value"]]], color="black", capsize=6, lw=1.6)
        ax.text(hi + 0.008, row, f"{m['value']:.3f}", va="center", fontsize=16)
    ax.set_yticks(range(len(MODELS)), [label for _, label, _ in MODELS], fontsize=16)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_xlim(0.4, 1.04)
    ax.set_xlabel("AUROC (95 % interval over test scenes)", fontsize=15)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "results_auroc.png")
    plt.close(fig)


def per_hazard(ci):
    keys = ["gonet_official", "gonet_own_labels", "gonet_retrained", "finetuned_resnet18"]
    chosen = [m for m in MODELS if m[0] in keys]
    fig, ax = plt.subplots(figsize=(13.33, 6.4), dpi=150)
    width = 0.8 / len(chosen)
    x = np.arange(len(HAZARDS))
    for j, (key, label, colour) in enumerate(chosen):
        vals = [ci["models"][key][f"flag_{h}"] for h, _ in HAZARDS]
        v = np.array([d["value"] for d in vals]) * 100
        lo = np.array([d["ci95"][0] for d in vals]) * 100
        hi = np.array([d["ci95"][1] for d in vals]) * 100
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, v, width * 0.92, color=colour, label=label)
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", color="black", capsize=3, lw=1.1)
    ax.axvline(len(HAZARDS) - 1.5, color="black", lw=0.8)
    ax.set_xticks(x, [name for _, name in HAZARDS], fontsize=17)
    ax.set_ylabel("% of photos flagged not traversable", fontsize=15)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=14, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.12))
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "results_per_hazard.png")
    plt.close(fig)


def labels_compare(summary):
    rows = []
    for key, name in HAZARDS[:4]:
        counts = summary[key]
        unsafe = {k: v for k, v in counts.items() if k.startswith("physics=0")}
        total = sum(v for k, v in unsafe.items() if not k.endswith("gray"))
        called_safe = unsafe.get("physics=0 rule=1", 0)
        rows.append((name, 100 * called_safe / total, 100 * (total - called_safe) / total, total))
    fig, ax = plt.subplots(figsize=(13.33, 6.0), dpi=150)
    y = np.arange(len(rows))[::-1]
    safe = [r[1] for r in rows]
    unsafe = [r[2] for r in rows]
    ax.barh(y, unsafe, color=RED, height=0.6, label="Motion rule: not traversable")
    ax.barh(y, safe, left=unsafe, color="#BBBBBB", height=0.6, label="Motion rule: traversable")
    for yi, (name, s, u, n) in zip(y, rows):
        ax.text(u + s / 2, yi, f"{s:.0f} %", va="center", ha="center", fontsize=18, color="black")
    ax.set_yticks(y, [f"{r[0]}  (n={r[3]})" for r in rows], fontsize=17)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Training photos our labels mark NOT traversable, as labelled by GONet's motion rule (%)", fontsize=14)
    ax.legend(fontsize=14, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "results_labels_compare.png")
    plt.close(fig)


def main() -> None:
    ci = json.loads(Path("runs/ci/ci.json").read_text())
    auroc(ci)
    per_hazard(ci)
    labels_compare(json.loads(Path("data/labels_gonet_rule/labels.summary.json").read_text()))
    print("written to", OUT)


if __name__ == "__main__":
    main()
