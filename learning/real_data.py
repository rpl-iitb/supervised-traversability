# Checked: split logic is tested in tests/test_real_data.py, the rest checked by inspection of outputs.
"""Real-robot frames for training: labels, leak-free run splits, and the masked floor input."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.episode_io import project_base_point

EXCLUDED_REASONS = {"not_moving", "beyond_route"}
INPUT_HW = {"mask": (112, 336), "full": (160, 288)}
CORRIDOR_HALF_WIDTH_M = 0.5  # the robot's width plus a margin, so the socket box (>= 1 m to the side) is hidden
CORRIDOR_FAR_M = 2.5
RGB_MEAN, RGB_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load_rows(data: Path) -> list[dict]:
    with (data / "labels.csv").open(newline="") as file:
        rows = [r for r in csv.DictReader(file) if r["reason"] not in EXCLUDED_REASONS]
    for r in rows:
        r["y"] = None if r["y"] == "" else int(r["y"])
        r["in_roi"] = set(filter(None, r["objects_in_roi"].split("+")))
        r["visible"] = set(filter(None, r["objects_visible"].split("+")))
    return rows


def run_table(rows: list[dict]) -> dict[str, dict]:
    runs = {}
    for r in rows:
        runs.setdefault(r["run"], {"session": r["session"], "direction": r["direction"], "condition": r["condition_run"]})
    return runs


def run_folds(runs: dict[str, dict], folds: int, seed: int = 0) -> list[set[str]]:
    """Split whole runs into folds, balanced over session, direction and clean/hazard."""
    strata = defaultdict(list)
    for name, info in sorted(runs.items()):
        strata[(info["condition"], info["session"], info["direction"])].append(name)
    rng = random.Random(seed)
    out = [set() for _ in range(folds)]
    position = 0
    for key in sorted(strata):
        names = strata[key]
        rng.shuffle(names)
        for name in names:
            out[position % folds].add(name)
            position += 1
    return out


def train_val_test(runs: dict[str, dict], fold: list[set[str]], index: int) -> tuple[set[str], set[str], set[str]]:
    """Fold ``index`` tests; the next fold validates (thresholds, epochs); the rest train."""
    test, val = fold[index], fold[(index + 1) % len(fold)]
    return set(runs) - test - val, val, test


def corridor_polygon(camera: dict) -> list[tuple[float, float]]:
    near = camera["mount_xyz_m"][0] + 0.05
    w, far = CORRIDOR_HALF_WIDTH_M, CORRIDOR_FAR_M
    return [project_base_point(camera, far, w, 0.0), project_base_point(camera, far, -w, 0.0),
            project_base_point(camera, near, -w, 0.0), project_base_point(camera, near, w, 0.0)]


def floor_mask(camera: dict) -> np.ndarray:
    from PIL import Image, ImageDraw

    mask = Image.new("L", (camera["width"], camera["height"]), 0)
    ImageDraw.Draw(mask).polygon(corridor_polygon(camera), fill=255)
    return np.asarray(mask) > 0


def preprocess(path: Path, mode: str, camera: dict, mask: np.ndarray) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    h, w = INPUT_HW[mode]
    if mode == "mask":
        pixels = np.asarray(image).copy()
        pixels[~mask] = 127
        rows = np.flatnonzero(mask.any(axis=1))
        crop = Image.fromarray(pixels[rows[0]:rows[-1] + 1])
    else:
        crop = image.crop((0, int(camera["cy"]) - 30, camera["width"], camera["height"]))
    return np.asarray(crop.resize((w, h), Image.BILINEAR))


def build_cache(data: Path, rows: list[dict], mode: str, cache: Path) -> np.ndarray:
    """uint8 array N x H x W x 3 aligned with ``rows``."""
    if cache.exists():
        images = np.load(cache, mmap_mode="r")
        if len(images) == len(rows):
            return images
    camera = json.loads((data / "episodes" / rows[0]["run"] / "meta.json").read_text())["camera"]
    mask = floor_mask(camera)
    h, w = INPUT_HW[mode]
    images = np.zeros((len(rows), h, w, 3), np.uint8)
    for i, r in enumerate(rows):
        images[i] = preprocess(data / "episodes" / r["rgb_path"], mode, camera, mask)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, images)
    return images
