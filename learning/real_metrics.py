# Checked: crossing and false-alarm logic tested in tests/test_real_metrics.py.
"""Metrics for real-robot experiments: frame, crossing and false-alarm level, with run-level bootstrap.

Convention: higher score = traversable; ``flag`` = model says NOT traversable.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

METRES_PER_FRAME = 0.04  # 0.4 m/s at 10 Hz
CONSECUTIVE = 3  # a stop gate needs this many flagged photos in a row, so one noisy photo cannot stop the robot
OBJECTS = ("lid", "pouch", "wood", "cable", "black_cable", "deskpad", "cardboard", "styrofoam", "cloth")


def choose_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    best, best_tau = -1.0, 0.5
    for tau in np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197))):
        keep = scores >= tau
        value = (keep[y == 1].mean() + (~keep[y == 0]).mean()) / 2
        if value > best:
            best, best_tau = value, float(tau)
    return best_tau


def has_run(flags: np.ndarray, length: int = CONSECUTIVE) -> bool:
    count = 0
    for flag in flags:
        count = count + 1 if flag else 0
        if count >= length:
            return True
    return False


def false_alarms(flags: np.ndarray, length: int = CONSECUTIVE) -> int:
    """Number of separate stops: each stretch of ``length``+ flagged photos counts once."""
    alarms, count = 0, 0
    for flag in flags:
        count = count + 1 if flag else 0
        if count == length:
            alarms += 1
    return alarms


def crossings(rows: list[dict], flag: np.ndarray) -> list[dict]:
    """One entry per object crossing: was it flagged while in the look-ahead strip, and was it really bad?"""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["hazard_instance"] and not np.isnan(flag[i]):
            groups[(r["run"], r["hazard_instance"])].append(i)
    out = []
    for (run, _), idx in groups.items():
        idx = sorted(idx, key=lambda i: rows[i]["frame_order"])
        out.append({"run": run, "kind": rows[idx[0]]["patch_class"], "detected": has_run(flag[idx] == 1),
                    "flag_rate": float(np.mean(flag[idx] == 1)), "bad": any(rows[i]["y"] == 0 for i in idx)})
    return out


def _safe_auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def frame_metrics(rows: list[dict], score: np.ndarray, flag: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score

    known = np.array([r["y"] is not None for r in rows]) & ~np.isnan(score)
    y = np.array([r["y"] if r["y"] is not None else -1 for r in rows])[known]
    s, f = score[known], flag[known] == 1
    tpr, tnr = float((~f)[y == 1].mean()), float(f[y == 0].mean())
    return {"frames": int(known.sum()), "auroc": _safe_auroc(y, s), "ap_non_traversable": float(average_precision_score(1 - y, -s)),
            "balanced_accuracy": (tpr + tnr) / 2, "keep_rate_traversable": tpr, "flag_rate_non_traversable": tnr}


def clean_run_metrics(rows: list[dict], flag: np.ndarray) -> dict:
    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        if r["condition_run"] == "clean" and not np.isnan(flag[i]):
            by_run[r["run"]].append(i)
    if not by_run:
        return {"clean_frame_flag_rate": float("nan"), "false_stops_per_100m": float("nan")}
    frames = alarms = flagged = 0
    for idx in by_run.values():
        idx = sorted(idx, key=lambda i: rows[i]["frame_order"])
        frames += len(idx); flagged += int(np.sum(flag[idx] == 1)); alarms += false_alarms(flag[idx] == 1)
    return {"clean_frame_flag_rate": flagged / frames, "false_stops_per_100m": 100.0 * alarms / (frames * METRES_PER_FRAME)}


def crossing_metrics(rows: list[dict], flag: np.ndarray) -> dict:
    items = crossings(rows, flag)
    per_object = {}
    for kind in OBJECTS:
        mine = [c for c in items if c["kind"] == kind]
        if mine:
            per_object[kind] = {"crossings": len(mine), "bad_crossings": sum(c["bad"] for c in mine),
                                "detected": float(np.mean([c["detected"] for c in mine])),
                                "strip_flag_rate": float(np.mean([c["flag_rate"] for c in mine]))}
    bad = [c["detected"] for c in items if c["bad"]]
    mild = [c["detected"] for c in items if not c["bad"]]
    return {"bad_crossing_detected": float(np.mean(bad)) if bad else float("nan"),
            "mild_crossing_flagged": float(np.mean(mild)) if mild else float("nan"),
            "bad_crossings": len(bad), "mild_crossings": len(mild), "per_object": per_object}


def summary(rows: list[dict], score: np.ndarray, flag: np.ndarray) -> dict:
    return {**frame_metrics(rows, score, flag), **clean_run_metrics(rows, flag), **crossing_metrics(rows, flag)}


def bootstrap(rows: list[dict], score: np.ndarray, flag: np.ndarray, reps: int = 500, seed: int = 0) -> dict:
    """95 % intervals resampling whole runs, the independent unit here (frames in a run are nearly duplicates)."""
    runs = sorted({r["run"] for i, r in enumerate(rows) if not np.isnan(score[i])})
    index = defaultdict(list)
    for i, r in enumerate(rows):
        if not np.isnan(score[i]):
            index[r["run"]].append(i)
    rng = np.random.default_rng(seed)
    keys = ("auroc", "balanced_accuracy", "clean_frame_flag_rate", "false_stops_per_100m", "bad_crossing_detected", "mild_crossing_flagged")
    samples = {k: [] for k in keys}
    y_all = np.array([r["y"] if r["y"] is not None else -1 for r in rows])
    cross_all = {run: [c for c in crossings([rows[i] for i in index[run]], flag[index[run]])] for run in runs}
    clean_all = {run: clean_run_metrics([rows[i] for i in index[run]], flag[index[run]]) for run in runs}
    frames_all = {run: sum(1 for i in index[run]) for run in runs}
    for _ in range(reps):
        picks = rng.choice(len(runs), len(runs))
        idx = np.concatenate([index[runs[p]] for p in picks])
        known = y_all[idx] >= 0
        y, s, f = y_all[idx][known], score[idx][known], flag[idx][known] == 1
        if 0 < y.sum() < len(y):
            samples["auroc"].append(_safe_auroc(y, s))
            samples["balanced_accuracy"].append(((~f)[y == 1].mean() + f[y == 0].mean()) / 2)
        cl = [clean_all[runs[p]] for p in picks if rows[index[runs[p]][0]]["condition_run"] == "clean"]
        if cl:
            weight = [frames_all[runs[p]] for p in picks if rows[index[runs[p]][0]]["condition_run"] == "clean"]
            samples["clean_frame_flag_rate"].append(np.average([c["clean_frame_flag_rate"] for c in cl], weights=weight))
            samples["false_stops_per_100m"].append(np.average([c["false_stops_per_100m"] for c in cl], weights=weight))
        items = [c for p in picks for c in cross_all[runs[p]]]
        bad, mild = [c["detected"] for c in items if c["bad"]], [c["detected"] for c in items if not c["bad"]]
        if bad:
            samples["bad_crossing_detected"].append(np.mean(bad))
        if mild:
            samples["mild_crossing_flagged"].append(np.mean(mild))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else None for k, v in samples.items()}
