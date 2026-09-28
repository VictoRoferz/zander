"""
Paired comparison of two eval runs on the SAME bucket and photos — the
report's G1 selection rule: a candidate only beats the runner-up if the paired
difference's 95 % CI excludes 0; otherwise the faster/simpler model wins.

    python cli.py compare <run_a> <run_b>

Paired = both models are scored on identical photos, so photo-to-photo
difficulty cancels out. Uncertainty comes from a cluster bootstrap over the
split_v2 board GROUPS (re-shots of one board are not independent) and, for the
photo verdicts, an exact McNemar test on the discordant photos. With few
discordant photos McNemar cannot reach significance (<= 5 discordant: never at
alpha 0.05) — which is the honest answer on this dataset size.
"""
from __future__ import annotations

import json
import logging
import math
import random

from config.settings import settings
from eval.photo_scorer import class_records, is_negative, is_positive, pscore
from eval.run_eval import load_predictions
from eval.scorer import (
    BOOTSTRAP_N,
    FPPI_OPERATING,
    _froc_from_stats,
    _photo_stats,
    cluster_resample,
    recall_at_fppi,
)

logger = logging.getLogger("ml.compare")


def _threshold(run: str) -> float | None:
    m = json.loads((settings.eval_dir / run / "metrics.json").read_text(encoding="utf-8"))
    return m["threshold"]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def run(run_a: str, run_b: str, class_index: int = 0, seed: int = 0) -> dict:
    da, db = load_predictions(run_a), load_predictions(run_b)
    ids_a = [p["capture_id"] for p in da["photos"]]
    if ids_a != [p["capture_id"] for p in db["photos"]] or da["bucket"] != db["bucket"]:
        raise ValueError("runs must cover the same bucket and photos in the same order")
    if da["class_names"][class_index] != db["class_names"][class_index]:
        raise ValueError("class index refers to different classes in the two runs")
    ta, tb = _threshold(run_a), _threshold(run_b)
    photos_a, photos_b = da["photos"], db["photos"]
    clusters = [p["group"] for p in photos_a]
    stats_a = _photo_stats(class_records(photos_a, class_index), 0.3)
    stats_b = _photo_stats(class_records(photos_b, class_index), 0.3)

    def recall(stats, idx):
        pts, n_gt = _froc_from_stats([stats[i] for i in idx])
        return recall_at_fppi(pts, FPPI_OPERATING)[0] if n_gt else None

    def escape(photos, t, idx):
        pos = [photos[i] for i in idx if is_positive(photos[i])]
        return sum(1 for p in pos if pscore(p) < t) / len(pos) if pos and t is not None else None

    full = list(range(len(photos_a)))
    point = {
        "recall_diff": recall(stats_a, full) - recall(stats_b, full),
        "escape_diff": (
            escape(photos_a, ta, full) - escape(photos_b, tb, full)
            if ta is not None and tb is not None else None
        ),
    }
    rng = random.Random(seed)
    boots = {"recall_diff": [], "escape_diff": []}
    for _ in range(BOOTSTRAP_N):
        idx = cluster_resample(len(photos_a), clusters, rng)
        ra, rb = recall(stats_a, idx), recall(stats_b, idx)
        if ra is not None and rb is not None:
            boots["recall_diff"].append(ra - rb)
        ea, eb = escape(photos_a, ta, idx), escape(photos_b, tb, idx)
        if ea is not None and eb is not None:
            boots["escape_diff"].append(ea - eb)

    def ci(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]

    # McNemar on photo verdict correctness (positives + iO negatives)
    b = c = 0
    for pa, pb in zip(photos_a, photos_b):
        if not (is_positive(pa) or is_negative(pa)) or ta is None or tb is None:
            continue
        truth = is_positive(pa)
        ok_a = (pscore(pa) >= ta) == truth
        ok_b = (pscore(pb) >= tb) == truth
        b += ok_a and not ok_b
        c += ok_b and not ok_a
    result = {
        "run_a": run_a, "run_b": run_b, "bucket": da["bucket"],
        "class": da["class_names"][class_index],
        "recall_at_0.5fppi_diff": round(point["recall_diff"], 4),
        "recall_diff_ci95": ci(boots["recall_diff"]),
        "escape_diff": None if point["escape_diff"] is None else round(point["escape_diff"], 4),
        "escape_diff_ci95": ci(boots["escape_diff"]),
        "verdict_discordant": {"a_right_b_wrong": b, "b_right_a_wrong": c},
        "mcnemar_p": round(mcnemar_exact(b, c), 4),
        "ci_unit": "board group",
    }
    rc = result["recall_diff_ci95"]
    decisive = rc is not None and (rc[0] > 0 or rc[1] < 0)
    result["g1_decision"] = (
        ("A better" if rc[0] > 0 else "B better") if decisive
        else "no significant difference -> prefer the faster/simpler model"
    )
    out = settings.eval_dir / f"compare_{run_a}__{run_b}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info(json.dumps(result, indent=1, ensure_ascii=False))
    return result
