"""
Frozen, group-aware, session-stratified splits.

Design rules (locked in the plan):
  - Whole groups only — no group ever spans two buckets (leakage test asserts).
  - test-v1 ≈ 20% of each session's positive images (and roughly 20% of its
    annotated images), FROZEN once committed; growth happens as additive
    test-v2, never by editing v1.
  - The remaining dev groups get a grouped K-fold assignment (default K=3),
    stratified so every fold holds positive groups from every session that
    has them.
  - Membership is keyed by capture_id only and label-snapshot-independent:
    unannotated and quarantined images ALSO get a bucket via their group, so
    later annotation slots in without re-splitting.
"""
from __future__ import annotations

import json
import logging

from pathlib import Path

from config.settings import settings

logger = logging.getLogger("ml.split")

SPLIT_ID = "split_v1"
K_FOLDS = 3
TEST_FRACTION = 0.2
MIN_POSITIVE_GROUPS_PER_FOLD = 3


def build_split(groups_doc: dict, k: int = K_FOLDS, seed: int | None = None) -> dict:
    seed = settings.seed if seed is None else seed
    groups = groups_doc["groups"]

    by_session: dict[str, list[str]] = {}
    for gid, g in groups.items():
        by_session.setdefault(g["session"], []).append(gid)

    # ---- test-v1: per-session greedy fill toward the 20% targets ----
    # Groups are lumpy (a burst can hold 17 positives), so the greedy walks
    # positive groups largest-first under an overshoot cap instead of taking
    # anything below target — otherwise one giant group makes test swallow a
    # whole session. Deterministic: sorted, no RNG (seed recorded anyway).
    test_groups: list[str] = []
    for session in sorted(by_session):
        gids = by_session[session]
        pos_total = sum(groups[g]["n_positive"] for g in gids)
        ann_total = sum(
            groups[g]["n_images"] for g in gids
        )  # session size proxy (incl. unannotated members)
        pos_target = round(TEST_FRACTION * pos_total)
        img_target = round(TEST_FRACTION * ann_total)
        pos_cap = max(pos_target * 1.3, pos_target + 2)
        img_cap = max(img_target * 1.3, img_target + 2)
        got_pos = got_img = 0
        pos_first = sorted(
            gids, key=lambda g: (-groups[g]["n_positive"], -groups[g]["n_images"], g)
        )
        for gid in pos_first:
            g = groups[gid]
            if g["n_positive"] > 0:
                take = got_pos < pos_target and got_pos + g["n_positive"] <= pos_cap
            else:
                take = got_img < img_target and got_img + g["n_images"] <= img_cap
            if take:
                test_groups.append(gid)
                got_pos += g["n_positive"]
                got_img += g["n_images"]
        logger.info(
            f"test-v1 {session}: {got_pos}/{pos_total} positive imgs "
            f"(target {pos_target}), {got_img}/{ann_total} imgs (target {img_target})"
        )

    # ---- K folds over the remaining dev groups ----
    dev_groups = [g for g in sorted(groups) if g not in set(test_groups)]
    folds: dict[str, list[str]] = {str(i): [] for i in range(k)}
    fold_pos = [0] * k
    fold_img = [0] * k
    # Biggest positive groups first so the balancer can spread them.
    for gid in sorted(
        dev_groups,
        key=lambda g: (-groups[g]["n_positive"], -groups[g]["n_images"], g),
    ):
        i = min(range(k), key=lambda j: (fold_pos[j], fold_img[j], j))
        folds[str(i)].append(gid)
        fold_pos[i] += groups[gid]["n_positive"]
        fold_img[i] += groups[gid]["n_images"]

    for i in range(k):
        n_pos_groups = sum(1 for g in folds[str(i)] if groups[g]["n_positive"] > 0)
        logger.info(
            f"fold {i}: {len(folds[str(i)])} groups, {fold_pos[i]} positive imgs, "
            f"{n_pos_groups} positive groups"
        )
        if n_pos_groups < MIN_POSITIVE_GROUPS_PER_FOLD:
            logger.warning(
                f"fold {i} has only {n_pos_groups} positive groups "
                f"(< {MIN_POSITIVE_GROUPS_PER_FOLD}) — fold metrics will be noisy"
            )

    assignment = {}
    bucket_of_group = {}
    for gid in test_groups:
        bucket_of_group[gid] = "test"
    for i, gids in folds.items():
        for gid in gids:
            bucket_of_group[gid] = f"fold{i}"
    for gid, g in groups.items():
        for cid in g["captures"]:
            assignment[cid] = {"group": gid, "bucket": bucket_of_group[gid]}

    return {
        "split_id": SPLIT_ID,
        "seed": seed,
        "k_folds": k,
        "test_fraction": TEST_FRACTION,
        "test_groups": sorted(test_groups),
        "folds": {i: sorted(g) for i, g in folds.items()},
        "assignment": assignment,
    }


def check_leakage(split: dict, groups_doc: dict) -> None:
    """Assert no group spans buckets and every capture is assigned exactly once."""
    seen_groups: dict[str, str] = {}
    for cid, a in split["assignment"].items():
        gid, bucket = a["group"], a["bucket"]
        if gid in seen_groups and seen_groups[gid] != bucket:
            raise AssertionError(f"group {gid} spans buckets {seen_groups[gid]} / {bucket}")
        seen_groups[gid] = bucket
    all_captures = {c for g in groups_doc["groups"].values() for c in g["captures"]}
    assigned = set(split["assignment"])
    if all_captures != assigned:
        raise AssertionError(
            f"assignment mismatch: {len(all_captures - assigned)} unassigned, "
            f"{len(assigned - all_captures)} unknown"
        )


def run(k: int = K_FOLDS) -> Path:
    groups_path = settings.datasets_dir / "groups.json"
    groups_doc = json.loads(groups_path.read_text(encoding="utf-8"))
    out = settings.datasets_dir / f"{SPLIT_ID}.json"
    if out.exists():
        raise RuntimeError(
            f"{out} already exists — {SPLIT_ID} is FROZEN. Growing the test set "
            "means creating an additive split_v2, never editing v1."
        )
    split = build_split(groups_doc, k=k)
    check_leakage(split, groups_doc)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"split written (frozen): {out}")
    return out


def load_split() -> dict:
    path = settings.datasets_dir / f"{SPLIT_ID}.json"
    if not path.exists():
        raise FileNotFoundError(f"run `python cli.py split` first — missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))
