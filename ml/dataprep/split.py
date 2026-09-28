"""
Frozen, board-aware split (split_v2) over the groups of dataprep/grouping.py.

split_v1 (2026-08-06, time-burst groups, NAL only, batch 1 only) is RETIRED:
on 2026-09-28 re-shots of the same boards (identical rare defects, taken > 30 s
apart) were found in fold0 / fold2 / test. No test evaluation had ever been
run on it (results.csv held one smoke row), so re-splitting was honest.
split_v1.json + groups.json stay committed as history.

Buckets:
  test          ~TEST_FRACTION of each session's NAL-positive photos + ~the same
                share of its images, whole groups, greedy per session. ONE-SHOT:
                final chosen model only, threshold frozen beforehand.
  fold0..K-1    grouped K-fold over the remaining groups, balanced by
                NAL-positive photos then images.
  train_always  groups that must be in EVERY training view and in no
                validation: board B is one panel carrying 99 % of the Brücke /
                94 % of the Fahne boxes — a fold model trained without it could
                not learn those classes at all. Never reported as validation.
                (Board variant A2 — one burst of ~3 boards — is a normal fold
                group: validating on it is an honest "unseen variant" check.
                It is kept out of `test` so the one-shot test represents the
                main product rather than 3 variant boards.)
  future        not a list but a rule: any capture after `frozen_at` that has
                no assignment belongs here — the chronological test.

Probe (box level, not a bucket): every Brücke/Fahne box on board A/A2 is held
out of ALL training views — the tiler drops every training tile that touches
one. The photos themselves stay in their buckets (they carry 56 % of the NAL
boxes). No model ever sees a board-A bridge/flag, so these ~17 boxes are the
only honest cross-product check for Brücke/Fahne (evaluate with
`--bucket probe_xprod`).

Membership is keyed by capture_id and label-independent: unannotated and
quarantined photos also get their group's bucket, so later labeling slots in
without re-splitting.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from dataprep.grouping import GROUPS_FILE, load_groups

logger = logging.getLogger("ml.split")

K_FOLDS = 3
TEST_FRACTION = 0.2
TEST_FRACTION_SEATED_IO = 0.35  # scarce production-pose negatives: a bit more for test
MIN_POSITIVE_GROUPS_PER_FOLD = 3
PROBE_CLASSES = ("Brücke", "Fahne")
PROBE_BOARDS = ("A", "A2")
TRAIN_ALWAYS_BOARDS = ("B",)
NO_TEST_BOARDS = ("A2",)


def _is_train_always(g: dict) -> bool:
    return bool(set(g["boards"]) & set(TRAIN_ALWAYS_BOARDS))


def probe_boxes(manifest: dict) -> list[dict]:
    """Board-A/A2 Brücke/Fahne boxes of usable photos (held out of training)."""
    from dataprep.ingest import usable

    out = []
    for e in manifest["entries"]:
        if not usable(e) or e["board"] not in PROBE_BOARDS:
            continue
        for i, (name, box) in enumerate(e["boxes"]):
            if name in PROBE_CLASSES:
                out.append(
                    {"capture_id": e["capture_id"], "index": i, "class": name,
                     "box": box, "board": e["board"], "pose": e["pose"]}
                )
    return out


def build_split(groups_doc: dict, manifest: dict, k: int = K_FOLDS) -> dict:
    groups = groups_doc["groups"]
    train_always = sorted(g for g in groups if _is_train_always(groups[g]))
    pool = [g for g in sorted(groups) if g not in set(train_always)]

    by_session: dict[str, list[str]] = {}
    for gid in pool:
        by_session.setdefault(groups[gid]["session"], []).append(gid)

    # ---- test: per-session greedy fill toward the targets ----
    # Groups are lumpy (one Jul-20 group holds 21 NAL photos, one holds 10 of
    # the only 23 seated iO photos), so the greedy walks groups largest-first
    # under overshoot caps instead of taking anything below target. Seated iO
    # photos get their own target: they are the only production-pose
    # negatives, needed both for the dev threshold and the test false-call
    # rate. Deterministic: sorted, no RNG.
    def _cap(target: int) -> float:
        return max(target * 1.3, target + 2)

    test_groups: list[str] = []
    for session in sorted(by_session):
        gids = by_session[session]
        tot = {
            key: sum(groups[g][key] for g in gids)
            for key in ("n_nal_photos", "n_usable", "n_io_seated")
        }
        target = {
            "n_nal_photos": round(TEST_FRACTION * tot["n_nal_photos"]),
            "n_usable": round(TEST_FRACTION * tot["n_usable"]),
            "n_io_seated": round(TEST_FRACTION_SEATED_IO * tot["n_io_seated"]),
        }
        got = {key: 0 for key in target}

        def fits(g: dict) -> bool:
            return all(got[key] + g[key] <= _cap(target[key]) for key in target)

        order = sorted(
            gids,
            key=lambda g: (-groups[g]["n_nal_photos"], -groups[g]["n_io_seated"],
                           -groups[g]["n_usable"], g),
        )
        for gid in order:
            g = groups[gid]
            if set(g["boards"]) & set(NO_TEST_BOARDS):
                continue
            if g["n_nal_photos"] > 0:
                wanted = got["n_nal_photos"] < target["n_nal_photos"]
            elif g["n_io_seated"] > 0:
                wanted = got["n_io_seated"] < target["n_io_seated"]
            else:
                wanted = g["n_usable"] > 0 and got["n_usable"] < target["n_usable"]
            if wanted and fits(g):
                test_groups.append(gid)
                for key in got:
                    got[key] += g[key]
        logger.info(
            f"test {session}: "
            + ", ".join(f"{key[2:]} {got[key]}/{tot[key]} (target {target[key]})"
                        for key in target)
        )

    # ---- K folds over the remaining groups (two greedy passes) ----
    # Pass 1 spreads the NAL-positive groups (largest first) by NAL photos;
    # pass 2 spreads everything else by seated-iO then all usable photos —
    # a single pass would dump every negative-only group into whichever fold
    # happens to trail by one positive, leaving the other folds' validation
    # (and one fold's TRAINING view) without negatives.
    dev = [g for g in pool if g not in set(test_groups)]
    folds: dict[str, list[str]] = {str(i): [] for i in range(k)}
    fold_pos, fold_img, fold_ios = [0] * k, [0] * k, [0] * k

    def _assign(gid: str, i: int) -> None:
        folds[str(i)].append(gid)
        fold_pos[i] += groups[gid]["n_nal_photos"]
        fold_img[i] += groups[gid]["n_usable"]
        fold_ios[i] += groups[gid]["n_io_seated"]

    positive = [g for g in dev if groups[g]["n_nal_photos"] > 0]
    for gid in sorted(positive, key=lambda g: (-groups[g]["n_nal_photos"], -groups[g]["n_usable"], g)):
        _assign(gid, min(range(k), key=lambda j: (fold_pos[j], fold_img[j], j)))
    rest = [g for g in dev if groups[g]["n_nal_photos"] == 0]
    for gid in sorted(rest, key=lambda g: (-groups[g]["n_io_seated"], -groups[g]["n_usable"], g)):
        if groups[gid]["n_io_seated"] > 0:
            key = lambda j: (fold_ios[j], fold_img[j], j)  # noqa: E731
        else:
            key = lambda j: (fold_img[j], j)  # noqa: E731
        _assign(gid, min(range(k), key=key))
    for i in range(k):
        n_pos_groups = sum(1 for g in folds[str(i)] if groups[g]["n_nal_photos"])
        logger.info(
            f"fold {i}: {len(folds[str(i)])} groups, {fold_pos[i]} NAL photos "
            f"({n_pos_groups} groups), {fold_img[i]} usable photos, "
            f"{fold_ios[i]} seated iO"
        )
        if n_pos_groups < MIN_POSITIVE_GROUPS_PER_FOLD:
            logger.warning(f"fold {i} has only {n_pos_groups} NAL groups — noisy")

    bucket_of_group = {g: "train_always" for g in train_always}
    bucket_of_group.update({g: "test" for g in test_groups})
    for i, gids in folds.items():
        bucket_of_group.update({g: f"fold{i}" for g in gids})
    assignment = {
        cid: {"group": gid, "bucket": bucket_of_group[gid]}
        for gid, g in groups.items()
        for cid in g["captures"]
    }
    return {
        "split_id": settings.split_id,
        "frozen_at": max(e["captured_at"] for e in manifest["entries"]),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": settings.seed,
        "snapshot_id": manifest["snapshot_id"],
        "groups_file": GROUPS_FILE,
        "k_folds": k,
        "test_fraction": TEST_FRACTION,
        "test_fraction_seated_io": TEST_FRACTION_SEATED_IO,
        "train_always_boards": list(TRAIN_ALWAYS_BOARDS),
        "no_test_boards": list(NO_TEST_BOARDS),
        "test_groups": sorted(test_groups),
        "folds": {i: sorted(g) for i, g in folds.items()},
        "train_always_groups": train_always,
        "probe_boxes": probe_boxes(manifest),
        "assignment": assignment,
    }


def check_leakage(split: dict, groups_doc: dict) -> None:
    """No group spans buckets; every capture assigned exactly once."""
    seen: dict[str, str] = {}
    for cid, a in split["assignment"].items():
        gid, bucket = a["group"], a["bucket"]
        if gid in seen and seen[gid] != bucket:
            raise AssertionError(f"group {gid} spans buckets {seen[gid]} / {bucket}")
        seen[gid] = bucket
    all_captures = {c for g in groups_doc["groups"].values() for c in g["captures"]}
    assigned = set(split["assignment"])
    if all_captures != assigned:
        raise AssertionError(
            f"assignment mismatch: {len(all_captures - assigned)} unassigned, "
            f"{len(assigned - all_captures)} unknown"
        )
    sets = [set(split["test_groups"]), set(split["train_always_groups"]),
            *[set(g) for g in split["folds"].values()]]
    union = set().union(*sets)
    if sum(len(s) for s in sets) != len(union):
        raise AssertionError("a group is listed in two buckets")
    if union != set(groups_doc["groups"]):
        raise AssertionError("buckets must cover every group")


def bucket_of(entry: dict, split: dict) -> str:
    """Bucket of a manifest entry; unassigned captures after the freeze = future."""
    a = split["assignment"].get(entry["capture_id"])
    if a is not None:
        return a["bucket"]
    if entry["captured_at"] > split["frozen_at"]:
        return "future"
    raise KeyError(
        f"capture {entry['capture_id']} predates the split freeze but has no "
        "assignment — the manifest and the frozen split disagree"
    )


def train_buckets(split: dict, holdout_fold: str | None) -> set[str]:
    """Buckets of one training view (holdout_fold=None -> dev_all)."""
    folds = {f"fold{i}" for i in split["folds"]}
    if holdout_fold is not None:
        folds.discard(f"fold{holdout_fold}")
    return folds | {"train_always"}


def run(k: int = K_FOLDS) -> Path:
    from dataprep.ingest import load_manifest

    out = settings.datasets_dir / f"{settings.split_id}.json"
    if out.exists():
        raise RuntimeError(
            f"{out} already exists — {settings.split_id} is FROZEN. New captures "
            "fall into the `future` bucket automatically; a new split needs a new "
            "split_id, never an edit."
        )
    groups_doc = load_groups()
    split = build_split(groups_doc, load_manifest(), k=k)
    check_leakage(split, groups_doc)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(
        f"split written (frozen): {out} — {len(split['probe_boxes'])} probe boxes, "
        f"train_always groups {split['train_always_groups']}"
    )
    return out


def load_split() -> dict:
    path = settings.datasets_dir / f"{settings.split_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"run `python cli.py split` first — missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))
