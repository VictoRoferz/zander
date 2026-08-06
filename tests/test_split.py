"""
Leakage + integrity tests for the committed frozen split (ml/datasets/).

These run against the REAL committed artifacts: if split_v1.json and
groups.json ever disagree, or a group leaks across buckets, every reported
metric is invalid — this is the test that guards the benchmark's honesty.

Run (no pytest needed):  python tests/test_split.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

DATASETS = ROOT / "ml" / "datasets"

with open(DATASETS / "groups.json", encoding="utf-8") as f:
    GROUPS_DOC = json.load(f)
with open(DATASETS / "split_v1.json", encoding="utf-8") as f:
    SPLIT = json.load(f)
with open(DATASETS / "manifest.json", encoding="utf-8") as f:
    MANIFEST = json.load(f)


def test_no_group_spans_buckets():
    from dataprep.split import check_leakage

    check_leakage(SPLIT, GROUPS_DOC)  # raises on leakage


def test_buckets_are_disjoint_and_complete():
    test_set = set(SPLIT["test_groups"])
    fold_sets = [set(g) for g in SPLIT["folds"].values()]
    all_sets = [test_set, *fold_sets]
    union = set().union(*all_sets)
    assert sum(len(s) for s in all_sets) == len(union), "overlapping buckets"
    assert union == set(GROUPS_DOC["groups"]), "buckets must cover every group"


def test_every_capture_assigned_once():
    captures = {e["capture_id"] for e in MANIFEST["entries"]}
    assert set(SPLIT["assignment"]) == captures


def test_groups_are_single_session():
    # Boards were not reused across days (user-confirmed) — a group spanning
    # sessions would mean the grouping code broke.
    session_of = {e["capture_id"]: e["session"] for e in MANIFEST["entries"]}
    for gid, g in GROUPS_DOC["groups"].items():
        sessions = {session_of[c] for c in g["captures"]}
        assert len(sessions) == 1, f"group {gid} spans sessions {sessions}"


def test_test_fraction_of_positives_is_sane():
    positives = {
        e["capture_id"]
        for e in MANIFEST["entries"]
        if e["label_state"] == "annotated_positive"
    }
    in_test = sum(
        1 for c in positives if SPLIT["assignment"][c]["bucket"] == "test"
    )
    frac = in_test / len(positives)
    assert 0.12 <= frac <= 0.32, f"test holds {frac:.0%} of positives"


def test_every_fold_has_positives():
    fold_pos = {i: 0 for i in SPLIT["folds"]}
    for e in MANIFEST["entries"]:
        if e["label_state"] != "annotated_positive" or e["quarantine_reason"]:
            continue
        bucket = SPLIT["assignment"][e["capture_id"]]["bucket"]
        if bucket.startswith("fold"):
            fold_pos[bucket.removeprefix("fold")] += 1
    for i, n in fold_pos.items():
        assert n >= 10, f"fold {i} has only {n} positive photos"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
