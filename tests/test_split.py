"""
Leakage + integrity tests for the committed frozen split_v2 (ml/datasets/).

These run against the REAL committed artifacts: if split_v2.json, groups_v2.json
and the manifest ever disagree, or a board group leaks across buckets, every
reported metric is invalid — this is the test that guards the benchmark's
honesty.

Run (no pytest needed):  python tests/test_split.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

DATASETS = ROOT / "ml" / "datasets"

with open(DATASETS / "groups_v2.json", encoding="utf-8") as f:
    GROUPS_DOC = json.load(f)
with open(DATASETS / "split_v2.json", encoding="utf-8") as f:
    SPLIT = json.load(f)
with open(DATASETS / "manifest.json", encoding="utf-8") as f:
    MANIFEST = json.load(f)
BY_ID = {e["capture_id"]: e for e in MANIFEST["entries"]}


def _usable(e):
    return e["quarantine_reason"] is None and e["label_state"] == "annotated"


def test_no_group_spans_buckets():
    from dataprep.split import check_leakage

    check_leakage(SPLIT, GROUPS_DOC)  # raises on leakage


def test_every_capture_assigned_once():
    assert set(SPLIT["assignment"]) == set(BY_ID)


def test_groups_are_single_session():
    # Boards were not reused across days (user-confirmed) — a group spanning
    # sessions would mean the grouping code broke.
    for gid, g in GROUPS_DOC["groups"].items():
        sessions = {BY_ID[c]["session"] for c in g["captures"]}
        assert len(sessions) == 1, f"group {gid} spans sessions {sessions}"


def test_reshots_share_a_group():
    # Known re-shot (same three rare NAL defects, taken 30 min apart on Jul-20):
    # split_v1 put them into different buckets — v2 must not.
    a = SPLIT["assignment"]["95dd9cc6" + next(
        c[8:] for c in BY_ID if c.startswith("95dd9cc6"))]
    b = SPLIT["assignment"][next(c for c in BY_ID if c.startswith("32a04ceb"))]
    assert a["group"] == b["group"]


def test_board_b_is_one_group_in_train_always():
    groups = {SPLIT["assignment"][c]["group"] for c, e in BY_ID.items() if e["board"] == "B"}
    assert len(groups) == 1
    assert SPLIT["assignment"][next(c for c, e in BY_ID.items() if e["board"] == "B")]["bucket"] == "train_always"


def test_probe_boxes_are_board_a_bruecke_fahne_outside_test():
    assert SPLIT["probe_boxes"], "probe set must not be empty"
    for p in SPLIT["probe_boxes"]:
        assert p["class"] in ("Brücke", "Fahne") and p["board"] in ("A", "A2")
        assert SPLIT["assignment"][p["capture_id"]]["bucket"] != "train_always"
        e = BY_ID[p["capture_id"]]
        assert e["boxes"][p["index"]] == [p["class"], p["box"]]


def test_test_fraction_of_nal_photos_is_sane():
    pos = [e for e in BY_ID.values() if _usable(e) and "Nicht ausreichend Lot" in e["classes_present"]]
    in_test = sum(1 for e in pos if SPLIT["assignment"][e["capture_id"]]["bucket"] == "test")
    frac = in_test / len(pos)
    assert 0.12 <= frac <= 0.32, f"test holds {frac:.0%} of NAL photos"


def test_every_fold_has_nal_photos_and_groups():
    for i in SPLIT["folds"]:
        es = [e for e in BY_ID.values() if _usable(e)
              and SPLIT["assignment"][e["capture_id"]]["bucket"] == f"fold{i}"]
        nal = [e for e in es if "Nicht ausreichend Lot" in e["classes_present"]]
        groups = {SPLIT["assignment"][e["capture_id"]]["group"] for e in nal}
        assert len(nal) >= 15, f"fold {i} has only {len(nal)} NAL photos"
        assert len(groups) >= 3, f"fold {i} has only {len(groups)} NAL groups"


def test_test_has_seated_negatives():
    n = sum(1 for e in BY_ID.values() if _usable(e) and e["verdict"] == "iO"
            and e["pose"] == "seated" and SPLIT["assignment"][e["capture_id"]]["bucket"] == "test")
    assert n >= 8, f"only {n} seated iO photos in test — false-call rate unmeasurable"


def test_future_rule():
    from dataprep.split import bucket_of

    late = {"capture_id": "new", "captured_at": "2099-01-01T00:00:00Z"}
    assert bucket_of(late, SPLIT) == "future"
    try:
        bucket_of({"capture_id": "ghost", "captured_at": "2000-01-01T00:00:00Z"}, SPLIT)
        raise AssertionError("an unassigned pre-freeze capture must raise")
    except KeyError:
        pass


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
