"""
Ingest v2 invariants on the REAL committed manifest (ml/datasets/manifest.json)
plus the schema remap and quarantine rules.

Run (no pytest needed):  python tests/test_ingest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from dataprep.quarantine import REVIEW, quarantine_reason  # noqa: E402
from dataprep.remap import project, schema_classes  # noqa: E402

with open(ROOT / "ml" / "datasets" / "manifest.json", encoding="utf-8") as f:
    MANIFEST = json.load(f)
E = MANIFEST["entries"]


def test_manifest_is_v2_and_multi_dir():
    assert MANIFEST["manifest_version"] == 2
    assert len(MANIFEST["source_images_dirs"]) == 2
    assert len(E) == 566
    assert len({e["capture_id"] for e in E}) == 566


def test_unannotated_is_never_a_negative():
    for e in E:
        if e["label_state"] == "unannotated":
            assert e["verdict"] is None and e["boxes"] == []


def test_verdict_matches_boxes():
    for e in E:
        if e["label_state"] == "annotated":
            assert e["verdict"] == ("NiO" if e["boxes"] else "iO")
            assert e["classes_present"] == sorted({n for n, _ in e["boxes"]})


def test_review_items_are_excluded():
    states = {e["capture_id"]: e["label_state"] for e in E}
    for cid in REVIEW:
        assert states[cid] == "review", cid


def test_no_pcb_frames_are_quarantined():
    for e in E:
        if e["content"] != "pcb":
            assert e["quarantine_reason"], e["capture_id"]
    labeled_junk = [e for e in E if e["quarantine_reason"] and e["label_state"] != "unannotated"]
    assert {e["capture_id"][:8] for e in labeled_junk} == {"7ed4ecf0", "fa64f74a"}


def test_usable_photos_have_board_and_pose():
    for e in E:
        if e["quarantine_reason"] is None and e["label_state"] == "annotated":
            assert e["board"] in ("A", "A2", "B") and e["pose"] in ("seated", "handheld")


def test_quarantine_rule_order():
    assert quarantine_reason("basler", 120.0, "pcb") is None
    assert "audit content" in quarantine_reason("basler", 120.0, "no_pcb")
    assert "non-camera" in quarantine_reason("fallback", 120.0, "pcb")
    assert "dark" in quarantine_reason("basler", 10.0, "pcb")


def test_schema_projection():
    boxes = [["Brücke", [0, 0, 10, 10]], ["Nicht ausreichend Lot", [5, 5, 9, 9]],
             ["Fahne", [1, 1, 2, 2]]]
    train, other = project(boxes, "def3")
    assert [c for c, _ in train] == [1, 0, 2] and other == []
    train, other = project(boxes, "nal1")
    assert [c for c, _ in train] == [0] and [n for n, _ in other] == ["Brücke", "Fahne"]
    assert schema_classes("def3")[0] == "Nicht ausreichend Lot"
    try:
        project([["Kratzer", [0, 0, 1, 1]]], "def3")
        raise AssertionError("unknown class must raise")
    except ValueError:
        pass


def test_image_paths_are_repo_relative_posix():
    for e in E:
        assert not e["image_path"].startswith("/") and "\\" not in e["image_path"]
        assert (ROOT / e["image_path"]).exists() or not (ROOT / "data_fotos").exists()


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
