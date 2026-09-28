"""
The photo decision rule (ml/infer/verdict.py) — shared by eval and inspector.

Required output: N.iO + detected class(es), or i.O. Run: python tests/test_verdict.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from infer.verdict import photo_verdict  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
B1, B2, B3 = (10, 10, 40, 40), (100, 100, 130, 130), (200, 200, 230, 230)


def test_io_when_nothing_reaches_threshold():
    v = photo_verdict([B1, B2], [0.2, 0.39], [0, 1], threshold=0.4, class_names=NAMES)
    assert v["verdict"] == "iO"
    assert v["classes"] == [] and v["defects"] == []
    assert v["photo_score"] == 0.39


def test_io_without_any_detection():
    v = photo_verdict([], [], [], threshold=0.4, class_names=NAMES)
    assert v["verdict"] == "iO" and v["photo_score"] == 0.0


def test_nio_with_class_set_in_schema_order():
    v = photo_verdict([B1, B2, B3], [0.5, 0.9, 0.1], [2, 0, 1], threshold=0.4, class_names=NAMES)
    assert v["verdict"] == "NiO"
    assert v["classes"] == ["Nicht ausreichend Lot", "Fahne"]  # Brücke below t
    assert [d["class"] for d in v["defects"]] == ["Nicht ausreichend Lot", "Fahne"]  # score-desc


def test_threshold_is_inclusive():
    v = photo_verdict([B1], [0.4], [1], threshold=0.4, class_names=NAMES)
    assert v["verdict"] == "NiO" and v["classes"] == ["Brücke"]


def test_duplicate_class_reported_once():
    v = photo_verdict([B1, B2], [0.8, 0.7], [1, 1], threshold=0.5, class_names=NAMES)
    assert v["classes"] == ["Brücke"] and len(v["defects"]) == 2


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
