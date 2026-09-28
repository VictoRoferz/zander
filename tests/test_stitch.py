"""
Class-aware cross-tile merge (ml/infer/stitch.py merge_by_class).

Run: python tests/test_stitch.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from infer.stitch import merge_by_class  # noqa: E402

A = (1540.0, 700.0, 1568.0, 728.0)
A_DUP = (1540.4, 700.2, 1568.5, 728.6)


def test_same_class_duplicates_merge():
    b, s, lab = merge_by_class([A, A_DUP], [0.85, 0.83], [0, 0])
    assert len(b) == 1 and s == [0.85] and lab == [0]


def test_different_classes_never_merge():
    # A bridge and an insufficient-solder call on overlapping pixels are two
    # decisions, not fragments of one defect.
    b, s, lab = merge_by_class([A, A_DUP], [0.85, 0.83], [0, 1])
    assert len(b) == 2 and sorted(lab) == [0, 1]


def test_output_is_score_descending_across_classes():
    far = (10.0, 10.0, 40.0, 40.0)
    b, s, lab = merge_by_class([A, far], [0.3, 0.9], [0, 2])
    assert s == [0.9, 0.3] and lab == [2, 0]


def test_per_class_cap():
    boxes = [(i * 100.0, 0.0, i * 100.0 + 30, 30.0) for i in range(6)]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    labels = [0, 0, 0, 1, 1, 1]
    b, s, lab = merge_by_class(boxes, scores, labels, max_per_class=2)
    assert lab.count(0) == 2 and lab.count(1) == 2
    assert s == [0.9, 0.8, 0.6, 0.5]  # the top-2 of each class survive


def test_empty_input():
    assert merge_by_class([], [], []) == ([], [], [])


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
