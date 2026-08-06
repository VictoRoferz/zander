"""
Hand-computed micro-fixture for the photo-level scorer (ml/eval/scorer.py).

Three photos, three GT boxes, four predictions — every number below is
derivable by hand, so a regression in matching, FROC, or AP is caught exactly.

Fixture:
  photo A: GT1 24px at (300,300,324,324) — pred hits it exactly, score 0.9
           GT2 40px at (100,100,140,140) — pred at (115,100,155,140),
             IoU = 1000/2200 = 0.4545 (match at IoU 0.3, NOT at 0.5), score 0.7
           plus one phantom FP at (800,800,830,830), score 0.8
  photo B: GT3 12px at (50,50,62,62) — no predictions (a miss)
  photo C: no GT; one FP at (505,505,532,532), score 0.95, overlapping a
           dropped other-defect box (500,500,530,530) with IoU 0.62 -> tagged

Hand results (IoU_op = 0.3):
  matched GT scores: 0.9, 0.7, None      FP scores: 0.8, 0.95
  FROC (t, recall, FPPI): (0.95, 0, 1/3) (0.9, 1/3, 1/3)
                          (0.8, 1/3, 2/3) (0.7, 2/3, 2/3)
  recall@FPPI<=0.5  = 1/3 at t* = 0.9 (ties resolve to the HIGHER
                      threshold — fewer false calls at equal recall)
  recall@FPPI<=0.125 = 0 (no threshold reaches so few FPs) -> threshold None
  recall@FPPI<=2.0  = 2/3
  fp_at_op = 1 (the 0.95 phantom), all of it other-defect-tagged
  AP@0.5: preds desc 0.95 FP, 0.9 TP, 0.8 FP, 0.7 FP -> envelope 1/2 up to
          recall 1/3 -> AP = 34 * (1/2) / 101 = 17/101
  size buckets at t*: 0-16px: 0/1, 16-32px: 1/1, 32-64px: 0/1

Run (no pytest needed):  python tests/test_scorer.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from eval.scorer import (  # noqa: E402
    average_precision,
    match_photo,
    score_photos,
)

PHOTOS = [
    {
        "capture_id": "photoA",
        "gt": [(300, 300, 324, 324), (100, 100, 140, 140)],
        "other_gt": [],
        "boxes": [(300, 300, 324, 324), (115, 100, 155, 140), (800, 800, 830, 830)],
        "scores": [0.9, 0.7, 0.8],
    },
    {
        "capture_id": "photoB",
        "gt": [(50, 50, 62, 62)],
        "other_gt": [],
        "boxes": [],
        "scores": [],
    },
    {
        "capture_id": "photoC",
        "gt": [],
        "other_gt": [(500, 500, 530, 530)],
        "boxes": [(505, 505, 532, 532)],
        "scores": [0.95],
    },
]


def _approx(a, b, tol=1e-3):
    assert abs(a - b) < tol, f"{a} != {b} (±{tol})"


def test_match_greedy_and_iou_threshold():
    p = PHOTOS[0]
    gm, pm = match_photo(p["boxes"], p["scores"], p["gt"], 0.3)
    assert gm == [0.9, 0.7]
    assert pm == [True, True, False]
    # at IoU 0.5 the shifted pred no longer matches
    gm50, pm50 = match_photo(p["boxes"], p["scores"], p["gt"], 0.5)
    assert gm50 == [0.9, None]
    assert pm50 == [True, False, False]


def test_one_pred_cannot_match_two_gt():
    gm, pm = match_photo(
        [(0, 0, 20, 20)], [0.9], [(0, 0, 20, 20), (1, 1, 21, 21)], 0.3
    )
    assert sum(1 for s in gm if s is not None) == 1


def test_operating_point_and_froc():
    m = score_photos(PHOTOS)
    assert m["n_photos"] == 3 and m["n_gt"] == 3
    _approx(m["recall_at_op"], 1 / 3)
    _approx(m["threshold_at_op"], 0.9)
    _approx(m["froc"]["2.0"]["recall"], 2 / 3)
    _approx(m["froc"]["0.125"]["recall"], 0.0)
    assert m["froc"]["0.125"]["threshold"] is None


def test_fp_tagging_against_other_defects():
    m = score_photos(PHOTOS)
    assert m["fp_at_op"] == 1
    assert m["fp_at_op_other_defect"] == 1


def test_ap50_hand_value():
    _approx(average_precision(PHOTOS, 0.5), 17 / 101)


def test_size_bucket_recall():
    m = score_photos(PHOTOS)
    buckets = m["size_bucket_recall_at_op"]
    _approx(buckets["0-16px"], 0.0)
    _approx(buckets["16-32px"], 1.0)
    _approx(buckets["32-64px"], 0.0)


def test_bootstrap_ci_is_sane():
    m = score_photos(PHOTOS)
    lo, hi = m["recall_ci95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_empty_predictions_all_recall_zero():
    empty = [dict(p, boxes=[], scores=[]) for p in PHOTOS]
    m = score_photos(empty)
    assert m["recall_at_op"] == 0.0
    assert m["fp_at_op"] == 0


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
