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



# ---------------------------------------------------------------------------
# Class-aware photo-level scorer (ml/eval/photo_scorer.py) — hand fixture.
#
# Classes [NAL, Brücke, Fahne]. Positives P1 (0.9), P2 (0.8), P5 (0.7);
# negatives P3 seated (0.6), P4 handheld (0.3).
#   t for sensitivity >= 0.95 = 3rd-highest positive score = 0.7
#   at t=0.7: predicted classes P1{NAL} P2{NAL} P3{} P4{} P5{Fahne}
#   escape 0/3 · false call seated 0/1 · handheld 0/1 · AUROC 1.0
#   NAL: sens 1/1, false alarm 1/4 (on P2) · Brücke: sens 0/2 · Fahne 1/1
#   exact class-set accuracy 1/3 (only P1)
#   box confusion: NAL->NAL 1, Brücke->NAL 1, Brücke->missed 1, Fahne->Fahne 1
#   class-agnostic recall@0.5FPPI = 3/4 (2 FPs over 5 photos = 0.4 FPPI)
#   probe (P2's Brücke on board A): found, wrong class (NAL)

from eval.photo_scorer import (  # noqa: E402
    auroc,
    score_all,
    threshold_for_sensitivity,
    wilson,
)

NAL_BOX, BR_BOX = (100, 100, 130, 130), (200, 200, 240, 220)
FA_BOX, BR5_BOX = (300, 300, 330, 330), (400, 400, 440, 420)


def _p(cid, board, pose, verdict, gt, boxes, scores, labels, group=None):
    return {"capture_id": cid, "group": group or cid, "board": board, "pose": pose,
            "gt_verdict": verdict, "gt": gt, "other_gt": [],
            "boxes": boxes, "scores": scores, "labels": labels}


MC = [
    _p("P1", "A", "seated", "NiO", [(0, NAL_BOX)], [NAL_BOX], [0.9], [0]),
    _p("P2", "A", "seated", "NiO", [(1, BR_BOX)], [BR_BOX], [0.8], [0]),
    _p("P3", "A", "seated", "iO", [], [(50, 50, 80, 80)], [0.6], [2]),
    _p("P4", "A", "handheld", "iO", [], [(60, 60, 90, 90)], [0.3], [0]),
    _p("P5", "B", "seated", "NiO", [(2, FA_BOX), (1, BR5_BOX)], [FA_BOX], [0.7], [2]),
]
NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
PROBE = [{"capture_id": "P2", "index": 0, "class": "Brücke", "box": BR_BOX,
          "board": "A", "pose": "seated"}]


def test_mc_threshold_and_verdict_level():
    assert threshold_for_sensitivity(MC) == 0.7
    m = score_all(MC, NAMES, None, "", probe=PROBE)
    l1 = m["level1_verdict"]
    assert m["threshold"] == 0.7 and "self-tuned" in m["threshold_source"]
    assert (l1["n_positive"], l1["n_negative"], l1["n_other_only"]) == (3, 2, 0)
    assert l1["photo_auroc"] == 1.0
    assert l1["escape"]["k"] == 0 and l1["escape"]["n"] == 3
    assert l1["false_call_seated"]["n"] == 1 and l1["false_call_seated"]["k"] == 0
    assert l1["false_call_handheld"]["n"] == 1


def test_mc_class_level():
    l2 = score_all(MC, NAMES, 0.7, "frozen")["level2_classes"]
    pc = l2["per_class"]
    assert (pc["Nicht ausreichend Lot"]["photo_sensitivity"]["k"],
            pc["Nicht ausreichend Lot"]["photo_sensitivity"]["n"]) == (1, 1)
    assert (pc["Nicht ausreichend Lot"]["class_false_alarm"]["k"],
            pc["Nicht ausreichend Lot"]["class_false_alarm"]["n"]) == (1, 4)
    assert (pc["Brücke"]["photo_sensitivity"]["k"], pc["Brücke"]["photo_sensitivity"]["n"]) == (0, 2)
    assert (pc["Fahne"]["photo_sensitivity"]["k"], pc["Fahne"]["photo_sensitivity"]["n"]) == (1, 1)
    assert (l2["exact_set_accuracy"]["k"], l2["exact_set_accuracy"]["n"]) == (1, 3)
    assert l2["photo_class_confusion"]["Brücke"] == {"Nicht ausreichend Lot": 1, "Fahne": 1}


def test_mc_wrong_class_is_not_an_escape():
    # P2 is a Brücke reported as NAL: the verdict is right (N.iO), the class wrong.
    m = score_all(MC, NAMES, 0.7, "frozen")
    assert m["level1_verdict"]["escape"]["k"] == 0
    assert m["level3_defects"]["box_confusion"]["rows"]["Brücke"] == {
        "Nicht ausreichend Lot": 1, "(missed)": 1}


def test_mc_box_confusion_and_defect_level():
    l3 = score_all(MC, NAMES, 0.7, "frozen")["level3_defects"]
    rows = l3["box_confusion"]["rows"]
    assert rows["Nicht ausreichend Lot"] == {"Nicht ausreichend Lot": 1}
    assert rows["Fahne"] == {"Fahne": 1}
    assert rows["(background)"] == {}  # P3's 0.6 FP is below t
    assert l3["per_class"]["Nicht ausreichend Lot"]["recall_at_op"] == 1.0
    assert l3["per_class"]["Nicht ausreichend Lot"]["fp_at_op_other_defect"] == 0
    _approx(l3["class_agnostic"]["recall_at_op"], 0.75)
    assert l3["class_agnostic"]["ci_unit"] == "group"


def test_mc_probe_hit_list():
    probe = score_all(MC, NAMES, 0.7, "frozen", probe=PROBE)["probe_xprod"]
    assert probe["summary"]["Brücke"] == {"n": 1, "found": 1, "class_correct": 0}
    assert probe["boxes"][0]["predicted_class"] == "Nicht ausreichend Lot"


def test_wilson_and_auroc_hand_values():
    lo, hi = wilson(18, 20)
    _approx(lo, 0.699, 2e-3)
    _approx(hi, 0.972, 2e-3)
    assert wilson(0, 0) is None
    assert auroc([0.5], [0.5]) == 0.5 and auroc([], [0.1]) is None


def test_cluster_bootstrap_is_deterministic_and_grouped():
    from eval.scorer import cluster_resample
    import random

    idx = cluster_resample(6, ["g1", "g1", "g2", "g2", "g2", "g3"], random.Random(0))
    idx2 = cluster_resample(6, ["g1", "g1", "g2", "g2", "g2", "g3"], random.Random(0))
    assert idx == idx2
    # whole groups only: g2's members always appear together
    assert idx.count(2) == idx.count(3) == idx.count(4)


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
