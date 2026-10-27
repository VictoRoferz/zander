"""
Class-aware photo-level scorer — measures exactly the required output:
i.O, or N.iO + which defect class(es) (infer/verdict.py decides; this scores).

Input: one record per photo
  {capture_id, group, board, pose, gt_verdict ("iO"|"NiO"),
   gt: [(class_idx, xyxy)]          boxes of the schema's classes
   other_gt: [(class_name, xyxy)]   real boxes of classes outside the schema
   boxes, scores, labels}           stitched predictions (labels = class idx)

Photo populations (per schema):
  positive  = has >= 1 schema-class GT box     -> escape rate denominators
  negative  = gt_verdict "iO" (no box at all)   -> false-call rate denominators
  photos whose only defects are outside the schema (nal1: Brücke/Fahne-only)
  are neither — counted, never scored as iO.

Three levels (plan 2026-09-28, Step 8):
  1 Verdict    photo AUROC; escape rate and false-call rate (seated vs
               handheld separately) at the threshold, Wilson 95 % CIs.
  2 Which      per class: photo sensitivity (class present -> reported) and
    error      class false alarm (reported but absent), exact-set accuracy on
               positives, photo-level class confusion. A wrong class is NOT an
               escape — verdict and class are scored separately.
  3 Defect     per class + class-agnostic: FROC / recall@0.5 FPPI (IoU 0.3)
               via eval/scorer.py with group-cluster bootstrap CIs; box-level
               confusion matrix at the threshold; AP50 per class.

The threshold is either FROZEN (operating_point.json from `cli.py choose-op`,
mandatory for test/probe/future) or, on folds only, self-tuned as the highest
t keeping photo sensitivity >= target (flagged `threshold_source`).

Sensitivity scope (user decision 2026-10-27): t* is fitted on the positives in
the production pose (`seated`) only. The handheld session is 9 NiO / 170 iO, so
its NiO photos score ≈ 0 and pulled the pooled t* down to the conf floor (every
seated iO flagged). Handheld photos are still scored and reported; they just
do not set t*. `all` = the rule before 2026-10-27.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from eval.scorer import IOU_OPERATING, iou, score_photos
from infer.verdict import photo_score

SENSITIVITY_TARGET = 0.95
SENSITIVITY_SCOPES = ("seated", "all")  # which positives t* must keep (see the docstring)
SENSITIVITY_SCOPE = "seated"


# ------------------------------------------------------------------ helpers


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUROC (ties count 1/2)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return round(wins / (len(pos) * len(neg)), 4)


def is_positive(p: dict) -> bool:
    return bool(p["gt"])


def is_negative(p: dict) -> bool:
    return p["gt_verdict"] == "iO"


def pscore(p: dict) -> float:
    return photo_score(p["scores"])


def in_scope(p: dict, scope: str) -> bool:
    if scope not in SENSITIVITY_SCOPES:
        raise ValueError(f"sensitivity scope {scope!r} not in {SENSITIVITY_SCOPES}")
    return scope == "all" or p.get("pose") == scope


def threshold_for_sensitivity(photos: list[dict], target: float = SENSITIVITY_TARGET,
                              scope: str = SENSITIVITY_SCOPE):
    """Highest t with photo sensitivity >= target on the positives in `scope`
    (fewest false calls at that sensitivity — Neyman-Pearson). None if there
    are no positives in scope."""
    pos = sorted((pscore(p) for p in photos if is_positive(p) and in_scope(p, scope)),
                 reverse=True)
    if not pos:
        return None
    k = math.ceil(target * len(pos))  # need >= k positives with score >= t
    t = pos[k - 1]
    return t if t > 0 else None


def predicted_classes(p: dict, t: float) -> set[int]:
    return {c for s, c in zip(p["scores"], p["labels"]) if s >= t}


# ------------------------------------------------------------------ levels


def level1_verdict(photos: list[dict], t: float | None) -> dict:
    pos = [p for p in photos if is_positive(p)]
    neg = [p for p in photos if is_negative(p)]
    out = {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "n_other_only": len(photos) - len(pos) - len(neg),
        "photo_auroc": auroc([pscore(p) for p in pos], [pscore(p) for p in neg]),
        "photo_auroc_seated": auroc(
            [pscore(p) for p in pos if p["pose"] == "seated"],
            [pscore(p) for p in neg if p["pose"] == "seated"],
        ),
    }
    if t is None:
        return out
    esc = sum(1 for p in pos if pscore(p) < t)
    out["escape"] = rate(esc, len(pos))
    sp = [p for p in pos if p["pose"] == "seated"]  # the population t* is fitted on
    out["escape_seated"] = rate(sum(1 for p in sp if pscore(p) < t), len(sp))
    for pose in ("seated", "handheld"):
        ns = [p for p in neg if p["pose"] == pose]
        out[f"false_call_{pose}"] = rate(sum(1 for p in ns if pscore(p) >= t), len(ns))
    flagged = [p for p in photos if pscore(p) >= t]
    out["false_call_share_of_flagged"] = rate(
        sum(1 for p in flagged if is_negative(p)), len(flagged)
    )
    # subgroups
    sub = {}
    for key in sorted({(p["board"], p["pose"]) for p in photos}):
        sp = [p for p in pos if (p["board"], p["pose"]) == key]
        sn = [p for p in neg if (p["board"], p["pose"]) == key]
        sub[f"{key[0]}/{key[1]}"] = {
            "escape": rate(sum(1 for p in sp if pscore(p) < t), len(sp)),
            "false_call": rate(sum(1 for p in sn if pscore(p) >= t), len(sn)),
        }
    out["by_board_pose"] = sub
    return out


def level2_classes(photos: list[dict], t: float, class_names: list[str]) -> dict:
    per_class = {}
    for c, name in enumerate(class_names):
        has = [p for p in photos if any(g[0] == c for g in p["gt"])]
        lacks = [p for p in photos if not any(g[0] == c for g in p["gt"])]
        alarms_io = defaultdict(lambda: [0, 0])
        for p in lacks:
            if is_negative(p):
                alarms_io[p["board"]][1] += 1
                alarms_io[p["board"]][0] += c in predicted_classes(p, t)
        per_class[name] = {
            "photo_sensitivity": rate(
                sum(1 for p in has if c in predicted_classes(p, t)), len(has)
            ),
            "class_false_alarm": rate(
                sum(1 for p in lacks if c in predicted_classes(p, t)), len(lacks)
            ),
            # background-shortcut test: this class reported on defect-free photos, per board
            "false_alarm_on_iO_by_board": {
                b: rate(k, n) for b, (k, n) in sorted(alarms_io.items())
            },
        }
    pos = [p for p in photos if is_positive(p)]
    exact = sum(
        1 for p in pos if predicted_classes(p, t) == {g[0] for g in p["gt"]}
    )
    confusion = {name: Counter() for name in class_names}
    for p in pos:
        pred = predicted_classes(p, t)
        for c in {g[0] for g in p["gt"]}:
            row = confusion[class_names[c]]
            if not pred:
                row["(none)"] += 1
            for q in pred:
                row[class_names[q]] += 1
    return {
        "per_class": per_class,
        "exact_set_accuracy": rate(exact, len(pos)),
        "photo_class_confusion": {k: dict(v) for k, v in confusion.items()},
    }


def class_records(photos: list[dict], c: int | None) -> list[dict]:
    """Single-class view for eval/scorer.py (c=None -> class-agnostic)."""
    out = []
    for p in photos:
        if c is None:
            gt = [b for _, b in p["gt"]]
            other = [b for _, b in p["other_gt"]]
            idx = range(len(p["boxes"]))
        else:
            gt = [b for k, b in p["gt"] if k == c]
            other = [b for k, b in p["gt"] if k != c] + [b for _, b in p["other_gt"]]
            idx = [i for i, lab in enumerate(p["labels"]) if lab == c]
        out.append({
            "capture_id": p["capture_id"],
            "gt": gt,
            "other_gt": other,
            "boxes": [p["boxes"][i] for i in idx],
            "scores": [p["scores"][i] for i in idx],
        })
    return out


def level3_defects(photos: list[dict], t: float | None, class_names: list[str]) -> dict:
    clusters = [p["group"] for p in photos]
    out = {"per_class": {}}
    for c, name in enumerate(class_names):
        recs = class_records(photos, c)
        if not any(r["gt"] for r in recs):
            out["per_class"][name] = {"n_gt": 0}
            continue
        m = score_photos(recs, cluster_ids=clusters)
        m.pop("fn_photos", None)
        out["per_class"][name] = m
    agn = score_photos(class_records(photos, None), cluster_ids=clusters)
    agn.pop("fn_photos", None)
    out["class_agnostic"] = agn
    if t is not None:
        out["box_confusion"] = box_confusion(photos, t, class_names)
    return out


def match_indices(boxes, scores, gt, iou_thr) -> tuple[list[int | None], list[bool]]:
    """Same greedy rule as eval/scorer.match_photo, returning GT->pred indices."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    gt_to_pred: list[int | None] = [None] * len(gt)
    matched = [False] * len(boxes)
    for i in order:
        best_j, best_iou = -1, iou_thr - 1e-12
        for j, g in enumerate(gt):
            if gt_to_pred[j] is not None:
                continue
            v = iou(boxes[i], g)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            gt_to_pred[best_j] = i
            matched[i] = True
    return gt_to_pred, matched


def box_confusion(photos: list[dict], t: float, class_names: list[str]) -> dict:
    """GT class x predicted class at threshold t, class-agnostic IoU matching."""
    cols = class_names + ["(missed)"]
    matrix = {name: Counter() for name in class_names + ["(background)"]}
    for p in photos:
        keep = [i for i, s in enumerate(p["scores"]) if s >= t]
        boxes = [p["boxes"][i] for i in keep]
        scores = [p["scores"][i] for i in keep]
        labels = [p["labels"][i] for i in keep]
        gt_to_pred, matched = match_indices(
            boxes, scores, [b for _, b in p["gt"]], IOU_OPERATING
        )
        for (c, _), j in zip(p["gt"], gt_to_pred):
            matrix[class_names[c]]["(missed)" if j is None else class_names[labels[j]]] += 1
        for j, m in enumerate(matched):
            if not m:
                matrix["(background)"][class_names[labels[j]]] += 1
    return {"columns": cols, "rows": {k: dict(v) for k, v in matrix.items()}}


def probe_hits(photos: list[dict], probe: list[dict], t: float | None,
               class_names: list[str]) -> list[dict]:
    """Per probe box: found by any class above t? found as its own class?"""
    by_id = {p["capture_id"]: p for p in photos}
    out = []
    for pb in probe:
        p = by_id.get(pb["capture_id"])
        if p is None:
            continue
        best = None
        for b, s, c in zip(p["boxes"], p["scores"], p["labels"]):
            if (t is None or s >= t) and iou(b, pb["box"]) >= IOU_OPERATING:
                if best is None or s > best[0]:
                    best = (s, class_names[c])
        out.append({
            "capture_id": pb["capture_id"], "class": pb["class"], "board": pb["board"],
            "found": best is not None,
            "predicted_class": best[1] if best else None,
            "score": round(best[0], 4) if best else None,
            "class_correct": bool(best and best[1] == pb["class"]),
        })
    return out


def score_all(photos: list[dict], class_names: list[str], threshold: float | None,
              threshold_source: str, probe: list[dict] | None = None) -> dict:
    if threshold is None:
        threshold = threshold_for_sensitivity(photos)
        threshold_source = (f"self-tuned on this bucket, {SENSITIVITY_SCOPE} positives "
                            "(diagnostic only)")
    result = {
        "class_names": class_names,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "level1_verdict": level1_verdict(photos, threshold),
        "level3_defects": level3_defects(photos, threshold, class_names),
    }
    if threshold is not None:
        result["level2_classes"] = level2_classes(photos, threshold, class_names)
    if probe:
        hits = probe_hits(photos, probe, threshold, class_names)
        summary = defaultdict(lambda: {"n": 0, "found": 0, "class_correct": 0})
        for h in hits:
            summary[h["class"]]["n"] += 1
            summary[h["class"]]["found"] += h["found"]
            summary[h["class"]]["class_correct"] += h["class_correct"]
        result["probe_xprod"] = {"summary": dict(summary), "boxes": hits}
    return result
