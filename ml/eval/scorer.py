"""
Photo-level scorer for stitched detections — the metric that picks the winner.

Primary operating metric: recall @ FPPI <= 0.5 (false positives per PHOTO),
FROC-style, matched at IoU 0.3. Rationale (plan): at a 28-px median box a
3-px localization error already drops IoU below 0.5, which would count a
found defect as a miss PLUS a false call; operators review crops, so loose
localization is acceptable. AP@0.5 / AP@[.5:.95] are reported for
comparability with the literature, never for selection.

Inputs are per-photo records:
  {capture_id, gt: [xyxy], other_gt: [xyxy],   # dropped Brücke/Fahne boxes
   boxes: [xyxy], scores: [float]}             # stitched predictions

False positives at the operating point that overlap a known other-defect
region (IoU >= 0.1 with other_gt) are tagged separately — the model firing on
a real-but-out-of-scope defect is a different failure than a phantom call.

Bootstrap CIs resample PHOTOS (the independent unit) and re-derive the
operating point per resample.
"""
from __future__ import annotations

import math
import random

IOU_OPERATING = 0.3
FPPI_TARGETS = (0.125, 0.25, 0.5, 1.0, 2.0)
FPPI_OPERATING = 0.5
SIZE_BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 10_000))
BOOTSTRAP_N = 1000


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )
    return inter / max(union, 1e-9)


def match_photo(boxes, scores, gt, iou_thr):
    """
    Greedy score-descending matching.

    Returns (gt_match_scores, pred_matched) where gt_match_scores[i] is the
    score of the prediction matched to gt[i] (or None) and pred_matched is a
    per-prediction bool aligned with the input order.
    """
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    gt_match: list[float | None] = [None] * len(gt)
    pred_matched = [False] * len(boxes)
    for i in order:
        # strict `>` with an epsilon-lowered start keeps the FIRST best GT on
        # exact ties — deterministic regardless of GT ordering
        best_j, best_iou = -1, iou_thr - 1e-12
        for j, g in enumerate(gt):
            if gt_match[j] is not None:
                continue
            v = iou(boxes[i], g)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            gt_match[best_j] = scores[i]
            pred_matched[i] = True
    return gt_match, pred_matched


def _photo_stats(photos, iou_thr):
    """Per photo: (gt match scores incl. None, unmatched-pred scores)."""
    stats = []
    for p in photos:
        gm, pm = match_photo(p["boxes"], p["scores"], p["gt"], iou_thr)
        stats.append((gm, [s for s, m in zip(p["scores"], pm) if not m]))
    return stats


def _froc_from_stats(stats):
    """All (threshold, recall, fppi) step points, thresholds descending.

    Vectorized (numpy searchsorted): at a 0.001 conf floor there can be
    thousands of scores, and the naive O(T*N) sweep dominated evaluation.
    """
    import numpy as np

    n_photos = len(stats)
    n_gt = sum(len(gm) for gm, _ in stats)
    matched = np.sort(
        np.array([s for gm, _ in stats for s in gm if s is not None], dtype=float)
    )
    fps = np.sort(np.array([s for _, f in stats for s in f], dtype=float))
    thresholds = np.unique(np.concatenate([matched, fps]))[::-1]
    if not len(thresholds):
        return [], n_gt
    tp = len(matched) - np.searchsorted(matched, thresholds, side="left")
    fp = len(fps) - np.searchsorted(fps, thresholds, side="left")
    recall = tp / n_gt if n_gt else np.zeros_like(tp, dtype=float)
    fppi = fp / n_photos if n_photos else np.zeros_like(fp, dtype=float)
    return (
        [(float(t), float(r), float(f)) for t, r, f in zip(thresholds, recall, fppi)],
        n_gt,
    )


def _froc_points(photos, iou_thr):
    return _froc_from_stats(_photo_stats(photos, iou_thr))


def recall_at_fppi(points, target):
    """Best (recall, threshold) with fppi <= target on the FROC step curve."""
    best = (0.0, None)
    for t, recall, fppi in points:
        if fppi <= target and recall > best[0]:
            best = (recall, t)
    return best


def average_precision(photos, iou_thr):
    """COCO-style 101-point interpolated AP, single class."""
    scored: list[tuple[float, bool]] = []
    n_gt = 0
    for p in photos:
        gm, pm = match_photo(p["boxes"], p["scores"], p["gt"], iou_thr)
        n_gt += len(p["gt"])
        scored.extend(zip(p["scores"], pm))
    if n_gt == 0:
        return 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    recalls, precisions = [], []
    for _, matched in scored:
        tp += matched
        fp += not matched
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    # precision envelope + 101-point sampling
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for r in [i / 100 for i in range(101)]:
        p_at = 0.0
        for rec, prec in zip(recalls, precisions):
            if rec >= r:
                p_at = prec
                break
        ap += p_at / 101
    return ap


def _box_side(b) -> float:
    return math.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 0.0))


def score_photos(photos, iou_op=IOU_OPERATING, seed=0) -> dict:
    points, n_gt = _froc_points(photos, iou_op)
    froc = {}
    for target in FPPI_TARGETS:
        r, t = recall_at_fppi(points, target)
        froc[str(target)] = {"recall": round(r, 4), "threshold": t}
    op_recall, op_t = recall_at_fppi(points, FPPI_OPERATING)

    # ---- operating-point diagnostics ----
    # op_t is None when NO threshold meets the FPPI budget (degenerate model):
    # the operating point is then "predict nothing" — every GT is a miss and
    # zero predictions count as operating-point FPs.
    op_eff = op_t if op_t is not None else float("inf")
    fn_photos, fp_total, fp_other = [], 0, 0
    bucket_tot = {b: 0 for b in SIZE_BUCKETS}
    bucket_hit = {b: 0 for b in SIZE_BUCKETS}
    for p in photos:
        gm, pm = match_photo(p["boxes"], p["scores"], p["gt"], iou_op)
        for g, ms in zip(p["gt"], gm):
            side = _box_side(g)
            for b in SIZE_BUCKETS:
                if b[0] <= side < b[1]:
                    bucket_tot[b] += 1
                    if ms is not None and ms >= op_eff:
                        bucket_hit[b] += 1
        missed = [
            g for g, ms in zip(p["gt"], gm) if ms is None or ms < op_eff
        ]
        if missed:
            fn_photos.append({"capture_id": p["capture_id"], "missed": missed})
        for box, s, m in zip(p["boxes"], p["scores"], pm):
            if m or s < op_eff:
                continue
            fp_total += 1
            if any(iou(box, og) >= 0.1 for og in p.get("other_gt", [])):
                fp_other += 1

    # ---- bootstrap CI on the operating recall (resample photos; the
    # per-photo match stats are computed once and reused) ----
    rng = random.Random(seed)
    stats = _photo_stats(photos, iou_op)
    boot = []
    for _ in range(BOOTSTRAP_N):
        sample = [stats[rng.randrange(len(stats))] for _ in stats]
        pts, sample_gt = _froc_from_stats(sample)
        if sample_gt == 0:
            continue
        boot.append(recall_at_fppi(pts, FPPI_OPERATING)[0])
    boot.sort()
    ci = (
        (boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1])
        if boot
        else (0.0, 0.0)
    )

    ap50 = average_precision(photos, 0.5)
    ap5095 = sum(
        average_precision(photos, 0.5 + 0.05 * i) for i in range(10)
    ) / 10

    return {
        "n_photos": len(photos),
        "n_gt": n_gt,
        "iou_operating": iou_op,
        "fppi_operating": FPPI_OPERATING,
        "recall_at_op": round(op_recall, 4),
        "recall_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "threshold_at_op": op_t,
        "froc": froc,
        "ap50": round(ap50, 4),
        "ap50_95": round(ap5095, 4),
        "fp_at_op": fp_total,
        "fp_at_op_other_defect": fp_other,
        "size_bucket_recall_at_op": {
            f"{lo}-{hi if hi < 10_000 else 'inf'}px": (
                round(bucket_hit[(lo, hi)] / bucket_tot[(lo, hi)], 4)
                if bucket_tot[(lo, hi)]
                else None
            )
            for lo, hi in SIZE_BUCKETS
        },
        "fn_photos": fn_photos,
    }
