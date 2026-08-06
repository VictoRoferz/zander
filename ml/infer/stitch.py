"""
Cross-tile merging of detections in photo coordinates.

Plain IoU-NMS is wrong here: a defect larger than one tile (max observed NAL
box is ~707 px vs 640 px tiles) is predicted as several non-identical
fragments whose pairwise IoU stays below any sane NMS threshold — it would be
counted 2-4 times, inflating the false-call rate. We therefore use greedy
non-max MERGING keyed on IoS (intersection over the SMALLER box): fragments
of one physical defect overlap heavily relative to the smaller fragment even
when their IoU is low.

Contract: greedy over descending score; a candidate with IoS >= threshold
against an already-kept box is absorbed into it (kept box grows to the union
rectangle, score stays the max = the kept one's score).
"""
from __future__ import annotations


def _ios(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = max(min(area_a, area_b), 1e-9)
    return inter / smaller


def merge_detections(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    ios_threshold: float = 0.5,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Greedy non-max merge in photo coords. Returns (boxes, scores), score-desc."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    for i in order:
        box = boxes[i]
        merged = False
        for kb in kept_boxes:
            if _ios(tuple(kb), box) >= ios_threshold:
                kb[0] = min(kb[0], box[0])
                kb[1] = min(kb[1], box[1])
                kb[2] = max(kb[2], box[2])
                kb[3] = max(kb[3], box[3])
                merged = True
                break
        if not merged:
            kept_boxes.append(list(box))
            kept_scores.append(scores[i])
    return [tuple(b) for b in kept_boxes], kept_scores
