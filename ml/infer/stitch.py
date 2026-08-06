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

Implementation is numpy-vectorized per candidate (kept-set IoS in one shot):
at the eval conf floor (0.001) a photo easily carries thousands of raw boxes
and a pure-python O(n*m) inner loop dominated the whole pipeline. Candidates
are additionally capped at MAX_MERGE_CANDIDATES by score — beyond the top
couple thousand, boxes can never influence any FROC point we report.
"""
from __future__ import annotations

import numpy as np

MAX_MERGE_CANDIDATES = 2000


def merge_detections(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    ios_threshold: float = 0.5,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Greedy non-max merge in photo coords. Returns (boxes, scores), score-desc."""
    if not boxes:
        return [], []
    b = np.asarray(boxes, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s)[:MAX_MERGE_CANDIDATES]
    b, s = b[order], s[order]

    kept = np.empty((0, 4), dtype=np.float64)
    kept_scores: list[float] = []
    kept_areas = np.empty((0,), dtype=np.float64)
    for i in range(len(b)):
        box = b[i]
        if len(kept):
            ix1 = np.maximum(kept[:, 0], box[0])
            iy1 = np.maximum(kept[:, 1], box[1])
            ix2 = np.minimum(kept[:, 2], box[2])
            iy2 = np.minimum(kept[:, 3], box[3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            area = (box[2] - box[0]) * (box[3] - box[1])
            smaller = np.maximum(np.minimum(kept_areas, area), 1e-9)
            ios = inter / smaller
            j = int(np.argmax(ios))
            if ios[j] >= ios_threshold:
                kept[j, 0] = min(kept[j, 0], box[0])
                kept[j, 1] = min(kept[j, 1], box[1])
                kept[j, 2] = max(kept[j, 2], box[2])
                kept[j, 3] = max(kept[j, 3], box[3])
                kept_areas[j] = (kept[j, 2] - kept[j, 0]) * (kept[j, 3] - kept[j, 1])
                continue
        kept = np.vstack([kept, box[None, :]])
        kept_areas = np.append(
            kept_areas, (box[2] - box[0]) * (box[3] - box[1])
        )
        kept_scores.append(float(s[i]))
    return [tuple(map(float, k)) for k in kept], kept_scores
