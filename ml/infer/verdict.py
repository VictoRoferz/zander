"""
The photo decision — ONE function shared by the evaluation harness and the
inspector, so the rule that is measured is exactly the rule that is deployed.

Required output (user, 2026-09-28): N.iO + which defect class(es) were
detected, or i.O if nothing was detected.

Rule:
  photo score   = max detection score over all classes (0 if none)
  verdict       = "NiO" if photo score >= threshold else "iO"   (inclusive)
  classes       = classes with >= 1 detection >= threshold
  defects       = those detections (class, box, score), score-desc

ONE threshold for all classes (the frozen t* from `cli.py choose-op`):
Brücke/Fahne have no honest validation data to tune their own thresholds on
(board B is a single panel in train_always), so per-class thresholds would be
fitted to noise. Per-class curves are reported as diagnostics only.
"""
from __future__ import annotations


def photo_score(scores: list[float]) -> float:
    return max(scores, default=0.0)


def photo_verdict(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    labels: list[int],
    threshold: float,
    class_names: list[str],
) -> dict:
    defects = sorted(
        (
            {"class": class_names[c], "box": [round(v, 1) for v in b], "score": round(s, 4)}
            for b, s, c in zip(boxes, scores, labels)
            if s >= threshold
        ),
        key=lambda d: -d["score"],
    )
    classes = sorted({d["class"] for d in defects}, key=class_names.index)
    return {
        "verdict": "NiO" if defects else "iO",
        "classes": classes,
        "photo_score": round(photo_score(scores), 4),
        "threshold": threshold,
        "defects": defects,
    }
