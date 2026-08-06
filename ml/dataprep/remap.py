"""
Class remap: Label Studio's 6-entry class list -> the single training class.

The LS YOLO export flattens BOTH control tags into classes.txt, so ids 2/4/5
(NiO / Weitere Überprüfung notwendig / iO) are image-level choices that never
appear as boxes. Of the real box classes, the locked project decision is to
train ONLY "Nicht ausreichend Lot"; Brücke and Fahne boxes are dropped to
background. Dropped boxes are still recorded in the manifest (`other_boxes`)
so the eval harness can tag false positives that overlap a known other-defect
region instead of counting them as unexplained false calls.

Re-adding a class later = extend TRAIN_CLASSES; nothing in Label Studio
changes.
"""
from __future__ import annotations

# name -> training class id
TRAIN_CLASSES: dict[str, int] = {
    "Nicht ausreichend Lot": 0,
}
TRAIN_CLASS_NAMES: list[str] = ["Nicht ausreichend Lot"]

# Box classes that exist in the snapshot but are deliberately not trained.
DROPPED_BOX_CLASSES = {"Brücke", "Fahne"}


def split_boxes(
    raw_boxes: list[tuple[str, tuple[float, float, float, float]]],
) -> tuple[
    list[tuple[float, float, float, float]],
    list[tuple[str, tuple[float, float, float, float]]],
]:
    """
    Split raw (class_name, xyxy) boxes into (train_boxes, other_boxes).

    Unknown class names raise — a new label class appearing in a future
    snapshot must be a conscious decision, not silently dropped.
    """
    train: list[tuple[float, float, float, float]] = []
    other: list[tuple[str, tuple[float, float, float, float]]] = []
    for name, box in raw_boxes:
        if name in TRAIN_CLASSES:
            train.append(box)
        elif name in DROPPED_BOX_CLASSES:
            other.append((name, box))
        else:
            raise ValueError(f"unexpected box class in snapshot: {name!r}")
    return train, other
