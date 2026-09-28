"""
Class schemas: which Label Studio box classes a training view uses.

The LS YOLO export flattens BOTH control tags into classes.txt, so ids 2/4/5
(NiO / Weitere Überprüfung notwendig / iO) are image-level choices that never
appear as boxes. The manifest stores every real box with its class NAME; a
schema decides at view-build time which classes become training classes.

  def3  — the product requirement (user, 2026-09-28): a photo is N.iO with the
          detected class(es), i.O otherwise -> three classes.
  nal1  — "Nicht ausreichend Lot" only: the Phase-1 benchmark view and the
          NAL-only comparison for the def3 model. Brücke/Fahne become
          background there and are kept as `other` boxes so the evaluator can
          tag false positives that sit on a real out-of-scope defect.

Class ids are the position in the schema list; COCO category_id = id + 1.
Unknown class names raise — a new label class appearing in a snapshot must be
a conscious decision, not silently dropped.
"""
from __future__ import annotations

BOX_CLASSES = ("Nicht ausreichend Lot", "Brücke", "Fahne")

SCHEMAS: dict[str, list[str]] = {
    "def3": ["Nicht ausreichend Lot", "Brücke", "Fahne"],
    "nal1": ["Nicht ausreichend Lot"],
}

# Short names for tables / galleries.
SHORT = {"Nicht ausreichend Lot": "NAL", "Brücke": "Brücke", "Fahne": "Fahne"}


def schema_classes(schema: str) -> list[str]:
    if schema not in SCHEMAS:
        raise KeyError(f"unknown schema {schema!r} — one of {sorted(SCHEMAS)}")
    return SCHEMAS[schema]


def check_class(name: str) -> str:
    if name not in BOX_CLASSES:
        raise ValueError(f"unexpected box class in snapshot: {name!r}")
    return name


def project(
    boxes: list[list], schema: str
) -> tuple[list[tuple[int, tuple[float, float, float, float]]], list[tuple[str, tuple]]]:
    """
    Split manifest boxes [[class_name, [x1,y1,x2,y2]], ...] into
    (train [(class_id, xyxy)], other [(class_name, xyxy)]) for one schema.
    """
    classes = schema_classes(schema)
    train, other = [], []
    for name, box in boxes:
        check_class(name)
        if name in classes:
            train.append((classes.index(name), tuple(box)))
        else:
            other.append((name, tuple(box)))
    return train, other
