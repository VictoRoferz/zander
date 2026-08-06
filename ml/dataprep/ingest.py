"""
Ingest: raw capture dump + label snapshot -> the committed manifest.

The manifest (ml/datasets/manifest.json) is the single source of truth for
every downstream step. One entry per photo:

  capture_id, captured_at, session, width/height, source, brightness,
  ls_task_id, sha256_ok,
  label_state  in {annotated_positive, annotated_negative, unannotated},
  quarantine_reason (None = usable), review_note,
  nal_boxes    [[x1,y1,x2,y2], ...]  pixel floats, photo coords,
  other_boxes  [[class_name, [x1,y1,x2,y2]], ...]  dropped Brücke/Fahne

Label semantics (critical): an image is a NEGATIVE only if an annotator
submitted it (a label file exists, possibly empty). The 183 images without a
label file are UNANNOTATED — they are excluded from training/eval, never fed
as implicit negatives.

Gates: hard-verified counts from the 2026-08-06 dataset analysis. A new label
snapshot that changes them must be a conscious re-baseline (--no-gates once,
then update EXPECTED).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from PIL import Image

from config.settings import settings
from dataprep.quarantine import REVIEW_NOTES, quarantine_reason
from dataprep.remap import split_boxes
from infer.tiling import yolo_to_xyxy

logger = logging.getLogger("ml.ingest")

# Verified against snapshot v1 (July-27 YOLO export) on 2026-08-06.
EXPECTED = {
    "total": 481,
    "annotated": 298,
    "annotated_positive": 99,
    "nal_boxes": 263,
    "annotated_negative": 199,  # 194 empty label files + 5 Brücke/Fahne-only
    "unannotated": 183,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean_brightness(path: Path) -> float:
    """Fast mean-gray via JPEG draft decode (matches the analysis method)."""
    with Image.open(path) as im:
        im.draft("L", (164, 123))
        gray = im.convert("L").resize((32, 24))
        pixels = list(gray.getdata())
    return sum(pixels) / len(pixels)


def load_yolo_snapshot(
    snapshot_dir: Path,
) -> tuple[dict[str, list[tuple[int, tuple[float, float, float, float]]]], list[str]]:
    """
    Read a Label Studio YOLO export: classes.txt + labels/<capture_id>.txt.

    Returns ({capture_id: [(class_id, (cx,cy,w,h) normalized), ...]}, classes).
    An empty label file yields an empty list — i.e. an annotated negative.
    """
    classes = (snapshot_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    labels: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for txt in sorted((snapshot_dir / "labels").glob("*.txt")):
        boxes = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"malformed YOLO line in {txt.name}: {line!r}")
            boxes.append(
                (int(parts[0]), tuple(float(v) for v in parts[1:]))
            )
        labels[txt.stem] = boxes
    return labels, classes


def build_manifest(verify_sha: bool = True) -> dict:
    src = settings.source_images_dir
    if not src.is_dir():
        raise FileNotFoundError(f"source images dir not found: {src}")
    snapshot_labels, snapshot_classes = load_yolo_snapshot(settings.snapshot_dir)
    logger.info(
        f"snapshot {settings.snapshot_id}: {len(snapshot_labels)} label files, "
        f"classes={snapshot_classes}"
    )

    entries = []
    for jpg in sorted(src.glob("*.jpg")):
        capture_id = jpg.stem
        sidecar = json.loads((src / f"{capture_id}.json").read_text(encoding="utf-8"))
        task_file = src / f"{capture_id}.task"
        ls_task_id = int(task_file.read_text().strip()) if task_file.exists() else None

        sha_ok = True
        if verify_sha:
            sha_ok = _sha256_file(jpg) == sidecar["sha256"]
            if not sha_ok:
                logger.warning(f"[{capture_id}] sha256 MISMATCH — corrupt copy?")

        brightness = _mean_brightness(jpg)
        width, height = int(sidecar["width"]), int(sidecar["height"])

        raw = snapshot_labels.get(capture_id)
        if raw is None:
            label_state = "unannotated"
            nal_boxes: list = []
            other_boxes: list = []
        else:
            named = [
                (snapshot_classes[cid], yolo_to_xyxy(ybox, width, height))
                for cid, ybox in raw
            ]
            nal, other = split_boxes(named)
            nal_boxes = [[round(v, 2) for v in b] for b in nal]
            other_boxes = [[name, [round(v, 2) for v in b]] for name, b in other]
            label_state = "annotated_positive" if nal_boxes else "annotated_negative"

        entries.append(
            {
                "capture_id": capture_id,
                "captured_at": sidecar["captured_at"],
                "session": sidecar["captured_at"][:10],
                "width": width,
                "height": height,
                "source": sidecar.get("source", "unknown"),
                "brightness": round(brightness, 1),
                "ls_task_id": ls_task_id,
                "sha256_ok": sha_ok,
                "label_state": label_state,
                "quarantine_reason": quarantine_reason(
                    sidecar.get("source", "unknown"), brightness
                ),
                "review_note": REVIEW_NOTES.get(capture_id),
                "nal_boxes": nal_boxes,
                "other_boxes": other_boxes,
            }
        )

    manifest = {
        "snapshot_id": settings.snapshot_id,
        "source_images_dir": str(src),
        "train_classes": ["Nicht ausreichend Lot"],
        "entries": entries,
    }
    return manifest


def check_gates(manifest: dict) -> dict:
    """Label-stat gates (computed pre-quarantine, matching the raw analysis)."""
    e = manifest["entries"]
    got = {
        "total": len(e),
        "annotated": sum(1 for x in e if x["label_state"] != "unannotated"),
        "annotated_positive": sum(1 for x in e if x["label_state"] == "annotated_positive"),
        "nal_boxes": sum(len(x["nal_boxes"]) for x in e),
        "annotated_negative": sum(1 for x in e if x["label_state"] == "annotated_negative"),
        "unannotated": sum(1 for x in e if x["label_state"] == "unannotated"),
    }
    return got


def run(verify_sha: bool = True, enforce_gates: bool = True) -> Path:
    manifest = build_manifest(verify_sha=verify_sha)
    got = check_gates(manifest)
    ok = got == EXPECTED
    for key in EXPECTED:
        marker = "ok" if got[key] == EXPECTED[key] else f"EXPECTED {EXPECTED[key]}"
        logger.info(f"gate {key}: {got[key]} ({marker})")

    quarantined = [x for x in manifest["entries"] if x["quarantine_reason"]]
    q_annotated = [x for x in quarantined if x["label_state"] != "unannotated"]
    logger.info(
        f"quarantined: {len(quarantined)} "
        f"(of which annotated: {len(q_annotated)} "
        f"{[x['capture_id'][:8] for x in q_annotated]})"
    )
    bad_sha = [x["capture_id"] for x in manifest["entries"] if not x["sha256_ok"]]
    if bad_sha:
        raise RuntimeError(f"sha256 mismatches — fix the copies first: {bad_sha}")
    if enforce_gates and not ok:
        raise RuntimeError(
            "label-stat gates failed (new snapshot? re-baseline EXPECTED "
            "deliberately or rerun with --no-gates once): "
            f"got {got} expected {EXPECTED}"
        )

    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    out = settings.datasets_dir / "manifest.json"
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(f"manifest written: {out} ({len(manifest['entries'])} entries)")
    return out


def load_manifest() -> dict:
    path = settings.datasets_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run `python cli.py ingest` first — missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))
