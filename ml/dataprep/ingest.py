"""
Ingest: raw capture dumps + label snapshot + photo attributes -> the committed
manifest (ml/datasets/manifest.json), the single source of truth for every
downstream step. One entry per photo:

  capture_id, captured_at, session, image_path (repo-relative, POSIX),
  width/height, source, brightness, ls_task_id, sha256_ok,
  board (A|A2|B), pose (seated|handheld), lighting, content,   # attributes table
  label_state  in {annotated, unannotated, review},
  verdict      "NiO" (>= 1 box) | "iO" (annotated, no box) | None,
  classes_present  sorted class names with >= 1 box,
  boxes        [[class_name, [x1,y1,x2,y2]], ...]  pixel floats, photo coords,
  quarantine_reason (None = usable), review_note

Class schemas (dataprep/remap.py) are applied later, per training view — the
manifest itself is schema-free.

Label semantics (critical): a photo is a NEGATIVE only if an annotator
submitted it (a label file exists, possibly empty). Photos without a label
file are UNANNOTATED — excluded from training/eval, never implicit negatives.
Photos on the review list (contradictory labels) are excluded as well.

Gates: hard-verified counts of the RAW snapshot (pre-quarantine, pre-review).
A new snapshot that changes them must be a conscious re-baseline (--no-gates
once, then update EXPECTED).
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from PIL import Image

from config.settings import REPO_ROOT, settings
from dataprep.attributes import load_attributes
from dataprep.quarantine import BOX_NOTES, REVIEW, quarantine_reason
from dataprep.remap import check_class
from infer.tiling import yolo_to_xyxy

logger = logging.getLogger("ml.ingest")

# Verified against snapshot v2 (Aug-17 YOLO export) on 2026-09-28.
EXPECTED = {
    "total": 566,
    "label_files": 331,
    "with_boxes": 136,
    "empty": 195,
    "unannotated": 235,
    "boxes_Nicht ausreichend Lot": 263,
    "boxes_Brücke": 522,
    "boxes_Fahne": 191,
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


def _collect_photos(dirs: list[Path]) -> dict[str, Path]:
    """capture_id -> jpg path across all source dirs; duplicates are an error."""
    photos: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"source images dir not found: {d}")
        for jpg in sorted(d.glob("*.jpg")):
            if jpg.stem in photos:
                raise ValueError(
                    f"capture {jpg.stem} exists in two source dirs: "
                    f"{photos[jpg.stem].parent} and {d}"
                )
            photos[jpg.stem] = jpg
    return photos


def build_manifest(verify_sha: bool = True) -> dict:
    photos = _collect_photos(settings.source_images_dirs)
    attributes = load_attributes()
    snapshot_labels, snapshot_classes = load_yolo_snapshot(settings.snapshot_dir)
    logger.info(
        f"snapshot {settings.snapshot_id}: {len(snapshot_labels)} label files, "
        f"classes={snapshot_classes}; {len(photos)} photos in "
        f"{len(settings.source_images_dirs)} dirs"
    )
    orphan_labels = set(snapshot_labels) - set(photos)
    if orphan_labels:
        raise ValueError(f"label files without a photo: {sorted(orphan_labels)[:5]}…")
    missing_attr = set(photos) - set(attributes)
    if missing_attr:
        raise ValueError(
            f"{len(missing_attr)} photos have no row in {settings.attributes_csv.name} "
            f"(e.g. {sorted(missing_attr)[:3]}) — audit the new captures first"
        )

    entries = []
    for capture_id, jpg in sorted(photos.items()):
        sidecar = json.loads(jpg.with_suffix(".json").read_text(encoding="utf-8"))
        task_file = jpg.with_suffix(".task")
        ls_task_id = int(task_file.read_text().strip()) if task_file.exists() else None
        attr = attributes[capture_id]

        sha_ok = True
        if verify_sha:
            sha_ok = _sha256_file(jpg) == sidecar["sha256"]
            if not sha_ok:
                logger.warning(f"[{capture_id}] sha256 MISMATCH — corrupt copy?")

        brightness = _mean_brightness(jpg)
        width, height = int(sidecar["width"]), int(sidecar["height"])

        raw = snapshot_labels.get(capture_id)
        if raw is None:
            label_state, boxes = "unannotated", []
        else:
            boxes = [
                [
                    check_class(snapshot_classes[cid]),
                    [round(v, 2) for v in yolo_to_xyxy(ybox, width, height)],
                ]
                for cid, ybox in raw
            ]
            label_state = "review" if capture_id in REVIEW else "annotated"
        classes_present = sorted({name for name, _ in boxes})
        verdict = (
            ("NiO" if boxes else "iO") if label_state == "annotated" else None
        )
        source = sidecar.get("source", "unknown")
        entries.append(
            {
                "capture_id": capture_id,
                "captured_at": sidecar["captured_at"],
                "session": sidecar["captured_at"][:10],
                "image_path": jpg.relative_to(REPO_ROOT).as_posix(),
                "width": width,
                "height": height,
                "source": source,
                "brightness": round(brightness, 1),
                "ls_task_id": ls_task_id,
                "sha256_ok": sha_ok,
                "board": attr["board"],
                "pose": attr["pose"],
                "lighting": attr["lighting"],
                "content": attr["content"],
                "label_state": label_state,
                "verdict": verdict,
                "classes_present": classes_present,
                "boxes": boxes,
                "quarantine_reason": quarantine_reason(
                    source, brightness, attr["content"]
                ),
                "review_note": REVIEW.get(capture_id) or BOX_NOTES.get(capture_id),
            }
        )

    return {
        "manifest_version": 2,
        "snapshot_id": settings.snapshot_id,
        "source_images_dirs": [
            p.relative_to(REPO_ROOT).as_posix() for p in settings.source_images_dirs
        ],
        "box_classes": sorted({n for e in entries for n, _ in e["boxes"]}),
        "entries": entries,
    }


def check_gates(manifest: dict) -> dict:
    """Raw-snapshot gates (pre-quarantine, pre-review), matching the export."""
    e = manifest["entries"]
    labeled = [x for x in e if x["label_state"] != "unannotated"]
    got = {
        "total": len(e),
        "label_files": len(labeled),
        "with_boxes": sum(1 for x in labeled if x["boxes"]),
        "empty": sum(1 for x in labeled if not x["boxes"]),
        "unannotated": len(e) - len(labeled),
    }
    per_class = Counter(n for x in e for n, _ in x["boxes"])
    for name in ("Nicht ausreichend Lot", "Brücke", "Fahne"):
        got[f"boxes_{name}"] = per_class.get(name, 0)
    return got


def usable(entry: dict) -> bool:
    """Train/eval eligible: not quarantined, annotated, not on the review list."""
    return entry["quarantine_reason"] is None and entry["label_state"] == "annotated"


def photo_path(entry: dict) -> Path:
    return REPO_ROOT / entry["image_path"]


def run(verify_sha: bool = True, enforce_gates: bool = True) -> Path:
    manifest = build_manifest(verify_sha=verify_sha)
    got = check_gates(manifest)
    ok = got == EXPECTED
    for key in EXPECTED:
        marker = "ok" if got[key] == EXPECTED.get(key) else f"EXPECTED {EXPECTED.get(key)}"
        logger.info(f"gate {key}: {got[key]} ({marker})")

    entries = manifest["entries"]
    quarantined = [x for x in entries if x["quarantine_reason"]]
    q_labeled = [x for x in quarantined if x["label_state"] != "unannotated"]
    logger.info(
        f"quarantined: {len(quarantined)} (of which labeled: {len(q_labeled)} "
        f"{[x['capture_id'][:8] for x in q_labeled]})"
    )
    for x in entries:
        if x["label_state"] == "review":
            logger.info(f"review (excluded): {x['capture_id'][:8]} — {x['review_note']}")
    n_usable = sum(1 for x in entries if usable(x))
    verdicts = Counter(x["verdict"] for x in entries if usable(x))
    logger.info(f"usable for train/eval: {n_usable} photos {dict(verdicts)}")

    bad_sha = [x["capture_id"] for x in entries if not x["sha256_ok"]]
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
    logger.info(f"manifest written: {out} ({len(entries)} entries)")
    return out


def load_manifest() -> dict:
    path = settings.datasets_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run `python cli.py ingest` first — missing {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 2:
        raise RuntimeError("manifest.json is v1 — rerun `python cli.py ingest`")
    return manifest
