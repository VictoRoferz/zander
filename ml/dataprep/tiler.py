"""
Cut annotated photos into 640-px tiles and emit COCO ground truth per bucket.

Output layout (gitignored, deterministic, hashed):
  ml_data/tiles/<dataset_id>/
    images/<capture_id>_x<xxxx>_y<yyyy>.jpg   one file per tile, written once
    coco/test.json                            photo bucket "test"
    coco/fold<i>_train.json                   dev folds != i, oversampled
    coco/fold<i>_val.json                     dev fold  == i, never oversampled
    coco/dev_all_train.json                   all dev folds, oversampled (final retrain)
    provenance.json                           params + counts + dataset_fingerprint

Rules pinned by the plan:
  - geometry/clipping from infer.tiling (the ONE slicer, TILER v1)
  - tiles re-encoded as JPEG quality=95 subsampling=0 (4:4:4) — never
    default-JPEG; 28-px defects and solder color must survive re-encoding
  - all tiles of annotated photos are kept (empty tiles = hard negatives);
    unannotated or quarantined photos produce no tiles
  - oversampling = duplicating positive-tile entries K× in TRAIN ground truth
    only (framework-agnostic; identical effect for Ultralytics and DEIM)
  - COCO category_id is 1 (COCO/pycocotools convention); the YOLO view maps
    it to class 0
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from PIL import Image

from config.settings import settings
from dataprep.ingest import load_manifest
from dataprep.split import load_split
from infer.tiling import (
    MIN_SIDE,
    MIN_VISIBILITY,
    STRIDE,
    TILE_SIZE,
    TILER_VERSION,
    clip_box_to_tile,
    make_tile_name,
    tile_origins,
    xyxy_to_coco,
)

logger = logging.getLogger("ml.tiler")

DATASET_ID = "ds_v1"
ENCODING = "jpeg95"  # or "png" — record in provenance either way
OVERSAMPLE_TARGET = 0.30  # aim: ~30% of train tiles contain a defect
CATEGORY = {"id": 1, "name": "Nicht ausreichend Lot"}


def _usable(entry: dict) -> bool:
    return entry["quarantine_reason"] is None and entry["label_state"] != "unannotated"


def _save_tile(im: Image.Image, x: int, y: int, out: Path) -> None:
    tile = im.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
    if ENCODING == "png":
        tile.save(out.with_suffix(".png"))
    else:
        tile.save(out.with_suffix(".jpg"), quality=95, subsampling=0)


def _tile_records(manifest: dict) -> list[dict]:
    """One record per tile of every usable photo: name, origin, GT boxes (tile coords)."""
    records = []
    for e in manifest["entries"]:
        if not _usable(e):
            continue
        for ox, oy in tile_origins(e["width"], e["height"]):
            boxes = []
            for b in e["nal_boxes"]:
                clipped = clip_box_to_tile(tuple(b), (ox, oy))
                if clipped is not None:
                    boxes.append(clipped)
            records.append(
                {
                    "capture_id": e["capture_id"],
                    "name": make_tile_name(e["capture_id"], ox, oy),
                    "origin": [ox, oy],
                    "boxes": boxes,
                }
            )
    return records


def _compute_oversample_k(records: list[dict], dev_captures: set[str]) -> int:
    pos = sum(1 for r in records if r["capture_id"] in dev_captures and r["boxes"])
    neg = sum(1 for r in records if r["capture_id"] in dev_captures and not r["boxes"])
    if pos == 0:
        raise RuntimeError("no positive tiles in the dev pool — check the split")
    k = round(OVERSAMPLE_TARGET * neg / ((1 - OVERSAMPLE_TARGET) * pos))
    return max(1, min(k, 20))


def _coco(records: list[dict], oversample_k: int, ext: str) -> dict:
    """Build one COCO doc; positive tiles duplicated oversample_k× total."""
    images, annotations = [], []
    img_id = ann_id = 0
    for r in sorted(records, key=lambda r: r["name"]):
        copies = oversample_k if (r["boxes"] and oversample_k > 1) else 1
        for _ in range(copies):
            img_id += 1
            images.append(
                {
                    "id": img_id,
                    "file_name": f"{r['name']}.{ext}",
                    "width": TILE_SIZE,
                    "height": TILE_SIZE,
                    "capture_id": r["capture_id"],
                    "origin": r["origin"],
                }
            )
            for b in r["boxes"]:
                ann_id += 1
                coco_box = xyxy_to_coco(b)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": CATEGORY["id"],
                        "bbox": [round(v, 2) for v in coco_box],
                        "area": round(coco_box[2] * coco_box[3], 2),
                        "iscrowd": 0,
                    }
                )
    return {"images": images, "annotations": annotations, "categories": [CATEGORY]}


def _fingerprint(records: list[dict], split_id: str, snapshot_id: str) -> str:
    h = hashlib.sha256()
    h.update(f"{TILER_VERSION}|{TILE_SIZE}|{STRIDE}|{MIN_VISIBILITY}|{MIN_SIDE}|".encode())
    h.update(f"{ENCODING}|{snapshot_id}|{split_id}|".encode())
    for r in sorted(records, key=lambda r: r["name"]):
        line = r["name"] + "|" + ";".join(
            ",".join(f"{v:.2f}" for v in b) for b in r["boxes"]
        )
        h.update(line.encode())
    return h.hexdigest()


def run(write_images: bool = True) -> Path:
    manifest = load_manifest()
    split = load_split()
    ext = "png" if ENCODING == "png" else "jpg"

    records = _tile_records(manifest)
    bucket_of = {cid: a["bucket"] for cid, a in split["assignment"].items()}
    fold_ids = sorted(split["folds"])
    dev_captures = {
        cid for cid, b in bucket_of.items() if b.startswith("fold")
    }
    oversample_k = _compute_oversample_k(records, dev_captures)

    ds_dir = settings.tiles_dir / DATASET_ID
    (ds_dir / "images").mkdir(parents=True, exist_ok=True)
    (ds_dir / "coco").mkdir(parents=True, exist_ok=True)

    # ---- write tile images (once, bucket-independent) ----
    if write_images:
        src = Path(manifest["source_images_dir"])
        by_capture: dict[str, list[dict]] = {}
        for r in records:
            by_capture.setdefault(r["capture_id"], []).append(r)
        for n, (cid, recs) in enumerate(sorted(by_capture.items()), 1):
            with Image.open(src / f"{cid}.jpg") as im:
                im = im.convert("RGB")
                for r in recs:
                    out = ds_dir / "images" / r["name"]
                    if not (out.with_suffix(f".{ext}")).exists():
                        _save_tile(im, r["origin"][0], r["origin"][1], out)
            if n % 50 == 0:
                logger.info(f"tiled {n}/{len(by_capture)} photos")

    # ---- COCO ground truth per bucket ----
    def bucket_records(buckets: set[str]) -> list[dict]:
        return [r for r in records if bucket_of[r["capture_id"]] in buckets]

    cocos = {"test": _coco(bucket_records({"test"}), 1, ext)}
    for i in fold_ids:
        train_buckets = {f"fold{j}" for j in fold_ids if j != i}
        cocos[f"fold{i}_train"] = _coco(bucket_records(train_buckets), oversample_k, ext)
        cocos[f"fold{i}_val"] = _coco(bucket_records({f"fold{i}"}), 1, ext)
    cocos["dev_all_train"] = _coco(
        bucket_records({f"fold{i}" for i in fold_ids}), oversample_k, ext
    )
    for name, doc in cocos.items():
        (ds_dir / "coco" / f"{name}.json").write_text(
            json.dumps(doc), encoding="utf-8"
        )
        logger.info(
            f"coco/{name}.json: {len(doc['images'])} tile entries, "
            f"{len(doc['annotations'])} boxes"
        )

    fingerprint = _fingerprint(records, split["split_id"], manifest["snapshot_id"])
    n_pos_tiles = sum(1 for r in records if r["boxes"])
    provenance = {
        "dataset_id": DATASET_ID,
        "tiler_version": TILER_VERSION,
        "tile_size": TILE_SIZE,
        "stride": STRIDE,
        "min_visibility": MIN_VISIBILITY,
        "min_side": MIN_SIDE,
        "encoding": ENCODING,
        "snapshot_id": manifest["snapshot_id"],
        "split_id": split["split_id"],
        "oversample_k": oversample_k,
        "oversample_target": OVERSAMPLE_TARGET,
        "n_photos_tiled": len({r["capture_id"] for r in records}),
        "n_tiles": len(records),
        "n_positive_tiles": n_pos_tiles,
        "n_tile_boxes": sum(len(r["boxes"]) for r in records),
        "dataset_fingerprint": fingerprint,
    }
    (ds_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8"
    )
    logger.info(
        f"dataset {DATASET_ID}: {provenance['n_tiles']} tiles "
        f"({n_pos_tiles} positive), oversample_k={oversample_k}, "
        f"fingerprint={fingerprint[:16]}…"
    )
    return ds_dir
