"""
Derive the Ultralytics YOLO view from the canonical COCO tiles.

One geometry code path: this module only re-expresses the COCO ground truth
(it never re-clips or re-computes boxes). Layout per subset:

  ml_data/yolo/<dataset_id>/
    images/<subset>/<tile>.jpg          symlink into tiles/<dataset_id>/images
    labels/<subset>/<tile>.txt          class 0 + YOLO-normalized boxes
    fold<i>.yaml / dev_all.yaml         Ultralytics data files

Oversampled duplicate entries become `<tile>_dup<N>.jpg` symlinks pointing at
the same physical tile plus a copied label file — physically distinct paths,
because Ultralytics caches labels per image path (duplicate lines in a txt
list would be collapsed).

Symlinks keep the view cheap; `export_zip` dereferences them so the zip works
anywhere (Colab/Drive).
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from config.settings import settings
from dataprep.tiler import DATASET_ID
from infer.tiling import TILE_SIZE

logger = logging.getLogger("ml.yolo")


def _link(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(target.resolve())
    except OSError:  # e.g. Windows without developer mode — fall back to copy
        import shutil

        shutil.copyfile(target, link)


def _write_subset(ds_dir: Path, out_dir: Path, subset: str) -> None:
    coco = json.loads((ds_dir / "coco" / f"{subset}.json").read_text(encoding="utf-8"))
    img_dir = out_dir / "images" / subset
    lbl_dir = out_dir / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    dup_count: dict[str, int] = defaultdict(int)
    for im in coco["images"]:
        src = ds_dir / "images" / im["file_name"]
        stem, ext = os.path.splitext(im["file_name"])
        n = dup_count[stem]
        dup_count[stem] += 1
        name = stem if n == 0 else f"{stem}_dup{n}"
        _link(src, img_dir / f"{name}{ext}")
        lines = []
        for a in anns_by_image.get(im["id"], []):
            x, y, w, h = a["bbox"]
            cx, cy = (x + w / 2) / TILE_SIZE, (y + h / 2) / TILE_SIZE
            lines.append(
                f"0 {cx:.6f} {cy:.6f} {w / TILE_SIZE:.6f} {h / TILE_SIZE:.6f}"
            )
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"yolo subset {subset}: {len(coco['images'])} entries")


def run() -> Path:
    ds_dir = settings.tiles_dir / DATASET_ID
    provenance = json.loads((ds_dir / "provenance.json").read_text(encoding="utf-8"))
    out_dir = settings.yolo_dir / DATASET_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = sorted(p.stem for p in (ds_dir / "coco").glob("*.json"))
    for subset in subsets:
        _write_subset(ds_dir, out_dir, subset)

    folds = sorted(
        {s.split("_")[0] for s in subsets if s.startswith("fold")}
    )
    for fold in folds:
        (out_dir / f"{fold}.yaml").write_text(
            "\n".join(
                [
                    f"# dataset {DATASET_ID} fingerprint {provenance['dataset_fingerprint']}",
                    f"path: {out_dir.resolve()}",
                    f"train: images/{fold}_train",
                    f"val: images/{fold}_val",
                    "names:",
                    "  0: Nicht ausreichend Lot",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    # Final-retrain view: train on all dev tiles, val on fold0_val as a
    # checkpoint monitor only (the honest number comes from test-v1).
    (out_dir / "dev_all.yaml").write_text(
        "\n".join(
            [
                f"# dataset {DATASET_ID} fingerprint {provenance['dataset_fingerprint']}",
                f"path: {out_dir.resolve()}",
                "train: images/dev_all_train",
                "val: images/fold0_val",
                "names:",
                "  0: Nicht ausreichend Lot",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"yolo view written: {out_dir}")
    return out_dir


def export_zip() -> Path:
    """
    Zip of the TILES dataset (images + COCO GT + provenance) for Drive/Colab.

    Deliberately NOT the YOLO view: dereferencing its oversampling symlinks
    would duplicate every positive tile K× and every tile once per subset
    (measured 4.3 GB vs ~1 GB). Colab rebuilds the symlinked YOLO view in
    seconds with `python cli.py tile --no-images` after unzipping.
    """
    ds_dir = settings.tiles_dir / DATASET_ID
    zip_path = settings.ml_data_root / f"{DATASET_ID}_tiles.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(ds_dir.rglob("*")):
            if path.is_file():
                zf.write(path, f"{DATASET_ID}/{path.relative_to(ds_dir)}")
    size_gb = zip_path.stat().st_size / 1e9
    logger.info(f"dataset zip: {zip_path} ({size_gb:.2f} GB)")
    return zip_path
