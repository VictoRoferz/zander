"""
Derive the Ultralytics YOLO view from the canonical COCO tiles.

One geometry code path: this module only re-expresses the COCO ground truth
(it never re-clips or re-computes boxes). Layout per subset:

  ml_data/yolo/<dataset_id>/
    images/<subset>/<tile>.jpg          symlink into tiles/<dataset_id>/images
    labels/<subset>/<tile>.txt          class (= category_id - 1) + YOLO-normalized box
    fold<i>.yaml / dev_all.yaml         Ultralytics data files (names from the schema)

Entries that share a physical file within one subset (only possible when an
ablation disables D4 and colour transfer) become `<tile>_dup<N>.jpg` symlinks
plus a copied label file — Ultralytics caches labels per image path.

Symlinks keep the view cheap; `export_zip` ships the TILES (not this view), and
Colab rebuilds the view in seconds with `python cli.py tile --no-images`.
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from config.settings import settings
from dataprep.tiler import DEFAULT_AUG, AugConfig, dataset_id, load_provenance
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
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():  # views are rebuilt, never merged
            old.unlink()

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
                f"{a['category_id'] - 1} {cx:.6f} {cy:.6f} "
                f"{w / TILE_SIZE:.6f} {h / TILE_SIZE:.6f}"
            )
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"yolo subset {subset}: {len(coco['images'])} entries")


def _data_yaml(out_dir: Path, provenance: dict, train: str, val: str) -> str:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(provenance["classes"]))
    return (
        f"# dataset {provenance['dataset_id']} fingerprint {provenance['dataset_fingerprint']}\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/{train}\n"
        f"val: images/{val}\n"
        f"names:\n{names}\n"
    )


def run(schema: str = "def3", aug: AugConfig = DEFAULT_AUG) -> Path:
    ds_id = dataset_id(schema, aug)
    ds_dir = settings.tiles_dir / ds_id
    provenance = load_provenance(ds_id)
    out_dir = settings.yolo_dir / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = sorted(p.stem for p in (ds_dir / "coco").glob("*.json"))
    for subset in subsets:
        _write_subset(ds_dir, out_dir, subset)

    folds = sorted({s.split("_")[0] for s in subsets if s.startswith("fold")})
    for fold in folds:
        (out_dir / f"{fold}.yaml").write_text(
            _data_yaml(out_dir, provenance, f"{fold}_train", f"{fold}_val"), encoding="utf-8"
        )
    # Final-retrain view: train on everything but test; fold0_val is only a
    # checkpoint monitor (the honest numbers come from the photo-level eval).
    (out_dir / "dev_all.yaml").write_text(
        _data_yaml(out_dir, provenance, "dev_all_train", "fold0_val"), encoding="utf-8"
    )
    logger.info(f"yolo view written: {out_dir}")
    return out_dir


def export_zip(schema: str = "def3", aug: AugConfig = DEFAULT_AUG) -> Path:
    """
    Zip of the TILES dataset (images + COCO GT + provenance) for Drive/Colab.

    Deliberately NOT the YOLO view: dereferencing its symlinks would copy every
    file once per subset. Colab rebuilds the symlinked view with
    `python cli.py tile --schema <schema> --no-images` after unzipping.
    """
    ds_id = dataset_id(schema, aug)
    ds_dir = settings.tiles_dir / ds_id
    zip_path = settings.ml_data_root / f"{ds_id}_tiles.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(ds_dir.rglob("*")):
            if path.is_file():
                zf.write(path, f"{ds_id}/{path.relative_to(ds_dir)}")
    logger.info(f"dataset zip: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
    return zip_path
