"""
DEIM / D-FINE-S onboarding — at arm's length (plan M3).

The DEIM repo (Apache-2.0, trains D-FINE variants with the improved CVPR'25
recipe) is used as an EXTERNAL tool: we generate its dataset layout + a config
override, then subprocess its own train/export scripts inside Colab. Its
dependencies never enter ml/requirements.txt, and our harness only ever
touches the resulting ONNX + bundle.

    python train/deim_run.py --fold fold0 --dry-run     # print the plan
    python train/deim_run.py --fold fold0               # run (Colab, GPU)

Pinned upstream: DEIM_REPO @ DEIM_COMMIT. If this integration stalls, the
Ultralytics matrix (M2) already yields a shippable winner — this is additive.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from dataprep.remap import schema_classes  # noqa: E402
from dataprep.tiler import dataset_id  # noqa: E402

logger = logging.getLogger("ml.deim")

DEIM_REPO = "https://github.com/ShihuaHuang95/DEIM"
# Pinned 2026-09-28 (HEAD of DEIM v1). LICENSE at this commit: Apache-2.0
# (checked). NOT DEIMv2 — that repo switched to a non-commercial license on
# 2026-08-24.
DEIM_COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"

# DEIM's default recipe is tuned for COCO: Mosaic at output_size 320 (halves
# object size), RandomZoomOut + RandomIoUCrop (rescale), mixup 0.5. For
# 28-px defects on native-resolution tiles those shrink/blend the signal, so
# the override mirrors the Ultralytics policy (train/ultralytics_run.py):
# photometric jitter + flips only; orientation (D4) and colour transfer are
# already baked into the tiles by the dataset builder.
TRAIN_OPS = """      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640], }
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        epoch: [4, 64, 120]
        ops: ['RandomPhotometricDistort']"""
BASE_CONFIG = "configs/deim_dfine/deim_hgnetv2_s_coco.yml"


def prepare_dataset_layout(fold: str, work: Path, schema: str) -> Path:
    """COCO layout the DEIM configs expect: images dir + annotation jsons."""
    tiles = settings.tiles_dir / dataset_id(schema)
    ds = work / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    if not (ds / "images").exists():
        (ds / "images").symlink_to((tiles / "images").resolve())
    for subset, out_name in (
        (f"{fold}_train", "train.json"),
        (f"{fold}_val", "val.json"),
    ):
        shutil.copyfile(tiles / "coco" / f"{subset}.json", ds / out_name)
    return ds


def write_override(fold: str, work: Path, ds: Path, epochs: int, schema: str) -> Path:
    """Minimal YAML override on top of DEIM's D-FINE-S COCO config."""
    override = work / f"zander_{fold}.yml"
    override.write_text(
        f"""__include__: [../{BASE_CONFIG}]
num_classes: {len(schema_classes(schema))}
remap_mscoco_category: False
train_dataloader:
  dataset:
    img_folder: {ds / 'images'}
    ann_file: {ds / 'train.json'}
    transforms:
{TRAIN_OPS}
  collate_fn:
    mixup_prob: 0.0
val_dataloader:
  dataset:
    img_folder: {ds / 'images'}
    ann_file: {ds / 'val.json'}
epoches: {epochs}
output_dir: {work / 'output'}
""",
        encoding="utf-8",
    )
    return override


def run(fold: str, epochs: int, dry_run: bool, schema: str = "def3") -> None:
    work = settings.run_dirs_dir / f"deim_s_{schema}_{fold}"
    work.mkdir(parents=True, exist_ok=True)
    repo = work / "DEIM"
    ds = prepare_dataset_layout(fold, work, schema)
    override = write_override(fold, work, ds, epochs, schema)

    steps = [
        ["git", "clone", DEIM_REPO, str(repo)],
        ["git", "-C", str(repo), "checkout", DEIM_COMMIT],
        [sys.executable, "-m", "pip", "install", "-r", str(repo / "requirements.txt")],
        [
            sys.executable, str(repo / "train.py"),
            "-c", str(override), "--use-amp", "--seed", str(settings.seed),
        ],
        [
            sys.executable, str(repo / "tools" / "deployment" / "export_onnx.py"),
            "-c", str(override), "-r", str(work / "output" / "best_stg2.pth"),
        ],
    ]
    if dry_run:
        print(f"# DEIM plan for {fold} (work dir {work}):")
        for s in steps:
            print("  $", " ".join(s))
        print(
            "# then bundle output/*.onnx + config + metrics and import with "
            "`python cli.py import-run`; evaluate with adapter 'deim'."
        )
        return
    for s in steps:
        logger.info("$ " + " ".join(s))
        subprocess.run(s, check=True)
    (work / "meta.json").write_text(
        json.dumps(
            {
                "framework": "deim",
                "adapter": "deim",
                "fold": fold,
                "deim_repo": DEIM_REPO,
                "deim_commit": DEIM_COMMIT,
                "dataset_id": dataset_id(schema),
                "schema": schema,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="fold0")
    parser.add_argument("--schema", default="def3", choices=["def3", "nal1"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    run(args.fold, args.epochs, args.dry_run, schema=args.schema)
