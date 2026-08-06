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
from dataprep.tiler import DATASET_ID  # noqa: E402

logger = logging.getLogger("ml.deim")

DEIM_REPO = "https://github.com/ShihuaHuang95/DEIM"
# Pin before first real run: `git ls-remote https://github.com/ShihuaHuang95/DEIM HEAD`
DEIM_COMMIT = "PIN_ME_BEFORE_FIRST_RUN"
BASE_CONFIG = "configs/deim_dfine/deim_hgnetv2_s_coco.yml"


def prepare_dataset_layout(fold: str, work: Path) -> Path:
    """COCO layout the DEIM configs expect: images dir + annotation jsons."""
    tiles = settings.tiles_dir / DATASET_ID
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


def write_override(fold: str, work: Path, ds: Path, epochs: int) -> Path:
    """Minimal YAML override on top of DEIM's D-FINE-S COCO config."""
    override = work / f"zander_{fold}.yml"
    override.write_text(
        f"""__include__: [../{BASE_CONFIG}]
num_classes: 1
remap_mscoco_category: False
train_dataloader:
  dataset:
    img_folder: {ds / 'images'}
    ann_file: {ds / 'train.json'}
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


def run(fold: str, epochs: int, dry_run: bool) -> None:
    work = settings.run_dirs_dir / f"deim_s_{fold}"
    work.mkdir(parents=True, exist_ok=True)
    repo = work / "DEIM"
    ds = prepare_dataset_layout(fold, work)
    override = write_override(fold, work, ds, epochs)

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
    if DEIM_COMMIT == "PIN_ME_BEFORE_FIRST_RUN":
        raise RuntimeError("pin DEIM_COMMIT before running for real")
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
                "dataset_id": DATASET_ID,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="fold0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    run(args.fold, args.epochs, args.dry_run)
