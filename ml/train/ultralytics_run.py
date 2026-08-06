"""
Ultralytics training runner: one run config YAML -> a portable run_dir bundle.

    python cli.py train --config configs/baseline.yaml [--device cpu]

Works identically locally (CPU smoke) and on Colab (T4). The durable output
is a BUNDLE — ml_data/run_dirs/<name>/bundle{.zip} containing config.yaml,
metrics.json (per-epoch), best.pt, model.onnx (nms=False), pip-freeze.txt and
meta.json (git sha, dataset fingerprint, versions). MLflow never syncs across
machines; bundles do (Drive), and `python cli.py import-run <bundle>` turns
one into a local MLflow run.

Locked augmentation policy for 28-px defects (plan): flips both axes, small
rotations, scale jitter capped at 0.2 (the 0.5 default would shrink 10-px
boxes to 5), mild HSV (solder color is signal), mosaic with close_mosaic,
no mixup. Tile duplication (oversample K) already lives in the dataset view.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from config.settings import settings
from dataprep.tiler import DATASET_ID

logger = logging.getLogger("ml.train")

DEFAULT_AUGMENT = {
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 10.0,
    "scale": 0.2,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.3,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
}


def _git_meta() -> dict:
    def _run(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent
            ).stdout.strip()
        except OSError:
            return "unknown"

    return {
        "git_sha": _run("rev-parse", "HEAD"),
        "git_dirty": bool(_run("status", "--porcelain")),
    }


def _write_data_yaml(fold: str, out: Path) -> Path:
    """Runtime data.yaml with absolute paths — survives Colab relocation."""
    yolo_root = (settings.yolo_dir / DATASET_ID).resolve()
    if not yolo_root.exists():
        raise FileNotFoundError(
            f"{yolo_root} missing — run `python cli.py tile` first (or set "
            "ML_DATA_ROOT to the unzipped dataset location)"
        )
    train = f"images/{fold}_train" if fold != "dev_all" else "images/dev_all_train"
    val = f"images/{fold}_val" if fold != "dev_all" else "images/fold0_val"
    out.write_text(
        "\n".join(
            [
                f"path: {yolo_root}",
                f"train: {train}",
                f"val: {val}",
                "names:",
                "  0: Nicht ausreichend Lot",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _metrics_from_results_csv(run_dir: Path) -> dict:
    results = run_dir / "results.csv"
    if not results.exists():
        return {"epochs": []}
    epochs = []
    with open(results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(
                {k.strip(): _maybe_float(v) for k, v in row.items() if k}
            )
    return {"epochs": epochs, "final": epochs[-1] if epochs else {}}


def _maybe_float(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def run(config_path: Path, device: str | None = None) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = cfg["name"]
    fold = str(cfg.get("fold", "fold0"))
    augment = {**DEFAULT_AUGMENT, **cfg.get("augment", {})}

    # Offline hygiene: no update checks / telemetry; cache weights in ml_data.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    settings.pretrained_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    try:
        yolo_settings.update({"sync": False, "weights_dir": str(settings.pretrained_dir)})
    except Exception:  # settings schema varies across versions — non-fatal
        pass

    weights = settings.pretrained_dir / cfg["model"]
    model = YOLO(str(weights) if weights.exists() else cfg["model"])

    run_root = settings.run_dirs_dir
    run_root.mkdir(parents=True, exist_ok=True)
    data_yaml = _write_data_yaml(fold, run_root / f"{name}_data.yaml")

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 120)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=cfg.get("batch", 16),
        seed=int(cfg.get("seed", settings.seed)),
        project=str(run_root),
        name=name,
        exist_ok=True,
        deterministic=True,
        **augment,
        **cfg.get("extra", {}),  # e.g. {fraction: 0.05} for smoke runs
    )
    if device is not None:
        train_kwargs["device"] = device
    logger.info(f"training {name}: model={cfg['model']} fold={fold}")
    model.train(**train_kwargs)

    run_dir = run_root / name
    best = run_dir / "weights" / "best.pt"

    # Raw-output ONNX for the shared eval harness (adapters expect nms=False).
    onnx_path = YOLO(str(best)).export(format="onnx", nms=False, imgsz=640)

    # ---- assemble the portable bundle ----
    bundle = run_dir / "bundle"
    bundle.mkdir(exist_ok=True)
    shutil.copyfile(config_path, bundle / "config.yaml")
    shutil.copyfile(best, bundle / "best.pt")
    shutil.copyfile(onnx_path, bundle / "model.onnx")
    (bundle / "metrics.json").write_text(
        json.dumps(_metrics_from_results_csv(run_dir), indent=1), encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    (bundle / "pip-freeze.txt").write_text(freeze, encoding="utf-8")

    provenance_path = settings.tiles_dir / DATASET_ID / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {}
    )
    import torch
    import ultralytics

    meta = {
        "run_name": name,
        "framework": "ultralytics",
        "adapter": "ultralytics",
        "fold": fold,
        "dataset_id": DATASET_ID,
        "dataset_fingerprint": provenance.get("dataset_fingerprint"),
        "snapshot_id": provenance.get("snapshot_id"),
        "split_id": provenance.get("split_id"),
        "oversample_k": provenance.get("oversample_k"),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        **_git_meta(),
    }
    (bundle / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    bundle_zip = run_root / f"{name}_bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(bundle.iterdir()):
            zf.write(f, f.name)
    logger.info(f"bundle ready: {bundle_zip} — import with `python cli.py import-run`")
    return bundle_zip
