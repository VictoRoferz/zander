"""
Stitched full-photo evaluation of one ONNX checkpoint on one split bucket.

    python cli.py eval --model <path>.onnx --adapter ultralytics --bucket fold0

Buckets: fold0|fold1|fold2 (that fold's photos), dev (all folds), test
(test-v1 — the one-shot honest number; use only for the final chosen model).

Writes ml_data/eval/<run_name>/{metrics.json, predictions.json, report.md,
gallery.html} and appends one row to the committed ml/reports/results.csv.
Model-vs-model decisions use ONLY these stitched photo-level numbers — never
per-tile/in-training val metrics.
"""
from __future__ import annotations

import csv
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from dataprep.ingest import load_manifest
from dataprep.split import load_split
from eval.gallery import write_gallery
from eval.scorer import iou, match_photo, score_photos
from infer.pipeline import TiledDetector

logger = logging.getLogger("ml.eval")

RESULTS_CSV_FIELDS = [
    "date", "run_name", "model", "adapter", "bucket", "n_photos", "n_gt",
    "recall_at_0.5fppi", "ci95_lo", "ci95_hi", "threshold", "ap50", "ap50_95",
    "fp_at_op", "fp_other_defect", "mean_ms_per_photo", "tiler_version",
    "split_id", "snapshot_id", "host",
]


def bucket_photos(bucket: str) -> list[dict]:
    manifest = load_manifest()
    split = load_split()
    wanted = (
        {f"fold{i}" for i in split["folds"]} if bucket == "dev" else {bucket}
    )
    photos = []
    for e in manifest["entries"]:
        if e["quarantine_reason"] or e["label_state"] == "unannotated":
            continue
        if split["assignment"][e["capture_id"]]["bucket"] in wanted:
            photos.append(e)
    if not photos:
        raise ValueError(f"no usable photos in bucket {bucket!r}")
    return photos


def run(
    model_path: Path,
    adapter_name: str,
    bucket: str,
    run_name: str | None = None,
    conf_floor: float = 0.001,
) -> dict:
    if bucket == "test":
        logger.warning(
            "TEST bucket: this is the one-shot honest number for a FINAL "
            "model. Do not use it for model selection or threshold tuning."
        )
    manifest = load_manifest()
    entries = bucket_photos(bucket)
    run_name = run_name or (
        f"{model_path.stem}_{bucket}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    out_dir = settings.ml_data_root / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = TiledDetector(model_path, adapter_name, conf_floor=conf_floor)
    source_dir = Path(manifest["source_images_dir"])

    photos, total_ms = [], 0.0
    for n, e in enumerate(entries, 1):
        boxes, scores, ms = detector.detect_file(source_dir / f"{e['capture_id']}.jpg")
        total_ms += ms
        photos.append(
            {
                "capture_id": e["capture_id"],
                "gt": [tuple(b) for b in e["nal_boxes"]],
                "other_gt": [tuple(b) for _, b in e["other_boxes"]],
                "boxes": boxes,
                "scores": scores,
            }
        )
        if n % 10 == 0 or n == len(entries):
            logger.info(f"evaluated {n}/{len(entries)} photos")

    metrics = score_photos(photos)
    metrics["mean_ms_per_photo"] = round(total_ms / len(photos), 1)
    metrics["model"] = str(model_path)
    metrics["adapter"] = adapter_name
    metrics["bucket"] = bucket

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1), encoding="utf-8"
    )
    (out_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "capture_id": p["capture_id"],
                    "boxes": [[round(v, 1) for v in b] for b in p["boxes"]],
                    "scores": [round(s, 4) for s in p["scores"]],
                }
                for p in photos
            ]
        ),
        encoding="utf-8",
    )

    # ---- gallery: FN + FP at the operating threshold ----
    t_op = metrics["threshold_at_op"]
    fn_items = [
        {"capture_id": f["capture_id"], "box": box}
        for f in metrics["fn_photos"]
        for box in f["missed"]
    ]
    fp_items = []
    for p in photos:
        _, matched = match_photo(p["boxes"], p["scores"], p["gt"], metrics["iou_operating"])
        for box, s, m in zip(p["boxes"], p["scores"], matched):
            if not m and t_op is not None and s >= t_op:
                fp_items.append(
                    {
                        "capture_id": p["capture_id"],
                        "box": box,
                        "score": s,
                        "other_defect": any(
                            iou(box, og) >= 0.1 for og in p["other_gt"]
                        ),
                    }
                )
    fp_items.sort(key=lambda x: -x["score"])
    write_gallery(
        out_dir / "gallery.html", source_dir, fn_items, fp_items,
        f"{run_name} — errors at recall@0.5FPPI",
    )

    # ---- report + committed results row ----
    summary = (
        f"# Eval {run_name}\n\n"
        f"- model `{model_path.name}` adapter `{adapter_name}` bucket `{bucket}`\n"
        f"- photos {metrics['n_photos']} · GT boxes {metrics['n_gt']}\n"
        f"- **recall@0.5FPPI = {metrics['recall_at_op']:.3f}** "
        f"(CI95 {metrics['recall_ci95'][0]:.3f}–{metrics['recall_ci95'][1]:.3f}, "
        f"t*={metrics['threshold_at_op']})\n"
        f"- AP50 {metrics['ap50']:.3f} · AP50-95 {metrics['ap50_95']:.3f}\n"
        f"- FP at op: {metrics['fp_at_op']} "
        f"(other-defect: {metrics['fp_at_op_other_defect']})\n"
        f"- FROC: "
        + " · ".join(
            f"{k}FPPI→{v['recall']:.3f}" for k, v in metrics["froc"].items()
        )
        + "\n"
        f"- size-bucket recall: {metrics['size_bucket_recall_at_op']}\n"
        f"- mean {metrics['mean_ms_per_photo']} ms/photo (this host)\n"
    )
    (out_dir / "report.md").write_text(summary, encoding="utf-8")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = settings.reports_dir / "results.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "run_name": run_name,
                "model": model_path.name,
                "adapter": adapter_name,
                "bucket": bucket,
                "n_photos": metrics["n_photos"],
                "n_gt": metrics["n_gt"],
                "recall_at_0.5fppi": metrics["recall_at_op"],
                "ci95_lo": metrics["recall_ci95"][0],
                "ci95_hi": metrics["recall_ci95"][1],
                "threshold": metrics["threshold_at_op"],
                "ap50": metrics["ap50"],
                "ap50_95": metrics["ap50_95"],
                "fp_at_op": metrics["fp_at_op"],
                "fp_other_defect": metrics["fp_at_op_other_defect"],
                "mean_ms_per_photo": metrics["mean_ms_per_photo"],
                "tiler_version": detector.tiler_version,
                "split_id": load_split()["split_id"],
                "snapshot_id": manifest["snapshot_id"],
                "host": platform.node(),
            }
        )
    logger.info(summary)
    logger.info(f"artifacts: {out_dir}")
    return metrics
