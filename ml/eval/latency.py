"""
Tiled-inference latency bench on the current host (CPU, onnxruntime).

    python cli.py latency --model <path>.onnx [--adapter ...] [--threads N]

Measures the full per-photo pipeline (20-tile grid, decode, IoS merge) on
real photos — the number that must stay under the <1 s interactive budget on
the deployment PC. Run it on the Mac as a proxy during development and on the
Windows hub before any deployment claim.
"""
from __future__ import annotations

import json
import logging
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import onnxruntime as ort

from config.settings import settings
from dataprep.ingest import load_manifest
from infer.pipeline import TiledDetector
from infer.tiling import TILE_SIZE, tile_origins

logger = logging.getLogger("ml.latency")

WARMUP = 2


def run(
    model_path: Path,
    adapter_name: str = "ultralytics",
    n_photos: int = 10,
    threads: int = 0,
) -> dict:
    manifest = load_manifest()
    usable = [
        e for e in manifest["entries"]
        if e["quarantine_reason"] is None and e["label_state"] != "unannotated"
    ]
    sample = usable[:: max(1, len(usable) // n_photos)][:n_photos]
    source_dir = Path(manifest["source_images_dir"])
    detector = TiledDetector(model_path, adapter_name, threads=threads)

    for e in sample[:WARMUP]:
        detector.detect_file(source_dir / f"{e['capture_id']}.jpg")

    times = []
    for e in sample:
        _, _, ms = detector.detect_file(source_dir / f"{e['capture_id']}.jpg")
        times.append(ms)

    n_tiles = len(tile_origins(sample[0]["width"], sample[0]["height"]))
    result = {
        "model": model_path.name,
        "adapter": adapter_name,
        "threads": threads or "ort-default",
        "n_photos": len(times),
        "tiles_per_photo": n_tiles,
        "tile_size": TILE_SIZE,
        "ms_per_photo": {
            "mean": round(statistics.mean(times), 1),
            "p50": round(statistics.median(times), 1),
            "min": round(min(times), 1),
            "max": round(max(times), 1),
        },
        "ms_per_tile_mean": round(statistics.mean(times) / n_tiles, 1),
        "host": platform.node(),
        "machine": platform.machine(),
        "onnxruntime": ort.__version__,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }
    settings.bench_dir.mkdir(parents=True, exist_ok=True)
    out = settings.bench_dir / f"latency_{model_path.stem}_{platform.node()}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    logger.info(json.dumps(result, indent=1))
    logger.info(f"written: {out}")
    return result
