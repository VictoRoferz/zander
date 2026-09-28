"""
Full-photo tiled detection: the ONE inference path shared by the evaluation
harness and the inspector service (torch-free: numpy + Pillow + onnxruntime).

photo -> pinned 640/512 tile grid -> per-tile ONNX -> conf floor + top-k
      -> offsets to photo coords -> per-class IoS non-max merge -> detections

Detections carry the model's class index (schema order); the photo decision
(i.O / N.iO + classes) is made by infer/verdict.py from these detections.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from infer.onnx_adapters import load_adapter
from infer.stitch import merge_by_class
from infer.tiling import TILE_SIZE, TILER_VERSION, tile_origins

PER_TILE_TOP_K = 300
IOS_MERGE_THRESHOLD = 0.5
# COCO-standard cap after stitching, per class; also bounds scorer cost at low
# conf floors.
MAX_DETECTIONS_PER_CLASS = 100


class Detections(NamedTuple):
    boxes: list[tuple[float, float, float, float]]  # photo-coordinate xyxy
    scores: list[float]
    labels: list[int]  # class index in the model's schema
    elapsed_ms: float


def load_photo_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class TiledDetector:
    def __init__(
        self,
        model_path: Path,
        adapter_name: str = "ultralytics",
        conf_floor: float = 0.001,
        threads: int = 0,
    ):
        self.adapter = load_adapter(adapter_name, str(model_path), threads=threads)
        self.conf_floor = conf_floor
        self.model_path = Path(model_path)
        self.tiler_version = TILER_VERSION

    def detect(self, photo_rgb: np.ndarray) -> Detections:
        h, w = photo_rgb.shape[:2]
        t0 = time.perf_counter()
        all_boxes: list[tuple[float, float, float, float]] = []
        all_scores: list[float] = []
        all_labels: list[int] = []
        for ox, oy in tile_origins(w, h):
            tile = photo_rgb[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE]
            boxes, scores, labels = self.adapter.run(tile, self.conf_floor)
            if len(scores) > PER_TILE_TOP_K:
                top = np.argsort(scores)[::-1][:PER_TILE_TOP_K]
                boxes, scores, labels = boxes[top], scores[top], labels[top]
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
                all_boxes.append(
                    (float(x1) + ox, float(y1) + oy, float(x2) + ox, float(y2) + oy)
                )
                all_scores.append(float(s))
                all_labels.append(int(c))
        boxes, scores, labels = merge_by_class(
            all_boxes, all_scores, all_labels,
            ios_threshold=IOS_MERGE_THRESHOLD, max_per_class=MAX_DETECTIONS_PER_CLASS,
        )
        return Detections(boxes, scores, labels, (time.perf_counter() - t0) * 1000.0)

    def detect_file(self, path: Path) -> Detections:
        return self.detect(load_photo_rgb(path))
