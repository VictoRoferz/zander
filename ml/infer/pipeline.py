"""
Full-photo tiled detection: the ONE inference path shared by the evaluation
harness and the inspector service (torch-free: numpy + Pillow + onnxruntime).

photo -> pinned 640/512 tile grid -> per-tile ONNX -> conf floor + top-k
      -> offsets to photo coords -> IoS non-max merge -> detections
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from infer.onnx_adapters import load_adapter
from infer.stitch import merge_detections
from infer.tiling import TILE_SIZE, TILER_VERSION, tile_origins

PER_TILE_TOP_K = 300
IOS_MERGE_THRESHOLD = 0.5
# COCO-standard cap after stitching; also bounds scorer cost at low conf floors.
MAX_DETECTIONS_PER_PHOTO = 100


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

    def detect(self, photo_rgb: np.ndarray) -> tuple[list, list, float]:
        """Returns (boxes photo-xyxy, scores, elapsed_ms) for one photo."""
        h, w = photo_rgb.shape[:2]
        t0 = time.perf_counter()
        all_boxes: list[tuple[float, float, float, float]] = []
        all_scores: list[float] = []
        for ox, oy in tile_origins(w, h):
            tile = photo_rgb[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE]
            boxes, scores = self.adapter.run(tile, self.conf_floor)
            if len(scores) > PER_TILE_TOP_K:
                top = np.argsort(scores)[::-1][:PER_TILE_TOP_K]
                boxes, scores = boxes[top], scores[top]
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                all_boxes.append(
                    (float(x1) + ox, float(y1) + oy, float(x2) + ox, float(y2) + oy)
                )
                all_scores.append(float(s))
        boxes, scores = merge_detections(
            all_boxes, all_scores, ios_threshold=IOS_MERGE_THRESHOLD
        )
        boxes = boxes[:MAX_DETECTIONS_PER_PHOTO]  # merge returns score-desc
        scores = scores[:MAX_DETECTIONS_PER_PHOTO]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, scores, elapsed_ms

    def detect_file(self, path: Path) -> tuple[list, list, float]:
        return self.detect(load_photo_rgb(path))
