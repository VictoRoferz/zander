"""
Detector service: wraps the shared torch-free tiled-inference pipeline
(ml/infer) around the exported winner ONNX.

The heavy lifting — pinned tile grid, ONNX adapters, IoS cross-tile merge —
is the SAME code the evaluation harness used to pick the model, imported from
ml/infer via a path insert (the repo's cwd-relative service convention; in
Docker the Dockerfile copies ml/infer to /app/ml/infer).
"""
import hashlib
import io
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import settings

# Make the shared inference core importable (repo layout or Docker layout).
_here = Path(__file__).resolve()
for candidate in (_here.parents[2] / "ml", _here.parents[1] / "ml"):
    if (candidate / "infer").is_dir():
        sys.path.insert(0, str(candidate))
        break

from infer.pipeline import TiledDetector  # noqa: E402
from infer.verdict import photo_verdict  # noqa: E402

logger = logging.getLogger("inspector.detector")


class DetectorService:
    def __init__(self) -> None:
        self._detector: TiledDetector | None = None
        self.model_sha8 = "unloaded"

    @property
    def ready(self) -> bool:
        return self._detector is not None

    def load(self) -> None:
        path = settings.model_path
        if not path.exists():
            logger.warning(
                f"model not found at {path} — /inspect returns 503 until a "
                "winner is exported there (see ml/README.md)"
            )
            return
        self._detector = TiledDetector(
            path,
            adapter_name=settings.adapter,
            conf_floor=settings.confidence_threshold,
            threads=settings.ort_threads,
        )
        self.model_sha8 = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        logger.info(f"model loaded: {path.name} ({self.model_sha8})")

    def inspect_bytes(self, image_bytes: bytes) -> dict:
        with Image.open(io.BytesIO(image_bytes)) as im:
            photo = np.asarray(im.convert("RGB"))
        return self.inspect_array(photo)

    def inspect_array(self, photo_rgb: np.ndarray) -> dict:
        assert self._detector is not None
        det = self._detector.detect(photo_rgb)
        # The same decision function the evaluation harness scores.
        decision = photo_verdict(
            det.boxes, det.scores, det.labels,
            threshold=settings.confidence_threshold, class_names=settings.class_names,
        )
        return {
            "verdict": decision["verdict"],
            "classes": decision["classes"],
            "detections": [
                {
                    "x1": d["box"][0],
                    "y1": d["box"][1],
                    "x2": d["box"][2],
                    "y2": d["box"][3],
                    "score": d["score"],
                    "label": d["class"],
                }
                for d in decision["defects"]
            ],
            "count": len(decision["defects"]),
            "latency_ms": round(det.elapsed_ms, 1),
            "model_version": f"{settings.model_path.name}@{self.model_sha8}",
            "tiler_version": self._detector.tiler_version,
            "confidence_threshold": settings.confidence_threshold,
        }


# Global service instance.
detector_service = DetectorService()
