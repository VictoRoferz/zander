"""
Detector service: wraps the shared torch-free tiled-inference pipeline
(ml/infer) around the exported detector ONNX and its card (detector.json).

The heavy lifting (pinned tile grid, ONNX adapters, IoS cross-tile merge) is
the SAME code the evaluation harness used to pick the model, imported from
ml/infer via a path insert (the repo's cwd-relative service convention; in
Docker the Dockerfile copies ml/infer to /app/ml/infer).

Loading (docs/CONTRACTS.md section 1) never raises. A model the inspector must
not serve leaves the service up with `load_error` set, so /inspect answers 503
and /status explains why:
  - card present -> the card is authoritative (adapter, class_names,
    threshold, conf_floor); env ADAPTER / CONFIDENCE_THRESHOLD / CLASS_NAMES
    are ignored with a WARNING. Refused on an invalid card, a sha256 mismatch,
    a non-null `verifier` (no verifier support in v1), or a class count that
    differs from what the model reports.
  - no card -> legacy env settings with a WARNING (threshold_source "env" or
    "default"), unless REQUIRE_CARD=true, which refuses instead.

Concurrency: ONE onnxruntime session serves every request. Decoding runs
outside the lock (in the caller's threadpool thread); `detect` is serialized
by `_lock`, so concurrent requests queue instead of oversubscribing the CPU.
"""
from __future__ import annotations

import io
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from config.settings import (
    DEFAULT_ADAPTER,
    DEFAULT_CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    settings,
)

# Make the shared inference core importable (repo layout or Docker layout).
_here = Path(__file__).resolve()
for candidate in (_here.parents[2] / "ml", _here.parents[1] / "ml"):
    if (candidate / "infer").is_dir():
        sys.path.insert(0, str(candidate))
        break

from infer.detector_card import (  # noqa: E402
    KNOWN_ADAPTERS,
    DetectorCardError,
    card_path_for,
    load_detector_card,
    sha256_file,
)
from infer.pipeline import TiledDetector, load_photo_rgb  # noqa: E402
from infer.verdict import photo_verdict  # noqa: E402

logger = logging.getLogger("inspector.detector")

NOT_LOADED = "model not loaded yet"


class ModelNotLoadedError(RuntimeError):
    """/inspect was called while no model is loaded (maps to HTTP 503)."""


class ImageDecodeError(ValueError):
    """The bytes are not a decodable image (maps to HTTP 422)."""


class ModelRefused(Exception):
    """Internal: the model must not be served; the message becomes load_error."""


class ModelMissing(ModelRefused):
    """Internal: no model file yet (expected before the first export)."""


@dataclass(frozen=True)
class LoadedModel:
    """Everything one loaded model needs, swapped in as a single reference."""

    detector: Any  # infer.pipeline.TiledDetector
    model_path: Path
    card_path: Optional[Path]  # the card the model was loaded with; None = legacy env
    model_sha256: str
    model_version: str  # "<file>@<sha256[:8]>", the receiver's idempotency key
    adapter: str
    schema: Optional[str]
    class_names: tuple[str, ...]
    threshold: float
    conf_floor: float
    threshold_source: str  # "detector.json" | "env" | "default"
    tiler_version: str
    meta: dict


def decode_image(image_bytes: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array, decoded exactly like the eval harness."""
    try:
        # load_photo_rgb is the eval harness's decoder; Image.open takes a file object.
        return load_photo_rgb(io.BytesIO(image_bytes))
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as e:
        # UnidentifiedImageError and truncated files are OSErrors.
        raise ImageDecodeError(f"cannot decode image: {e}") from e


def _clean_class_names(names: Any, origin: str) -> tuple[str, ...]:
    if (
        not isinstance(names, (list, tuple))
        or not names
        or not all(isinstance(n, str) and n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ModelRefused(f"{origin} must be a non-empty list of unique, non-empty strings")
    return tuple(names)


class DetectorService:
    def __init__(self) -> None:
        self._state: Optional[LoadedModel] = None
        self._lock = threading.Lock()  # serializes detect() on the one ORT session
        self.load_error: Optional[str] = NOT_LOADED

    @property
    def ready(self) -> bool:
        return self._state is not None

    # ---- Loading ------------------------------------------------------------

    def load(self) -> None:
        """Load the model named by MODEL_PATH (+ its card). Never raises."""
        try:
            state = self._build_state()
        except ModelMissing as e:
            self._fail(str(e), level=logging.WARNING)
            return
        except ModelRefused as e:
            self._fail(str(e))
            return
        except Exception as e:  # anything else (e.g. onnxruntime rejects the file)
            logger.exception("model load crashed")
            self._fail(f"model failed to load: {type(e).__name__}: {e}")
            return
        self._state = state
        self.load_error = None
        logger.info(
            f"model loaded: {state.model_version} adapter={state.adapter} "
            f"schema={state.schema} classes={list(state.class_names)!a} "
            f"threshold={state.threshold} ({state.threshold_source}) "
            f"conf_floor={state.conf_floor} tiler={state.tiler_version}"
        )

    def _fail(self, reason: str, level: int = logging.ERROR) -> None:
        self._state = None
        self.load_error = reason
        logger.log(level, f"no model served: {reason!a} (/inspect answers 503)")

    def _build_state(self) -> LoadedModel:
        path = settings.model_path
        card_path = card_path_for(path)
        if not path.exists():
            raise ModelMissing(
                f"model not found at {path} (export one with `cli.py export`, "
                "see ml/README.md)"
            )

        if card_path.exists():
            card_used: Optional[Path] = card_path
            try:
                # Same validator as the exporter, including the sha256 check.
                card = load_detector_card(path)
            except DetectorCardError as e:
                raise ModelRefused(f"invalid detector card {card_path.name}: {e}") from e
            if card.verifier is not None:
                raise ModelRefused(
                    f"detector card {card_path.name} requires a verifier, which "
                    "inspector v1 does not support"
                )
            ignored = [
                name
                for name, value in (
                    ("ADAPTER", settings.adapter),
                    ("CONFIDENCE_THRESHOLD", settings.confidence_threshold),
                    ("CLASS_NAMES", settings.class_names),
                )
                if value is not None
            ]
            if ignored:
                logger.warning(
                    f"{card_path.name} is authoritative: ignoring env {', '.join(ignored)}"
                )
            adapter = card.adapter
            schema = card.schema
            class_names = card.class_names  # validated by load_detector_card
            threshold = card.threshold
            conf_floor = card.conf_floor
            threshold_source = "detector.json"
            model_sha256 = card.model_sha256  # verified against the file above
            source = card.raw.get("source") or {}
            meta = {
                "split_id": source.get("split_id"),
                "dataset_fingerprint": source.get("dataset_fingerprint"),
                "run_name": source.get("bundle_run"),
                "created": card.raw.get("exported_at"),
            }
        else:
            if settings.require_card:
                raise ModelRefused(
                    f"REQUIRE_CARD=true but there is no detector card at {card_path}"
                )
            card_used = None
            adapter = settings.adapter or DEFAULT_ADAPTER
            if adapter not in KNOWN_ADAPTERS:
                raise ModelRefused(
                    f"ADAPTER must be one of {list(KNOWN_ADAPTERS)}, got {adapter!r}"
                )
            schema = None
            class_names = _clean_class_names(
                settings.class_names if settings.class_names is not None else list(DEFAULT_CLASS_NAMES),
                "CLASS_NAMES",
            )
            if settings.confidence_threshold is not None:
                threshold, threshold_source = float(settings.confidence_threshold), "env"
            else:
                threshold, threshold_source = DEFAULT_CONFIDENCE_THRESHOLD, "default"
            if not 0.0 < threshold < 1.0:
                raise ModelRefused(f"CONFIDENCE_THRESHOLD must be in (0, 1), got {threshold}")
            conf_floor = threshold  # no card: the floor is the threshold
            model_sha256 = sha256_file(path)
            meta = {"split_id": None, "dataset_fingerprint": None, "run_name": None, "created": None}
            logger.warning(
                f"no detector card at {card_path}: using legacy env settings "
                f"(adapter={adapter}, threshold={threshold} from {threshold_source}); "
                "export the model with `cli.py export` to ship its card"
            )

        detector = TiledDetector(
            path, adapter_name=adapter, conf_floor=conf_floor, threads=settings.ort_threads
        )
        n_model = getattr(detector.adapter, "num_classes", None)  # None = not knowable
        if n_model is not None and n_model != len(class_names):
            raise ModelRefused(
                f"class count mismatch: the model outputs {n_model} classes but "
                f"{'the card' if card_used else 'CLASS_NAMES'} lists {len(class_names)}"
            )
        return LoadedModel(
            detector=detector,
            model_path=path,
            card_path=card_used,
            model_sha256=model_sha256,
            model_version=f"{path.name}@{model_sha256[:8]}",
            adapter=adapter,
            schema=schema,
            class_names=class_names,
            threshold=threshold,
            conf_floor=conf_floor,
            threshold_source=threshold_source,
            tiler_version=detector.tiler_version,
            meta=meta,
        )

    # ---- Inference ----------------------------------------------------------

    def inspect_bytes(self, image_bytes: bytes) -> dict:
        """Decode (outside the lock), then detect. Blocking: call from a thread."""
        if self._state is None:  # fail fast, before paying for the decode
            raise ModelNotLoadedError(self.load_error or NOT_LOADED)
        return self.inspect_array(decode_image(image_bytes))

    def inspect_array(self, photo_rgb: np.ndarray) -> dict:
        state = self._state  # one consistent snapshot for the whole request
        if state is None:
            raise ModelNotLoadedError(self.load_error or NOT_LOADED)
        height, width = photo_rgb.shape[:2]
        with self._lock:
            det = state.detector.detect(photo_rgb)
        class_names = list(state.class_names)
        bad = [c for c in det.labels if not 0 <= c < len(class_names)]
        if bad:
            raise RuntimeError(
                f"model emitted class index {max(bad)} but only {len(class_names)} "
                "class names are configured"
            )
        # The same decision function the evaluation harness scores.
        decision = photo_verdict(
            det.boxes, det.scores, det.labels,
            threshold=state.threshold, class_names=class_names,
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
                for d in decision["defects"]  # score-desc, all >= threshold
            ],
            "count": len(decision["defects"]),
            "latency_ms": round(det.elapsed_ms, 1),
            "model_version": state.model_version,
            "tiler_version": state.tiler_version,
            "confidence_threshold": state.threshold,
            "photo_score": decision["photo_score"],
            "image_width": int(width),
            "image_height": int(height),
            "model_sha256": state.model_sha256,
            "schema": state.schema,
            "adapter": state.adapter,
            "class_names": class_names,
            "threshold_source": state.threshold_source,
        }

    # ---- Status -------------------------------------------------------------

    def status(self) -> dict:
        state = self._state
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "model_loaded": state is not None,
            "model_path": str(settings.model_path),
            "meta_path": str(state.card_path) if state and state.card_path else None,
            "model_version": state.model_version if state else None,
            "model_sha256": state.model_sha256 if state else None,
            "adapter": state.adapter if state else None,
            "schema": state.schema if state else None,
            "class_names": list(state.class_names) if state else None,
            "confidence_threshold": state.threshold if state else None,
            "threshold_source": state.threshold_source if state else None,
            "tiler_version": state.tiler_version if state else None,
            "load_error": self.load_error,
            "meta": dict(state.meta) if state else {
                "split_id": None, "dataset_fingerprint": None, "run_name": None, "created": None,
            },
        }


# Global service instance.
detector_service = DetectorService()
