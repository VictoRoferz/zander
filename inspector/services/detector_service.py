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

Several models (v0.3.0): every <name>.onnx in the models folder is loaded
with the rules above, each with its own card; one that is refused never
stops the others. ONE of them is active and serves /inspect:
  - the saved choice (active_model.json in the models folder), else
    MODEL_PATH, else (MODEL_PATH missing) the first model by name;
  - a wanted model that is refused is never replaced by another one: no model
    is served (503) and load_error says why, as with a single model.
`set_active` switches the model for the next request and saves the choice
(a read-only folder only loses the save, never the switch).

Concurrency: one onnxruntime session per model. Decoding runs outside the
lock (in the caller's threadpool thread); `detect` is serialized by `_lock`
across all models, so concurrent requests queue instead of oversubscribing
the CPU. Each request reads the active model once (`self._state`), so a
switch never mixes two models within one photo.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from config.settings import (
    ACTIVE_MODEL_FILE,
    DEFAULT_ADAPTER,
    DEFAULT_CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    settings,
)

# Make the shared inference core importable (repo layout or Docker layout).
# APPENDED, never inserted in front: ml/ has its own `config` namespace package
# (ml/config/settings.py) and so does the inspector (config/settings.py). With
# ml/ first, `config.settings` would resolve to the ML toolchain's settings
# whenever it was imported after this line; last on the path, ml/ only supplies
# `infer`, which nothing else provides.
_here = Path(__file__).resolve()
for candidate in (_here.parents[2] / "ml", _here.parents[1] / "ml"):
    if (candidate / "infer").is_dir():
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
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


class UnknownModelError(LookupError):
    """set_active named a model that is not in the models folder (HTTP 404)."""


class ModelUnavailableError(RuntimeError):
    """set_active named a model that was refused at load (HTTP 409)."""


class ModelRefused(Exception):
    """Internal: the model must not be served; the message becomes load_error."""


class ModelMissing(ModelRefused):
    """Internal: no model file yet (expected before the first export)."""


@dataclass(frozen=True)
class LoadedModel:
    """Everything one loaded model needs, swapped in as a single reference."""

    name: str  # the file stem: what the dashboard shows and PUT /models/active takes
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


_REPLACE_RETRIES = 5
_REPLACE_RETRY_S = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(target: Path, doc: dict) -> None:
    """Temp file in the target dir -> fsync -> 0644 -> os.replace (as the receiver)."""
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, 0o644)
        for attempt in range(_REPLACE_RETRIES + 1):
            try:
                os.replace(tmp_name, target)
                break
            except PermissionError:  # Windows: the target is open in another process
                if attempt >= _REPLACE_RETRIES:
                    raise
                time.sleep(_REPLACE_RETRY_S)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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
        # The active model: /inspect reads this one reference per request.
        self._state: Optional[LoadedModel] = None
        # Every servable model by name and why the others are not served.
        # load() replaces both as a whole; they are never mutated.
        self._models: dict[str, LoadedModel] = {}
        self._refused: dict[str, str] = {}
        # The active model's name, also when it cannot be served (load_error says why).
        self._active: Optional[str] = None
        self._selection: dict[str, Optional[str]] = {"selected_by": None, "selected_at": None}
        self._lock = threading.Lock()  # serializes detect() across every ORT session
        self._switch_lock = threading.Lock()  # one switch (and its save) at a time
        self.load_error: Optional[str] = NOT_LOADED

    @property
    def ready(self) -> bool:
        return self._state is not None

    # ---- Loading ------------------------------------------------------------

    def load(self) -> None:
        """Load every model in the models folder (+ cards), then pick the active one.

        Never raises. See the module docstring for which model becomes active.
        """
        models: dict[str, LoadedModel] = {}
        refused: dict[str, str] = {}
        for path in self._model_files():
            name = path.stem
            try:
                state = self._build_state(path)
            except ModelRefused as e:
                refused[name] = str(e)
                logger.error(f"model {name!a} refused: {str(e)!a}")
                continue
            except Exception as e:  # anything else (e.g. onnxruntime rejects the file)
                logger.exception(f"model {name!a} load crashed")
                refused[name] = f"model failed to load: {type(e).__name__}: {e}"
                continue
            models[name] = state
            logger.info(
                f"model loaded: {state.model_version} adapter={state.adapter} "
                f"schema={state.schema} classes={list(state.class_names)!a} "
                f"threshold={state.threshold} ({state.threshold_source}) "
                f"conf_floor={state.conf_floor} tiler={state.tiler_version}"
            )
        with self._switch_lock:
            self._models, self._refused = models, refused
            saved = self._read_saved_choice()
            name, error = self._pick_active(saved)
            picked_saved = saved is not None and name == saved["name"]
            self._selection = {
                key: (saved.get(key) if picked_saved and isinstance(saved.get(key), str) else None)
                for key in ("selected_by", "selected_at")
            }
            if error is None:
                self._activate(name)
            else:
                self._active = name
                self._fail(error, level=logging.WARNING if name is None else logging.ERROR)

    def _model_files(self) -> list[Path]:
        """MODEL_PATH first (if it exists), then every other *.onnx in the models folder by name."""
        default = settings.model_path
        files: list[Path] = [default] if default.is_file() else []
        folder = settings.models_folder
        try:
            found = sorted(p for p in folder.glob("*.onnx") if p.is_file())
        except OSError as e:
            logger.warning(f"cannot list the models folder {folder}: {e}")
            found = []
        default_resolved = default.resolve()
        names = {p.stem for p in files}
        for path in found:
            if path.resolve() == default_resolved:
                continue
            if path.stem in names:
                logger.warning(f"{path} skipped: a model named {path.stem!a} is already loaded")
                continue
            names.add(path.stem)
            files.append(path)
        return files

    def _pick_active(self, saved: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
        """(active name or None, why it cannot be served or None)."""
        wanted = ([saved["name"]] if saved else []) + [settings.model_path.stem]
        for i, name in enumerate(wanted):
            if name in self._models or name in self._refused:
                return name, self._refused.get(name)
            if saved and i == 0:
                logger.warning(
                    f"the saved model choice {name!a} is not in {settings.models_folder} "
                    f"any more; using MODEL_PATH ({settings.model_path.name})"
                )
        if self._models:
            name = min(self._models)
            logger.warning(
                f"MODEL_PATH {settings.model_path} not found: the active model is "
                f"{name!a}, the first one in {settings.models_folder}"
            )
            return name, None
        return None, (
            f"model not found at {settings.model_path} (export one with `cli.py export`, "
            "see ml/README.md)"
        )

    def _activate(self, name: str, selected_by: Optional[str] = None) -> None:
        state = self._models[name]
        self._state = state
        self._active = name
        self.load_error = None
        by = f" (chosen by {selected_by!a})" if selected_by else ""
        logger.info(f"active model: {state.model_version}{by}")

    def _fail(self, reason: str, level: int = logging.ERROR) -> None:
        self._state = None
        self.load_error = reason
        logger.log(level, f"no model served: {reason!a} (/inspect answers 503)")

    # ---- Switching ------------------------------------------------------------

    def set_active(self, name: str, selected_by: Optional[str] = None) -> bool:
        """Make `name` the model /inspect uses and save the choice.

        Raises UnknownModelError / ModelUnavailableError. Returns False when
        the choice could not be saved: it still applies until the next restart.
        """
        with self._switch_lock:
            if name in self._refused:
                raise ModelUnavailableError(f"model {name!r} cannot be served: {self._refused[name]}")
            if name not in self._models:
                raise UnknownModelError(f"unknown model {name!r}")
            self._activate(name, selected_by)
            now = _utc_now()
            self._selection = {"selected_by": selected_by, "selected_at": now}
            return self._save_choice({
                "name": name,
                "model_version": self._models[name].model_version,
                "selected_by": selected_by,
                "selected_at": now,
            })

    def _save_choice(self, doc: dict) -> bool:
        target = settings.active_model_file
        try:
            _atomic_write_json(target, doc)
        except OSError as e:
            logger.warning(
                f"model choice not saved to {target} ({type(e).__name__}: {e}); "
                "it applies until the inspector restarts"
            )
            return False
        return True

    def _read_saved_choice(self) -> Optional[dict]:
        path = settings.active_model_file
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:  # unreadable or not JSON
            logger.warning(f"ignoring the saved model choice {path}: {type(e).__name__}: {e}")
            return None
        if not isinstance(doc, dict) or not isinstance(doc.get("name"), str) or not doc["name"]:
            logger.warning(f"ignoring the saved model choice {path}: no model name in it")
            return None
        return doc

    def _build_state(self, path: Path) -> LoadedModel:
        card_path = card_path_for(path)
        if not path.exists():
            raise ModelMissing(
                f"model not found at {path} (export one with `cli.py export`, "
                "see ml/README.md)"
            )
        if card_path.name == ACTIVE_MODEL_FILE:
            raise ModelRefused(
                f"the model name {path.stem!r} is reserved: its card would be "
                f"{ACTIVE_MODEL_FILE}, the saved model choice. Rename the model."
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
            name=path.stem,
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

    def models(self) -> dict:
        """Every model in the models folder and which one is active (GET /api/v1/models)."""
        models, refused = self._models, self._refused
        state, active, selection = self._state, self._active, self._selection
        items = []
        for name in sorted(set(models) | set(refused)):
            m = models.get(name)
            items.append({
                "name": name,
                "loaded": m is not None,
                "active": name == active,
                "error": refused.get(name),
                "model_version": m.model_version if m else None,
                "model_sha256": m.model_sha256 if m else None,
                "adapter": m.adapter if m else None,
                "schema": m.schema if m else None,
                "class_names": list(m.class_names) if m else None,
                "threshold": m.threshold if m else None,
                "threshold_source": m.threshold_source if m else None,
                "run_name": m.meta.get("run_name") if m else None,
                "created": m.meta.get("created") if m else None,
            })
        return {
            "active": active,
            "model_loaded": state is not None,
            "load_error": self.load_error,
            "selected_by": selection["selected_by"],
            "selected_at": selection["selected_at"],
            "models_dir": str(settings.models_folder),
            "models": items,
        }

    def status(self) -> dict:
        state = self._state
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "model_loaded": state is not None,
            "active_model": self._active,
            "model_path": str(state.model_path if state else settings.model_path),
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
