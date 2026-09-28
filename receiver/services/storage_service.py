"""
Persistent storage for the receiver (laptop).

Owns all filesystem paths for capture artifacts on the laptop. Receives raw
bytes over HTTP (not numpy frames), so it never imports OpenCV. Atomic writes
keep half-written files invisible to Label Studio and the dashboard.

Layout under data_root:
  unlabeled/<id>.jpg    image served by LS local-files + the dashboard
  unlabeled/<id>.json   capture metadata sidecar (from the Pi)
  unlabeled/<id>.task   idempotency marker — holds the created LS task id
  labeled/<id>.jpg      durable labeled export (training dataset)
  labeled/<id>.json     merged metadata + LS annotation
  predictions/<id>.pred.json  model prediction (pred.json v1, docs/CONTRACTS.md
                        section 2); own folder so no *.jpg / <id>.json glob and
                        no LS storage ever sees it
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from config.settings import settings
from models.schemas import CaptureMetadata

logger = logging.getLogger("receiver.storage")

# Windows refuses os.replace onto a file another process holds open
# (PermissionError, WinError 5 / 32) until that reader closes it: the dashboard
# reading the previous pred.json, an antivirus scan. Retry briefly there. POSIX
# renames over open files, so it tries exactly once (behaviour unchanged).
# Keep in sync with camerapi/services/storage_service.py.
_REPLACE_RETRIES = 5 if os.name == "nt" else 0
_REPLACE_RETRY_S = 0.05


def _replace(src: str, dst: Path) -> None:
    """os.replace, retried while Windows reports the target as in use."""
    for attempt in range(_REPLACE_RETRIES + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt >= _REPLACE_RETRIES:
                raise
            time.sleep(_REPLACE_RETRY_S)


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomic write: temp file in the target dir → fsync → os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the file 0600 (owner-only). In Docker the receiver
        # writes as root but Label Studio reads as a different (non-root) user,
        # so it must be world-readable or LS fails with "error opening file".
        os.chmod(tmp_name, 0o644)
        _replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class StorageService:
    """Owns all filesystem paths for capture artifacts on the laptop."""

    def __init__(self) -> None:
        self.unlabeled_dir: Path = settings.unlabeled_dir
        self.labeled_dir: Path = settings.labeled_dir
        self.predictions_dir: Path = settings.predictions_dir
        logger.info(
            f"StorageService initialized: unlabeled={self.unlabeled_dir}, "
            f"labeled={self.labeled_dir}, predictions={self.predictions_dir}"
        )

    # ---- Paths ----------------------------------------------------------

    def unlabeled_image_path(self, capture_id: str) -> Path:
        return self.unlabeled_dir / f"{capture_id}.jpg"

    def unlabeled_metadata_path(self, capture_id: str) -> Path:
        return self.unlabeled_dir / f"{capture_id}.json"

    def task_marker_path(self, capture_id: str) -> Path:
        return self.unlabeled_dir / f"{capture_id}.task"

    def labeled_image_path(self, capture_id: str) -> Path:
        return self.labeled_dir / f"{capture_id}.jpg"

    def labeled_annotation_path(self, capture_id: str) -> Path:
        return self.labeled_dir / f"{capture_id}.json"

    def prediction_path(self, capture_id: str) -> Path:
        return self.predictions_dir / f"{capture_id}.pred.json"

    # ---- Idempotency marker --------------------------------------------

    def has_task_marker(self, capture_id: str) -> bool:
        return self.task_marker_path(capture_id).exists()

    def read_task_marker(self, capture_id: str) -> Optional[str]:
        try:
            return self.task_marker_path(capture_id).read_text("utf-8").strip()
        except OSError:
            return None

    def write_task_marker(self, capture_id: str, task_id: Any) -> None:
        _atomic_write_bytes(self.task_marker_path(capture_id), str(task_id).encode("utf-8"))

    # ---- Unlabeled ------------------------------------------------------

    def save_unlabeled(
        self,
        capture_id: str,
        image_bytes: bytes,
        metadata: CaptureMetadata,
    ) -> Path:
        """Write the unlabeled image + metadata sidecar atomically."""
        _atomic_write_bytes(self.unlabeled_image_path(capture_id), image_bytes)
        _atomic_write_bytes(
            self.unlabeled_metadata_path(capture_id),
            metadata.model_dump_json(indent=2).encode("utf-8"),
        )
        return self.unlabeled_image_path(capture_id)

    def load_unlabeled_metadata(self, capture_id: str) -> Optional[CaptureMetadata]:
        path = self.unlabeled_metadata_path(capture_id)
        if not path.exists():
            return None
        try:
            return CaptureMetadata.model_validate_json(path.read_text("utf-8"))
        except Exception as e:
            logger.warning(f"Could not parse metadata for {capture_id}: {e}")
            return None

    # ---- Labeled --------------------------------------------------------

    def save_labeled(
        self,
        capture_id: str,
        image_bytes: bytes,
        annotation: dict[str, Any],
        capture_metadata: Optional[CaptureMetadata] = None,
    ) -> tuple[Path, Path]:
        """Write the labeled image + merged annotation/metadata JSON. Idempotent."""
        _atomic_write_bytes(self.labeled_image_path(capture_id), image_bytes)
        merged: dict[str, Any] = {"capture_id": capture_id, "annotation": annotation}
        if capture_metadata is not None:
            merged["capture_metadata"] = capture_metadata.model_dump(mode="json")
        _atomic_write_bytes(
            self.labeled_annotation_path(capture_id),
            json.dumps(merged, indent=2, default=str).encode("utf-8"),
        )
        logger.info(f"Saved labeled export {capture_id}")
        return self.labeled_image_path(capture_id), self.labeled_annotation_path(capture_id)

    # ---- Predictions ----------------------------------------------------

    def has_prediction(self, capture_id: str) -> bool:
        return self.prediction_path(capture_id).exists()

    def read_prediction(self, capture_id: str) -> Optional[dict[str, Any]]:
        """The stored pred.json, or None if absent/unreadable (bad JSON is logged)."""
        path = self.prediction_path(capture_id)
        try:
            text = path.read_text("utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.warning(f"[{capture_id}] prediction unreadable: {e}")
            return None
        try:
            doc = json.loads(text)
        except ValueError as e:
            logger.warning(f"[{capture_id}] prediction is not valid JSON: {e}")
            return None
        if not isinstance(doc, dict):
            logger.warning(f"[{capture_id}] prediction is not a JSON object")
            return None
        return doc

    def write_prediction(self, capture_id: str, doc: Mapping[str, Any]) -> Path:
        """Atomic write (temp -> fsync -> 0644 -> os.replace), UTF-8, not ASCII-escaped."""
        path = self.prediction_path(capture_id)
        data = json.dumps(dict(doc), indent=2, ensure_ascii=False) + "\n"
        _atomic_write_bytes(path, data.encode("utf-8"))
        return path

    # ---- Counts (for /status) ------------------------------------------

    def count_unlabeled(self) -> int:
        try:
            return sum(1 for _ in self.unlabeled_dir.glob("*.jpg"))
        except FileNotFoundError:
            return 0

    def count_labeled(self) -> int:
        try:
            return sum(1 for _ in self.labeled_dir.glob("*.jpg"))
        except FileNotFoundError:
            return 0

    def count_predictions(self) -> int:
        try:
            return sum(1 for _ in self.predictions_dir.glob("*.pred.json"))
        except FileNotFoundError:
            return 0


# Global instance.
storage_service = StorageService()
