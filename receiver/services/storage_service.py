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
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from models.schemas import CaptureMetadata

logger = logging.getLogger("receiver.storage")


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
        os.replace(tmp_name, target)
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
        logger.info(
            f"StorageService initialized: unlabeled={self.unlabeled_dir}, "
            f"labeled={self.labeled_dir}"
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


# Global instance.
storage_service = StorageService()
