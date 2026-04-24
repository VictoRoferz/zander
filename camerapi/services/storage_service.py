"""
Persistent storage for camerapi.

Responsibilities:
- Atomic JPEG/JSON writes (no half-written files visible to readers).
- SHA256 hashing for dedup/verification (not primary ID — see capture_id).
- Paths for unlabeled and labeled artifacts.

All disk I/O for the app lives here. Camera, HTTP, and LS services never
touch the filesystem directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config.settings import settings
from models.schemas import CaptureMetadata
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


def new_capture_id() -> str:
    """Generate a fresh capture ID. UUID4 — collision-free in practice."""
    return uuid.uuid4().hex


def compute_sha256(path: Path) -> str:
    """Stream the file in chunks to stay memory-friendly on the Pi."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """
    Write bytes atomically: write to a sibling temp file, then rename.
    Rename within the same directory is atomic on POSIX.
    """
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
        # Best-effort cleanup of the temp file on failure
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class StorageService:
    """Owns all filesystem paths for capture artifacts."""

    def __init__(self) -> None:
        self.unlabeled_dir: Path = settings.unlabeled_dir
        self.labeled_dir: Path = settings.labeled_dir
        logger.info(
            f"StorageService initialized: unlabeled={self.unlabeled_dir}, "
            f"labeled={self.labeled_dir}"
        )

    # ---- Unlabeled ------------------------------------------------------

    def unlabeled_image_path(self, capture_id: str) -> Path:
        return self.unlabeled_dir / f"{capture_id}.jpg"

    def unlabeled_metadata_path(self, capture_id: str) -> Path:
        return self.unlabeled_dir / f"{capture_id}.json"

    def save_unlabeled(
        self,
        frame: np.ndarray,
        capture_id: str,
        source: str,
        camera_serial: Optional[str] = None,
        camera_model: Optional[str] = None,
    ) -> tuple[Path, CaptureMetadata]:
        """
        Encode frame as JPEG, write image + metadata sidecar atomically.
        Returns (image_path, metadata).
        """
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame")
        data = buf.tobytes()

        image_path = self.unlabeled_image_path(capture_id)
        _atomic_write_bytes(image_path, data)

        height, width = frame.shape[:2]
        metadata = CaptureMetadata(
            capture_id=capture_id,
            sha256=hashlib.sha256(data).hexdigest(),
            captured_at=datetime.now(timezone.utc),
            source=source,
            width=int(width),
            height=int(height),
            size_bytes=len(data),
            camera_serial=camera_serial,
            camera_model=camera_model,
        )

        self._save_metadata(self.unlabeled_metadata_path(capture_id), metadata)
        logger.info(
            f"Saved unlabeled capture {capture_id} "
            f"({width}x{height}, {len(data) / 1024:.1f} KB)"
        )
        return image_path, metadata

    # ---- Labeled --------------------------------------------------------

    def labeled_image_path(self, capture_id: str) -> Path:
        return self.labeled_dir / f"{capture_id}.jpg"

    def labeled_annotation_path(self, capture_id: str) -> Path:
        return self.labeled_dir / f"{capture_id}.json"

    def save_labeled(
        self,
        capture_id: str,
        image_bytes: bytes,
        annotation: dict[str, Any],
        capture_metadata: Optional[CaptureMetadata] = None,
    ) -> tuple[Path, Path]:
        """
        Write labeled image + merged annotation/metadata JSON atomically.
        Idempotent — repeated calls overwrite.
        """
        image_path = self.labeled_image_path(capture_id)
        _atomic_write_bytes(image_path, image_bytes)

        merged: dict[str, Any] = {"capture_id": capture_id, "annotation": annotation}
        if capture_metadata is not None:
            merged["capture_metadata"] = capture_metadata.model_dump(mode="json")

        annotation_path = self.labeled_annotation_path(capture_id)
        _atomic_write_bytes(
            annotation_path,
            json.dumps(merged, indent=2, default=str).encode("utf-8"),
        )
        logger.info(f"Saved labeled capture {capture_id}")
        return image_path, annotation_path

    # ---- Metadata helpers ----------------------------------------------

    def load_unlabeled_metadata(self, capture_id: str) -> Optional[CaptureMetadata]:
        path = self.unlabeled_metadata_path(capture_id)
        if not path.exists():
            return None
        try:
            return CaptureMetadata.model_validate_json(path.read_text("utf-8"))
        except Exception as e:
            logger.warning(f"Could not parse metadata for {capture_id}: {e}")
            return None

    def _save_metadata(self, path: Path, metadata: CaptureMetadata) -> None:
        data = metadata.model_dump_json(indent=2).encode("utf-8")
        _atomic_write_bytes(path, data)


# Global instance — imported by routes / webhooks.
storage_service = StorageService()
