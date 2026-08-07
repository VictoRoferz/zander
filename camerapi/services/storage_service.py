"""
Local spool storage for camerapi.

camerapi writes each capture (image + JSON sidecar) into the spool, and the
background uploader (services/spool_uploader.py) ships it to the ingestion
hub. An entry stays in the spool until the receiver ACKs it, so the Docker
side being down or still booting never loses a capture.

All disk I/O for the app lives here. Camera and HTTP code never touch the
filesystem directly.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from config.settings import settings
from models.schemas import CaptureMetadata
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


def new_capture_id() -> str:
    """Generate a fresh capture ID. UUID4 — collision-free in practice."""
    return uuid.uuid4().hex


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """
    Write bytes atomically: write to a sibling temp file, then rename.
    Rename within the same directory is atomic on POSIX and on Windows
    (os.replace maps to MoveFileEx with replace semantics).
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
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()


class StorageService:
    """Owns the local spool for capture artifacts."""

    def __init__(self) -> None:
        self.spool_dir: Path = settings.spool_dir
        self.failed_dir: Path = settings.spool_failed_dir
        logger.info(f"StorageService initialized: spool={self.spool_dir}")

    # ---- Paths ----------------------------------------------------------

    def spool_image_path(self, capture_id: str) -> Path:
        return self.spool_dir / f"{capture_id}.jpg"

    def spool_metadata_path(self, capture_id: str) -> Path:
        return self.spool_dir / f"{capture_id}.json"

    # ---- Write ----------------------------------------------------------

    def save_to_spool(
        self,
        frame: np.ndarray,
        capture_id: str,
        source: str,
        camera_serial: Optional[str] = None,
        camera_model: Optional[str] = None,
        triggered_by: Optional[str] = None,
    ) -> tuple[Path, CaptureMetadata]:
        """
        Encode the frame as JPEG and write image + metadata sidecar into the
        spool. Returns (image_path, metadata).
        """
        data = _encode_jpeg(frame)
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
            triggered_by=triggered_by,
        )

        # Image first, then sidecar. iter_pending() only yields an entry once
        # BOTH exist, so a crash between the two writes can never ship a
        # half-written capture.
        _atomic_write_bytes(self.spool_image_path(capture_id), data)
        _atomic_write_bytes(
            self.spool_metadata_path(capture_id),
            metadata.model_dump_json(indent=2).encode("utf-8"),
        )
        logger.info(
            f"Spooled capture {capture_id} "
            f"({width}x{height}, {len(data) / 1024:.1f} KB)"
        )
        return self.spool_image_path(capture_id), metadata

    def save_test(self, frame: np.ndarray, capture_id: str) -> tuple[Path, int, int, int]:
        """
        Encode + save a frame into a throwaway `test/` dir (NOT the spool, so
        it is never uploaded). Used by /test-camera to verify the camera path
        end-to-end without touching downstream. Returns (path, bytes, w, h).
        """
        data = _encode_jpeg(frame)
        path = settings.data_root / "test" / f"{capture_id}.jpg"
        _atomic_write_bytes(path, data)
        height, width = frame.shape[:2]
        return path, len(data), int(width), int(height)

    # ---- Read (uploader) ------------------------------------------------

    def iter_pending(self) -> Iterator[str]:
        """
        Yield capture_ids ready to upload: a `<id>.jpg` with a sibling
        `<id>.json`. `failed/` is a subdirectory so its entries are never
        matched by the top-level glob.
        """
        try:
            entries = sorted(self.spool_dir.glob("*.jpg"))
        except FileNotFoundError:
            return
        for jpg in entries:
            capture_id = jpg.stem
            if self.spool_metadata_path(capture_id).exists():
                yield capture_id

    def read_entry(self, capture_id: str) -> tuple[bytes, str]:
        """Return (image_bytes, metadata_json) for a spooled entry."""
        image_bytes = self.spool_image_path(capture_id).read_bytes()
        metadata_json = self.spool_metadata_path(capture_id).read_text("utf-8")
        return image_bytes, metadata_json

    def pending_count(self) -> int:
        return sum(1 for _ in self.iter_pending())

    def failed_count(self) -> int:
        try:
            return sum(1 for _ in self.failed_dir.glob("*.jpg"))
        except FileNotFoundError:
            return 0

    # ---- Lifecycle ------------------------------------------------------

    def mark_uploaded(self, capture_id: str) -> None:
        """Delete the spooled image + sidecar after a successful upload."""
        for path in (
            self.spool_image_path(capture_id),
            self.spool_metadata_path(capture_id),
        ):
            try:
                path.unlink()
            except OSError:
                pass

    def move_to_failed(self, capture_id: str) -> None:
        """Park a poison entry (laptop rejected it with a 4xx) for inspection."""
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.spool_image_path(capture_id),
            self.spool_metadata_path(capture_id),
        ):
            if path.exists():
                try:
                    os.replace(path, self.failed_dir / path.name)
                except OSError as e:
                    logger.warning(f"Could not move {path} to failed/: {e}")


# Global instance — imported by routes / uploader.
storage_service = StorageService()
