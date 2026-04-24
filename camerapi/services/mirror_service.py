"""
Laptop mirror service.

Sends a copy of each capture (and each labeled result) to a fixed laptop
receiver over HTTP. Failure to reach the laptop is logged but never
bubbles up to fail the capture flow — the Pi is authoritative storage.

Enabled/disabled by `LAPTOP_MIRROR_ENABLED` in .env.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import requests

from config.settings import settings
from models.schemas import CaptureMetadata
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


class MirrorService:
    """HTTP client that pushes artifacts to the laptop receiver."""

    def __init__(self) -> None:
        self.enabled: bool = settings.laptop_mirror_enabled
        self.unlabeled_url: str = settings.laptop_unlabeled_url
        self.labeled_url: str = settings.laptop_labeled_url
        self.timeout: int = settings.laptop_mirror_timeout
        self.retries: int = settings.laptop_mirror_retries
        logger.info(
            f"MirrorService initialized: enabled={self.enabled}, "
            f"unlabeled={self.unlabeled_url}, labeled={self.labeled_url}"
        )

    # ---- Unlabeled ------------------------------------------------------

    def mirror_unlabeled(
        self,
        image_path: Path,
        metadata: CaptureMetadata,
    ) -> tuple[bool, Optional[str]]:
        """
        Upload unlabeled image + metadata to the laptop.
        Returns (ok, detail).
        """
        if not self.enabled:
            return True, "mirror disabled"

        metadata_json = metadata.model_dump_json()
        with image_path.open("rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            data = {
                "capture_id": metadata.capture_id,
                "metadata_json": metadata_json,
            }
            return self._post_with_retry(self.unlabeled_url, files, data)

    # ---- Labeled --------------------------------------------------------

    def mirror_labeled(
        self,
        image_path: Path,
        annotation_path: Path,
        capture_id: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Upload labeled image + annotation JSON to the laptop.
        Returns (ok, detail).
        """
        if not self.enabled:
            return True, "mirror disabled"

        with image_path.open("rb") as img, annotation_path.open("rb") as ann:
            files = {
                "file": (image_path.name, img, "image/jpeg"),
                "annotation": (annotation_path.name, ann, "application/json"),
            }
            data = {"capture_id": capture_id}
            return self._post_with_retry(self.labeled_url, files, data)

    # ---- Internals ------------------------------------------------------

    def _post_with_retry(
        self,
        url: str,
        files: dict[str, Any],
        data: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        POST with simple exponential backoff. Never raises — returns (ok, detail).
        """
        last_error: Optional[str] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    url, files=files, data=data, timeout=self.timeout
                )
                if 200 <= resp.status_code < 300:
                    return True, f"HTTP {resp.status_code}"
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"

            # Rewind file-like objects if retrying
            for key, fileinfo in files.items():
                fileobj = fileinfo[1] if isinstance(fileinfo, tuple) else fileinfo
                if hasattr(fileobj, "seek"):
                    try:
                        fileobj.seek(0)
                    except Exception:
                        pass

            if attempt < self.retries:
                delay = 1.0 * (2 ** (attempt - 1))
                logger.warning(
                    f"Mirror to {url} failed (attempt {attempt}/{self.retries}): "
                    f"{last_error}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        logger.error(f"Mirror to {url} failed after {self.retries} attempts: {last_error}")
        return False, last_error


mirror_service = MirrorService()
