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
        self.unlabeled_urls: list[str] = settings.laptop_unlabeled_urls
        self.labeled_urls: list[str] = settings.laptop_labeled_urls
        self.timeout: int = settings.laptop_mirror_timeout
        self.retries: int = settings.laptop_mirror_retries
        logger.info(
            f"MirrorService initialized: enabled={self.enabled}, "
            f"targets={settings.laptop_mirror_base_urls}"
        )

    # ---- Unlabeled ------------------------------------------------------

    def mirror_unlabeled(
        self,
        image_path: Path,
        metadata: CaptureMetadata,
    ) -> tuple[bool, Optional[str]]:
        """
        Upload unlabeled image + metadata to every configured receiver.
        Returns (ok, detail) aggregated across all targets.
        """
        if not self.enabled:
            return True, "mirror disabled"

        metadata_json = metadata.model_dump_json()
        # Read once; reuse the same bytes for every target (and for retries).
        image_bytes = image_path.read_bytes()

        def build():
            files = {"file": (image_path.name, image_bytes, "image/jpeg")}
            data = {
                "capture_id": metadata.capture_id,
                "metadata_json": metadata_json,
            }
            return files, data

        return self._fan_out(self.unlabeled_urls, build)

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

        image_bytes = image_path.read_bytes()
        annotation_bytes = annotation_path.read_bytes()

        def build():
            files = {
                "file": (image_path.name, image_bytes, "image/jpeg"),
                "annotation": (annotation_path.name, annotation_bytes, "application/json"),
            }
            data = {"capture_id": capture_id}
            return files, data

        return self._fan_out(self.labeled_urls, build)

    # ---- Internals ------------------------------------------------------

    def _fan_out(self, urls, build) -> tuple[bool, Optional[str]]:
        """
        POST to every target URL (each with its own retry budget).

        `build` returns a fresh (files, data) pair per target so file payloads
        aren't consumed across requests. Best-effort: an unreachable target is
        logged but doesn't sink the others. Overall ok = at least one target
        succeeded (so a single online receiver still counts as success even
        when the other machine is offline).
        """
        if not urls:
            return True, "no mirror targets configured"

        successes: list[str] = []
        failures: list[str] = []
        for url in urls:
            files, data = build()
            ok, detail = self._post_with_retry(url, files, data)
            if ok:
                successes.append(f"{url} -> {detail}")
            else:
                failures.append(f"{url} -> {detail}")

        summary = f"{len(successes)}/{len(urls)} ok"
        if failures:
            summary += f"; failed: {'; '.join(failures)}"
        # ok if at least one receiver got it.
        return (len(successes) > 0), summary

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
