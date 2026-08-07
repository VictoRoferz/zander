"""
Background spool uploader for camerapi.

Ships spooled captures to the laptop ingestion hub (receiver /api/v1/ingest),
retrying until the laptop ACKs. Replaces the old best-effort mirror_service:
there is exactly one upload target now, and nothing is dropped on a transient
failure.

Runs as a daemon thread started/stopped by main.lifespan. Uses the blocking
`requests` library on its own thread so it never blocks the event loop.

Retry policy per entry:
  - 2xx or 409 (already ingested) -> delete from spool (success / idempotent).
  - other 4xx                     -> poison; move to spool/failed/ (never retried).
  - 5xx / connection error        -> keep; exponential backoff, capped.
A Pi reboot loses only the in-memory backoff timers; the first loop iteration
re-scans the spool and re-uploads everything still there.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import requests

from config.settings import settings
from services.storage_service import storage_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


class SpoolUploader:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Per-capture_id next-eligible-attempt (monotonic) + current backoff.
        # In-memory only; reset on process restart (fine — we just retry sooner).
        self._next_attempt: Dict[str, float] = {}
        self._backoff: Dict[str, float] = {}

    # ---- Lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="spool-uploader", daemon=True
        )
        self._thread.start()
        logger.info(f"Spool uploader started -> {settings.ingest_endpoint}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Spool uploader stopped")

    # ---- Loop -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain_once()
            except Exception as e:
                logger.error(f"Uploader loop error: {e}", exc_info=True)
            self._stop.wait(settings.upload_poll_interval)

    def _drain_once(self) -> None:
        now = time.monotonic()
        for capture_id in storage_service.iter_pending():
            if self._stop.is_set():
                return
            if now < self._next_attempt.get(capture_id, 0.0):
                continue
            self._upload(capture_id)

    def _upload(self, capture_id: str) -> None:
        try:
            image_bytes, metadata_json = storage_service.read_entry(capture_id)
        except FileNotFoundError:
            return  # entry vanished (already uploaded / moved); nothing to do

        files = {"file": (f"{capture_id}.jpg", image_bytes, "image/jpeg")}
        data = {"capture_id": capture_id, "metadata_json": metadata_json}
        try:
            resp = requests.post(
                settings.ingest_endpoint,
                files=files,
                data=data,
                timeout=settings.upload_timeout,
            )
        except requests.RequestException as e:
            self._defer(capture_id, f"{type(e).__name__}: {e}")
            return

        if resp.status_code in (200, 201, 409):
            storage_service.mark_uploaded(capture_id)
            self._clear(capture_id)
            logger.info(f"[{capture_id}] ingested ok (HTTP {resp.status_code})")
        elif 400 <= resp.status_code < 500:
            logger.error(
                f"[{capture_id}] poison (HTTP {resp.status_code}: "
                f"{resp.text[:200]}); moving to spool/failed/"
            )
            storage_service.move_to_failed(capture_id)
            self._clear(capture_id)
        else:
            self._defer(capture_id, f"HTTP {resp.status_code}: {resp.text[:200]}")

    # ---- Backoff bookkeeping -------------------------------------------

    def _defer(self, capture_id: str, reason: str) -> None:
        prev = self._backoff.get(capture_id, 0.0)
        delay = min(settings.upload_max_backoff, prev * 2 if prev else 1.0)
        self._backoff[capture_id] = delay
        self._next_attempt[capture_id] = time.monotonic() + delay
        logger.warning(
            f"[{capture_id}] ingest failed ({reason}); retrying in {delay:.1f}s"
        )

    def _clear(self, capture_id: str) -> None:
        self._next_attempt.pop(capture_id, None)
        self._backoff.pop(capture_id, None)


spool_uploader = SpoolUploader()
