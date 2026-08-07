"""
HTTP routes for camerapi.

camerapi is a thin capture node: grab → write to the local spool → return.
The background spool_uploader ships spooled captures to the ingestion hub;
their outcome shows up in the camerapi logs (look for "[<capture_id>]").

Triggers:
  - Production: USB keypad → scripts/usb_button_listener.py → POST /api/v1/capture
                (or the dashboard "Capture" button, which adds X-Triggered-By)
  - Testing:    curl -X POST http://localhost:8001/api/v1/capture
"""
from __future__ import annotations

from typing import Optional

import requests
from fastapi import APIRouter, Header, HTTPException

from config.settings import settings
from models.schemas import CaptureResult, StepResult
from services.camera_service import camera_service
from services.storage_service import new_capture_id, storage_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

router = APIRouter(prefix="/api/v1", tags=["camera"])


def _ask_dashboard_for_current_user() -> Optional[str]:
    """
    Best-effort identity probe used when the request has no X-Triggered-By
    (e.g. the USB button listener posted it). Never raises — returns None
    on any failure so the capture flow stays unblocked.
    """
    url = f"{settings.dashboard_url.rstrip('/')}/api/current-user"
    try:
        resp = requests.get(url, timeout=settings.dashboard_timeout)
        if resp.status_code != 200:
            return None
        email = resp.json().get("email")
        if isinstance(email, str) and email:
            return email
    except Exception as e:
        logger.debug(f"dashboard current-user probe failed: {e}")
    return None


@router.post("/capture", response_model=CaptureResult, status_code=202)
async def capture(
    x_triggered_by: Optional[str] = Header(default=None),
) -> CaptureResult:
    """
    Capture one image and write it to the local spool, then return 202.

    The background uploader ships the spooled capture to the ingestion hub
    and retries until it ACKs — so the response returns immediately and a
    receiver outage never blocks (or loses) a capture. Watch the logs for
    "[<capture_id>] ingested ok".

    Attribution lookup, in order:
      1. `X-Triggered-By` request header (the dashboard's "Capture" button
         and any explicit caller set this).
      2. The dashboard's `GET /api/current-user` endpoint (used when the
         USB button listener posts here with no header).
      3. None (button fired with no logged-in user / dashboard unreachable).
    """
    capture_id = new_capture_id()
    triggered_by = x_triggered_by or _ask_dashboard_for_current_user()
    logger.info(
        f"Capture requested: capture_id={capture_id} "
        f"triggered_by={triggered_by or '<unknown>'}"
    )

    # ---- Step 1: camera (fatal on failure) -----------------------------
    try:
        frame, source_info = camera_service.capture_frame()
    except Exception as e:
        logger.error(f"[{capture_id}] Camera capture failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"camera capture failed: {e}")

    # ---- Step 2: spool (fatal on failure) ------------------------------
    try:
        image_path, metadata = storage_service.save_to_spool(
            frame=frame,
            capture_id=capture_id,
            source=source_info.get("source", "unknown"),
            camera_serial=source_info.get("camera_serial"),
            camera_model=source_info.get("camera_model"),
            triggered_by=triggered_by,
        )
    except Exception as e:
        logger.error(f"[{capture_id}] Spooling image failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"spool failed: {e}")

    return CaptureResult(
        capture_id=capture_id,
        camera=StepResult(
            ok=True,
            detail=f"spooled {image_path.name} ({metadata.size_bytes // 1024} KB)",
        ),
        spooled=True,
        metadata=metadata,
    )


@router.get("/status")
async def get_status() -> dict:
    """Service status — camera, spool depth, and the ingest target."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "camera": camera_service.get_status(),
        "ingest": {
            "url": settings.ingest_url,
            "endpoint": settings.ingest_endpoint,
        },
        "spool": {
            "dir": str(settings.spool_dir),
            "pending": storage_service.pending_count(),
            "failed": storage_service.failed_count(),
        },
    }


@router.get("/health")
async def health() -> dict:
    """Liveness probe: returns 200 as long as the process is running."""
    return {"status": "healthy", "service": settings.service_name}


@router.post("/test-camera")
async def test_camera() -> dict:
    """
    Capture + save into a throwaway `test/` dir on the Pi — NOT the spool, so
    it is never uploaded. Use to verify the camera end-to-end without touching
    downstream.
    """
    capture_id = new_capture_id()
    logger.info(f"Camera test: capture_id={capture_id}")
    try:
        frame, source_info = camera_service.capture_frame()
        image_path, size_bytes, width, height = storage_service.save_test(
            frame, capture_id
        )
        return {
            "status": "ok",
            "capture_id": capture_id,
            "image_path": str(image_path),
            "size_kb": round(size_bytes / 1024, 1),
            "width": width,
            "height": height,
            "source": source_info.get("source", "unknown"),
            "note": "Saved to test/ only — not spooled, not uploaded.",
        }
    except Exception as e:
        logger.error(f"Camera test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"camera test failed: {e}")
