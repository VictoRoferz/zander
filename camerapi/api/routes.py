"""
HTTP routes for camerapi.

Orchestration pattern: capture → store → LS task → laptop mirror.
Each step's result is reported independently (fault isolation).

Triggers:
  - Production: GPIO button → button_listener.py → POST /api/v1/capture
  - Testing:    curl -X POST http://<pi>:8001/api/v1/capture
"""
from __future__ import annotations

from typing import Optional

import requests
from fastapi import APIRouter, Header, HTTPException

from config.settings import settings
from models.schemas import CaptureResult, StepResult
from services.camera_service import camera_service
from services.labelstudio_service import labelstudio_service
from services.mirror_service import mirror_service
from services.storage_service import new_capture_id, storage_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

router = APIRouter(prefix="/api/v1", tags=["camera"])


def _ask_dashboard_for_current_user() -> Optional[str]:
    """
    Best-effort identity probe used when the request has no X-Triggered-By
    (e.g. the GPIO button posted from the Pi). Never raises — returns None
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


@router.post("/capture", response_model=CaptureResult)
async def capture(
    x_triggered_by: Optional[str] = Header(default=None),
) -> CaptureResult:
    """
    Capture one image, store it, send to Label Studio, mirror to laptop.

    Attribution lookup, in order:
      1. `X-Triggered-By` request header (the dashboard's "Capture" button
         and any explicit caller set this).
      2. The dashboard's `GET /api/current-user` endpoint (used when the
         GPIO button posts here directly with no header).
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

    try:
        image_path, metadata = storage_service.save_unlabeled(
            frame=frame,
            capture_id=capture_id,
            source=source_info.get("source", "unknown"),
            camera_serial=source_info.get("camera_serial"),
            camera_model=source_info.get("camera_model"),
            triggered_by=triggered_by,
        )
    except Exception as e:
        logger.error(f"[{capture_id}] Saving image failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"save failed: {e}")

    camera_step = StepResult(
        ok=True,
        detail=f"saved {image_path.name} ({metadata.size_bytes // 1024} KB)",
    )

    # ---- Step 2: Label Studio (best-effort) ----------------------------
    try:
        ls_result = labelstudio_service.create_task_from_image(image_path, metadata)
        ls_step = StepResult(
            ok=True,
            detail=f"task_id={ls_result['task_id']} project_id={ls_result['project_id']}",
        )
    except Exception as e:
        logger.error(f"[{capture_id}] LS task creation failed: {e}", exc_info=True)
        ls_step = StepResult(ok=False, detail=f"{type(e).__name__}: {e}")

    # ---- Step 3: laptop mirror (best-effort) ---------------------------
    mirror_ok, mirror_detail = mirror_service.mirror_unlabeled(image_path, metadata)
    mirror_step = StepResult(ok=mirror_ok, detail=mirror_detail)

    return CaptureResult(
        capture_id=capture_id,
        camera=camera_step,
        label_studio=ls_step,
        laptop_mirror=mirror_step,
        metadata=metadata,
    )


@router.get("/status")
async def get_status() -> dict:
    """Service status — camera, LS, laptop mirror reachability."""
    camera_status = camera_service.get_status()
    ls_ok = labelstudio_service.is_healthy()

    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "camera": camera_status,
        "label_studio": {"url": settings.labelstudio_url, "healthy": ls_ok},
        "laptop_mirror": {
            "enabled": settings.laptop_mirror_enabled,
            "url": settings.laptop_mirror_url,
        },
        "storage": {
            "unlabeled_dir": str(settings.unlabeled_dir),
            "labeled_dir": str(settings.labeled_dir),
        },
    }


@router.get("/health")
async def health() -> dict:
    """Liveness probe: returns 200 as long as the process is running."""
    return {"status": "healthy", "service": settings.service_name}


@router.post("/test-camera")
async def test_camera() -> dict:
    """
    Capture + store on the Pi, but do NOT send to LS or laptop.
    Use to verify the camera end-to-end without touching downstream.
    """
    capture_id = new_capture_id()
    logger.info(f"Camera test: capture_id={capture_id}")
    try:
        frame, source_info = camera_service.capture_frame()
        image_path, metadata = storage_service.save_unlabeled(
            frame=frame,
            capture_id=capture_id,
            source=source_info.get("source", "unknown"),
            camera_serial=source_info.get("camera_serial"),
            camera_model=source_info.get("camera_model"),
        )
        return {
            "status": "ok",
            "capture_id": capture_id,
            "image_path": str(image_path),
            "size_kb": round(metadata.size_bytes / 1024, 1),
            "width": metadata.width,
            "height": metadata.height,
            "note": "Stored locally on Pi only — LS and laptop mirror skipped.",
        }
    except Exception as e:
        logger.error(f"Camera test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"camera test failed: {e}")
