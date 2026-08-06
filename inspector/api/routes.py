"""
Inspector API.

POST /api/v1/inspect — run defect detection on one photo. Accepts either a
multipart file upload (`file`) or a known capture id (`capture_id`, resolved
against DATA_ROOT/unlabeled/). Returns stitched photo-coordinate detections.
"""
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config.settings import settings
from services.detector_service import detector_service

logger = logging.getLogger("inspector.api")

router = APIRouter(prefix="/api/v1", tags=["inspect"])


@router.post("/inspect")
async def inspect(
    file: Optional[UploadFile] = File(default=None),
    capture_id: Optional[str] = Form(default=None),
) -> dict:
    if not detector_service.ready:
        raise HTTPException(
            status_code=503,
            detail=f"no model loaded (expected at {settings.model_path})",
        )
    if file is None and not capture_id:
        raise HTTPException(
            status_code=422, detail="provide a file upload or a capture_id"
        )
    if file is not None:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="empty upload")
        source = file.filename or "upload"
    else:
        if "/" in capture_id or "\\" in capture_id:
            raise HTTPException(status_code=422, detail="invalid capture_id")
        path = settings.unlabeled_dir / f"{capture_id}.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"unknown capture {capture_id}")
        image_bytes = path.read_bytes()
        source = capture_id

    result = detector_service.inspect_bytes(image_bytes)
    logger.info(
        f"[{source}] {result['count']} detection(s) in {result['latency_ms']} ms"
    )
    return {"status": "ok", "source": source, **result}
