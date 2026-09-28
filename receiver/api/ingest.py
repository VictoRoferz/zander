"""
Ingestion endpoint for the receiver (laptop).

POST /api/v1/ingest — the Pi's spool uploader sends each capture here. We
store the image + metadata sidecar on the laptop and create the Label Studio
task. This is the single intake path for unlabeled captures.

Idempotency: the Pi retries until it gets an ACK, so the same capture_id may
arrive more than once. A per-capture `.task` marker (holding the LS task id)
lets a re-send short-circuit to a 200 without creating a duplicate task.

Status contract (drives the Pi's retry policy):
  200 → stored + LS task created (or already existed). Pi deletes the spool entry.
  422 → bad payload (invalid capture_id / unparseable metadata / empty image /
        sha mismatch). Poison; the Pi parks it in spool/failed/. The capture_id
        is checked before anything else: it names the files on disk.
  503 → Label Studio not reachable. Pi keeps the entry and retries later.

Predictions (docs/CONTRACTS.md section 6): once the capture is stored and its
task marker written, it is handed to the prediction worker; a re-send that is
deduped is handed over only if it has no pred.json yet. The hand-off is an
O(1) non-blocking enqueue wrapped so it can never change the ACK.
"""
from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import CaptureMetadata
from services.labelstudio_service import labelstudio_service
from services.prediction_service import CAPTURE_ID_PATTERN, prediction_service
from services.storage_service import storage_service

logger = logging.getLogger("receiver.ingest")

router = APIRouter(prefix="/api/v1", tags=["ingest"])


def _queue_prediction(capture_id: str, only_if_missing: bool = False) -> None:
    """Hand the capture to the prediction worker. Never raises, never blocks."""
    try:
        if only_if_missing and storage_service.has_prediction(capture_id):
            return
        prediction_service.enqueue(capture_id)
    except Exception as e:  # the ingest ACK must not depend on predictions
        logger.warning(f"[{capture_id}] prediction not queued: {e}")


@router.post("/ingest")
async def ingest(
    # Validated by FastAPI before this body runs (422, a validation error), so an
    # id like "../x" or "abc\n" never reaches a file path. Same rule as the
    # dashboard and the inspector; camerapi sends uuid4 hex (32 characters).
    capture_id: str = Form(..., pattern=CAPTURE_ID_PATTERN),
    metadata_json: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    # ---- Idempotent short-circuit --------------------------------------
    if storage_service.has_task_marker(capture_id):
        existing = storage_service.read_task_marker(capture_id)
        logger.info(f"[{capture_id}] already ingested (task {existing}); ack")
        _queue_prediction(capture_id, only_if_missing=True)
        return {
            "status": "ok",
            "capture_id": capture_id,
            "task_id": existing,
            "deduped": True,
        }

    # ---- Validate payload (422 = poison) -------------------------------
    try:
        metadata = CaptureMetadata.model_validate_json(metadata_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"metadata_json invalid: {e}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="empty image")

    if metadata.sha256:
        actual = hashlib.sha256(image_bytes).hexdigest()
        if actual != metadata.sha256:
            raise HTTPException(
                status_code=422,
                detail=f"sha256 mismatch (expected {metadata.sha256[:12]}…, got {actual[:12]}…)",
            )

    # ---- Label Studio must be reachable (503 = retry) ------------------
    if not labelstudio_service.ensure_initialized():
        raise HTTPException(status_code=503, detail="label studio not ready")

    # ---- Store, then create the LS task --------------------------------
    image_path = storage_service.save_unlabeled(capture_id, image_bytes, metadata)
    try:
        result = labelstudio_service.create_task_from_image(image_path, metadata)
    except Exception as e:
        logger.error(f"[{capture_id}] LS task creation failed: {e}", exc_info=True)
        # Image is safely on disk (atomic, idempotent overwrite); ask the Pi to
        # retry so the task eventually gets created.
        raise HTTPException(
            status_code=503, detail=f"label studio task creation failed: {e}"
        )

    storage_service.write_task_marker(capture_id, result["task_id"])
    logger.info(f"[{capture_id}] ingested -> LS task {result['task_id']}")
    _queue_prediction(capture_id)
    return {
        "status": "ok",
        "capture_id": capture_id,
        "task_id": result["task_id"],
        "image_path": str(image_path),
    }
