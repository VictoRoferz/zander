"""
Manual prediction trigger (docs/CONTRACTS.md section 6, optional endpoint).

POST /api/v1/predict/{capture_id}?force=false
  Queue one stored capture for the inspector again: after an "error" or
  "no_model" prediction, or with force=true to re-run even when an "ok" from
  the same model exists. 202 {"queued": bool} (false = predictions off, queue
  full or already queued); 404 if the capture has no image; 422 for an
  invalid id.
"""
from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam

from services.prediction_service import CAPTURE_ID_PATTERN, prediction_service
from services.storage_service import storage_service

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict/{capture_id}", status_code=202)
async def predict(
    capture_id: str = PathParam(..., pattern=CAPTURE_ID_PATTERN),
    force: bool = False,
) -> dict:
    if not storage_service.unlabeled_image_path(capture_id).exists():
        raise HTTPException(status_code=404, detail=f"unknown capture {capture_id}")
    return {"queued": prediction_service.enqueue(capture_id, force=force)}
