"""
Label Studio webhook handler.

Configured in the LS UI under Project → Webhooks:
    URL:     http://localhost:8001/api/v1/webhook/annotation-created
    Events:  Annotation created, Annotation updated

On each event we:
  1. Parse the payload (tolerating unknown fields LS may add over time).
  2. Recover `capture_id` from task.meta (set by camerapi when it uploaded).
  3. Fetch the image bytes from LS and save a labeled copy on the Pi.
  4. Mirror the labeled copy to the laptop (best-effort).

Handler is idempotent: repeated calls overwrite. LS may retry on timeouts,
and ANNOTATION_UPDATED lands here too — both should be safe.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from config.settings import settings
from models.schemas import AnnotationWebhookPayload, CaptureMetadata
from services.labelstudio_service import labelstudio_service
from services.mirror_service import mirror_service
from services.storage_service import storage_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])


@router.post("/annotation-created")
async def annotation_created(payload: AnnotationWebhookPayload) -> dict:
    """
    Handle Label Studio ANNOTATION_CREATED / ANNOTATION_UPDATED.

    Returns quickly (< 5 s) to stay within LS's webhook timeout.
    """
    logger.info(
        f"Webhook received: action={payload.action} "
        f"task={payload.task.id} annotation={payload.annotation.id}"
    )

    capture_id = payload.task.meta.get("capture_id") if payload.task.meta else None
    if not capture_id:
        logger.warning(
            f"Task {payload.task.id} has no capture_id in meta — "
            "cannot match to unlabeled image. Skipping."
        )
        raise HTTPException(
            status_code=422,
            detail="task.meta.capture_id missing; nothing to match",
        )

    # Pull image bytes from LS (LS holds the original; we never ask the
    # annotator's browser for anything).
    try:
        image_bytes = labelstudio_service.fetch_task_image_bytes(
            payload.task.model_dump()
        )
    except Exception as e:
        logger.error(f"Failed to fetch image for task {payload.task.id}: {e}")
        raise HTTPException(status_code=502, detail=f"fetch image failed: {e}")

    # Pull the original capture metadata if we still have it, so the merged
    # JSON can include camera serial, resolution, etc.
    original_metadata: CaptureMetadata | None = storage_service.load_unlabeled_metadata(
        capture_id
    )

    # Write labeled artifacts atomically.
    annotation_dict = payload.annotation.model_dump(mode="json")
    image_path, annotation_path = storage_service.save_labeled(
        capture_id=capture_id,
        image_bytes=image_bytes,
        annotation=annotation_dict,
        capture_metadata=original_metadata,
    )

    # Best-effort mirror to laptop. Log but don't fail the webhook — LS
    # shouldn't retry on laptop outages.
    mirror_ok, mirror_detail = mirror_service.mirror_labeled(
        image_path=image_path,
        annotation_path=annotation_path,
        capture_id=capture_id,
    )
    if not mirror_ok:
        logger.warning(
            f"Labeled mirror failed for {capture_id}: {mirror_detail} "
            "(Pi copy is still saved)"
        )

    return {
        "status": "ok",
        "capture_id": capture_id,
        "labeled_image": str(image_path),
        "labeled_annotation": str(annotation_path),
        "laptop_mirror": {"ok": mirror_ok, "detail": mirror_detail},
    }
