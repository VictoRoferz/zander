"""
Label Studio webhook handler.

Configured in the LS UI under Project → Webhooks:
    URL:     http://localhost:8001/api/v1/webhook/annotation-created
    Events:  Annotation created, Annotation updated

On each event we:
  1. Parse the payload (tolerating unknown fields LS may add over time).
  2. Recover `capture_id` from task.meta (set by camerapi when it created
     the task).
  3. Read the original image from disk (unlabeled_dir/<capture_id>.jpg).
     No HTTP fetch — LS and camerapi share the same filesystem.
  4. Save a labeled copy + annotation JSON on the Pi.
  5. Mirror the labeled copy to the laptop (best-effort).

Handler is idempotent: repeated calls overwrite. LS may retry on timeouts,
and ANNOTATION_UPDATED lands here too — both are safe.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from config.settings import settings
from models.schemas import AnnotationWebhookPayload, CaptureMetadata
from services.mirror_service import mirror_service
from services.storage_service import storage_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])


@router.post("/annotation-created")
async def annotation_created(payload: AnnotationWebhookPayload) -> dict:
    """Handle Label Studio ANNOTATION_CREATED / ANNOTATION_UPDATED."""
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

    # Read image bytes directly from the shared filesystem. LS was pointed
    # at the same folder via local-files storage, so this is the same file.
    unlabeled_path = storage_service.unlabeled_image_path(capture_id)
    if not unlabeled_path.exists():
        logger.error(f"Unlabeled image missing for {capture_id}: {unlabeled_path}")
        raise HTTPException(
            status_code=404,
            detail=f"unlabeled image not found on disk: {unlabeled_path.name}",
        )
    image_bytes = unlabeled_path.read_bytes()

    # Load original capture metadata so labeled JSON preserves camera info.
    original_metadata: CaptureMetadata | None = storage_service.load_unlabeled_metadata(
        capture_id
    )

    annotation_dict = payload.annotation.model_dump(mode="json")
    image_path, annotation_path = storage_service.save_labeled(
        capture_id=capture_id,
        image_bytes=image_bytes,
        annotation=annotation_dict,
        capture_metadata=original_metadata,
    )

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
