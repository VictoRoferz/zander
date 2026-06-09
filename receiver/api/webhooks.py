"""
Label Studio webhook handler (runs on the laptop, next to LS).

Configured in the LS UI under Project → Webhooks:
    URL:     http://localhost:8002/api/v1/webhook/annotation-created
    Events:  Annotation created, Annotation updated

Role: this is the durable **labeled-dataset export** sink. It copies the
labeled image + annotation into data_root/labeled/ for downstream CV training.
The dashboard does NOT depend on this — it reads label state live from the LS
API — so a missed webhook only delays the export, never the dashboard view.

We ack in <100 ms and do the file work in a FastAPI background task (LS's
webhook timeout defaults to ~1s).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

import logging

from models.schemas import CaptureMetadata
from services.storage_service import storage_service

logger = logging.getLogger("receiver.webhook")

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])


def _dig(obj: Any, *keys: str) -> Any:
    """Walk nested dicts safely; return None if any key is missing."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_capture_id(payload: dict[str, Any]) -> Optional[str]:
    """
    Find capture_id in the webhook payload. Two known locations:
      - payload["task"]["meta"]["capture_id"] (what we set at task creation)
      - payload["task"]["data"]["image"] filename (parse from local-files URL)
    """
    cap = _dig(payload, "task", "meta", "capture_id")
    if isinstance(cap, str) and cap:
        return cap

    image_url = _dig(payload, "task", "data", "image")
    if isinstance(image_url, str) and image_url:
        # /data/local-files/?d=unlabeled/<capture_id>.jpg
        tail = image_url.rsplit("/", 1)[-1]
        stem = tail.split("?", 1)[0]
        if stem.lower().endswith(".jpg"):
            return stem[:-4]
    return None


def _process_annotation(capture_id: str, annotation: dict[str, Any]) -> None:
    """Heavy work, runs in a background task after the webhook responds."""
    try:
        unlabeled_path = storage_service.unlabeled_image_path(capture_id)
        if not unlabeled_path.exists():
            logger.error(
                f"[bg] Unlabeled image missing for {capture_id}: {unlabeled_path}"
            )
            return
        image_bytes = unlabeled_path.read_bytes()

        original_metadata: CaptureMetadata | None = (
            storage_service.load_unlabeled_metadata(capture_id)
        )
        storage_service.save_labeled(
            capture_id=capture_id,
            image_bytes=image_bytes,
            annotation=annotation,
            capture_metadata=original_metadata,
        )
        logger.info(f"[bg] Saved labeled export {capture_id}")
    except Exception as e:
        logger.error(
            f"[bg] Unhandled error exporting labeled {capture_id}: {e}",
            exc_info=True,
        )


@router.post("/annotation-created")
async def annotation_created(request: Request, background: BackgroundTasks) -> dict:
    """Acknowledge fast and queue the labeled-export work as a background task."""
    try:
        payload = await request.json()
    except Exception as e:
        logger.warning(f"Webhook body is not JSON: {e}")
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")

    action = payload.get("action") if isinstance(payload, dict) else None
    task_id = _dig(payload, "task", "id")
    annotation_id = _dig(payload, "annotation", "id")
    logger.info(
        f"Webhook received: action={action} task={task_id} annotation={annotation_id}"
    )

    capture_id = _extract_capture_id(payload)
    if not capture_id:
        logger.warning(
            f"Webhook task {task_id} has no capture_id (meta or filename); ignoring"
        )
        return {"status": "ignored", "reason": "no capture_id"}

    annotation_dict = payload.get("annotation") or {}
    background.add_task(_process_annotation, capture_id, annotation_dict)

    return {
        "status": "queued",
        "capture_id": capture_id,
        "task_id": task_id,
        "annotation_id": annotation_id,
    }
