"""
Label Studio webhook handler.

Configured in the LS UI under Project → Webhooks:
    URL:     http://localhost:8001/api/v1/webhook/annotation-created
    Events:  Annotation created, Annotation updated

We accept the raw request body and dig out what we need (action,
annotation, task, capture_id) defensively, because Label Studio's webhook
payload shape varies between versions and event types. Strict Pydantic
parsing was rejecting valid payloads with 422 before our handler could
log anything.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from config.settings import settings
from models.schemas import CaptureMetadata
from services.mirror_service import mirror_service
from services.storage_service import storage_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

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
      - payload["task"]["meta"]["capture_id"] (what we set at upload time)
      - payload["task"]["data"]["image"] filename (parse from local-files URL)
    """
    cap = _dig(payload, "task", "meta", "capture_id")
    if isinstance(cap, str) and cap:
        return cap

    image_url = _dig(payload, "task", "data", "image")
    if isinstance(image_url, str) and image_url:
        # /data/local-files/?d=unlabeled/<capture_id>.jpg
        # take the part after the final slash, strip extension
        tail = image_url.rsplit("/", 1)[-1]
        stem = tail.split("?", 1)[0]
        if stem.lower().endswith(".jpg"):
            return stem[:-4]
    return None


@router.post("/annotation-created")
async def annotation_created(request: Request) -> dict:
    """Handle Label Studio ANNOTATION_CREATED / ANNOTATION_UPDATED."""
    try:
        payload = await request.json()
    except Exception as e:
        logger.warning(f"Webhook body is not JSON: {e}")
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")

    action = payload.get("action") if isinstance(payload, dict) else None
    task_id = _dig(payload, "task", "id")
    annotation_id = _dig(payload, "annotation", "id")
    logger.info(
        f"Webhook received: action={action} task={task_id} "
        f"annotation={annotation_id}"
    )
    logger.debug(f"webhook payload keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}")

    capture_id = _extract_capture_id(payload)
    if not capture_id:
        logger.warning(
            f"Webhook task {task_id} has no capture_id (meta or filename). "
            f"task.meta = {_dig(payload, 'task', 'meta')!r} "
            f"task.data = {_dig(payload, 'task', 'data')!r}"
        )
        raise HTTPException(
            status_code=422,
            detail="capture_id missing in task.meta and not derivable from image URL",
        )

    unlabeled_path = storage_service.unlabeled_image_path(capture_id)
    if not unlabeled_path.exists():
        logger.error(f"Unlabeled image missing for {capture_id}: {unlabeled_path}")
        raise HTTPException(
            status_code=404,
            detail=f"unlabeled image not found on disk: {unlabeled_path.name}",
        )
    image_bytes = unlabeled_path.read_bytes()

    original_metadata: CaptureMetadata | None = storage_service.load_unlabeled_metadata(
        capture_id
    )

    annotation_dict = payload.get("annotation") or {}
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
