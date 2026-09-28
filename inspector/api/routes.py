"""
Inspector API (docs/CONTRACTS.md section 3).

POST /api/v1/inspect: run defect detection on one photo. Accepts a multipart
file upload (`file`) or a known capture id (`capture_id`, resolved against
DATA_ROOT/unlabeled/<id>.jpg). When both are sent (the receiver does), the
upload is inspected and the id only labels the request. Returns stitched
photo-coordinate detections plus the verdict.

Errors: 503 no model loaded, 422 missing/empty/undecodable image or invalid
capture_id, 404 unknown capture_id, 500 inference failed.

Blocking work (file read, decode, ONNX inference) runs in the threadpool so
the event loop keeps answering /status and /health during a slow photo.
"""
import logging
import re
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from config.settings import settings
from services.detector_service import (
    ImageDecodeError,
    ModelNotLoadedError,
    detector_service,
)

logger = logging.getLogger("inspector.api")

router = APIRouter(prefix="/api/v1", tags=["inspect"])

# Same id rule as the dashboard and the receiver: safe as a file name on every
# OS. Always fullmatch: with re.match, "$" also accepts a trailing newline.
CAPTURE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _no_model() -> HTTPException:
    return HTTPException(
        status_code=503, detail=f"no model loaded: {detector_service.load_error}"
    )


@router.post("/inspect")
async def inspect(
    file: Optional[UploadFile] = File(default=None),
    capture_id: Optional[str] = Form(default=None),
) -> dict:
    if not detector_service.ready:
        raise _no_model()
    if file is None and not capture_id:
        raise HTTPException(
            status_code=422, detail="provide a file upload or a capture_id"
        )
    if capture_id and not CAPTURE_ID_RE.fullmatch(capture_id):
        raise HTTPException(status_code=422, detail="invalid capture_id")

    if file is not None:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="empty upload")
        source = capture_id or file.filename or "upload"
    else:
        path = settings.unlabeled_dir / f"{capture_id}.jpg"
        try:
            image_bytes = await run_in_threadpool(path.read_bytes)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"unknown capture {capture_id}")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"cannot read capture {capture_id}: {e}")
        if not image_bytes:
            raise HTTPException(status_code=422, detail=f"capture {capture_id} is empty")
        source = capture_id

    try:
        result = await run_in_threadpool(detector_service.inspect_bytes, image_bytes)
    except ImageDecodeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelNotLoadedError:
        raise _no_model()
    except Exception as e:
        logger.exception(f"[{source}] inference failed")
        raise HTTPException(
            status_code=500, detail=f"inference failed: {type(e).__name__}: {e}"
        )
    logger.info(
        f"[{source}] {result['verdict']}: {result['count']} detection(s) "
        f"in {result['latency_ms']} ms"
    )
    return {"status": "ok", "source": source, **result}
