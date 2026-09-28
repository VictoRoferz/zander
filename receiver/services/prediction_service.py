"""
Prediction worker for the receiver (docs/CONTRACTS.md sections 2 and 6).

Ingest hands every stored capture to `enqueue()`, which is O(1) and never
blocks. ONE daemon thread (the camerapi spool_uploader pattern) takes ids off
a bounded, id-deduplicated queue and runs `process_one`, which never raises:

  1. GET  <INSPECTOR_URL>/api/v1/status
       unreachable         -> pred.json "error" ("inspector unreachable")
       model_loaded false  -> "no_model" (error = the inspector's load_error)
  2. an existing "ok" from the same model_version -> skip (unless force)
  3. POST <INSPECTOR_URL>/api/v1/inspect with the stored photo + capture_id
       200 -> validated "ok" document (confidence_threshold -> threshold)
       503 -> "no_model";  any other status or a timeout -> "error"
  4. an existing "ok" is never replaced by a non-ok result

Results go to DATA_ROOT/predictions/<capture_id>.pred.json through
storage_service (atomic, 0644). The receiver is the only writer; "pending" is
never written (the dashboard derives it from a missing file).

Blocking `requests` calls run on the worker thread only, so neither the event
loop nor the ingest ACK ever waits for the inspector. The HTTP client is
injectable (`http=`) for tests.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from pydantic import ValidationError

from config.settings import settings
from models.schemas import PredictionDoc
from services.storage_service import storage_service

logger = logging.getLogger("receiver.predictions")

# Same id rule as the dashboard and the inspector: safe as a file name on every OS.
CAPTURE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_CAPTURE_ID_RE = re.compile(CAPTURE_ID_PATTERN)


def is_valid_capture_id(capture_id: Any) -> bool:
    return isinstance(capture_id, str) and bool(_CAPTURE_ID_RE.match(capture_id))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_object(resp: Any) -> Optional[dict]:
    try:
        body = resp.json()
    except ValueError:  # requests' JSONDecodeError is a ValueError
        return None
    return body if isinstance(body, dict) else None


def _detail(resp: Any) -> str:
    """Best-effort error text of an HTTP response (FastAPI puts it in `detail`)."""
    body = _json_object(resp)
    if body is not None and body.get("detail") is not None:
        detail = body["detail"]
        return detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    text = (getattr(resp, "text", "") or "").strip()
    return text[:200] or "no detail"


def _validation_summary(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts)


class PredictionService:
    def __init__(self, http: Any = None, queue_max: Optional[int] = None) -> None:
        # Anything with requests' get/post signature; the module by default.
        self.http = http if http is not None else requests
        self._queue: "queue.Queue[str]" = queue.Queue(
            maxsize=queue_max if queue_max is not None else settings.prediction_queue_max
        )
        self._pending: dict[str, bool] = {}  # queued capture_id -> force
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return bool(settings.inspector_url)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ---- Lifecycle ------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            logger.info("predictions off (INSPECTOR_URL is empty)")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="prediction-worker", daemon=True
        )
        self._thread.start()
        logger.info(f"prediction worker started -> {settings.inspector_url}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            # An /inspect call in flight may outlive this; the thread is a daemon.
            self._thread.join(timeout=5)
            self._thread = None
            logger.info("prediction worker stopped")

    # ---- Queue ----------------------------------------------------------

    def enqueue(self, capture_id: str, force: bool = False) -> bool:
        """Queue one capture. Never blocks. False = not queued: predictions off,
        invalid id, queue full, or already queued (a queued run is upgraded to
        forced when `force` is set, which counts as queued)."""
        if not self.enabled:
            return False
        if not is_valid_capture_id(capture_id):
            logger.warning(f"prediction not queued: invalid capture id {capture_id!a}")
            return False
        with self._pending_lock:
            if capture_id in self._pending:
                if force and not self._pending[capture_id]:
                    self._pending[capture_id] = True
                    return True
                return False
            try:
                self._queue.put_nowait(capture_id)
            except queue.Full:
                logger.warning(
                    f"[{capture_id}] prediction queue full ({self._queue.maxsize}); not queued"
                )
                return False
            self._pending[capture_id] = force
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                capture_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._pending_lock:
                force = self._pending.pop(capture_id, False)
            try:
                self.process_one(capture_id, force=force)
            except Exception:  # process_one never raises; keep the thread alive regardless
                logger.exception(f"[{capture_id}] prediction worker error")
            finally:
                self._queue.task_done()

    # ---- One capture ----------------------------------------------------

    def process_one(self, capture_id: str, force: bool = False) -> str:
        """Predict one capture and store its pred.json. Never raises.

        Returns the outcome: "ok" | "no_model" | "error" (file written),
        "skipped" (an ok from the same model exists), "kept" (a non-ok result
        did not replace an existing ok), "disabled" or "invalid" (bad id).
        """
        if not self.enabled:
            return "disabled"
        if not is_valid_capture_id(capture_id):
            logger.warning(f"prediction skipped: invalid capture id {capture_id!a}")
            return "invalid"
        try:
            return self._process(capture_id, force)
        except Exception as e:
            logger.exception(f"[{capture_id}] prediction failed unexpectedly")
            try:
                return self._store(
                    capture_id,
                    self._failure(capture_id, "error", f"receiver error: {type(e).__name__}: {e}"),
                    storage_service.read_prediction(capture_id),
                )
            except Exception:
                logger.exception(f"[{capture_id}] could not record the prediction failure")
                return "error"

    def _process(self, capture_id: str, force: bool) -> str:
        base = settings.inspector_url
        existing = storage_service.read_prediction(capture_id)

        # 1. Is the inspector up, and which model does it serve?
        try:
            resp = self.http.get(
                f"{base}/api/v1/status", timeout=settings.inspector_status_timeout
            )
        except requests.RequestException as e:
            logger.warning(f"[{capture_id}] inspector unreachable at {base}: {type(e).__name__}")
            return self._store(
                capture_id, self._failure(capture_id, "error", "inspector unreachable"), existing
            )
        info = _json_object(resp)
        if resp.status_code != 200 or info is None:
            return self._store(
                capture_id,
                self._failure(
                    capture_id, "error",
                    f"inspector status HTTP {resp.status_code}: {_detail(resp)}",
                ),
                existing,
            )
        if not info.get("model_loaded"):
            return self._store(
                capture_id,
                self._failure(capture_id, "no_model", info.get("load_error") or "no model loaded"),
                existing,
            )

        # 2. Idempotent on model_version.
        if (
            not force
            and existing is not None
            and existing.get("status") == "ok"
            and existing.get("model_version") == info.get("model_version")
        ):
            logger.info(
                f"[{capture_id}] prediction up to date ({existing.get('model_version')}); skipped"
            )
            return "skipped"

        # 3. Inspect the stored photo.
        try:
            image = storage_service.unlabeled_image_path(capture_id).read_bytes()
        except FileNotFoundError:
            return self._store(
                capture_id,
                self._failure(capture_id, "error", f"image not found: unlabeled/{capture_id}.jpg"),
                existing,
            )
        try:
            resp = self.http.post(
                f"{base}/api/v1/inspect",
                files={"file": (f"{capture_id}.jpg", image, "image/jpeg")},
                data={"capture_id": capture_id},
                timeout=settings.inspector_timeout,
            )
        except requests.Timeout:
            doc = self._failure(
                capture_id, "error",
                f"inspector timed out after {settings.inspector_timeout:g} s",
            )
        except requests.RequestException as e:
            doc = self._failure(
                capture_id, "error", f"inspector request failed: {type(e).__name__}: {e}"
            )
        else:
            if resp.status_code == 200:
                doc = self._ok_doc(capture_id, resp)
            elif resp.status_code == 503:
                doc = self._failure(capture_id, "no_model", _detail(resp))
            else:
                doc = self._failure(
                    capture_id, "error", f"HTTP {resp.status_code}: {_detail(resp)}"
                )
        return self._store(capture_id, doc, existing, force=force)

    # ---- Documents ------------------------------------------------------

    @staticmethod
    def _failure(capture_id: str, status: str, error: str) -> PredictionDoc:
        return PredictionDoc(
            capture_id=capture_id, status=status, created_at=_utc_now(), error=error
        )

    def _ok_doc(self, capture_id: str, resp: Any) -> PredictionDoc:
        body = _json_object(resp)
        if body is None:
            return self._failure(
                capture_id, "error", "invalid inspector response: not a JSON object"
            )
        try:
            return PredictionDoc.model_validate({
                "capture_id": capture_id,
                "status": "ok",
                "created_at": _utc_now(),
                "error": None,
                "model_version": body.get("model_version"),
                "model_sha256": body.get("model_sha256"),
                "schema": body.get("schema"),
                "tiler_version": body.get("tiler_version"),
                "class_names": body.get("class_names"),
                "threshold": body.get("confidence_threshold"),
                "threshold_source": body.get("threshold_source"),
                "image_width": body.get("image_width"),
                "image_height": body.get("image_height"),
                "verdict": body.get("verdict"),
                "classes": body.get("classes") or [],
                "photo_score": body.get("photo_score"),
                "detections": body.get("detections") or [],
                "latency_ms": body.get("latency_ms"),
            })
        except ValidationError as e:
            return self._failure(
                capture_id, "error", f"invalid inspector response: {_validation_summary(e)}"
            )

    def _store(
        self,
        capture_id: str,
        doc: PredictionDoc,
        existing: Optional[dict],
        force: bool = False,
    ) -> str:
        if existing is not None and existing.get("status") == "ok":
            if doc.status != "ok":
                logger.warning(
                    f"[{capture_id}] prediction {doc.status} ({doc.error!a}); "
                    "keeping the existing ok result"
                )
                return "kept"
            if not force and existing.get("model_version") == doc.model_version:
                return "skipped"  # an ok is only replaced by an ok from another model
        storage_service.write_prediction(capture_id, doc.model_dump(mode="json", by_alias=True))
        if doc.status == "ok":
            logger.info(
                f"[{capture_id}] prediction ok: {doc.verdict}, {len(doc.detections)} "
                f"defect(s), {doc.model_version}, {doc.latency_ms} ms"
            )
        elif doc.status == "no_model":
            logger.info(f"[{capture_id}] prediction no_model: {doc.error!a}")
        else:
            logger.warning(f"[{capture_id}] prediction error: {doc.error!a}")
        return doc.status


# Global service instance.
prediction_service = PredictionService()
