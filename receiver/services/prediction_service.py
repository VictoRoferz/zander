"""
Prediction worker for the receiver (docs/CONTRACTS.md sections 2 and 6).

Ingest hands every stored capture to `enqueue()`, which never blocks. ONE
daemon thread (the camerapi spool_uploader pattern) takes ids from a bounded,
id-deduplicated work list and runs one inspector request at a time:

  1. GET  <INSPECTOR_URL>/api/v1/status
       unreachable / 5xx   -> pred.json "error" ("inspector unreachable", ...)
       model_loaded false  -> "no_model" (error = the inspector's load_error)
  2. an existing "ok" from the same model_version -> skip (unless force)
  3. POST <INSPECTOR_URL>/api/v1/inspect with the stored photo + capture_id
       200 -> validated "ok" document (confidence_threshold -> threshold)
       503 -> "no_model";  any other status or a timeout -> "error"
  4. an existing "ok" is never replaced by a non-ok result, and a failure
     that repeats the stored one does not rewrite the file

Transient failures are retried, and the error document stays visible in the
dashboard meanwhile:
  - the inspector did not answer GET /status (down, restarting, 5xx): the
    capture waits for it. While it is down ONE request probes it after
    PREDICTION_RETRY_INITIAL_S, doubling to PREDICTION_RETRY_MAX_S; the first
    answer releases every waiting capture. New captures are always tried at
    once (and show the error at once).
  - /inspect itself failed (timeout, connection lost, 5xx other than 503):
    that capture alone backs off on the same schedule, at most
    PREDICTION_RETRY_MAX_ATTEMPTS attempts, so a photo that keeps failing
    cannot load the inspector forever.
Order: new captures first, then retries, then rescan backfill.

`rescan()` (at startup in a background thread, and POST /api/v1/predict/rescan)
queues, as backfill, every ingested capture (task marker) that still needs a
prediction: no pred.json, a transient error, or no_model (so deploying a
model and restarting the receiver classifies older captures). Newest first by
the sidecar's captured_at; backfill never takes a new capture's place.

Results go to DATA_ROOT/predictions/<capture_id>.pred.json through
storage_service (atomic, 0644). The receiver is the only writer; "pending" is
never written (the dashboard derives it from a missing file). The work list
is in memory; the startup rescan picks up whatever a restart dropped.

Blocking `requests` calls run on the worker thread only, so neither the event
loop nor the ingest ACK ever waits for the inspector. The HTTP client is
injectable (`http=`) for tests.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, NamedTuple, Optional

import requests
from pydantic import ValidationError

from config.settings import settings
from models.schemas import PredictionDoc
from services.storage_service import storage_service

logger = logging.getLogger("receiver.predictions")

# Same id rule as the dashboard and the inspector: safe as a file name on every OS
# (camerapi's ids are uuid4 hex, 32 characters). Anchored because FastAPI's
# `pattern=` (pydantic) searches; checked here with fullmatch, because re's "$"
# also matches before a trailing newline ("abc\n" passes re.match).
CAPTURE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_CAPTURE_ID_RE = re.compile(CAPTURE_ID_PATTERN)

# Retry classes of a failed attempt.
RETRY_UNAVAILABLE = "unavailable"  # GET /status got no (usable) answer: wait for the inspector
RETRY_FAILED = "failed"            # /inspect failed for this photo: per-capture backoff, capped

UNREACHABLE = "inspector unreachable"
# The error texts this worker writes for RETRY_* failures, i.e. failures of the
# inspector rather than of the photo. A rescan re-queues captures whose
# pred.json carries one of them (tested against every retryable path).
_TRANSIENT_ERROR_PREFIXES = (
    UNREACHABLE,                 # GET /status: no answer
    "inspector status HTTP 5",   # GET /status: 5xx
    "inspector timed out",       # POST /inspect: timeout
    "inspector request failed",  # POST /inspect: connection lost mid-request
    "HTTP 5",                    # POST /inspect: 5xx other than 503 (= no_model)
)


def is_valid_capture_id(capture_id: Any) -> bool:
    return isinstance(capture_id, str) and _CAPTURE_ID_RE.fullmatch(capture_id) is not None


def _is_transient(status: Any, error: Any) -> bool:
    return status == "error" and isinstance(error, str) and error.startswith(_TRANSIENT_ERROR_PREFIXES)


def is_transient_error(doc: Optional[Mapping[str, Any]]) -> bool:
    """A stored pred.json that records a failure of the inspector (worth retrying)."""
    return doc is not None and _is_transient(doc.get("status"), doc.get("error"))


def needs_prediction(doc: Optional[Mapping[str, Any]]) -> bool:
    """The rescan / re-send rule: no (readable) pred.json, no_model, or a transient error."""
    return doc is None or doc.get("status") == "no_model" or is_transient_error(doc)


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


class _Result(NamedTuple):
    outcome: str             # what process_one returns
    retry: Optional[str]     # None, RETRY_UNAVAILABLE or RETRY_FAILED
    reached: Optional[bool]  # GET /status answered (True) / did not (False) / not asked (None)


@dataclass
class _Item:
    """One capture waiting for the worker."""

    force: bool
    backfill: bool      # queued by a rescan: runs only when no new work is waiting
    due: float          # time.monotonic(); later than now = a per-capture backoff
    seq: int            # FIFO tie-break
    attempts: int = 0   # failed attempts so far (0 = never tried)
    parked: bool = False  # waits for the inspector to answer again


class PredictionService:
    def __init__(
        self,
        http: Any = None,
        queue_max: Optional[int] = None,
        retry_initial_s: Optional[float] = None,
        retry_max_s: Optional[float] = None,
        retry_max_attempts: Optional[int] = None,
    ) -> None:
        # Anything with requests' get/post signature; the module by default.
        self.http = http if http is not None else requests
        limit = queue_max if queue_max is not None else settings.prediction_queue_max
        # At most this many new captures AND this many backfill captures are
        # tracked (0 must not mean unbounded); backfill never takes a new one's place.
        self._max = max(1, int(limit))
        self.retry_initial_s = float(
            retry_initial_s if retry_initial_s is not None else settings.prediction_retry_initial_s
        )
        self.retry_max_s = max(
            self.retry_initial_s,
            float(retry_max_s if retry_max_s is not None else settings.prediction_retry_max_s),
        )
        self.retry_max_attempts = max(1, int(
            retry_max_attempts if retry_max_attempts is not None
            else settings.prediction_retry_max_attempts
        ))
        self._cond = threading.Condition()   # guards everything below
        self._items: dict[str, _Item] = {}
        self._counts = {False: 0, True: 0}   # tracked ids by `backfill`
        self._seq = itertools.count()
        # Inspector gate: while it does not answer, waiting captures are held
        # until _probe_at; the first answer reopens it.
        self._probe_at = 0.0
        self._probe_backoff = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rescan_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(settings.inspector_url)

    def queue_depth(self) -> int:
        """Captures still waiting for a prediction (new, retrying and backfill)."""
        with self._cond:
            return len(self._items)

    def retrying(self) -> int:
        """Of queue_depth: captures that failed at least once and wait to be retried."""
        with self._cond:
            return sum(1 for item in self._items.values() if item.attempts)

    def backfill_depth(self) -> int:
        """Of queue_depth: captures queued by a rescan."""
        with self._cond:
            return self._counts[True]

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
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            # An /inspect call in flight may outlive this; the thread is a daemon.
            self._thread.join(timeout=5)
            self._thread = None
            logger.info("prediction worker stopped")

    def start_rescan(self) -> None:
        """rescan() once in a background thread (receiver startup): captures
        whose queued work a restart dropped, or that still say no_model."""
        if not self.enabled:
            return

        def run() -> None:
            try:
                self.rescan()
            except Exception:
                logger.exception("prediction rescan failed")

        threading.Thread(target=run, name="prediction-rescan", daemon=True).start()

    # ---- Queue ----------------------------------------------------------

    def enqueue(self, capture_id: str, force: bool = False) -> bool:
        """Queue one capture. Never blocks. False = not queued: predictions off,
        invalid id, queue full, or already queued (a queued run is upgraded to
        forced when `force` is set, and a capture waiting for a retry is tried
        now; both count as queued)."""
        return self._enqueue(capture_id, force, backfill=False) in ("queued", "promoted", "upgraded")

    def _enqueue(self, capture_id: str, force: bool, backfill: bool) -> str:
        if not self.enabled:
            return "disabled"
        if not is_valid_capture_id(capture_id):
            logger.warning(f"prediction not queued: invalid capture id {capture_id!a}")
            return "invalid"
        now = time.monotonic()
        with self._cond:
            item = self._items.get(capture_id)
            if item is None:
                if self._counts[backfill] >= self._max:
                    if not backfill:
                        logger.warning(
                            f"[{capture_id}] prediction queue full ({self._max}); not queued"
                        )
                    return "full"
                self._add(capture_id, _Item(force, backfill, now, next(self._seq)))
                return "queued"
            waiting = item.due > now or item.parked
            to_live = item.backfill and not backfill
            if waiting or to_live:
                # Asked again while it waits for a retry, or a rescan item is now
                # wanted by an ingest / the predict endpoint: try it now.
                if to_live:
                    self._counts[True] -= 1
                    self._counts[False] += 1
                    item.backfill = False
                item.force = item.force or force
                item.due, item.parked, item.seq = now, False, next(self._seq)
                self._cond.notify()
                return "promoted"
            if force and not item.force:
                item.force = True
                return "upgraded"
            return "duplicate"

    def _add(self, capture_id: str, item: _Item) -> None:
        """Track a new item (caller holds the lock)."""
        self._items[capture_id] = item
        self._counts[item.backfill] += 1
        self._cond.notify()

    def _select(self, now: float) -> tuple[Optional[str], float]:
        """(the next capture to run or None, when to look again). New captures
        first, then retries, then backfill; FIFO within each. While the
        inspector is unreachable only new captures run (each probes it);
        waiting ones and backfill are held until the next probe time."""
        gate_open = now >= self._probe_at
        best_id: Optional[str] = None
        best_key: Optional[tuple] = None
        wake = math.inf
        for capture_id, item in self._items.items():
            if item.due > now:
                wake = min(wake, item.due)
                continue
            if (item.parked or item.backfill) and not gate_open:
                wake = min(wake, self._probe_at)
                continue
            key = (item.backfill, item.attempts > 0, item.due, item.seq)
            if best_key is None or key < best_key:
                best_id, best_key = capture_id, key
        return best_id, wake

    def _next_job(self, timeout: float) -> Optional[tuple[str, _Item]]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._stop.is_set():
                now = time.monotonic()
                capture_id, wake = self._select(now)
                if capture_id is not None:
                    item = self._items.pop(capture_id)
                    self._counts[item.backfill] -= 1
                    return capture_id, item
                wait = min(wake, deadline) - now
                if wait <= 0:
                    return None
                self._cond.wait(wait)
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._next_job(timeout=0.5)
            if job is None:
                continue
            capture_id, item = job
            try:
                result = self._attempt(capture_id, item.force)
            except Exception:  # _attempt never raises; keep the thread alive regardless
                logger.exception(f"[{capture_id}] prediction worker error")
                result = _Result("error", None, None)
            self._settle(capture_id, item, result)

    def _settle(self, capture_id: str, item: _Item, result: _Result) -> None:
        """After one attempt: update the inspector gate, schedule a retry."""
        now = time.monotonic()
        with self._cond:
            if result.reached is True and self._probe_backoff:
                logger.info("inspector reachable again; resuming the waiting predictions")
                self._probe_at = self._probe_backoff = 0.0
            elif result.reached is False:
                if not self._probe_backoff:
                    logger.warning(
                        f"inspector unreachable at {settings.inspector_url}; predictions "
                        f"wait for it (checked again every {self.retry_initial_s:g} "
                        f"to {self.retry_max_s:g} s)"
                    )
                    self._probe_backoff = self.retry_initial_s
                elif item.attempts or item.backfill:
                    # A probe failed: back off further. (A burst of new captures
                    # does not: each of them is tried at once anyway.)
                    self._probe_backoff = min(self.retry_max_s, self._probe_backoff * 2.0)
                self._probe_at = now + self._probe_backoff

            if result.retry is None:
                if item.attempts:
                    logger.info(
                        f"[{capture_id}] prediction {result.outcome} after "
                        f"{item.attempts + 1} attempts"
                    )
                return
            attempts = item.attempts + 1
            if result.retry == RETRY_FAILED and attempts >= self.retry_max_attempts:
                logger.warning(
                    f"[{capture_id}] prediction failed {attempts} times; not retried again "
                    f"(POST /api/v1/predict/{capture_id} or /api/v1/predict/rescan re-queues it)"
                )
                return
            if self._stop.is_set():
                return  # shutting down: the startup rescan picks it up again
            current = self._items.get(capture_id)
            if current is not None:
                # Queued again while it ran (re-send, predict endpoint, rescan):
                # that entry runs; it keeps the retry count and the priority.
                current.attempts = max(current.attempts, attempts)
                current.force = current.force or item.force
                if current.backfill and not item.backfill:
                    self._counts[True] -= 1
                    self._counts[False] += 1
                    current.backfill = False
                return
            if self._counts[item.backfill] >= self._max:
                logger.warning(f"[{capture_id}] prediction queue full ({self._max}); retry dropped")
                return
            if result.retry == RETRY_UNAVAILABLE:
                retry = _Item(item.force, item.backfill, now, next(self._seq), attempts, parked=True)
                when = "when the inspector answers"
            else:
                delay = min(self.retry_max_s, self.retry_initial_s * 2.0 ** (attempts - 1))
                retry = _Item(item.force, item.backfill, now + delay, next(self._seq), attempts)
                when = f"in {delay:g} s"
            self._add(capture_id, retry)
        log = logger.info if attempts == 1 else logger.debug
        log(f"[{capture_id}] prediction attempt {attempts} failed ({result.outcome}); retrying {when}")

    # ---- Rescan ---------------------------------------------------------

    def rescan(self) -> int:
        """Queue (as backfill) every ingested capture that still needs a
        prediction: a task marker and no pred.json, a transient error or
        no_model. Newest first by the sidecar's captured_at, as many as the
        backfill has room for. A capture that waits for a retry is tried now.
        Returns how many were queued."""
        if not self.enabled:
            return 0
        with self._rescan_lock:
            candidates = [
                capture_id for capture_id in storage_service.task_marker_ids()
                if is_valid_capture_id(capture_id)
                and needs_prediction(storage_service.read_prediction(capture_id))
            ]
            candidates.sort(key=self._capture_time, reverse=True)
            queued = 0
            left = 0
            for n, capture_id in enumerate(candidates):
                outcome = self._enqueue(capture_id, False, backfill=True)
                if outcome == "full":
                    left = len(candidates) - n
                    break
                if outcome in ("queued", "promoted"):
                    queued += 1
            message = (
                f"prediction rescan: {len(candidates)} capture(s) need a prediction, "
                f"{queued} queued"
            )
            if left:
                message += (
                    f", {left} left for a later rescan (PREDICTION_QUEUE_MAX={self._max})"
                )
            logger.info(message)
            return queued

    @staticmethod
    def _capture_time(capture_id: str) -> float:
        """The sidecar's captured_at as a POSIX timestamp (naive = UTC); the
        image mtime without a usable sidecar; 0 if neither exists."""
        metadata = storage_service.load_unlabeled_metadata(capture_id)
        if metadata is not None:
            captured_at = metadata.captured_at
            if captured_at.tzinfo is None:
                captured_at = captured_at.replace(tzinfo=timezone.utc)
            try:
                return captured_at.timestamp()
            except (OverflowError, OSError, ValueError):
                pass
        try:
            return storage_service.unlabeled_image_path(capture_id).stat().st_mtime
        except OSError:
            return 0.0

    # ---- One capture ----------------------------------------------------

    def process_one(self, capture_id: str, force: bool = False) -> str:
        """Predict one capture and store its pred.json. Never raises.

        Returns the outcome: "ok" | "no_model" | "error" (file written, or an
        identical failure already stored), "skipped" (an ok from the same
        model exists), "kept" (a non-ok result did not replace an existing
        ok), "disabled" or "invalid" (bad id).
        """
        return self._attempt(capture_id, force).outcome

    def _attempt(self, capture_id: str, force: bool) -> _Result:
        if not self.enabled:
            return _Result("disabled", None, None)
        if not is_valid_capture_id(capture_id):
            logger.warning(f"prediction skipped: invalid capture id {capture_id!a}")
            return _Result("invalid", None, None)
        try:
            return self._process(capture_id, force)
        except Exception as e:
            logger.exception(f"[{capture_id}] prediction failed unexpectedly")
            try:
                outcome = self._store(
                    capture_id,
                    self._failure(capture_id, "error", f"receiver error: {type(e).__name__}: {e}"),
                    storage_service.read_prediction(capture_id),
                )
            except Exception:
                logger.exception(f"[{capture_id}] could not record the prediction failure")
                outcome = "error"
            return _Result(outcome, None, None)

    def _process(self, capture_id: str, force: bool) -> _Result:
        base = settings.inspector_url
        existing = storage_service.read_prediction(capture_id)

        # 1. Is the inspector up, and which model does it serve?
        try:
            resp = self.http.get(
                f"{base}/api/v1/status", timeout=settings.inspector_status_timeout
            )
        except requests.RequestException as e:
            logger.debug(f"[{capture_id}] inspector unreachable at {base}: {type(e).__name__}")
            doc = self._failure(capture_id, "error", UNREACHABLE)
            return _Result(self._store(capture_id, doc, existing), RETRY_UNAVAILABLE, False)
        info = _json_object(resp)
        if resp.status_code != 200 or info is None:
            doc = self._failure(
                capture_id, "error", f"inspector status HTTP {resp.status_code}: {_detail(resp)}"
            )
            if 500 <= resp.status_code < 600:  # up but not serving yet: wait for it
                return _Result(self._store(capture_id, doc, existing), RETRY_UNAVAILABLE, False)
            return _Result(self._store(capture_id, doc, existing), None, None)
        if not info.get("model_loaded"):
            doc = self._failure(capture_id, "no_model", info.get("load_error") or "no model loaded")
            return _Result(self._store(capture_id, doc, existing), None, True)

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
            return _Result("skipped", None, True)

        # 3. Inspect the stored photo.
        try:
            image = storage_service.unlabeled_image_path(capture_id).read_bytes()
        except FileNotFoundError:
            doc = self._failure(capture_id, "error", f"image not found: unlabeled/{capture_id}.jpg")
            return _Result(self._store(capture_id, doc, existing), None, True)
        retry: Optional[str] = None
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
            retry = RETRY_FAILED
        except requests.RequestException as e:
            doc = self._failure(
                capture_id, "error", f"inspector request failed: {type(e).__name__}: {e}"
            )
            retry = RETRY_FAILED
        else:
            if resp.status_code == 200:
                doc = self._ok_doc(capture_id, resp)
            elif resp.status_code == 503:
                doc = self._failure(capture_id, "no_model", _detail(resp))
            else:
                doc = self._failure(
                    capture_id, "error", f"HTTP {resp.status_code}: {_detail(resp)}"
                )
                if 500 <= resp.status_code < 600:
                    retry = RETRY_FAILED
        return _Result(self._store(capture_id, doc, existing, force=force), retry, True)

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
        if (
            doc.status != "ok"
            and existing is not None
            and existing.get("schema_version") == 1
            and existing.get("status") == doc.status
            and existing.get("error") == doc.error
        ):
            # A retry that failed exactly like the stored result: keep the file
            # (and its created_at, the first failure) instead of rewriting it.
            logger.debug(f"[{capture_id}] prediction {doc.status} unchanged ({doc.error!a})")
            return doc.status
        storage_service.write_prediction(capture_id, doc.model_dump(mode="json", by_alias=True))
        if doc.status == "ok":
            logger.info(
                f"[{capture_id}] prediction ok: {doc.verdict}, {len(doc.detections)} "
                f"defect(s), {doc.model_version}, {doc.latency_ms} ms"
            )
        elif doc.status == "no_model":
            logger.info(f"[{capture_id}] prediction no_model: {doc.error!a}")
        elif _is_transient(doc.status, doc.error):
            logger.info(f"[{capture_id}] prediction error (will be retried): {doc.error!a}")
        else:
            logger.warning(f"[{capture_id}] prediction error: {doc.error!a}")
        return doc.status


# Global service instance.
prediction_service = PredictionService()
