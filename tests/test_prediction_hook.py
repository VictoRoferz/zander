"""
Receiver prediction hook (receiver/): the worker's process_one state machine,
the enqueue rules and the ingest hand-off (docs/CONTRACTS.md sections 2 and 6).

Run (from the repo root): ../venv/bin/python -B tests/test_prediction_hook.py
Skips (exit 0) when fastapi / httpx / requests / label_studio_sdk are missing.

The inspector is a scripted fake with requests' get/post signature; Label
Studio is patched out. The app's lifespan is NOT run (TestClient without
`with`), so no LS connection and no background worker unless a test starts one.
"""
import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401  (TestClient)
    import label_studio_sdk  # noqa: F401
    import requests
except ImportError as e:
    print(f"SKIP: {e} (run with the services venv: ../venv/bin/python)")
    sys.exit(0)

# ---- Environment BEFORE importing the service (settings are read at import) --
TMP = Path(tempfile.mkdtemp(prefix="zander-receiver-test-"))
os.environ.update({
    "DATA_ROOT": str(TMP / "data"),
    "INSPECTOR_URL": "http://inspector.test",
    "INSPECTOR_TIMEOUT": "60",
    "INSPECTOR_STATUS_TIMEOUT": "5",
    "PREDICTION_QUEUE_MAX": "1000",
    "LABELSTUDIO_URL": "http://127.0.0.1:9",
    "LABELSTUDIO_API_KEY": "",
    "LOG_LEVEL": "CRITICAL",
})
os.chdir(TMP)  # the service reads ./.env; make sure there is none
sys.path.insert(0, str(ROOT / "receiver"))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import services.prediction_service as prediction_service_module  # noqa: E402
import services.storage_service as storage_module  # noqa: E402
from config.settings import settings  # noqa: E402
from services.labelstudio_service import labelstudio_service  # noqa: E402
from services.prediction_service import PredictionService, prediction_service  # noqa: E402
from services.storage_service import storage_service  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
MODEL_VERSION = "detector.onnx@abcd1234"
PRED_KEYS = {  # CONTRACTS section 2, the full table
    "schema_version", "capture_id", "status", "created_at", "error", "model_version",
    "model_sha256", "schema", "tiler_version", "class_names", "threshold",
    "threshold_source", "image_width", "image_height", "verdict", "classes",
    "photo_score", "detections", "latency_ms",
}


# ---- Fake inspector -------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code, body=None, text=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


def status_body(loaded=True, model_version=MODEL_VERSION, load_error=None):
    return {"model_loaded": loaded, "model_version": model_version if loaded else None,
            "load_error": load_error}


def inspect_body(capture_id="x", model_version=MODEL_VERSION):
    return {
        "status": "ok", "source": capture_id, "verdict": "NiO",
        "classes": ["Nicht ausreichend Lot", "Brücke"],
        "detections": [
            {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 150.0, "score": 0.9, "label": "Brücke"},
            {"x1": 10.0, "y1": 10.0, "x2": 40.0, "y2": 40.0, "score": 0.5,
             "label": "Nicht ausreichend Lot"},
        ],
        "count": 2, "latency_ms": 812.3, "model_version": model_version,
        "tiler_version": "v2", "confidence_threshold": 0.34, "photo_score": 0.9,
        "image_width": 2616, "image_height": 1960, "model_sha256": "ab" * 32,
        "schema": "def3", "adapter": "ultralytics", "class_names": NAMES,
        "threshold_source": "detector.json",
    }


class FakeHttp:
    """Scripted stand-in for `requests`: each reply is a FakeResponse or an
    exception, or a callable returning one (status: no argument; inspect: the
    capture_id), so a test can change the inspector's answers over time."""

    def __init__(self, status=None, inspect=None):
        self.status = status if status is not None else FakeResponse(200, status_body())
        self.inspect = inspect if inspect is not None else FakeResponse(200, inspect_body())
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, {"timeout": timeout}))
        reply = self.status() if callable(self.status) else self.status
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def post(self, url, files=None, data=None, timeout=None):
        self.calls.append(("POST", url, {"files": files, "data": data, "timeout": timeout}))
        reply = self.inspect(data["capture_id"]) if callable(self.inspect) else self.inspect
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def gets(self):
        return [c for c in self.calls if c[0] == "GET"]

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]

    def posted_ids(self):
        return [c[2]["data"]["capture_id"] for c in self.posts()]


# ---- Helpers ----------------------------------------------------------------------
IMAGE = b"\xff\xd8 not really a jpeg, the inspector is fake"


def _capture(with_image=True) -> str:
    capture_id = str(uuid.uuid4())
    if with_image:
        storage_service.unlabeled_image_path(capture_id).write_bytes(IMAGE)
    return capture_id


def _existing(capture_id, status="ok", model_version=MODEL_VERSION, error=None) -> dict:
    doc = {
        "schema_version": 1, "capture_id": capture_id, "status": status,
        "created_at": "2026-09-28T10:00:00.000Z", "error": error,
        "model_version": model_version if status == "ok" else None,
        "model_sha256": None, "schema": "def3", "tiler_version": "v2",
        "class_names": NAMES, "threshold": 0.34, "threshold_source": "detector.json",
        "image_width": 2616, "image_height": 1960,
        "verdict": "iO" if status == "ok" else None, "classes": [], "photo_score": 0.1,
        "detections": [], "latency_ms": 700.0,
    }
    storage_service.write_prediction(capture_id, doc)
    return storage_service.read_prediction(capture_id)


def _pred(capture_id):
    return storage_service.read_prediction(capture_id)


# ---- process_one ------------------------------------------------------------------------
def test_ok_writes_the_contract_document():
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id)))
    assert PredictionService(http=http).process_one(capture_id) == "ok"

    path = settings.predictions_dir / f"{capture_id}.pred.json"
    assert path == settings.data_root / "predictions" / f"{capture_id}.pred.json"
    raw = path.read_text("utf-8")
    doc = json.loads(raw)
    assert set(doc) == PRED_KEYS, set(doc) ^ PRED_KEYS
    assert doc["schema_version"] == 1 and doc["capture_id"] == capture_id
    assert doc["status"] == "ok" and doc["error"] is None
    assert doc["created_at"].endswith("Z")
    datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))
    assert doc["threshold"] == 0.34 and doc["threshold_source"] == "detector.json"
    assert doc["verdict"] == "NiO" and doc["classes"] == ["Nicht ausreichend Lot", "Brücke"]
    assert doc["detections"][0] == {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 150.0,
                                    "score": 0.9, "label": "Brücke"}
    assert doc["photo_score"] == 0.9 and doc["latency_ms"] == 812.3
    assert doc["image_width"] == 2616 and doc["image_height"] == 1960
    assert doc["model_version"] == MODEL_VERSION and doc["schema"] == "def3"
    assert doc["class_names"] == NAMES and doc["tiler_version"] == "v2"
    assert "Brücke" in raw, "pred.json must be UTF-8, not ASCII-escaped"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o644, oct(path.stat().st_mode)
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")], "temp file left"

    (get, post) = http.calls
    assert get[1] == "http://inspector.test/api/v1/status" and get[2]["timeout"] == 5
    assert post[1] == "http://inspector.test/api/v1/inspect" and post[2]["timeout"] == 60
    assert post[2]["data"] == {"capture_id": capture_id}
    name, data, ctype = post[2]["files"]["file"]
    assert name == f"{capture_id}.jpg" and data == IMAGE and ctype == "image/jpeg"


def test_no_model_via_status_skips_the_post():
    capture_id = _capture()
    http = FakeHttp(status=FakeResponse(200, status_body(
        loaded=False, load_error="model not found at /data/models/detector.onnx")))
    assert PredictionService(http=http).process_one(capture_id) == "no_model"
    doc = _pred(capture_id)
    assert doc["status"] == "no_model"
    assert doc["error"] == "model not found at /data/models/detector.onnx"
    assert doc["verdict"] is None and doc["classes"] == [] and doc["detections"] == []
    assert not http.posts()


def test_post_503_is_no_model():
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(503, {"detail": "no model loaded: sha256 mismatch"}))
    assert PredictionService(http=http).process_one(capture_id) == "no_model"
    doc = _pred(capture_id)
    assert doc["status"] == "no_model" and "sha256 mismatch" in doc["error"]


def test_post_500_is_error_with_code_and_detail():
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(500, {"detail": "inference failed: boom"}))
    assert PredictionService(http=http).process_one(capture_id) == "error"
    doc = _pred(capture_id)
    assert doc["status"] == "error" and doc["error"] == "HTTP 500: inference failed: boom"
    assert doc["verdict"] is None and doc["detections"] == []


def test_non_json_error_body_uses_text():
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(502, None, text="Bad Gateway"))
    PredictionService(http=http).process_one(capture_id)
    assert _pred(capture_id)["error"] == "HTTP 502: Bad Gateway"


def test_connection_error_is_inspector_unreachable():
    capture_id = _capture()
    http = FakeHttp(status=requests.ConnectionError("connection refused"))
    assert PredictionService(http=http).process_one(capture_id) == "error"
    doc = _pred(capture_id)
    assert doc["status"] == "error" and doc["error"] == "inspector unreachable"
    assert not http.posts()


def test_inspect_timeout_is_error():
    capture_id = _capture()
    http = FakeHttp(inspect=requests.Timeout("read timed out"))
    assert PredictionService(http=http).process_one(capture_id) == "error"
    assert "timed out" in _pred(capture_id)["error"]


def test_invalid_ok_body_is_error():
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(200, {"verdict": "maybe"}))
    assert PredictionService(http=http).process_one(capture_id) == "error"
    assert _pred(capture_id)["error"].startswith("invalid inspector response")


def test_long_error_is_capped_at_500_chars():
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(500, {"detail": "x" * 5000}))
    PredictionService(http=http).process_one(capture_id)
    assert len(_pred(capture_id)["error"]) == 500


def test_same_model_version_is_skipped():
    capture_id = _capture()
    before = _existing(capture_id)
    http = FakeHttp()
    assert PredictionService(http=http).process_one(capture_id) == "skipped"
    assert not http.posts()
    assert _pred(capture_id) == before


def test_force_reruns_the_same_model():
    capture_id = _capture()
    before = _existing(capture_id)
    http = FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id)))
    assert PredictionService(http=http).process_one(capture_id, force=True) == "ok"
    assert len(http.posts()) == 1
    after = _pred(capture_id)
    assert after["verdict"] == "NiO" and after["created_at"] != before["created_at"]


def test_new_model_version_replaces_an_ok():
    capture_id = _capture()
    _existing(capture_id, model_version="old.onnx@11111111")
    http = FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id)))
    assert PredictionService(http=http).process_one(capture_id) == "ok"
    assert _pred(capture_id)["model_version"] == MODEL_VERSION


def test_never_downgrades_an_ok():
    capture_id = _capture()
    before = _existing(capture_id, model_version="old.onnx@11111111")
    for http in (
        FakeHttp(inspect=FakeResponse(500, {"detail": "boom"})),
        FakeHttp(inspect=FakeResponse(503, {"detail": "no model loaded: x"})),
        FakeHttp(inspect=requests.Timeout("slow")),
        FakeHttp(status=requests.ConnectionError("down")),
        FakeHttp(status=FakeResponse(200, status_body(loaded=False, load_error="gone"))),
    ):
        assert PredictionService(http=http).process_one(capture_id) == "kept"
        assert _pred(capture_id) == before


def test_existing_error_is_retried():
    capture_id = _capture()
    _existing(capture_id, status="error", error="inspector unreachable")
    http = FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id)))
    assert PredictionService(http=http).process_one(capture_id) == "ok"
    assert len(http.posts()) == 1 and _pred(capture_id)["status"] == "ok"


def test_existing_no_model_is_retried():
    capture_id = _capture()
    _existing(capture_id, status="no_model", error="model not found")
    http = FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id)))
    assert PredictionService(http=http).process_one(capture_id) == "ok"


def test_missing_image_is_error():
    capture_id = _capture(with_image=False)
    http = FakeHttp()
    assert PredictionService(http=http).process_one(capture_id) == "error"
    assert "image not found" in _pred(capture_id)["error"]
    assert not http.posts()


def test_unexpected_exception_is_recorded_not_raised():
    capture_id = _capture()
    http = FakeHttp(status=RuntimeError("not a requests error"))
    assert PredictionService(http=http).process_one(capture_id) == "error"
    assert _pred(capture_id)["error"].startswith("receiver error: RuntimeError")


def test_invalid_capture_id_is_rejected_without_io():
    http = FakeHttp()
    svc = PredictionService(http=http)
    for bad in ("../escape", "a b", "", "x" * 65, "a.b", "abc\n", "abc\r\n", None, 42):
        assert svc.process_one(bad) == "invalid", repr(bad)
        assert svc.enqueue(bad) is False, repr(bad)
    assert not http.calls


def test_capture_id_rule_is_a_fullmatch():
    # re.match with "$" accepts a trailing newline; the rule must not.
    assert prediction_service_module.is_valid_capture_id("abc\n") is False
    assert prediction_service_module.is_valid_capture_id(uuid.uuid4().hex) is True  # camerapi's ids
    assert prediction_service_module.is_valid_capture_id("x" * 64) is True


class LockedReplace:
    """os.replace stand-in: the first `times` calls fail like Windows does while
    another process has the target open (WinError 32 is a PermissionError)."""

    def __init__(self, times):
        self.times = times
        self.calls = []
        self.real = os.replace

    def __call__(self, src, dst):
        self.calls.append(str(dst))
        if len(self.calls) <= self.times:
            raise PermissionError(13, "The process cannot access the file because it is "
                                      "being used by another process", str(dst))
        return self.real(src, dst)

    def __enter__(self):
        self.saved_retries = storage_module._REPLACE_RETRIES
        os.replace = self
        return self

    def __exit__(self, *exc):
        os.replace = self.real
        storage_module._REPLACE_RETRIES = self.saved_retries
        return False


def test_prediction_write_survives_a_briefly_locked_file():
    # Windows: the dashboard reading the old pred.json made os.replace fail, and
    # process_one then recorded an error doc instead of the ok result.
    capture_id = _capture()
    http = FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id)))
    with LockedReplace(times=2) as locked:
        storage_module._REPLACE_RETRIES = 5  # the Windows setting
        assert PredictionService(http=http).process_one(capture_id) == "ok"
    assert len(locked.calls) == 3, locked.calls
    assert all(c.endswith(f"{capture_id}.pred.json") for c in locked.calls)
    assert _pred(capture_id)["status"] == "ok"


def test_atomic_write_gives_up_on_a_lasting_lock_and_cleans_up():
    target = TMP / "atomic" / "x.bin"
    for retries in (5, 0):  # the Windows setting, then the POSIX one
        with LockedReplace(times=10**6) as locked:
            storage_module._REPLACE_RETRIES = retries
            try:
                storage_module._atomic_write_bytes(target, b"data")
                raise AssertionError("expected PermissionError")
            except PermissionError:
                pass
        assert len(locked.calls) == retries + 1, (retries, locked.calls)
        assert not [p for p in target.parent.iterdir() if p.name.endswith(".tmp")], "temp file left"
    # Only Windows retries; POSIX keeps its single attempt.
    assert storage_module._REPLACE_RETRIES == (5 if os.name == "nt" else 0)


def test_disabled_does_nothing():
    saved = settings.inspector_url
    settings.inspector_url = ""
    try:
        capture_id = _capture()
        http = FakeHttp()
        svc = PredictionService(http=http)
        assert svc.enabled is False
        assert svc.enqueue(capture_id) is False and svc.queue_depth() == 0
        assert svc.process_one(capture_id) == "disabled"
        assert not http.calls and _pred(capture_id) is None
    finally:
        settings.inspector_url = saved


# ---- Queue + worker -------------------------------------------------------------------------
def test_enqueue_dedupes_and_is_bounded():
    svc = PredictionService(http=FakeHttp(), queue_max=2)
    a, b, c = _capture(), _capture(), _capture()
    assert svc.enqueue(a) is True
    assert svc.enqueue(a) is False  # duplicate
    assert svc.enqueue(b) is True
    assert svc.enqueue(c) is False  # full
    assert svc.queue_depth() == 2
    tiny = PredictionService(http=FakeHttp(), queue_max=0)  # 0 must not mean unbounded
    assert tiny.enqueue(a) is True and tiny.enqueue(b) is False


def test_worker_drains_the_queue_and_honours_a_force_upgrade():
    http = FakeHttp(inspect=FakeResponse(200, inspect_body()))
    svc = PredictionService(http=http)
    a, b = _capture(), _capture()
    _existing(a)  # an ok from the same model: only a forced run re-inspects it
    assert svc.enqueue(a) is True
    assert svc.enqueue(a, force=True) is True  # upgrades the queued run
    assert svc.enqueue(b) is True
    svc.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (svc.queue_depth() or len(http.posts()) < 2):
            time.sleep(0.02)
    finally:
        svc.stop()
    posted = [c[2]["data"]["capture_id"] for c in http.posts()]
    assert posted == [a, b], posted
    assert _pred(b)["status"] == "ok" and svc.queue_depth() == 0
    assert svc.enqueue(a) is True, "a processed id can be queued again"


# ---- Ingest hand-off (HTTP) -------------------------------------------------------------------
labelstudio_service.ensure_initialized = lambda: True
labelstudio_service.create_task_from_image = lambda image_path, metadata: {"task_id": 4242}
labelstudio_service.is_healthy = lambda: False
client = TestClient(main.app)  # no `with`: the lifespan (LS init, worker) is not run


class Recorder:
    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises

    def __call__(self, capture_id, force=False):
        self.calls.append((capture_id, force, storage_service.has_task_marker(capture_id)))
        if self.raises is not None:
            raise self.raises
        return True


def _ingest(capture_id, image=IMAGE, sha=None):
    metadata = {
        "capture_id": capture_id,
        "sha256": sha or hashlib.sha256(image).hexdigest(),
        "captured_at": "2026-09-28T10:00:00Z", "source": "camera",
        "width": 2616, "height": 1960, "size_bytes": len(image),
    }
    return client.post(
        "/api/v1/ingest",
        data={"capture_id": capture_id, "metadata_json": json.dumps(metadata)},
        files={"file": (f"{capture_id}.jpg", image, "image/jpeg")},
    )


def _with_recorder(fn, raises=None):
    rec = Recorder(raises)
    prediction_service.enqueue = rec  # instance attribute shadows the method
    try:
        fn(rec)
    finally:
        del prediction_service.enqueue
    return rec


def test_ingest_enqueues_once_after_the_task_marker():
    capture_id = str(uuid.uuid4())

    def run(rec):
        r = _ingest(capture_id)
        assert r.status_code == 200 and r.json()["task_id"] == 4242, r.text
        assert rec.calls == [(capture_id, False, True)], rec.calls

    _with_recorder(run)


def test_ingest_dedupe_without_prediction_enqueues():
    capture_id = str(uuid.uuid4())
    _with_recorder(lambda rec: _ingest(capture_id))

    def run(rec):
        r = _ingest(capture_id)
        assert r.status_code == 200 and r.json()["deduped"] is True, r.text
        assert [c[0] for c in rec.calls] == [capture_id], rec.calls

    _with_recorder(run)


def test_ingest_dedupe_with_prediction_does_not_enqueue():
    capture_id = str(uuid.uuid4())
    _with_recorder(lambda rec: _ingest(capture_id))
    _existing(capture_id)

    def run(rec):
        r = _ingest(capture_id)
        assert r.status_code == 200 and r.json()["deduped"] is True, r.text
        assert rec.calls == [], rec.calls

    _with_recorder(run)


def test_ingest_ack_survives_a_failing_enqueue():
    for _ in range(2):  # fresh ingest, then the dedupe branch
        capture_id = str(uuid.uuid4())

        def run(rec):
            assert _ingest(capture_id).status_code == 200
            assert _ingest(capture_id).status_code == 200
            assert len(rec.calls) == 2

        _with_recorder(run, raises=RuntimeError("queue exploded"))


def test_rejected_ingest_does_not_enqueue():
    capture_id = str(uuid.uuid4())

    def run(rec):
        r = _ingest(capture_id, sha="0" * 64)
        assert r.status_code == 422, r.text
        assert rec.calls == []

    _with_recorder(run)


def test_ingest_rejects_an_invalid_capture_id_before_any_work():
    touched = []
    storage_service.has_task_marker = lambda cid: touched.append(("marker", cid)) or False
    storage_service.save_unlabeled = lambda cid, *a, **k: touched.append(("save", cid))
    labelstudio_service.ensure_initialized = lambda: touched.append(("ls", None)) or True
    before = sorted(p.name for p in settings.unlabeled_dir.iterdir())

    def run(rec):
        for bad in ("abc\n", "../escape", "..\\escape", "a b", "x" * 65, "a.b", ""):
            r = _ingest(bad)
            assert r.status_code == 422, (repr(bad), r.status_code, r.text)
        assert rec.calls == [] and touched == [], (rec.calls, touched)

    try:
        _with_recorder(run)
    finally:
        del storage_service.has_task_marker, storage_service.save_unlabeled
        labelstudio_service.ensure_initialized = lambda: True
    assert sorted(p.name for p in settings.unlabeled_dir.iterdir()) == before

    good = uuid.uuid4().hex  # the id format camerapi sends

    def accepted(rec):
        r = _ingest(good)
        assert r.status_code == 200, r.text
        assert [c[0] for c in rec.calls] == [good], rec.calls

    _with_recorder(accepted)
    assert storage_service.has_task_marker(good)


def test_webhook_ignores_an_invalid_capture_id():
    # The webhook builds unlabeled/ and labeled/ paths from the id it is sent; the
    # receiver's port is published, so the id gets the same rule as /ingest.
    good = uuid.uuid4().hex
    _with_recorder(lambda rec: _ingest(good))
    sidecar = storage_service.unlabeled_metadata_path(good)
    sidecar_before = sidecar.read_bytes()
    labeled_before = sorted(p.name for p in settings.labeled_dir.iterdir()) if settings.labeled_dir.exists() else []
    annotation = {"id": 7, "result": [{"value": {"rectanglelabels": ["Brücke"]}}]}

    for task in (
        {"id": 1, "meta": {"capture_id": f"../unlabeled/{good}"}},  # would overwrite the sidecar
        {"id": 2, "meta": {"capture_id": good + "\n"}},
        {"id": 3, "meta": {}, "data": {"image": "/data/local-files/?d=unlabeled/a.b.jpg"}},
    ):
        r = client.post("/api/v1/webhook/annotation-created",
                        json={"action": "ANNOTATION_CREATED", "task": task, "annotation": annotation})
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ignored", "reason": "invalid capture_id"}, (task, r.json())
    assert sidecar.read_bytes() == sidecar_before
    now = sorted(p.name for p in settings.labeled_dir.iterdir()) if settings.labeled_dir.exists() else []
    assert now == labeled_before, now

    # A valid id still exports (TestClient runs the background task before returning).
    r = client.post("/api/v1/webhook/annotation-created",
                    json={"action": "ANNOTATION_CREATED", "task": {"id": 4, "meta": {"capture_id": good}},
                          "annotation": annotation})
    assert r.status_code == 200 and r.json()["status"] == "queued", r.text
    assert storage_service.labeled_image_path(good).exists()
    assert storage_service.labeled_annotation_path(good).exists()


def test_ingest_does_no_inspector_work_inline():
    saved_http = prediction_service.http
    http = FakeHttp()
    prediction_service.http = http
    try:
        assert prediction_service._thread is None, "the worker must not run in this test"
        depth = prediction_service.queue_depth()
        capture_id = str(uuid.uuid4())
        assert _ingest(capture_id).status_code == 200
        assert http.calls == [], "the inspector was called before the ACK"
        assert prediction_service.queue_depth() == depth + 1
    finally:
        prediction_service.http = saved_http


def test_log_level_is_case_insensitive():
    assert main._log_level("debug") == logging.DEBUG
    assert main._log_level(" Warning ") == logging.WARNING
    assert main._log_level("INFO") == logging.INFO
    assert main._log_level("bogus") is None and main._log_level("basic_format") is None
    # The real failure mode: logging.basicConfig("debug") crashed the import.
    for value, warns in (("debug", False), ("bogus", True)):
        proc = subprocess.run(
            [sys.executable, "-B", "-c", "import main"],
            cwd=TMP, capture_output=True, text=True, timeout=120,
            env={**os.environ, "LOG_LEVEL": value, "PYTHONPATH": str(ROOT / "receiver"),
                 "DATA_ROOT": str(TMP / "log-level")},
        )
        assert proc.returncode == 0, (value, proc.stderr[-800:])
        assert ("is not a log level; using INFO" in proc.stderr) is warns, (value, proc.stderr[-800:])


def test_status_reports_predictions():
    _existing(_capture())  # at least one pred.json, whatever ran before
    r = client.get("/api/v1/status")
    assert r.status_code == 200, r.text
    block = r.json()["predictions"]
    assert set(block) == {"enabled", "inspector_url", "queue_depth", "retrying", "backfill", "count"}
    assert block["enabled"] is True and block["inspector_url"] == "http://inspector.test"
    assert all(isinstance(block[k], int) for k in ("queue_depth", "retrying", "backfill"))
    assert block["count"] == storage_service.count_predictions() > 0


# ---- Retries (transient inspector failures) ------------------------------------------------
PS = prediction_service_module
FAST = dict(retry_initial_s=0.05, retry_max_s=0.2)


def wait_for(predicate, timeout=5.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


def test_retry_classes_match_the_stored_error_texts():
    # Every retryable path writes an error the rescan recognises as transient,
    # and no permanent one does (the rescan rule reads only pred.json).
    ok = FakeResponse(200, inspect_body())
    cases = [
        # (status reply, inspect reply, retry class, transient?)
        (requests.ConnectionError("refused"), ok, PS.RETRY_UNAVAILABLE, True),
        (requests.Timeout("status slow"), ok, PS.RETRY_UNAVAILABLE, True),
        (FakeResponse(502, None, text="Bad Gateway"), ok, PS.RETRY_UNAVAILABLE, True),
        (FakeResponse(404, {"detail": "Not Found"}), ok, None, False),
        (None, requests.Timeout("read timed out"), PS.RETRY_FAILED, True),
        (None, requests.ConnectionError("RemoteDisconnected"), PS.RETRY_FAILED, True),
        (None, FakeResponse(500, {"detail": "inference failed: boom"}), PS.RETRY_FAILED, True),
        (None, FakeResponse(504, None, text="Gateway Timeout"), PS.RETRY_FAILED, True),
        (None, FakeResponse(422, {"detail": "cannot decode image"}), None, False),
    ]
    for status, inspect, retry, transient in cases:
        capture_id = _capture()
        http = FakeHttp(status=status, inspect=inspect)
        result = PredictionService(http=http)._attempt(capture_id, False)
        doc = _pred(capture_id)
        assert result.outcome == "error" and result.retry == retry, (status, inspect, result)
        assert PS.is_transient_error(doc) is transient, (doc["error"], transient)
        assert PS.needs_prediction(doc) is transient, doc["error"]
    # Not errors of the inspector: the photo is gone, an unexpected receiver bug.
    missing = _capture(with_image=False)
    assert PredictionService(http=FakeHttp())._attempt(missing, False).retry is None
    assert not PS.is_transient_error(_pred(missing))
    bug = _capture()
    assert PredictionService(http=FakeHttp(status=RuntimeError("x")))._attempt(bug, False).retry is None
    assert not PS.is_transient_error(_pred(bug))
    # no_model is not retried, but a rescan re-queues it (a model may have arrived).
    nm = _capture()
    result = PredictionService(http=FakeHttp(inspect=FakeResponse(503, {"detail": "x"})))._attempt(nm, False)
    assert result.outcome == "no_model" and result.retry is None
    assert PS.needs_prediction(_pred(nm)) and PS.needs_prediction(None)
    assert not PS.needs_prediction(_existing(_capture()))  # ok


def test_failed_photo_backoff_doubles_and_is_capped():
    svc = PredictionService(http=FakeHttp(), retry_initial_s=0.5, retry_max_s=3.0, retry_max_attempts=6)
    capture_id = _capture()
    item = PS._Item(False, False, 0.0, 0)
    delays = []
    for _ in range(5):
        svc._settle(capture_id, item, PS._Result("error", PS.RETRY_FAILED, True))
        item = svc._items.pop(capture_id)
        svc._counts[item.backfill] -= 1
        delays.append(round(item.due - time.monotonic(), 1))
    assert delays == [0.5, 1.0, 2.0, 3.0, 3.0], delays
    # the 6th failure is the last attempt: no retry is scheduled
    svc._settle(capture_id, item, PS._Result("error", PS.RETRY_FAILED, True))
    assert capture_id not in svc._items and item.attempts == 5


def test_worker_retries_until_the_inspector_is_back():
    # The documented model-update step: `docker compose restart inspector`.
    down = {"on": True}
    http = FakeHttp(status=lambda: requests.ConnectionError("refused") if down["on"]
                    else FakeResponse(200, status_body()))
    svc = PredictionService(http=http, **FAST)
    capture_id = _capture()
    svc.enqueue(capture_id)
    svc.start()
    try:
        assert wait_for(lambda: (_pred(capture_id) or {}).get("status") == "error")
        assert _pred(capture_id)["error"] == "inspector unreachable"  # shown meanwhile
        assert wait_for(lambda: len(http.gets()) >= 3)                 # probed again
        assert svc.queue_depth() == 1 and svc.retrying() == 1
        down["on"] = False
        assert wait_for(lambda: (_pred(capture_id) or {}).get("status") == "ok"), _pred(capture_id)
        assert wait_for(lambda: svc.queue_depth() == 0)
    finally:
        svc.stop()
    assert len(http.posts()) == 1


def test_unreachable_inspector_is_probed_once_per_backoff_not_per_capture():
    down = {"on": True}
    http = FakeHttp(status=lambda: requests.ConnectionError("refused") if down["on"]
                    else FakeResponse(200, status_body()))
    svc = PredictionService(http=http, **FAST)
    ids = [_capture() for _ in range(6)]
    for capture_id in ids:
        svc.enqueue(capture_id)
    svc.start()
    try:
        # Each new capture is tried at once (and gets its error document)...
        assert wait_for(lambda: all((_pred(c) or {}).get("status") == "error" for c in ids))
        first_round = len(http.gets())
        assert first_round == len(ids), first_round
        # ...then they wait together: ONE probe per backoff step (0.05, 0.1, 0.2, 0.2 ... s).
        time.sleep(0.8)
        probes = len(http.gets()) - first_round
        assert 2 <= probes <= 7, probes  # per-capture retries would be ~6 x that
        down["on"] = False
        assert wait_for(lambda: all((_pred(c) or {}).get("status") == "ok" for c in ids), timeout=5)
    finally:
        svc.stop()
    assert sorted(http.posted_ids()) == sorted(ids)


def test_a_failing_photo_never_blocks_new_captures():
    bad, good = _capture(), _capture()
    http = FakeHttp(inspect=lambda cid: FakeResponse(500, {"detail": "inference failed: boom"})
                    if cid == bad else FakeResponse(200, inspect_body(cid)))
    svc = PredictionService(http=http, retry_max_attempts=4, **FAST)
    svc.enqueue(bad)
    svc.enqueue(good)
    svc.start()
    try:
        assert wait_for(lambda: (_pred(good) or {}).get("status") == "ok")
        # good ran right after bad's first failure, not after bad's retries
        assert http.posted_ids()[:2] == [bad, good], http.posted_ids()
        assert wait_for(lambda: http.posted_ids().count(bad) == 4)
        time.sleep(0.5)  # no 5th attempt: RETRY_MAX_ATTEMPTS
        assert http.posted_ids().count(bad) == 4 and svc.queue_depth() == 0
    finally:
        svc.stop()
    assert _pred(bad)["status"] == "error" and _pred(bad)["error"].startswith("HTTP 500")


def test_a_repeated_failure_is_not_rewritten_and_an_ok_is_never_downgraded():
    capture_id = _capture()
    writes = []
    real = storage_service.write_prediction
    storage_service.write_prediction = lambda cid, doc: writes.append(doc["status"]) or real(cid, doc)
    try:
        down = FakeHttp(status=requests.ConnectionError("refused"))
        svc = PredictionService(http=down)
        for _ in range(3):
            assert svc.process_one(capture_id) == "error"
        assert writes == ["error"], writes  # created_at stays the first failure
        assert svc.process_one(capture_id, force=True) == "error" and writes == ["error"]
        up = PredictionService(http=FakeHttp(inspect=FakeResponse(200, inspect_body(capture_id))))
        assert up.process_one(capture_id) == "ok" and writes == ["error", "ok"]
        # an ok survives any number of failed (forced) retries
        for _ in range(2):
            assert svc.process_one(capture_id, force=True) == "kept"
        assert writes == ["error", "ok"] and _pred(capture_id)["status"] == "ok"
    finally:
        del storage_service.write_prediction


def test_enqueue_tries_a_waiting_capture_now():
    svc = PredictionService(http=FakeHttp(), **FAST)
    capture_id = _capture()
    svc._settle(capture_id, PS._Item(False, False, 0.0, 0), PS._Result("error", PS.RETRY_FAILED, True))
    assert svc._items[capture_id].due > time.monotonic()      # backing off
    assert svc.enqueue(capture_id) is True                    # e.g. POST /predict/<id>
    assert svc._items[capture_id].due <= time.monotonic() and svc._items[capture_id].attempts == 1
    assert svc.enqueue(capture_id) is False                   # now simply queued: a duplicate


# ---- Rescan ------------------------------------------------------------------------------------
@contextlib.contextmanager
def isolated_store():
    """storage_service on fresh unlabeled/ + predictions/ folders."""
    saved = storage_service.unlabeled_dir, storage_service.predictions_dir
    root = Path(tempfile.mkdtemp(prefix="rescan-", dir=TMP))
    storage_service.unlabeled_dir, storage_service.predictions_dir = root / "unlabeled", root / "predictions"
    storage_service.unlabeled_dir.mkdir()
    storage_service.predictions_dir.mkdir()
    try:
        yield
    finally:
        storage_service.unlabeled_dir, storage_service.predictions_dir = saved


def _ingested(name_age, pred=None, marker=True):
    """A stored capture (image, sidecar, task marker) captured `name_age[1]` s ago."""
    capture_id = f"{name_age[0]}-{uuid.uuid4().hex[:8]}"
    storage_service.unlabeled_image_path(capture_id).write_bytes(IMAGE)
    captured_at = datetime.fromtimestamp(time.time() - name_age[1], tz=timezone.utc)
    storage_service.unlabeled_metadata_path(capture_id).write_text(json.dumps({
        "capture_id": capture_id, "sha256": "0" * 64,
        "captured_at": captured_at.isoformat().replace("+00:00", "Z"), "source": "basler",
        "width": 2616, "height": 1960, "size_bytes": len(IMAGE),
    }), "utf-8")
    if marker:
        storage_service.write_task_marker(capture_id, 1)
    if pred == "ok":
        _existing(capture_id)
    elif pred is not None:
        status, error = pred
        _existing(capture_id, status=status, error=error)
    return capture_id


def _queued_order(svc):
    return [cid for cid, _ in sorted(svc._items.items(), key=lambda kv: kv[1].seq)]


def test_rescan_queues_what_needs_a_prediction_newest_first():
    with isolated_store():
        missing = _ingested(("missing", 50))
        no_model = _ingested(("nomodel", 10), pred=("no_model", "model not found"))
        unreachable = _ingested(("unreach", 30), pred=("error", "inspector unreachable"))
        http500 = _ingested(("http500", 40), pred=("error", "HTTP 500: inference failed: x"))
        _ingested(("ok", 5), pred="ok")
        _ingested(("gone", 20), pred=("error", "image not found: unlabeled/x.jpg"))
        _ingested(("undecodable", 25), pred=("error", "HTTP 422: cannot decode image"))
        _ingested(("not-ingested", 1), marker=False)  # no task marker: ingest still running
        svc = PredictionService(http=FakeHttp())
        assert svc.rescan() == 4
        assert _queued_order(svc) == [no_model, unreachable, http500, missing]  # newest first
        assert svc.backfill_depth() == 4 and svc.queue_depth() == 4
        assert svc.rescan() == 0  # already queued


def test_rescan_is_bounded_and_backfill_never_takes_a_new_captures_place():
    with isolated_store():
        a, b = _ingested(("a", 3)), _ingested(("b", 2))
        _ingested(("c", 9))
        svc = PredictionService(http=FakeHttp(), queue_max=2)
        assert svc.rescan() == 2 and _queued_order(svc) == [b, a]  # the newest two
        fresh = _capture()
        assert svc.enqueue(fresh) is True   # a new capture still fits
        assert svc.backfill_depth() == 2 and svc.queue_depth() == 3


def test_worker_runs_new_captures_before_backfill():
    with isolated_store():
        old1, old2 = _ingested(("old1", 60)), _ingested(("old2", 90))
        http = FakeHttp()
        svc = PredictionService(http=http, **FAST)
        assert svc.rescan() == 2
        fresh = _capture()
        svc.enqueue(fresh)
        svc.start()
        try:
            assert wait_for(lambda: len(http.posts()) == 3)
        finally:
            svc.stop()
        assert http.posted_ids() == [fresh, old1, old2], http.posted_ids()


def test_rescan_after_a_model_deploy_classifies_older_captures():
    # Captures taken without a model say no_model; deploying one and restarting
    # the receiver (the startup rescan) or calling the rescan endpoint fixes them.
    with isolated_store():
        ids = [_ingested((f"nm{n}", 10 + n), pred=("no_model", "model not found")) for n in range(3)]
        http = FakeHttp(inspect=lambda cid: FakeResponse(200, inspect_body(cid)))
        svc = PredictionService(http=http, **FAST)
        svc.start()
        try:
            svc.start_rescan()
            assert wait_for(lambda: all(_pred(c)["status"] == "ok" for c in ids))
        finally:
            svc.stop()
        assert http.posted_ids() == ids  # newest first


def test_receiver_startup_runs_the_rescan():
    calls = []
    prediction_service.start = lambda: calls.append("start")
    prediction_service.start_rescan = lambda: calls.append("rescan")
    prediction_service.stop = lambda: calls.append("stop")
    try:
        with TestClient(main.app):
            pass
    finally:
        del prediction_service.start, prediction_service.start_rescan, prediction_service.stop
    assert calls == ["start", "rescan", "stop"], calls


def test_rescan_endpoint():
    with isolated_store():
        _ingested(("r1", 5))
        _ingested(("r2", 6), pred=("error", "inspector timed out after 60 s"))
        _ingested(("r3", 7), pred="ok")
        depth = prediction_service.queue_depth()
        r = client.post("/api/v1/predict/rescan")
        assert r.status_code == 202 and r.json() == {"queued": 2}, r.text
        assert prediction_service.queue_depth() == depth + 2
        assert client.post("/api/v1/predict/rescan").json() == {"queued": 0}


def test_ingest_resend_requeues_only_what_needs_a_prediction():
    for pred, expected in (
        (("error", "inspector unreachable"), 1),
        (("no_model", "model not found"), 1),
        (("error", "image not found: unlabeled/x.jpg"), 0),
        ("ok", 0),
    ):
        capture_id = str(uuid.uuid4())
        _with_recorder(lambda rec: _ingest(capture_id))
        if pred == "ok":
            _existing(capture_id)
        else:
            _existing(capture_id, status=pred[0], error=pred[1])

        def run(rec):
            r = _ingest(capture_id)
            assert r.status_code == 200 and r.json()["deduped"] is True, r.text
            assert len(rec.calls) == expected, (pred, rec.calls)

        _with_recorder(run)


def test_predict_endpoint():
    capture_id = _capture()

    def run(rec):
        r = client.post(f"/api/v1/predict/{capture_id}?force=true")
        assert r.status_code == 202 and r.json() == {"queued": True}, r.text
        assert rec.calls == [(capture_id, True, False)]
        assert client.post(f"/api/v1/predict/{uuid.uuid4()}").status_code == 404
        assert client.post("/api/v1/predict/bad.id").status_code == 422
        assert client.post(f"/api/v1/predict/{capture_id}%0A").status_code == 422

    _with_recorder(run)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    try:
        for t in tests:
            try:
                t()
                print(f"PASS {t.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {t.__name__}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    finally:
        os.chdir(ROOT)
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
