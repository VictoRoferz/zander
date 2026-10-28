"""
Dashboard: the latest-capture view, the live-preview relay and the
labeled-cache lock (docs/CONTRACTS.md sections 2, 4 and 5).

- /api/latest and /api/captures/{id}: newest capture by mtime, pending -> none
  after PREDICTION_PENDING_TIMEOUT_S, ok/no_model/error passed through,
  schema_version > 1 -> error, width/height precedence, id validation, login.
- /api/live/status and /api/live.mjpg against a mocked camerapi
  (httpx.MockTransport): status mapping, token header, byte-exact relay,
  error mapping, and the upstream is closed when the browser disconnects
  (real uvicorn server, raw socket).
- The Label Studio cache makes one LS call for concurrent pollers.
- The live view switch (POST /api/live/mode) against the mocked camerapi:
  token + user forwarded, body validated, 401/404/unreachable -> 502.
- The model chooser against a mocked inspector + receiver: /api/models
  (listing, unreachable, too old, predictions off), PUT /api/models/active
  (forwards the user, maps 404/409/502, validates the body) and
  POST /api/captures/{id}/recheck (forwards to the receiver's /predict).

Run from the repo root (no pytest needed):
    ../venv/bin/python -B tests/test_dashboard_latest.py
"""
import contextlib
import inspect
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# The dashboard reads its settings at import time: set them all first.
TMP = Path(tempfile.mkdtemp(prefix="zander-dash-latest-"))
os.environ["DATA_ROOT"] = str(TMP)
os.environ["PREDICTIONS_DIR"] = str(TMP / "predictions")
os.environ["CAMERAPI_URL"] = "http://127.0.0.1:9"   # discard port: nothing listens
os.environ["INSPECTOR_URL"] = "http://inspector.test:8004"   # always mocked below
os.environ["RECEIVER_URL"] = "http://receiver.test:8002"
os.environ["PREDICTION_PENDING_TIMEOUT_S"] = "5"
os.environ["PREVIEW_TOKEN"] = ""
os.environ["LABELSTUDIO_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "ERROR"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import anyio  # noqa: E402
import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main as dash  # noqa: E402
import auth  # noqa: E402  (the dashboard's own modules, loaded by main)
import db  # noqa: E402

db.initialize()
db.add_user("tester@example.com", auth.hash_password("pw"), "Tester")
SESSION_ID, _ = auth.login("tester@example.com")
AUTH = {"Cookie": f"{auth.SESSION_COOKIE}={SESSION_ID}"}

UNLABELED = dash.UNLABELED_DIR
PREDICTIONS = dash.PREDICTIONS_DIR
PREDICTIONS.mkdir(parents=True, exist_ok=True)
CAPTURE_KEYS = {"capture_id", "captured_at", "triggered_by", "source", "image_url", "width", "height"}
FRAME = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n\xff\xd8\xff\xd9\r\n"


# ---- fixtures -------------------------------------------------------------

def set_age(path, age_s):
    t = time.time() - age_s
    os.utime(path, (t, t))


def iso(age_s):
    """captured_at the way camerapi writes it (pydantic: UTC, "Z"), age_s ago."""
    t = datetime.fromtimestamp(time.time() - age_s, tz=timezone.utc)
    return t.isoformat().replace("+00:00", "Z")


def write_sidecar(cid, captured_at, source="basler", directory=None):
    (directory or UNLABELED).joinpath(f"{cid}.json").write_text(json.dumps({
        "capture_id": cid,
        "captured_at": captured_at,
        "triggered_by": "op@example.com",
        "source": source,
        "width": 2616,
        "height": 1960,
    }), "utf-8")


def make_capture(cid, age_s=0.0, sidecar=True, source="basler", directory=None,
                 captured_at=None):
    """An image aged age_s (+ sidecar). captured_at defaults to the image's
    own age: camerapi stamps the photo a moment before the receiver stores it."""
    directory = directory or UNLABELED
    jpg = directory / f"{cid}.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xd9")   # never decoded by the dashboard
    if sidecar:
        write_sidecar(cid, iso(age_s) if captured_at is None else captured_at, source, directory)
    set_age(jpg, age_s)
    return jpg


def pred_doc(cid, **fields):
    doc = {
        "schema_version": 1, "capture_id": cid, "status": "ok",
        "created_at": "2026-09-28T10:00:01Z", "error": None,
        "model_version": "detector.onnx@0123abcd", "model_sha256": "0123abcd" * 8,
        "schema": "def3", "tiler_version": "v2",
        "class_names": ["Nicht ausreichend Lot", "Brücke", "Fahne"],
        "threshold": 0.3412, "threshold_source": "detector.json",
        "image_width": 1000, "image_height": 800,
        "verdict": "NiO", "classes": ["Nicht ausreichend Lot"], "photo_score": 0.91,
        "detections": [{"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 45.0,
                        "score": 0.91, "label": "Nicht ausreichend Lot"}],
        "latency_ms": 512.3,
    }
    doc.update(fields)
    return doc


def write_pred(cid, doc=None, raw=None):
    path = PREDICTIONS / f"{cid}.pred.json"
    if raw is not None:
        path.write_bytes(raw)
    else:
        doc = doc if doc is not None else pred_doc(cid)
        path.write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
    return path


def view_status(cid):
    """HTTP status _capture_view raises for this id, or None if it succeeds."""
    try:
        dash._capture_view(cid)
    except HTTPException as e:
        return e.status_code
    return None


class FakeStream(httpx.AsyncByteStream):
    """camerapi's MJPEG body; records whether the relay closed it."""

    def __init__(self, frames, forever=False):
        self.frames = frames
        self.forever = forever
        self.closed = False

    async def __aiter__(self):
        while True:
            for frame in self.frames:
                yield frame
                if self.forever:
                    await anyio.sleep(0.02)
            if not self.forever:
                return

    async def aclose(self):
        # A checkpoint: a cancelled caller without a shield never gets past it.
        await anyio.sleep(0)
        self.closed = True


class FakeCamerapi:
    """camerapi's preview endpoints behind httpx.MockTransport."""

    def __init__(self):
        self.status_code = 200
        self.status_body = {"enabled": True, "state": "streaming", "boot_id": "boot-1",
                            "fps_measured": 4.9, "clients": 1}
        self.stream_code = 200
        self.stream_type = "multipart/x-mixed-replace; boundary=frame"
        self.frames = [FRAME, FRAME]
        self.forever = False
        self.requests = []
        self.streams = []
        self.mode_code = 200

    async def handler(self, request):
        self.requests.append(request)
        if request.url.path == "/api/v1/preview/status":
            return httpx.Response(self.status_code, json=self.status_body)
        if request.url.path == "/api/v1/preview/mode" and request.method == "POST":
            if self.mode_code != 200:
                return httpx.Response(self.mode_code, json={"detail": "nope"})
            enabled = json.loads(request.content)["enabled"]
            body = {**self.status_body, "enabled": enabled,
                    "state": "idle" if enabled else "disabled", "saved": True}
            return httpx.Response(200, json=body)
        if request.url.path == "/api/v1/preview/stream.mjpg":
            if self.stream_code != 200:
                return httpx.Response(self.stream_code, json={"detail": "nope"})
            stream = FakeStream(self.frames, self.forever)
            self.streams.append(stream)
            return httpx.Response(200, headers={"content-type": self.stream_type}, stream=stream)
        return httpx.Response(404)


@contextlib.contextmanager
def fake_camerapi_client(fake):
    """Make the dashboard's shared HTTP client talk to `fake` instead of the network."""
    original = dash._new_http_client
    dash._new_http_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    try:
        yield
    finally:
        dash._new_http_client = original


@contextlib.contextmanager
def client(fake=None):
    """TestClient with the lifespan (shared HTTP client); camerapi mocked if `fake`."""
    with contextlib.ExitStack() as stack:
        if fake is not None:
            stack.enter_context(fake_camerapi_client(fake))
        yield stack.enter_context(TestClient(dash.app))


# ---- newest capture ------------------------------------------------------------

def test_newest_capture_by_mtime():
    d = Path(tempfile.mkdtemp(prefix="index-", dir=TMP))
    index = dash._CaptureIndex(d)
    assert index.newest() is None
    make_capture("a", age_s=30, directory=d)
    make_capture("b", age_s=10, directory=d)
    make_capture("c", age_s=20, directory=d)
    # Newer, but not captures: a temp file, a sidecar, an id the API rejects.
    for name in (".c.jpg.x1y2.tmp", "e.json", "bad name.jpg"):
        (d / name).write_bytes(b"x")
        set_age(d / name, 0)
    assert index.newest() == "b"
    make_capture("d", age_s=0, directory=d)   # new file: the directory mtime changes
    assert index.newest() == "d"
    (d / "d.jpg").unlink()
    assert index.newest() == "b"


def test_newest_capture_is_cached_on_directory_mtime():
    d = Path(tempfile.mkdtemp(prefix="index-", dir=TMP))
    make_capture("a", age_s=30, directory=d)
    make_capture("b", age_s=10, directory=d)
    index = dash._CaptureIndex(d, ttl_s=3600.0, full_restat_s=3600.0)
    scans = []
    real_rescan = index._rescan

    def counting_rescan(full):
        scans.append(full)
        real_rescan(full)

    index._rescan = counting_rescan
    assert index.newest() == "b" and index.newest() == "b"
    assert scans == [True], scans          # one full scan, then served from the cache
    make_capture("c", age_s=0, directory=d)
    assert index.newest() == "c"
    assert scans == [True, False], scans   # directory changed: incremental rescan


def test_newest_capture_ttl_and_full_restat():
    d = Path(tempfile.mkdtemp(prefix="index-", dir=TMP))
    # No sidecars: these rank by mtime, which only a full re-stat refreshes.
    make_capture("a", age_s=30, sidecar=False, directory=d)
    make_capture("b", age_s=10, sidecar=False, directory=d)
    index = dash._CaptureIndex(d, ttl_s=0.0, full_restat_s=0.0)
    assert index.newest() == "b"
    set_age(d / "a.jpg", -2)   # touched in place: no directory change
    assert index.newest() == "a"


def test_newest_capture_missing_directory():
    assert dash._CaptureIndex(TMP / "does-not-exist").newest() is None


def test_newest_capture_by_capture_time_not_arrival():
    # A spool backlog drains in uuid order: the photo taken last can arrive
    # first. The panel must show the photo taken last.
    d = Path(tempfile.mkdtemp(prefix="index-", dir=TMP))
    index = dash._CaptureIndex(d)
    make_capture("taken-last", age_s=30, captured_at=iso(300), directory=d)   # arrived first
    make_capture("taken-first", age_s=10, captured_at=iso(500), directory=d)  # arrived last
    make_capture("taken-mid", age_s=20, captured_at=iso(400), directory=d)
    assert index.newest() == "taken-last"
    make_capture("live", age_s=0, captured_at=iso(1), directory=d)            # a new capture
    assert index.newest() == "live"


def test_newest_capture_without_a_usable_time_ranks_by_mtime():
    d = Path(tempfile.mkdtemp(prefix="index-", dir=TMP))
    index = dash._CaptureIndex(d)
    make_capture("dated", age_s=5, captured_at=iso(100), directory=d)
    # The receiver writes the image before its sidecar: until then the new
    # capture ranks by its (fresh) mtime...
    make_capture("arriving", age_s=0, sidecar=False, directory=d)
    assert index.newest() == "arriving"
    # ...and once the sidecar lands (the directory changes), by its captured_at.
    write_sidecar("arriving", iso(200), directory=d)
    assert index.newest() == "dated"
    # An unparseable captured_at falls back to the mtime too.
    make_capture("garbled", age_s=0, captured_at="yesterday-ish", directory=d)
    assert index.newest() == "garbled"


def test_newest_capture_reads_each_sidecar_once():
    d = Path(tempfile.mkdtemp(prefix="index-", dir=TMP))
    for n, cid in enumerate(("a", "b", "c")):
        make_capture(cid, age_s=30 - n, captured_at=iso(300 - n), directory=d)
    index = dash._CaptureIndex(d, ttl_s=0.0, full_restat_s=0.0)   # every call rescans, fully
    reads = []
    real = dash._sidecar_captured_at
    dash._sidecar_captured_at = lambda path: reads.append(path.name) or real(path)
    try:
        assert index.newest() == "c" and sorted(reads) == ["a.json", "b.json", "c.json"]
        reads.clear()
        assert index.newest() == "c" and reads == [], reads   # known: never re-read
        make_capture("d", age_s=0, captured_at=iso(1), directory=d)
        assert index.newest() == "d" and reads == ["d.json"], reads
    finally:
        dash._sidecar_captured_at = real


def test_iso_timestamp():
    assert dash._iso_timestamp("2026-09-28T10:00:00Z") == 1790589600.0
    assert dash._iso_timestamp("2026-09-28T10:00:00.250000+00:00") == 1790589600.25
    assert dash._iso_timestamp("2026-09-28T12:00:00+02:00") == 1790589600.0
    assert dash._iso_timestamp("2026-09-28T10:00:00") == 1790589600.0   # no offset = UTC
    for bad in (None, "", "   ", "yesterday", 1790589600, "2026-13-01T00:00:00Z"):
        assert dash._iso_timestamp(bad) is None, repr(bad)


# ---- capture view -----------------------------------------------------------------

def test_capture_id_validation():
    for bad in ("a.b", "x" * 65, "", "abc\n", "../x", "a b", "ä"):
        assert view_status(bad) == 400, repr(bad)
    assert view_status("x" * 64) == 404
    assert view_status("unknown-id") == 404


def test_capture_fields():
    make_capture("fields", source="fallback", captured_at="2026-09-28T10:00:00Z")
    view = dash._capture_view("fields")
    assert set(view) == {"capture", "state", "prediction"}
    cap = view["capture"]
    assert set(cap) == CAPTURE_KEYS
    assert cap["capture_id"] == "fields"
    assert cap["image_url"] == "/thumb/unlabeled/fields.jpg"
    assert cap["captured_at"] == "2026-09-28T10:00:00Z"
    assert cap["triggered_by"] == "op@example.com"
    assert cap["source"] == "fallback"


def test_pending_then_none():
    jpg = make_capture("pend1", age_s=0)
    try:
        view = dash._capture_view("pend1")
        assert view["state"] == "pending" and view["prediction"] is None
        set_age(jpg, 10)   # older than PREDICTION_PENDING_TIMEOUT_S=5
        view = dash._capture_view("pend1")
        assert view["state"] == "none" and view["prediction"] is None
        set_age(jpg, -3600)   # clock skew: an image "from the future" is not pending forever
        assert dash._capture_view("pend1")["state"] == "none"
    finally:
        set_age(jpg, 10)   # never leave a future mtime behind (it would be "newest")


def test_prediction_status_passed_through():
    for status in ("ok", "no_model", "error"):
        cid = f"status-{status}"
        make_capture(cid)
        fields = {}
        if status != "ok":
            fields = {"verdict": None, "classes": [], "detections": [], "error": f"{status} text"}
        doc = pred_doc(cid, status=status, **fields)
        write_pred(cid, doc)
        view = dash._capture_view(cid)
        assert view["state"] == status, (status, view["state"])
        assert view["prediction"] == doc, status


def test_unsupported_schema_version_is_error():
    make_capture("schema-v2")
    doc = pred_doc("schema-v2", schema_version=2, image_width=4000, image_height=3000)
    write_pred("schema-v2", doc)
    view = dash._capture_view("schema-v2")
    assert view["state"] == "error"
    assert view["prediction"] == doc   # as stored; the page shows "unsupported"
    # Nothing from an unsupported document is used: the size comes from the sidecar.
    assert (view["capture"]["width"], view["capture"]["height"]) == (2616, 1960)


def test_invalid_prediction_files_are_errors():
    cases = {
        "bad-json": (b"{not json", False),
        "bad-shape": (b"[1, 2]", False),
        "bad-utf8": (b"\xff\xfe\x00", False),
        "no-version": (json.dumps({"status": "ok"}).encode(), True),
        "bool-version": (json.dumps(pred_doc("x", schema_version=True)).encode(), True),
        "bad-status": (json.dumps(pred_doc("x", status="pending")).encode(), True),
    }
    for cid, (raw, keeps_document) in cases.items():
        make_capture(cid)
        write_pred(cid, raw=raw)
        view = dash._capture_view(cid)
        assert view["state"] == "error", (cid, view["state"])
        assert isinstance(view["prediction"], dict) is keeps_document, cid


def test_nan_in_prediction_is_serializable():
    make_capture("nan-doc")
    write_pred("nan-doc", raw=(
        b'{"schema_version": 1, "status": "ok", "created_at": "x", "detections": [],'
        b' "latency_ms": NaN, "photo_score": Infinity}'
    ))
    view = dash._capture_view("nan-doc")
    assert view["state"] == "ok"
    assert view["prediction"]["latency_ms"] is None
    assert view["prediction"]["photo_score"] is None
    json.dumps(view, allow_nan=False)   # what Starlette's JSONResponse requires


def test_width_height_precedence():
    make_capture("wh-pred")
    write_pred("wh-pred")   # prediction 1000x800, sidecar 2616x1960
    cap = dash._capture_view("wh-pred")["capture"]
    assert (cap["width"], cap["height"]) == (1000, 800)

    make_capture("wh-no-model")
    write_pred("wh-no-model", pred_doc(
        "wh-no-model", status="no_model", image_width=None, image_height=None,
        verdict=None, classes=[], detections=[],
    ))
    cap = dash._capture_view("wh-no-model")["capture"]
    assert (cap["width"], cap["height"]) == (2616, 1960)

    make_capture("wh-half")
    write_pred("wh-half", pred_doc("wh-half", image_height=None))   # never mix sources
    cap = dash._capture_view("wh-half")["capture"]
    assert (cap["width"], cap["height"]) == (2616, 1960)

    make_capture("wh-none", sidecar=False)   # jpg written, sidecar not (yet)
    cap = dash._capture_view("wh-none")["capture"]
    assert (cap["width"], cap["height"]) == (None, None)
    assert cap["captured_at"] is None and cap["source"] is None


# ---- Label Studio cache + concurrency --------------------------------------------------

def test_labeled_cache_single_flight():
    calls = []

    def slow_ls():
        calls.append(threading.get_ident())
        time.sleep(0.3)
        return [{"capture_id": "c1"}], {"c1"}

    original = dash._labeled_from_ls
    dash._labeled_from_ls = slow_ls
    try:
        dash._labeled_cache = {"ts": -1e9, "items": [], "ids": set(), "ok": False}
        barrier = threading.Barrier(3)
        results = []

        def poller():
            barrier.wait()
            results.append(dash._get_labeled_cached())

        threads = [threading.Thread(target=poller) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert len(calls) == 1, f"{len(calls)} Label Studio calls"
        assert len(results) == 3
        assert all(r["ok"] and r["ids"] == {"c1"} for r in results)
        dash._labeled_cache["ts"] = -1e9   # TTL expired: the next poll asks LS again
        dash._get_labeled_cached()
        assert len(calls) == 2
    finally:
        dash._labeled_from_ls = original


def test_log_level_is_case_insensitive():
    assert dash._log_level("debug") == logging.DEBUG
    assert dash._log_level(" Warning ") == logging.WARNING
    assert dash._log_level("bogus") is None and dash._log_level("basic_format") is None
    # The real failure mode: logging.basicConfig("debug") crashed the import.
    for value, warns in (("debug", False), ("bogus", True)):
        proc = subprocess.run(
            [sys.executable, "-B", "-c", "import main"],
            cwd=TMP, capture_output=True, text=True, timeout=120,
            env={**os.environ, "LOG_LEVEL": value, "PYTHONPATH": str(ROOT / "dashboard"),
                 "DATA_ROOT": str(TMP / "log-level")},
        )
        assert proc.returncode == 0, (value, proc.stderr[-800:])
        assert ("is not a log level; using INFO" in proc.stderr) is warns, (value, proc.stderr[-800:])


def test_polled_endpoints_run_in_the_threadpool():
    # Blocking Label Studio / disk / bcrypt work must not run on the event loop.
    for fn in (dash.stats, dash.api_unlabeled, dash.api_labeled, dash.api_latest,
               dash.api_capture, dash.thumb, dash.home, dash.login_submit):
        assert not inspect.iscoroutinefunction(fn), fn.__name__


# ---- HTTP -------------------------------------------------------------------------

def test_http_requires_login():
    with client() as c:
        for path in ("/api/latest", "/api/captures/abc", "/api/live.mjpg", "/api/live/status"):
            r = c.get(path)
            assert r.status_code == 401, (path, r.status_code)


def test_http_latest_and_captures():
    make_capture("http-latest")   # the newest capture of this run
    write_pred("http-latest")
    with client() as c:
        r = c.get("/api/latest", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"capture", "state", "prediction"}
        assert set(body["capture"]) == CAPTURE_KEYS
        assert body["capture"]["capture_id"] == "http-latest"
        assert body["state"] == "ok" and body["prediction"]["verdict"] == "NiO"
        r = c.get("/api/captures/http-latest", headers=AUTH)
        assert r.status_code == 200 and r.json() == body
        assert c.get("/api/captures/a.b", headers=AUTH).status_code == 400
        assert c.get("/api/captures/" + "x" * 65, headers=AUTH).status_code == 400
        assert c.get("/api/captures/no-such-capture", headers=AUTH).status_code == 404
        # The page draws the image from the existing thumb route.
        r = c.get(body["capture"]["image_url"], headers=AUTH)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_http_latest_without_captures():
    original = dash._capture_index
    dash._capture_index = dash._CaptureIndex(Path(tempfile.mkdtemp(prefix="empty-", dir=TMP)))
    try:
        with client() as c:
            r = c.get("/api/latest", headers=AUTH)
            assert r.status_code == 200
            assert r.json() == {"capture": None, "state": "none", "prediction": None}
    finally:
        dash._capture_index = original


def test_home_page_panels_outside_the_grids():
    with client() as c:
        r = c.get("/", headers=AUTH)
        assert r.status_code == 200
        html = r.text
        for element_id in ("live-img", "live-overlay", "live-status", "result-verdict",
                           "result-figure", "result-crops", "result-meta", "result-message",
                           "capture-btn", "model-select", "recheck-btn", "model-note",
                           "live-toggle"):
            assert f'id="{element_id}"' in html, element_id
        assert html.index('class="live-row"') < html.index('id="grid-unlabeled"')
        assert '<div class="grid" id="grid-unlabeled"></div>' in html
        assert '<div class="grid" id="grid-labeled"></div>' in html
        assert f"/static/app.js?v={dash.SERVICE_VERSION}" in html


def test_http_live_camerapi_unreachable():
    with client() as c:
        r = c.get("/api/live/status", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["reachable"] is False
        assert isinstance(body["error"], str) and body["error"]
        assert c.get("/api/live.mjpg", headers=AUTH).status_code == 502


def test_http_live_status_mapping():
    fake = FakeCamerapi()
    with client(fake) as c:
        body = c.get("/api/live/status", headers=AUTH).json()
        assert body["reachable"] is True and body["error"] is None
        assert body["enabled"] is True and body["state"] == "streaming"
        assert body["boot_id"] == "boot-1"
        for code, state in ((401, "auth_error"), (403, "auth_error"), (404, "unsupported"), (500, "error")):
            fake.status_code = code
            body = c.get("/api/live/status", headers=AUTH).json()
            assert body["reachable"] is True, code
            assert body["state"] == state, (code, body)
            assert isinstance(body["error"], str) and body["error"], code


def test_http_live_token_header():
    fake = FakeCamerapi()
    original = dash.PREVIEW_TOKEN
    try:
        with client(fake) as c:
            dash.PREVIEW_TOKEN = ""
            c.get("/api/live/status", headers=AUTH)
            assert "x-preview-token" not in fake.requests[-1].headers
            dash.PREVIEW_TOKEN = "s3cret"
            c.get("/api/live/status", headers=AUTH)
            assert fake.requests[-1].headers["x-preview-token"] == "s3cret"
            c.get("/api/live.mjpg", headers=AUTH)
            assert fake.requests[-1].headers["x-preview-token"] == "s3cret"
            # Relayed raw, so the stream must not come back content-encoded.
            assert fake.requests[-1].headers["accept-encoding"] == "identity"
            # A token that cannot be a header value is a 502, never a 500.
            dash.PREVIEW_TOKEN = "töken"
            count = len(fake.requests)
            assert c.get("/api/live.mjpg", headers=AUTH).status_code == 502
            body = c.get("/api/live/status", headers=AUTH).json()
            assert body["reachable"] is False and body["error"]
            assert len(fake.requests) == count   # nothing was sent
    finally:
        dash.PREVIEW_TOKEN = original


def test_http_live_stream_relay():
    fake = FakeCamerapi()
    with client(fake) as c:
        r = c.get("/api/live.mjpg", headers=AUTH)
        assert r.status_code == 200
        assert r.headers["content-type"] == fake.stream_type
        assert r.headers["cache-control"] == "no-store"
        assert r.content == FRAME * 2   # relayed byte for byte
        assert fake.streams and all(s.closed for s in fake.streams), "upstream left open"
        for code, expected in ((404, 404), (503, 503), (401, 502), (403, 502), (500, 502)):
            fake.stream_code = code
            r = c.get("/api/live.mjpg", headers=AUTH)
            assert r.status_code == expected, (code, r.status_code)
        fake.stream_code, fake.stream_type = 200, "text/html"   # never relay a non-stream
        assert c.get("/api/live.mjpg", headers=AUTH).status_code == 502
        assert fake.streams[-1].closed


def test_relay_response_closes_upstream_when_never_started():
    # ASGI >= 2.4 servers make send() raise once the browser is gone; Starlette
    # then raises ClientDisconnect before the relay generator ever ran.
    async def scenario():
        stream = FakeStream([FRAME], forever=True)
        upstream = httpx.Response(200, stream=stream)
        started = []

        async def relay():
            started.append(True)
            yield FRAME

        async def receive():
            await anyio.sleep(10)
            return {"type": "http.disconnect"}

        async def send(message):
            raise OSError("client gone")

        response = dash._RelayResponse(upstream, relay(), media_type="multipart/x-mixed-replace; boundary=frame")
        with contextlib.suppress(Exception):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        return started, stream.closed

    started, closed = anyio.run(scenario)
    assert not started, "the scenario should never start the generator"
    assert closed, "upstream left open"


def test_relay_closes_upstream_when_browser_disconnects():
    fake = FakeCamerapi()
    fake.forever = True
    fake.frames = [FRAME]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with fake_camerapi_client(fake):
        server = uvicorn.Server(uvicorn.Config(
            dash.app, host="127.0.0.1", port=port, log_level="error", timeout_graceful_shutdown=1,
        ))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert time.monotonic() < deadline and thread.is_alive(), "server did not start"
                time.sleep(0.02)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall((
                    f"GET /api/live.mjpg HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                    f"Cookie: {AUTH['Cookie']}\r\n\r\n"
                ).encode())
                received = b""
                while received.count(b"--frame") < 2:
                    chunk = sock.recv(65536)
                    assert chunk, "the relay ended early"
                    received += chunk
                assert received.startswith(b"HTTP/1.1 200"), received[:80]
                assert b"multipart/x-mixed-replace" in received
            # The browser is gone: the relay must close camerapi's response.
            deadline = time.monotonic() + 5
            while not fake.streams[0].closed:
                assert time.monotonic() < deadline, "upstream still open after the disconnect"
                time.sleep(0.02)
        finally:
            server.should_exit = True
            thread.join(10)


# ---- model chooser (inspector + receiver mocked) ------------------------------

MODELS_LISTING = {
    "active": "detector", "model_loaded": True, "load_error": None,
    "selected_by": None, "selected_at": None, "models_dir": "/data/models",
    "models": [
        {"name": "detector", "loaded": True, "active": True, "error": None,
         "model_version": "detector.onnx@a68b57d9", "threshold": 0.0575},
        {"name": "fold1", "loaded": True, "active": False, "error": None,
         "model_version": "fold1.onnx@cdb12d97", "threshold": 0.0575},
    ],
}


class FakeUpstreams:
    """The inspector's /models endpoints and the receiver's /predict behind MockTransport."""

    def __init__(self):
        self.requests = []
        self.down = set()          # hosts that refuse connections
        self.models_code = 200
        self.switch_code = 200
        self.switch_body = None
        self.predict_code = 202

    async def handler(self, request):
        self.requests.append(request)
        if request.url.host in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if request.url.host == "inspector.test" and path == "/api/v1/models":
            return httpx.Response(self.models_code, json=MODELS_LISTING)
        if request.url.host == "inspector.test" and path == "/api/v1/models/active":
            name = json.loads(request.content)["name"]
            if self.switch_code != 200:
                return httpx.Response(self.switch_code, json={"detail": f"model {name!r}: nope"})
            body = self.switch_body or {**MODELS_LISTING, "active": name, "saved": True}
            return httpx.Response(200, json=body)
        if request.url.host == "receiver.test" and path.startswith("/api/v1/predict/"):
            if self.predict_code == 404:
                return httpx.Response(404, json={"detail": "unknown capture"})
            return httpx.Response(self.predict_code, json={"queued": True})
        return httpx.Response(404)


def test_http_models_listing_and_unavailable_cases():
    fake = FakeUpstreams()
    with client(fake) as c:
        assert c.get("/api/models").status_code == 401
        body = c.get("/api/models", headers=AUTH).json()
        assert body["enabled"] is True and body["reachable"] is True and body["error"] is None
        assert [m["name"] for m in body["models"]] == ["detector", "fold1"]
        assert body["active"] == "detector"
        fake.down.add("inspector.test")
        body = c.get("/api/models", headers=AUTH).json()
        assert body["reachable"] is False and body["models"] == [], body
        assert "inspector unreachable" in body["error"]
        fake.down.clear()
        fake.models_code = 404   # an inspector before v0.3.0
        body = c.get("/api/models", headers=AUTH).json()
        assert body["reachable"] is False and "too old" in body["error"], body
        fake.models_code = 500
        body = c.get("/api/models", headers=AUTH).json()
        assert body["reachable"] is False and "HTTP 500" in body["error"], body
        original = dash.INSPECTOR_URL
        dash.INSPECTOR_URL = ""  # predictions off
        try:
            n = len(fake.requests)
            body = c.get("/api/models", headers=AUTH).json()
            assert body["enabled"] is False and "predictions are off" in body["error"], body
            r = c.put("/api/models/active", json={"name": "fold1"}, headers=AUTH)
            assert r.status_code == 409, r.text
            assert len(fake.requests) == n, "no upstream call while predictions are off"
        finally:
            dash.INSPECTOR_URL = original


def test_http_switch_model_forwards_the_user_and_maps_errors():
    fake = FakeUpstreams()
    with client(fake) as c:
        assert c.put("/api/models/active", json={"name": "fold1"}).status_code == 401
        r = c.put("/api/models/active", json={"name": "fold1"}, headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["active"] == "fold1" and body["saved"] is True and body["reachable"] is True
        sent = fake.requests[-1]
        assert sent.method == "PUT" and sent.url.path == "/api/v1/models/active"
        assert json.loads(sent.content) == {"name": "fold1", "selected_by": "tester@example.com"}
        for code in (404, 409):
            fake.switch_code = code
            r = c.put("/api/models/active", json={"name": "fold1"}, headers=AUTH)
            assert r.status_code == code and "nope" in r.json()["detail"], (code, r.text)
        n = len(fake.requests)
        for bad in ({"name": ""}, {}, {"name": "x" * 201}):
            r = c.put("/api/models/active", json=bad, headers=AUTH)
            assert r.status_code == 422, (bad, r.status_code)
        assert len(fake.requests) == n, "an invalid body never reaches the inspector"
        fake.down.add("inspector.test")
        r = c.put("/api/models/active", json={"name": "fold1"}, headers=AUTH)
        assert r.status_code == 502 and "inspector unreachable" in r.json()["detail"], r.text


def test_http_recheck_forwards_to_the_receiver():
    fake = FakeUpstreams()
    with client(fake) as c:
        assert c.post("/api/captures/abc123/recheck").status_code == 401
        r = c.post("/api/captures/abc123/recheck", headers=AUTH)
        assert r.status_code == 202 and r.json() == {"queued": True}, r.text
        sent = fake.requests[-1]
        assert sent.method == "POST" and sent.url.host == "receiver.test"
        assert sent.url.path == "/api/v1/predict/abc123" and not sent.url.query, sent.url
        n = len(fake.requests)
        assert c.post("/api/captures/bad.id/recheck", headers=AUTH).status_code == 400
        assert len(fake.requests) == n
        fake.predict_code = 404
        assert c.post("/api/captures/abc123/recheck", headers=AUTH).status_code == 404
        fake.down.add("receiver.test")
        r = c.post("/api/captures/abc123/recheck", headers=AUTH)
        assert r.status_code == 502 and "receiver unreachable" in r.json()["detail"], r.text


def test_http_live_mode_switch_forwards_the_token_and_the_user():
    fake = FakeCamerapi()
    original = dash.PREVIEW_TOKEN
    try:
        with client(fake) as c:
            dash.PREVIEW_TOKEN = "s3cret"
            assert c.post("/api/live/mode", json={"enabled": False}).status_code == 401
            n = len(fake.requests)
            for bad in ({"enabled": "yes"}, {}, {"enabled": 1}):
                assert c.post("/api/live/mode", json=bad, headers=AUTH).status_code == 422, bad
            assert len(fake.requests) == n, "an invalid body never reaches camerapi"
            r = c.post("/api/live/mode", json={"enabled": False}, headers=AUTH)
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["enabled"] is False and body["state"] == "disabled" and body["saved"] is True
            assert body["reachable"] is True and body["error"] is None
            sent = fake.requests[-1]
            assert sent.method == "POST" and sent.url.path == "/api/v1/preview/mode"
            assert sent.headers["x-preview-token"] == "s3cret"
            assert json.loads(sent.content) == {"enabled": False, "changed_by": "tester@example.com"}
            for code, words in ((401, "PREVIEW_TOKEN"), (403, "PREVIEW_TOKEN"), (404, "3.3.0"), (500, "HTTP 500")):
                fake.mode_code = code
                r = c.post("/api/live/mode", json={"enabled": True}, headers=AUTH)
                assert r.status_code == 502 and words in r.json()["detail"], (code, r.text)
        with client() as c:  # camerapi unreachable
            r = c.post("/api/live/mode", json={"enabled": True}, headers=AUTH)
            assert r.status_code == 502 and "unreachable" in r.json()["detail"], r.text
    finally:
        dash.PREVIEW_TOKEN = original


TESTS = [
    test_newest_capture_by_mtime,
    test_newest_capture_is_cached_on_directory_mtime,
    test_newest_capture_ttl_and_full_restat,
    test_newest_capture_missing_directory,
    test_newest_capture_by_capture_time_not_arrival,
    test_newest_capture_without_a_usable_time_ranks_by_mtime,
    test_newest_capture_reads_each_sidecar_once,
    test_iso_timestamp,
    test_capture_id_validation,
    test_capture_fields,
    test_pending_then_none,
    test_prediction_status_passed_through,
    test_unsupported_schema_version_is_error,
    test_invalid_prediction_files_are_errors,
    test_nan_in_prediction_is_serializable,
    test_width_height_precedence,
    test_labeled_cache_single_flight,
    test_log_level_is_case_insensitive,
    test_polled_endpoints_run_in_the_threadpool,
    test_http_requires_login,
    test_http_latest_and_captures,   # needs "http-latest" to be the newest capture
    test_http_latest_without_captures,
    test_home_page_panels_outside_the_grids,
    test_http_live_camerapi_unreachable,
    test_http_live_status_mapping,
    test_http_live_token_header,
    test_http_live_stream_relay,
    test_relay_response_closes_upstream_when_never_started,
    test_relay_closes_upstream_when_browser_disconnects,
    test_http_models_listing_and_unavailable_cases,
    test_http_switch_model_forwards_the_user_and_maps_errors,
    test_http_recheck_forwards_to_the_receiver,
    test_http_live_mode_switch_forwards_the_token_and_the_user,
]


if __name__ == "__main__":
    failed = 0
    try:
        for t in TESTS:
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
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
