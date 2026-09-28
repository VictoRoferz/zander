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
from pathlib import Path

# The dashboard reads its settings at import time: set them all first.
TMP = Path(tempfile.mkdtemp(prefix="zander-dash-latest-"))
os.environ["DATA_ROOT"] = str(TMP)
os.environ["PREDICTIONS_DIR"] = str(TMP / "predictions")
os.environ["CAMERAPI_URL"] = "http://127.0.0.1:9"   # discard port: nothing listens
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


def make_capture(cid, age_s=0.0, sidecar=True, source="basler", directory=None):
    directory = directory or UNLABELED
    jpg = directory / f"{cid}.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xd9")   # never decoded by the dashboard
    if sidecar:
        (directory / f"{cid}.json").write_text(json.dumps({
            "capture_id": cid,
            "captured_at": "2026-09-28T10:00:00Z",
            "triggered_by": "op@example.com",
            "source": source,
            "width": 2616,
            "height": 1960,
        }), "utf-8")
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

    async def handler(self, request):
        self.requests.append(request)
        if request.url.path == "/api/v1/preview/status":
            return httpx.Response(self.status_code, json=self.status_body)
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
    make_capture("a", age_s=30, directory=d)
    make_capture("b", age_s=10, directory=d)
    index = dash._CaptureIndex(d, ttl_s=0.0, full_restat_s=0.0)
    assert index.newest() == "b"
    set_age(d / "a.jpg", -2)   # touched in place: no directory change
    assert index.newest() == "a"


def test_newest_capture_missing_directory():
    assert dash._CaptureIndex(TMP / "does-not-exist").newest() is None


# ---- capture view -----------------------------------------------------------------

def test_capture_id_validation():
    for bad in ("a.b", "x" * 65, "", "abc\n", "../x", "a b", "ä"):
        assert view_status(bad) == 400, repr(bad)
    assert view_status("x" * 64) == 404
    assert view_status("unknown-id") == 404


def test_capture_fields():
    make_capture("fields", source="fallback")
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
                           "capture-btn"):
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


TESTS = [
    test_newest_capture_by_mtime,
    test_newest_capture_is_cached_on_directory_mtime,
    test_newest_capture_ttl_and_full_restat,
    test_newest_capture_missing_directory,
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
