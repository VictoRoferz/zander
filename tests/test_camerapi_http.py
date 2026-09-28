"""
camerapi over HTTP (FastAPI TestClient, no camera, no network): the preview
access rule, the 404/503 switches, and POST /api/v1/capture with the preview
off (today's path) and on (camera-owner thread over sample.jpg).

The TestClient is used WITHOUT `with`, so the lifespan (uploader + camera
owner threads) does not start; tests start the owner themselves when needed.

Run from the repo root (no pytest needed):
    ../venv/bin/python -B tests/test_camerapi_http.py
Skips (exit 0) if fastapi/httpx/opencv are not installed.
"""
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMERAPI = ROOT / "camerapi"
DATA_ROOT = Path(tempfile.mkdtemp(prefix="zander-camhttp-"))
TOKEN = "test-preview-token-0123456789"

# Settings are read (and folders created) at import: configure first.
os.environ.update({
    "USE_CAMERA": "false",
    "LIVE_PREVIEW_ENABLED": "true",
    "PREVIEW_TOKEN": TOKEN,
    "PREVIEW_FPS": "20",
    "PREVIEW_MAX_WIDTH": "320",
    "PREVIEW_IDLE_STOP_S": "0.3",
    "PREVIEW_MAX_CLIENTS": "8",
    "CAMERA_TRANSPORT": "ethernet",
    "FALLBACK_IMAGE_PATH": str(CAMERAPI / "sample.jpg"),
    "DATA_ROOT": str(DATA_ROOT),
    "INGEST_URL": "http://127.0.0.1:9",
    "DASHBOARD_URL": "http://127.0.0.1:9",
    "DASHBOARD_TIMEOUT": "0.5",
    "LOG_LEVEL": os.environ.get("TEST_LOG_LEVEL", "CRITICAL"),
})

try:
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
except ImportError as e:
    print(f"SKIP: {e} (install camerapi/requirements.txt + httpx to run this test)")
    sys.exit(0)

import _fake_pylon as fake  # noqa: E402  (tests/ is sys.path[0]; never touch a real camera)

fake.install()
sys.path.insert(0, str(CAMERAPI))

import main  # noqa: E402
from config.settings import settings  # noqa: E402
from services.live_camera import live_camera  # noqa: E402

AUTH = {"X-Preview-Token": TOKEN}
STATUS_KEYS = {"enabled", "state", "source", "camera_model", "camera_serial", "profile",
               "fps_target", "fps_measured", "clients", "frame_seq", "last_frame_age_s",
               "frames_incomplete", "reconnects", "last_error", "preview_width",
               "preview_height", "boot_id"}

client = TestClient(main.app)  # peer "testclient", Host "testserver"
local = TestClient(main.app, base_url="http://localhost", client=("127.0.0.1", 50000))


@contextmanager
def configured(**values):
    old = {k: getattr(settings, k) for k in values}
    for k, v in values.items():
        setattr(settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


@contextmanager
def owner_running():
    live_camera.start()
    try:
        deadline = time.monotonic() + 5
        while live_camera.state not in ("idle", "streaming") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert live_camera.state in ("idle", "streaming"), live_camera.state
        yield live_camera
    finally:
        live_camera.stop()


def decode(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


# ---- Access rule ------------------------------------------------------------------


def test_token_rules_on_status():
    with configured(preview_token=TOKEN, live_preview_enabled=True):
        assert client.get("/api/v1/preview/status").status_code == 401
        assert client.get("/api/v1/preview/status", headers={"X-Preview-Token": "wrong"}).status_code == 401
        # a configured token is required even from localhost
        assert local.get("/api/v1/preview/status").status_code == 401
        r = client.get("/api/v1/preview/status", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == STATUS_KEYS, set(body) ^ STATUS_KEYS
        assert body["enabled"] is True and body["fps_target"] == 20.0


def test_token_rules_on_frame():
    with configured(preview_token=TOKEN, live_preview_enabled=True), owner_running():
        assert client.get("/api/v1/preview/frame.jpg").status_code == 401
        assert client.get("/api/v1/preview/frame.jpg", headers={"X-Preview-Token": "wrong"}).status_code == 401
        r = client.get("/api/v1/preview/frame.jpg", headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/jpeg"
        assert r.headers["cache-control"] == "no-store"
        img = decode(r.content)
        assert img is not None and img.shape[1] == 320, None if img is None else img.shape
        assert live_camera.status()["clients"] == 0  # frame.jpg released its slot


def test_empty_token_means_local_only():
    with configured(preview_token="", live_preview_enabled=True):
        assert client.get("/api/v1/preview/status").status_code == 403  # peer "testclient"
        assert local.get("/api/v1/preview/status").status_code == 200
        v4 = TestClient(main.app, base_url="http://127.0.0.1:8001", client=("127.0.0.1", 50000))
        assert v4.get("/api/v1/preview/status").status_code == 200
        # (TestClient cannot parse an IPv6 base_url, so set the Host header)
        v6 = TestClient(main.app, base_url="http://localhost", client=("::1", 50000))
        assert v6.get("/api/v1/preview/status", headers={"Host": "[::1]:8001"}).status_code == 200
        mapped = TestClient(main.app, base_url="http://localhost", client=("::ffff:127.0.0.1", 50000))
        assert mapped.get("/api/v1/preview/status").status_code == 200
        # DNS rebinding: loopback peer, foreign Host header
        rebound = TestClient(main.app, base_url="http://evil.example", client=("127.0.0.1", 50000))
        assert rebound.get("/api/v1/preview/status").status_code == 403
        # a LAN peer that claims Host: localhost
        lan = TestClient(main.app, base_url="http://localhost", client=("192.168.177.20", 50000))
        assert lan.get("/api/v1/preview/status").status_code == 403
        # cross-site browser request to localhost (CORS is "*")
        assert local.get("/api/v1/preview/status", headers={"Origin": "https://evil.example"}).status_code == 403
        assert local.get("/api/v1/preview/status", headers={"Origin": "http://localhost:8003"}).status_code == 200
        # sending some token does not unlock remote access when none is configured
        assert client.get("/api/v1/preview/status", headers={"X-Preview-Token": "x"}).status_code == 403


# ---- Preview switches -------------------------------------------------------------


def test_flag_off_preview_is_404_and_status_says_disabled():
    with configured(live_preview_enabled=False, preview_token=TOKEN):
        assert client.get("/api/v1/preview/stream.mjpg", headers=AUTH).status_code == 404
        assert client.get("/api/v1/preview/frame.jpg", headers=AUTH).status_code == 404
        r = client.get("/api/v1/preview/status", headers=AUTH)
        assert r.status_code == 200 and r.json()["enabled"] is False and r.json()["state"] == "disabled"
        # auth still comes first
        assert client.get("/api/v1/preview/stream.mjpg").status_code == 401


def test_client_limit_gives_503():
    with configured(live_preview_enabled=True, preview_token=TOKEN, preview_max_clients=1):
        live_camera.add_client()
        try:
            assert client.get("/api/v1/preview/stream.mjpg", headers=AUTH).status_code == 503
            assert client.get("/api/v1/preview/frame.jpg", headers=AUTH).status_code == 503
        finally:
            live_camera.remove_client()


def test_stream_headers_when_the_owner_stops():
    # The TestClient buffers whole bodies, so an endless stream would hang it;
    # a stopped owner ends the stream at once and the response completes.
    with configured(live_preview_enabled=True, preview_token=TOKEN):
        with owner_running():
            pass
        assert live_camera.stopping
        r = client.get("/api/v1/preview/stream.mjpg", headers=AUTH)
        assert r.status_code == 200
        assert r.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
        assert "no-store" in r.headers["cache-control"]
        assert live_camera.status()["clients"] == 0


# ---- Capture ------------------------------------------------------------------------


def _spooled(capture_id: str):
    spool = DATA_ROOT / "spool"
    jpg, meta = spool / f"{capture_id}.jpg", spool / f"{capture_id}.json"
    assert jpg.exists() and meta.exists(), sorted(p.name for p in spool.iterdir())
    return jpg, json.loads(meta.read_text("utf-8"))


def test_flag_off_capture_is_202_and_spools():
    with configured(live_preview_enabled=False):
        r = client.post("/api/v1/capture")
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["spooled"] is True and body["camera"]["ok"] is True
        assert body["metadata"]["source"] == "fallback"
        assert (body["metadata"]["width"], body["metadata"]["height"]) == (1920, 1080)
        jpg, meta = _spooled(body["capture_id"])
        assert meta["capture_id"] == body["capture_id"] and meta["triggered_by"] is None
        assert decode(jpg.read_bytes()).shape[:2] == (1080, 1920)

        r = client.post("/api/v1/capture", headers={"X-Triggered-By": "anna@example.com"})
        assert r.status_code == 202
        assert _spooled(r.json()["capture_id"])[1]["triggered_by"] == "anna@example.com"

        status = client.get("/api/v1/status").json()
        assert status["version"] == "3.1.0"
        assert status["live"]["enabled"] is False and set(status["live"]) == STATUS_KEYS
        assert status["camera"]["use_camera"] is False and "owner" not in status["camera"]
        assert status["spool"]["pending"] >= 2


def test_flag_on_capture_goes_through_the_owner():
    with configured(live_preview_enabled=True), owner_running():
        r = client.post("/api/v1/capture")
        assert r.status_code == 202, r.text
        meta = r.json()["metadata"]
        # full-resolution sample, not the 320 px preview
        assert meta["source"] == "fallback" and (meta["width"], meta["height"]) == (1920, 1080)
        _spooled(r.json()["capture_id"])
        status = client.get("/api/v1/status").json()
        assert status["live"]["enabled"] is True and status["live"]["source"] == "sample"
        assert status["live"]["state"] in ("idle", "streaming")
        assert status["camera"]["owner"] == "live"
        r = client.post("/api/v1/test-camera")
        assert r.status_code == 200 and r.json()["source"] == "fallback"


def test_spool_write_survives_a_briefly_locked_file():
    # Windows: os.replace onto a file another process has open raises
    # PermissionError (WinError 32) for a moment; the capture must still spool.
    import services.storage_service as ss

    real, calls = os.replace, []

    def locked(times):
        def replace(src, dst):
            calls.append(str(dst))
            if len(calls) <= times:
                raise PermissionError(13, "being used by another process", str(dst))
            return real(src, dst)
        return replace

    saved = ss._REPLACE_RETRIES
    try:
        ss._REPLACE_RETRIES = 5  # the Windows setting
        os.replace = locked(times=1)
        with configured(live_preview_enabled=False):
            r = client.post("/api/v1/capture")
        assert r.status_code == 202, r.text
        _spooled(r.json()["capture_id"])
        assert len(calls) == 3, calls  # the image (locked once, then ok) + its sidecar

        # A lock that never goes away: the error surfaces, no temp file is left.
        target = DATA_ROOT / "atomic" / "x.bin"
        for retries in (5, 0):  # Windows, then POSIX
            calls.clear()
            ss._REPLACE_RETRIES = retries
            os.replace = locked(times=10**6)
            try:
                ss._atomic_write_bytes(target, b"data")
                raise AssertionError("expected PermissionError")
            except PermissionError:
                pass
            assert len(calls) == retries + 1, (retries, calls)
            assert not [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    finally:
        os.replace = real
        ss._REPLACE_RETRIES = saved
    assert ss._REPLACE_RETRIES == (5 if os.name == "nt" else 0)  # POSIX: one attempt


def test_root_lists_the_preview_endpoints():
    body = client.get("/").json()
    assert body["endpoints"]["preview_stream"] == "GET /api/v1/preview/stream.mjpg"
    assert body["endpoints"]["preview_frame"] == "GET /api/v1/preview/frame.jpg"
    assert body["endpoints"]["preview_status"] == "GET /api/v1/preview/status"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.monotonic()
        try:
            t()
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.2f} s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    live_camera.stop()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
