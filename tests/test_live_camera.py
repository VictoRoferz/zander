"""
camerapi camera paths without a camera: the live camera owner
(services/live_camera.py) with a scripted frame source, BaslerSource and the
legacy one-shot path (services/camera_service.py) against a fake pylon, and
the MJPEG part generator (api/preview.py).

Run from the repo root (no pytest needed):
    ../venv/bin/python -B tests/test_live_camera.py
Skips (exit 0) if numpy/opencv are not installed.
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMERAPI = ROOT / "camerapi"

# Settings are read (and folders created) at import: configure first.
os.environ.update({
    "USE_CAMERA": "false",
    "PREVIEW_FPS": "20",
    "PREVIEW_MAX_WIDTH": "320",
    "PREVIEW_IDLE_STOP_S": "0.3",
    "PREVIEW_MAX_CLIENTS": "8",
    "CAMERA_RECONNECT_MAX_BACKOFF_S": "0.2",
    "LIVE_GRAB_TIMEOUT_MS": "200",
    "LIVE_CAPTURE_TIMEOUT_S": "5",
    "CAMERA_TRANSPORT": "ethernet",
    "FALLBACK_IMAGE_PATH": str(CAMERAPI / "sample.jpg"),
    "DATA_ROOT": tempfile.mkdtemp(prefix="zander-livecam-"),
    "INGEST_URL": "http://127.0.0.1:9",
    "DASHBOARD_URL": "http://127.0.0.1:9",
    "LOG_LEVEL": os.environ.get("TEST_LOG_LEVEL", "CRITICAL"),
})

try:
    import cv2
    import numpy as np
    import pydantic_settings  # noqa: F401
except ImportError as e:
    print(f"SKIP: {e} (install camerapi/requirements.txt to run this test)")
    sys.exit(0)

import _fake_pylon as fake  # noqa: E402  (tests/ is sys.path[0])

fake.install()  # before anything imports services.camera_service
sys.path.insert(0, str(CAMERAPI))

from config.settings import settings  # noqa: E402
from services import live_camera as lc  # noqa: E402
from services.camera_service import CameraService  # noqa: E402
from services.live_camera import (  # noqa: E402
    BaslerSource,
    GrabFailed,
    LiveCamera,
    SampleSource,
    SourceLost,
    TooManyClients,
)

SAMPLE = cv2.imread(str(CAMERAPI / "sample.jpg"))


# ---- Helpers --------------------------------------------------------------------


def wait_until(predicate, timeout=3.0, step=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


def decode(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class ScriptedSource:
    """A frame source whose behaviour the test controls."""

    name = "fake"
    preview_label = None
    profile = "ethernet"
    camera_model = "FakeCam"
    camera_serial = "F0001"

    def __init__(self, streamable=True):
        self.streamable = streamable
        self.is_open = False
        self.streaming = False
        self.calls = []            # method names, in order
        self.open_times = []
        self.open_failures = 0     # next N open() calls raise
        self.lose_on_read = False  # next read_stream raises SourceLost
        self.lose_always = False   # every read_stream raises SourceLost (flapping)
        self.grab_delay = 0.0
        self.grab_error = None
        self.no_frames = False     # read_stream times out forever (firewall case)
        self.frames_incomplete = 0
        self.last_error = None
        self.last_streamed = 0     # stamp of the last frame read_stream returned

    def open(self):
        self.calls.append("open")
        self.open_times.append(time.monotonic())
        if self.open_failures > 0:
            self.open_failures -= 1
            raise RuntimeError("fake camera not found")
        self.is_open = True

    def close(self):
        self.calls.append("close")
        self.is_open = False
        self.streaming = False

    def start_stream(self):
        self.calls.append("start_stream")
        self.streaming = True

    def stop_stream(self):
        if self.streaming:
            self.calls.append("stop_stream")
        self.streaming = False

    def read_stream(self, timeout_s):
        if self.lose_on_read or self.lose_always:
            self.lose_on_read = False
            raise SourceLost("fake cable pulled")
        if self.no_frames:  # like BaslerSource when no packet ever arrives
            time.sleep(min(timeout_s, 0.02))
            self.frames_incomplete += 1
            self.last_error = f"no frame within {timeout_s:.1f} s"
            return None
        time.sleep(0.002)
        frame, n = fake.next_frame(640, 480)
        self.last_streamed = n
        self.calls.append("read")
        return frame

    def grab_fresh(self):
        self.calls.append("grab_fresh")
        if self.grab_delay:
            time.sleep(self.grab_delay)
        if self.grab_error is not None:
            raise self.grab_error
        frame, _ = fake.next_frame(640, 480)
        return frame, {"source": "basler", "camera_serial": self.camera_serial,
                       "camera_model": self.camera_model}


@contextmanager
def running(source_or_factory):
    """A started LiveCamera over a source instance (or a factory); always stopped."""
    if callable(source_or_factory):
        factory = source_or_factory
    else:
        factory = lambda: source_or_factory  # noqa: E731
    cam = LiveCamera(source_factory=factory)
    cam.start()
    try:
        yield cam
    finally:
        cam.stop()


@contextmanager
def counting_encodes():
    original = lc._encode_jpeg
    box = {"n": 0}

    def counted(image, quality):
        box["n"] += 1
        return original(image, quality)

    lc._encode_jpeg = counted
    try:
        yield box
    finally:
        lc._encode_jpeg = original


@contextmanager
def setting(name, value):
    old = getattr(settings, name)
    setattr(settings, name, value)
    try:
        yield
    finally:
        setattr(settings, name, old)


# ---- LiveCamera with a scripted source ---------------------------------------------


def test_no_clients_means_no_stream_and_no_encoding():
    src = ScriptedSource()
    with counting_encodes() as encodes, running(src) as cam:
        assert wait_until(lambda: cam.state == "idle"), cam.state
        time.sleep(0.4)
        assert "start_stream" not in src.calls, src.calls
        assert cam.latest_preview() is None
        assert encodes["n"] == 0
        st = cam.status()
        assert st["clients"] == 0 and st["frame_seq"] == 0 and st["fps_measured"] == 0.0
        assert st["source"] == "fake" and st["enabled"] is True


def test_one_client_gets_downscaled_jpegs_at_the_target_rate():
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        cam.add_client()
        try:
            assert wait_until(lambda: cam.latest_preview() is not None)
            first = cam.latest_preview()
            img = decode(first.jpeg)
            assert img is not None and img.shape[:2] == (240, 320), None if img is None else img.shape
            assert (first.w, first.h) == (320, 240)
            t0, s0 = time.monotonic(), first.seq
            time.sleep(1.0)
            last, t1 = cam.latest_preview(), time.monotonic()
            rate = (last.seq - s0) / (t1 - t0)
            target = settings.preview_fps
            assert last.seq > s0
            assert rate <= 1.3 * target, f"{rate:.1f} fps for a {target:g} fps target"
            assert rate >= 0.5 * target, f"stream too slow: {rate:.1f} fps"
            st = cam.status()
            assert st["state"] == "streaming" and st["clients"] == 1
            assert 0 < st["fps_measured"] <= 1.3 * target, st["fps_measured"]
            assert (st["preview_width"], st["preview_height"]) == (320, 240)
            assert st["last_frame_age_s"] is not None and st["last_frame_age_s"] < 1.0
        finally:
            cam.remove_client()
        # PREVIEW_IDLE_STOP_S=0.3 after the last viewer left: the stream stops.
        assert wait_until(lambda: cam.state == "idle", timeout=2.0), cam.state
        assert "stop_stream" in src.calls
        assert cam.status()["fps_measured"] == 0.0


def test_stream_without_frames_reports_its_age():
    # The Windows-firewall case: the stream runs, but no frame ever arrives.
    # last_frame_age_s stays null, so the dashboard needs stream_age_s to tell
    # "no frame for N s" from "just started".
    src = ScriptedSource()
    src.no_frames = True
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        cam.add_client()
        try:
            assert wait_until(lambda: cam.state == "streaming")
            time.sleep(0.3)
            st = cam.status()
            assert st["state"] == "streaming" and st["last_frame_age_s"] is None, st
            assert st["fps_measured"] == 0.0 and st["frame_seq"] >= 0
            assert isinstance(st["stream_age_s"], float) and st["stream_age_s"] >= 0.25, st
            assert st["frames_incomplete"] > 0 and "no frame within" in st["last_error"], st
            src.no_frames = False  # frames arrive (the firewall rule was added)
            assert wait_until(lambda: cam.status()["last_frame_age_s"] is not None)
            st = cam.status()
            assert st["stream_age_s"] >= 0.3 and st["last_frame_age_s"] < 1.0, st
        finally:
            cam.remove_client()
        assert wait_until(lambda: cam.state == "idle", timeout=2.0)
        assert cam.status()["stream_age_s"] is None  # only reported while streaming


def test_idle_grace_keeps_the_stream_warm_but_encodes_nothing():
    src = ScriptedSource()
    with setting("preview_idle_stop_s", 1.0), running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        cam.add_client()
        assert wait_until(lambda: cam.latest_preview() is not None)
        cam.remove_client()
        time.sleep(0.05)  # let an in-flight publish finish
        with counting_encodes() as encodes:
            reads = src.calls.count("read")
            time.sleep(0.4)
            assert cam.state == "streaming"  # still warm for a quick re-attach
            assert src.calls.count("read") > reads
            assert encodes["n"] == 0, encodes["n"]
        assert wait_until(lambda: cam.state == "idle", timeout=2.0)


def test_capture_is_the_first_frame_after_the_trigger():
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        cam.add_client()
        try:
            assert wait_until(lambda: src.streaming and cam.latest_preview() is not None)
            time.sleep(0.1)
            before = src.last_streamed
            frame, info = cam.capture_frame()
            assert fake.read_stamp(frame) > before, (fake.read_stamp(frame), before)
            assert frame.shape == (480, 640, 3)  # full resolution, not the preview
            assert info == {"source": "basler", "camera_serial": "F0001", "camera_model": "FakeCam"}
            assert wait_until(lambda: src.streaming)  # resumed for the viewer
            names = src.calls
            i = names.index("grab_fresh")
            assert names[i - 1] == "stop_stream", names[max(0, i - 3): i + 3]
            assert names[i + 1] == "start_stream", names[max(0, i - 3): i + 3]
        finally:
            cam.remove_client()


def test_capture_while_idle_does_not_start_the_stream():
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        frame, info = cam.capture_frame()
        assert info["source"] == "basler" and fake.read_stamp(frame) > 0
        time.sleep(0.1)
        assert "grab_fresh" in src.calls
        assert "start_stream" not in src.calls and "stop_stream" not in src.calls
        assert cam.state == "idle"


def test_concurrent_captures_each_get_their_own_grab():
    src = ScriptedSource()
    src.grab_delay = 0.05
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        results = []
        threads = [threading.Thread(target=lambda: results.append(cam.capture_frame())) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        stamps = sorted(fake.read_stamp(f) for f, _ in results)
        assert len(stamps) == 3 and len(set(stamps)) == 3, stamps
        assert all(info["source"] == "basler" for _, info in results)
        assert src.calls.count("grab_fresh") == 3


def test_source_lost_reconnects_with_bounded_backoff():
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        cam.add_client()
        try:
            assert wait_until(lambda: cam.state == "streaming" and cam.latest_preview() is not None)
            opens_before = len(src.open_times)
            seen = set()
            src.open_failures = 2
            t_lost = time.monotonic()
            src.lose_on_read = True
            recovered = wait_until(
                lambda: seen.add(cam.state) or (
                    cam.state == "streaming" and len(src.open_times) >= opens_before + 3),
                timeout=3.0, step=0.002)
            assert recovered, (cam.state, seen, len(src.open_times) - opens_before)
            assert "reconnecting" in seen, seen
            st = cam.status()
            assert st["reconnects"] == 1, st
            assert st["last_error"], st
            assert "close" in src.calls
            opens = src.open_times[opens_before:]
            gaps = [b - a for a, b in zip(opens, opens[1:])]
            cap = settings.camera_reconnect_max_backoff_s
            assert opens[0] - t_lost < 0.5, "a healthy camera is re-opened at once"
            assert all(g <= cap + 0.25 for g in gaps), gaps
            seq = cam.latest_preview().seq
            assert wait_until(lambda: cam.latest_preview().seq > seq)
        finally:
            cam.remove_client()


def test_flapping_camera_is_throttled_by_the_backoff():
    src = ScriptedSource()
    src.lose_always = True  # opens fine, dies on the first frame, every time
    with running(src) as cam:
        cam.add_client()
        try:
            time.sleep(1.0)
            opens = len(src.open_times)
            cap = settings.camera_reconnect_max_backoff_s
            # without backoff this would be hundreds of open/close cycles
            assert 2 <= opens <= 1.0 / cap + 3, opens
            assert cam.status()["reconnects"] >= 2
        finally:
            cam.remove_client()


def test_capture_during_an_outage_falls_back_quickly():
    src = ScriptedSource()
    src.open_failures = 10**9  # camera never comes back
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "reconnecting")
        t0 = time.monotonic()
        frame, info = cam.capture_frame()
        elapsed = time.monotonic() - t0
        assert info["source"] == "fallback", info
        assert np.array_equal(frame, SAMPLE)
        assert elapsed < 1.0, f"{elapsed:.2f} s (LIVE_CAPTURE_TIMEOUT_S={settings.live_capture_timeout_s:g})"


def test_placeholder_while_reconnecting_with_a_viewer():
    src = ScriptedSource()
    src.open_failures = 10**9
    with running(src) as cam:
        cam.add_client()
        try:
            assert wait_until(lambda: cam.latest_preview() is not None)
            first = cam.latest_preview()
            assert decode(first.jpeg) is not None
            # re-published about every second while the viewer waits
            assert wait_until(lambda: cam.latest_preview().seq > first.seq, timeout=2.5)
        finally:
            cam.remove_client()


def test_slow_capture_times_out_to_the_fallback_and_the_late_frame_is_dropped():
    src = ScriptedSource()
    with setting("live_capture_timeout_s", 0.3), running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        src.grab_delay = 0.8
        t0 = time.monotonic()
        frame, info = cam.capture_frame()
        elapsed = time.monotonic() - t0
        assert info["source"] == "fallback" and 0.25 <= elapsed < 0.75, (info, elapsed)
        src.grab_delay = 0.0
        time.sleep(0.7)  # the owner finishes the slow grab and drops it
        frame, info = cam.capture_frame()
        assert info["source"] == "basler", info


def test_failed_grab_falls_back_without_reconnecting():
    # Timeouts only (no image arrived): straight to sample.jpg, no one-shot
    # retake on top (the keypad's 40 s budget), and no reconnect.
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        src.grab_error = GrabFailed("3 timeouts", timed_out=True)
        frame, info = cam.capture_frame()
        assert info["source"] == "fallback"
        assert cam.status()["reconnects"] == 0 and "close" not in src.calls
        src.grab_error = None
        assert cam.capture_frame()[1]["source"] == "basler"


def test_unexpected_grab_error_falls_back_and_reconnects():
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")
        src.grab_error = RuntimeError("driver hiccup")
        frame, info = cam.capture_frame()
        assert info["source"] == "fallback"
        src.grab_error = None
        assert wait_until(lambda: cam.state == "idle" and cam.status()["reconnects"] == 1)
        assert "close" in src.calls
        assert cam.capture_frame()[1]["source"] == "basler"


def test_wifi_profile_is_unavailable_but_captures_work():
    # WiFi transport profile (2 Mbit/s): a full frame takes ~8 s, longer than a
    # LIVE_GRAB_TIMEOUT_MS attempt (5 s) and than LIVE_CAPTURE_TIMEOUT_S would
    # allow for three of them. Scaled down: 200 ms attempts (this file's
    # LIVE_GRAB_TIMEOUT_MS), a 0.3 s caller budget, and a fake GrabOne that
    # needs 0.5 s and, like pylon, times out after its timeout. The capture
    # must still be the camera's photo, not sample.jpg.
    fake.reset(devices=1)
    with setting("camera_transport", "wifi"), setting("live_capture_timeout_s", 0.3):
        cam = LiveCamera(source_factory=BaslerSource)
        cam.start()
        try:
            assert wait_until(lambda: cam.state == "unavailable"), cam.state
            camera = fake.last_camera()
            cam.add_client()
            try:
                assert wait_until(lambda: cam.latest_preview() is not None)
                assert decode(cam.latest_preview().jpeg) is not None  # the placeholder
                camera.grab_delay = 0.5
                frame, info = cam.capture_frame()
                assert info["source"] == "basler", info
                assert frame.shape == (48, 64, 3)
                grabs = fake.calls(camera, "GrabOne")
                # ONE attempt with the one-shot path's 30 s budget
                assert [g[2][0] for g in grabs] == [30000], grabs
                assert "StartGrabbing" not in fake.methods(camera)  # never streams on WiFi
                assert cam.status()["profile"] == "wifi" and cam.state == "unavailable"
            finally:
                cam.remove_client()
        finally:
            cam.stop()


def test_wifi_capture_queued_before_the_camera_opened_gets_the_longer_budget():
    # The first capture after a (re)connect: the caller starts waiting with the
    # normal budget; the owner then opens the camera onto the WiFi profile, and
    # the caller must keep waiting for the (slow) grab instead of giving up.
    gate = threading.Event()
    fake.reset(devices=1, open_gate=gate)
    with setting("camera_transport", "wifi"), setting("live_capture_timeout_s", 0.3):
        cam = LiveCamera(source_factory=BaslerSource)
        cam.start()
        try:
            assert wait_until(lambda: fake.CAMERAS and fake.methods(fake.last_camera()))
            camera = fake.last_camera()  # Open() is blocked on the gate
            camera.grab_delay = 0.5
            results = []
            caller = threading.Thread(target=lambda: results.append(cam.capture_frame()))
            caller.start()
            time.sleep(0.1)
            gate.set()
            caller.join(10)
            assert results, "capture did not return"
            assert results[0][1]["source"] == "basler", results[0][1]
        finally:
            gate.set()
            cam.stop()


def test_sample_source_overlays_the_preview_but_never_the_capture():
    with running(lambda: SampleSource()) as cam:
        assert wait_until(lambda: cam.state == "idle")
        cam.add_client()
        try:
            assert wait_until(lambda: cam.latest_preview() is not None)
            preview = decode(cam.latest_preview().jpeg)
            assert preview.shape[:2] == (180, 320), preview.shape  # 1920x1080 -> 320 wide

            def yellow_pixels(img):  # the label text colour (BGR 0, 215, 255)
                b, g, r = (img[:24, :160, c].astype(int) for c in range(3))
                return int(((b < 110) & (g > 120) & (r > 150)).sum())

            plain = cv2.resize(SAMPLE, (320, 180), interpolation=cv2.INTER_AREA)
            assert yellow_pixels(plain) == 0
            assert yellow_pixels(preview) >= 10, "no SAMPLE label on the preview"
            frame, info = cam.capture_frame()
            assert info == {"source": "fallback", "camera_serial": None, "camera_model": None}
            assert np.array_equal(frame, SAMPLE), "the capture must be the untouched sample"
            assert cam.status()["source"] == "sample"
        finally:
            cam.remove_client()


def test_client_limit():
    cam = LiveCamera(source_factory=ScriptedSource)
    with setting("preview_max_clients", 2):
        cam.add_client()
        cam.add_client()
        assert not cam.has_capacity()
        try:
            cam.add_client()
            raise AssertionError("expected TooManyClients")
        except TooManyClients:
            pass
        cam.remove_client()
        assert cam.has_capacity()
        cam.remove_client()
        cam.remove_client()  # never goes negative
        assert cam.status()["clients"] == 0


def test_stop_is_fast_and_closes_the_source():
    src = ScriptedSource()
    cam = LiveCamera(source_factory=lambda: src)
    cam.start()
    assert wait_until(lambda: cam.state == "idle")
    cam.add_client()
    assert wait_until(lambda: cam.state == "streaming")
    t0 = time.monotonic()
    cam.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"stop took {elapsed:.2f} s"
    assert "close" in src.calls and not src.is_open
    assert cam.state == "stopped" and not cam.running and cam.stopping
    frame, info = cam.capture_frame()  # after stop: straight to the fallback
    assert info["source"] == "fallback"


def test_status_shapes_match_the_contract():
    keys = {"enabled", "state", "source", "camera_model", "camera_serial", "profile",
            "fps_target", "fps_measured", "clients", "frame_seq", "last_frame_age_s",
            "stream_age_s", "frames_incomplete", "reconnects", "one_shot_captures", "last_error",
            "preview_width", "preview_height", "boot_id"}
    cam = LiveCamera(source_factory=ScriptedSource)
    assert set(cam.status()) == keys
    off = cam.disabled_status()
    assert set(off) == keys and off["enabled"] is False and off["state"] == "disabled"
    assert off["stream_age_s"] is None
    assert cam.status()["state"] == "stopped" and cam.status()["stream_age_s"] is None


# ---- MJPEG generator (api/preview.py) ------------------------------------------------


def _preview_module():
    try:
        import api.preview as preview  # noqa: WPS433
    except ImportError as e:  # fastapi missing
        print(f"  (skipped: {e})")
        return None
    return preview


def _check_part(part: bytes) -> None:
    head, _, rest = part.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    assert lines[0] == b"--frame", lines
    assert lines[1] == b"Content-Type: image/jpeg", lines
    assert lines[2].startswith(b"Content-Length: "), lines
    n = int(lines[2].split(b": ", 1)[1])
    assert len(rest) == n + 2 and rest.endswith(b"\r\n"), (len(rest), n)
    assert decode(rest[:n]) is not None


def test_mjpeg_generator_parts_and_cleanup():
    preview = _preview_module()
    if preview is None:
        return
    src = ScriptedSource()
    with running(src) as cam:
        assert wait_until(lambda: cam.state == "idle")

        async def two_parts():
            gen = preview.mjpeg_parts(cam)
            parts = [await gen.__anext__(), await gen.__anext__()]
            during = cam.status()["clients"]
            await gen.aclose()
            return parts, during

        parts, during = asyncio.run(two_parts())
        assert during == 1
        for part in parts:
            _check_part(part)
        assert cam.status()["clients"] == 0

        # A cancelled consumer (client gone mid-wait) also releases the slot.
        async def cancelled():
            async def consume():
                async for _ in preview.mjpeg_parts(cam):
                    pass
            task = asyncio.ensure_future(consume())
            await asyncio.sleep(0.2)
            assert cam.status()["clients"] == 1
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(cancelled())
        assert cam.status()["clients"] == 0


def test_mjpeg_generator_ends_when_the_camera_stops_and_respects_the_limit():
    preview = _preview_module()
    if preview is None:
        return
    src = ScriptedSource()
    cam = LiveCamera(source_factory=lambda: src)
    cam.start()
    try:
        async def until_stopped():
            n = 0
            async for part in preview.mjpeg_parts(cam):
                _check_part(part)
                n += 1
                if n == 1:
                    threading.Thread(target=cam.stop).start()
            return n

        t0 = time.monotonic()
        n = asyncio.run(until_stopped())
        assert n >= 1 and time.monotonic() - t0 < 3.0
        assert cam.status()["clients"] == 0
    finally:
        cam.stop()

    with setting("preview_max_clients", 1):
        cam2 = LiveCamera(source_factory=ScriptedSource)
        cam2.add_client()

        async def refused():
            return [p async for p in preview.mjpeg_parts(cam2)]

        assert asyncio.run(refused()) == []
        assert cam2.status()["clients"] == 1
        cam2.remove_client()


# ---- BaslerSource against the fake pylon ----------------------------------------------


def test_basler_source_caps_the_frame_rate_on_both_node_names():
    for node_name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
        fake.reset(devices=1, frame_rate_node=node_name)
        src = BaslerSource()
        src.open()
        try:
            camera = fake.last_camera()
            assert camera.node("AcquisitionFrameRateEnable").value is True
            assert camera.node(node_name).value == settings.preview_fps, camera.node(node_name).value
            assert src.streamable and src.profile == "ethernet"
            assert (src.camera_model, src.camera_serial) == ("acA2500-14gc", "20000000")
        finally:
            src.close()
    # Neither node: still opens (the owner thread paces the preview).
    fake.reset(devices=1, frame_rate_node=None)
    src = BaslerSource()
    src.open()
    assert src.is_open
    src.close()


def test_basler_stream_is_latest_image_only_and_capture_stops_it_first():
    fake.reset(devices=1)
    src = BaslerSource()
    src.open()
    try:
        camera = fake.last_camera()
        src.start_stream()
        assert camera.grabbing and camera.strategy == fake.GrabStrategy_LatestImageOnly
        streamed = src.read_stream(1.0)
        assert streamed is not None and streamed.shape == (48, 64, 3)
        frame, info = src.grab_fresh()
        assert fake.read_stamp(frame) > fake.read_stamp(streamed)
        assert info == {"source": "basler", "camera_serial": "20000000", "camera_model": "acA2500-14gc"}
        m = fake.methods(camera)
        i = m.index("GrabOne")
        stream_calls = [x for x in m[:i] if x in ("StartGrabbing", "StopGrabbing", "RetrieveResult")]
        assert stream_calls[-1] == "StopGrabbing", stream_calls
        assert fake.calls(camera, "GrabOne")[0][2][0] == settings.live_grab_timeout_ms
        assert not camera.grabbing
    finally:
        src.close()


def test_basler_grab_retries_a_timeout():
    fake.reset(devices=1)
    src = BaslerSource()
    src.open()
    try:
        camera = fake.last_camera()
        camera.grabone_timeouts = 1
        frame, info = src.grab_fresh()
        assert info["source"] == "basler"
        assert len(fake.calls(camera, "GrabOne")) == 2
        camera.grabone_timeouts = 3
        try:
            src.grab_fresh()
            raise AssertionError("expected GrabFailed")
        except GrabFailed:
            pass
        assert src.is_open and not camera.destroyed  # a failed grab is not a lost camera
    finally:
        src.close()


def test_basler_stream_counts_timeouts_and_incomplete_frames():
    fake.reset(devices=1)
    src = BaslerSource()
    src.open()
    try:
        camera = fake.last_camera()
        src.start_stream()
        camera.retrieve_script.extend(["timeout", "incomplete"])
        assert src.read_stream(0.05) is None
        assert src.read_stream(0.05) is None
        assert src.frames_incomplete == 2 and "incomplete" in src.last_error
        assert src.read_stream(0.05) is not None
        camera.grabbing = False  # grab engine died underneath us
        try:
            src.read_stream(0.05)
            raise AssertionError("expected SourceLost")
        except SourceLost:
            pass
    finally:
        src.close()


def test_basler_device_removal_raises_source_lost():
    fake.reset(devices=1)
    src = BaslerSource()
    src.open()
    camera = fake.last_camera()
    src.start_stream()
    camera.removed = True
    for step in (lambda: src.read_stream(0.1), src.grab_fresh, src.check_alive):
        try:
            step()
            raise AssertionError("expected SourceLost")
        except SourceLost:
            pass
    src.close()
    assert camera.destroyed and not src.is_open


def test_basler_wifi_profile_is_not_streamable():
    fake.reset(devices=1)
    with setting("camera_transport", "wifi"):
        src = BaslerSource()
        src.open()
        try:
            assert src.profile == "wifi" and src.streamable is False
        finally:
            src.close()


def test_basler_open_failure_leaves_no_half_open_handle():
    fake.reset(devices=1, fail_open=fake.AccessException("camera in use by another app"))
    src = BaslerSource()
    try:
        src.open()
        raise AssertionError("expected AccessException")
    except fake.AccessException:
        pass
    camera = fake.last_camera()
    assert not src.is_open and camera.destroyed and src._svc._camera is None
    fake.CONFIG.pop("fail_open")
    src.open()
    assert src.is_open and fake.last_camera() is not camera
    src.close()

    fake.reset(devices=0)
    try:
        BaslerSource().open()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "No Basler camera" in str(e)


def test_owner_with_basler_source_end_to_end():
    fake.reset(devices=1)
    cam = LiveCamera(source_factory=BaslerSource)
    cam.start()
    try:
        assert wait_until(lambda: cam.state == "idle"), cam.state
        cam.add_client()
        assert wait_until(lambda: cam.latest_preview() is not None and cam.state == "streaming")
        assert decode(cam.latest_preview().jpeg).shape[:2] == (48, 64)
        frame, info = cam.capture_frame()
        assert info["source"] == "basler" and frame.shape == (48, 64, 3)
        first = fake.last_camera()
        m = fake.methods(first)
        i = m.index("GrabOne")
        before = [x for x in m[:i] if x in ("StartGrabbing", "StopGrabbing", "RetrieveResult")]
        assert before[-1] == "StopGrabbing", before
        assert wait_until(lambda: "StartGrabbing" in fake.methods(first)[i:])  # resumed

        # Unplug: the device disappears from the network, then comes back.
        fake.DEVICES.clear()
        first.removed = True
        assert wait_until(lambda: cam.state == "reconnecting"), cam.state
        assert first.destroyed
        fake.DEVICES.append(fake.DeviceInfo(serial="20000000"))
        assert wait_until(lambda: cam.state == "streaming" and fake.last_camera() is not first, timeout=3.0)
        assert cam.status()["reconnects"] == 1
        assert cam.capture_frame()[1]["source"] == "basler"
    finally:
        cam.stop()
    assert fake.last_camera().destroyed  # released on stop


# ---- Legacy one-shot path (LIVE_PREVIEW_ENABLED=false) -------------------------------


def _legacy():
    svc = CameraService()
    svc.use_camera = True  # env says false; this instance drives the fake camera
    return svc


def test_legacy_init_failure_releases_the_handle_and_retries():
    fake.reset(devices=1, fail_open=fake.AccessException("in use"))
    svc = _legacy()
    frame, info = svc.capture_frame()
    assert info["source"] == "fallback"
    camera = fake.last_camera()
    assert svc._camera is None and camera.destroyed
    fake.CONFIG.pop("fail_open")
    frame, info = svc.capture_frame()
    assert info["source"] == "basler" and frame.shape == (48, 64, 3)
    assert svc.profile_label == "ethernet"


def test_legacy_grab_timeout_falls_back_after_one_attempt():
    # Flag-off behaviour must stay as before: one 30 s GrabOne, then sample.jpg.
    # Retrying timeouts would push a blocked stream past the keypad (40 s) and
    # dashboard (60 s) request timeouts.
    fake.reset(devices=1)
    svc = _legacy()
    svc.capture_frame()
    camera = fake.last_camera()
    camera.grabone_timeouts = 2
    frame, info = svc.capture_frame()
    assert info["source"] == "fallback", info
    grabs = fake.calls(camera, "GrabOne")
    assert len(grabs) == 1 + 1, grabs
    assert all(c[2] == (30000, fake.TimeoutHandling_ThrowException) for c in grabs)
    assert "StartGrabbing" not in fake.methods(camera)  # never streams


def test_legacy_status_does_not_wait_for_a_capture():
    fake.reset(devices=1)
    svc = _legacy()
    svc.capture_frame()
    camera = fake.last_camera()
    camera.grab_delay = 1.0
    worker = threading.Thread(target=svc.capture_frame)
    worker.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    busy = svc.get_status()
    elapsed = time.monotonic() - t0
    worker.join()
    assert busy.get("camera_busy") is True and elapsed < 0.9, (busy, elapsed)
    idle = svc.get_status()
    assert "camera_busy" not in idle and idle["camera_available"] is True


def test_legacy_captures_are_serialized():
    fake.reset(devices=1)
    svc = _legacy()
    svc.capture_frame()
    fake.last_camera().grab_delay = 0.2
    t0 = time.monotonic()
    workers = [threading.Thread(target=svc.capture_frame) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert time.monotonic() - t0 >= 0.38, "two grabs overlapped"


def test_legacy_removed_device_is_reopened_on_the_next_capture():
    fake.reset(devices=1)
    svc = _legacy()
    svc.capture_frame()
    camera = fake.last_camera()
    camera.removed = True
    frame, info = svc.capture_frame()
    assert info["source"] == "fallback" and svc._camera is None and camera.destroyed
    frame, info = svc.capture_frame()
    assert info["source"] == "basler" and fake.last_camera() is not camera


# ---- The one-shot rescue and the runtime switch (services/live_mode.py) ----------------


class FakeOneShot:
    """camera_service's part in a rescue: capture_frame() + release()."""

    def __init__(self, live_source=None, source="basler"):
        self.live_source = live_source
        self.source = source
        self.events = []

    def capture_frame(self):
        # The live owner must have closed the camera before this opens it.
        self.events.append(("capture", self.live_source.is_open if self.live_source else None))
        frame, _ = fake.next_frame(640, 480)
        return frame, {"source": self.source, "camera_serial": "F0001", "camera_model": "FakeCam"}

    def release(self):
        self.events.append(("release", None))


def test_incomplete_live_grab_is_retaken_with_the_one_shot_path():
    src = ScriptedSource()
    one_shot = FakeOneShot(live_source=src)
    cam = LiveCamera(source_factory=lambda: src, one_shot=one_shot)
    cam.start()
    try:
        assert wait_until(lambda: cam.state == "idle")
        src.grab_error = GrabFailed("incompletely grabbed", timed_out=False)
        frame, info = cam.capture_frame()
        assert info["source"] == "basler" and frame is not None, info
        # owner closed its handle -> one-shot grabbed -> one-shot released -> owner back
        assert one_shot.events == [("capture", False), ("release", None)], one_shot.events
        assert "close" in src.calls
        assert wait_until(lambda: cam.running and cam.state == "idle"), cam.state
        st = cam.status()
        assert st["one_shot_captures"] == 1 and "one-shot" in st["last_error"], st
        src.grab_error = None
        assert cam.capture_frame()[1]["source"] == "basler"
        assert len(one_shot.events) == 2, "a good live grab needs no rescue"
    finally:
        cam.stop()


def test_one_shot_rescue_can_still_end_in_the_sample_image():
    src = ScriptedSource()
    one_shot = FakeOneShot(live_source=src, source="fallback")
    cam = LiveCamera(source_factory=lambda: src, one_shot=one_shot)
    cam.start()
    try:
        assert wait_until(lambda: cam.state == "idle")
        src.grab_error = GrabFailed("incompletely grabbed", timed_out=False)
        assert cam.capture_frame()[1]["source"] == "fallback"
        assert wait_until(lambda: cam.running and cam.state == "idle")  # preview resumes anyway
    finally:
        cam.stop()


def test_basler_grab_failure_says_whether_an_image_arrived():
    fake.reset(devices=1)
    src = BaslerSource()
    src.open()
    try:
        camera = fake.last_camera()
        camera.grabone_timeouts = 3
        try:
            src.grab_fresh()
            raise AssertionError("expected GrabFailed")
        except GrabFailed as e:
            assert e.timed_out is True, e
        camera.grabone_incomplete = 1
        camera.grabone_timeouts = 2
        try:
            src.grab_fresh()
            raise AssertionError("expected GrabFailed")
        except GrabFailed as e:
            assert e.timed_out is False and "incompletely grabbed" in str(e), e
        camera.grabone_incomplete = 1
        frame, info = src.grab_fresh()  # the 2nd attempt is complete
        assert info["source"] == "basler"
    finally:
        src.close()


def test_owner_with_basler_source_rescues_incomplete_grabs_with_the_legacy_path():
    fake.reset(devices=1)
    legacy = _legacy()
    cam = LiveCamera(source_factory=BaslerSource, one_shot=legacy)
    cam.start()
    try:
        assert wait_until(lambda: cam.state == "idle"), cam.state
        live_handle = fake.last_camera()
        live_handle.grabone_incomplete = 3
        frame, info = cam.capture_frame()
        assert info["source"] == "basler" and frame.shape == (48, 64, 3), info
        assert live_handle.destroyed, "the owner released the camera before the one-shot grab"
        assert legacy._camera is None, "the one-shot handle was released afterwards"
        assert wait_until(lambda: cam.state == "idle" and fake.last_camera() is not live_handle)
        assert cam.status()["one_shot_captures"] == 1
    finally:
        cam.stop()


class FakeOwner:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append("owner.start")

    def stop(self):
        self.events.append("owner.stop")


class FakeRelease:
    def __init__(self, events):
        self.events = events

    def release(self):
        self.events.append("one_shot.release")


def _mode(tmp, events):
    from services.live_mode import LiveMode
    return LiveMode(camera=FakeOwner(events), one_shot=FakeRelease(events),
                    state_path=Path(tmp) / "live_preview.json")


def test_live_mode_default_is_the_env_and_a_saved_choice_wins():
    tmp = tempfile.mkdtemp(prefix="zander-livemode-")
    events = []
    with setting("live_preview_enabled", True):
        mode = _mode(tmp, events)
        mode.start()
        assert mode.enabled and mode.origin == "env" and events == ["owner.start"]
        (Path(tmp) / "live_preview.json").write_text('{"enabled": false, "changed_by": "op"}', encoding="utf-8")
        events.clear()
        mode = _mode(tmp, events)
        mode.start()
        assert not mode.enabled and mode.origin == "saved" and mode.changed_by == "op"
        assert events == [], "a saved 'off' never starts the owner"
        for raw in ("{broken", "[]", '{"enabled": "yes"}'):
            (Path(tmp) / "live_preview.json").write_text(raw, encoding="utf-8")
            events.clear()
            mode = _mode(tmp, events)
            mode.start()
            assert mode.enabled and mode.origin == "env", raw


def test_live_mode_switch_hands_the_camera_over_and_is_saved():
    tmp = tempfile.mkdtemp(prefix="zander-livemode-")
    events = []
    with setting("live_preview_enabled", False):
        mode = _mode(tmp, events)
        mode.start()
        assert not mode.enabled and events == []
        assert mode.set(True, "anna@example.com") is True
        assert events == ["one_shot.release", "owner.start"], events  # one owner at a time
        assert mode.enabled and mode.origin == "switch" and mode.changed_by == "anna@example.com"
        saved = json.loads((Path(tmp) / "live_preview.json").read_text(encoding="utf-8"))
        assert saved["enabled"] is True and saved["changed_by"] == "anna@example.com"
        events.clear()
        assert mode.set(True) is True and events == [], "no-op when already on"
        assert mode.set(False) is True and events == ["owner.stop"]
        mode2 = _mode(tmp, [])
        mode2.start()
        assert not mode2.enabled and mode2.origin == "saved"


def test_live_mode_switch_waits_for_a_capture_and_survives_a_failed_save():
    tmp = tempfile.mkdtemp(prefix="zander-livemode-")
    events = []
    with setting("live_preview_enabled", False):
        mode = _mode(tmp, events)
        mode.start()
        held = threading.Event()
        release = threading.Event()

        def capture():  # what api/routes.py does around a grab
            with mode.lock:
                held.set()
                release.wait(2)

        worker = threading.Thread(target=capture)
        worker.start()
        held.wait(2)
        switcher = threading.Thread(target=lambda: mode.set(True))
        switcher.start()
        time.sleep(0.15)
        assert events == [], "the switch must wait for the capture"
        release.set()
        worker.join()
        switcher.join(2)
        assert events == ["one_shot.release", "owner.start"] and mode.enabled
        (Path(tmp) / "live_preview.json").unlink()
        (Path(tmp) / "live_preview.json").mkdir()  # os.replace onto a directory fails
        assert mode.set(False) is False and not mode.enabled  # applies anyway


# ---- GigE network settings (profile + CAMERA_* overrides) -----------------------------


def _set_values(camera):
    return {c[2][0]: c[2][1] for c in fake.calls(camera, "SetValue")}


def test_ethernet_profile_resends_in_time_and_throttles_below_line_rate():
    fake.reset(devices=1)
    svc = _legacy()
    svc.capture_frame()
    values = _set_values(fake.last_camera())
    # PacketTimeout far below FrameRetention, both in ms: a lost packet is re-sent in time.
    assert values["PacketTimeout"] == 40 and values["FrameRetention"] == 1000, values
    assert values["DeviceLinkThroughputLimit"] == 100_000_000, values  # bytes/s = 800 Mbit/s
    assert values["DeviceLinkThroughputLimitMode"] == "On", values  # else ace 2 ignores the limit


def test_env_overrides_the_network_settings():
    fake.reset(devices=1)
    with setting("camera_packet_timeout_ms", 80), setting("camera_frame_retention_ms", 2000), \
            setting("camera_throughput_limit_mbps", 400.0), setting("camera_inter_packet_delay", 1000), \
            setting("camera_packet_size", 8192):
        svc = _legacy()
        svc.capture_frame()
    values = _set_values(fake.last_camera())
    assert values["PacketTimeout"] == 80 and values["FrameRetention"] == 2000, values
    assert values["DeviceLinkThroughputLimit"] == 50_000_000, values  # 400 Mbit/s in bytes/s
    assert values["GevSCPD"] == 1000 and values["GevSCPSPacketSize"] == 8192, values
    from services.camera_service import ETHERNET_PROFILE
    assert ETHERNET_PROFILE["stream_grabber"]["PacketTimeout"] == 40, "overrides never edit the profile"


def test_limit_at_the_camera_max_leaves_the_mode_alone():
    fake.reset(devices=1)
    with setting("camera_throughput_limit_mbps", 5000.0):  # above the 1000 Mbit/s max
        _legacy().capture_frame()
    values = _set_values(fake.last_camera())
    assert values["DeviceLinkThroughputLimit"] == 125_000_000, values
    assert "DeviceLinkThroughputLimitMode" not in values, values


def test_stream_statistics_are_read_best_effort():
    fake.reset(devices=1)
    svc = _legacy()
    svc.capture_frame()
    stats = CameraService.stream_statistics(fake.last_camera())
    assert stats == {"Failed_Packet_Count": 7, "Resend_Request_Count": 3}, stats

    class NoNodeMap:
        def GetStreamGrabberNodeMap(self):
            raise RuntimeError("not open")
    assert CameraService.stream_statistics(NoNodeMap()) == {}


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
