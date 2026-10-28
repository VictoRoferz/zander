"""
Live camera owner for camerapi (only used while the live preview is on: see
services/live_mode.py, which starts and stops it at runtime).

With the preview on, ONE daemon thread ("camera-owner") is the only code that
touches pylon. It:
  - opens the camera: BaslerSource (pypylon) or, on dev machines,
    SampleSource (sample.jpg with a "SAMPLE hh:mm:ss" overlay on the preview);
  - streams a downscaled MJPEG preview while viewers are connected
    (StartGrabbing(LatestImageOnly); each frame is resized + JPEG-encoded
    ONCE and shared by all viewers; nothing is encoded without viewers);
  - serves every capture by stopping the stream and calling GrabOne, so the
    capture is the first frame exposed after the trigger, never a buffered
    preview frame; the stream then resumes;
  - hands a capture whose live grabs came back INCOMPLETE to the one-shot
    path (the owner thread stops and closes the camera, camera_service grabs
    the photo exactly as with the preview off, then the owner starts again),
    so the preview never costs a photo. Timeouts are not handed over: the
    stream is not arriving at all, and a 30 s one-shot grab on top would
    break the keypad's 40 s request budget;
  - reconnects with exponential backoff when the camera goes away.

Other threads use the thread-safe API: capture_frame() (same return shape and
sample.jpg fallback semantics as CameraService.capture_frame), add_client() /
remove_client(), latest_preview(), status(). The contract lives in
docs/CONTRACTS.md section 4. With the preview off nothing here runs;
captures use the one-shot path in camera_service.py.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import cv2
import numpy as np

from config.settings import settings
from services.camera_service import (
    ONE_SHOT_GRAB_TIMEOUT_MS,
    PYPYLON_AVAILABLE,
    CameraService,
    camera_service,
)
from utils.logger import setup_logger

if PYPYLON_AVAILABLE:
    from pypylon import pylon
else:  # dev machine without the Basler SDK: only SampleSource is used
    pylon = None

logger = setup_logger(__name__, level=settings.log_level)

PLACEHOLDER_REFRESH_S = 1.0  # placeholders are re-published this often
IDLE_POLL_S = 1.0            # owner wake-up period while not streaming
STABLE_OPEN_S = 10.0         # open this long without trouble = not flapping
LIVE_GRAB_ATTEMPTS = 3       # GrabOne attempts per capture on a streamable link
# A capture caller waits at least this long on the WiFi profile, where the
# owner makes ONE GrabOne of ONE_SHOT_GRAB_TIMEOUT_MS (a frame takes ~8 s+).
SLOW_LINK_CAPTURE_WAIT_S = 35.0
RECONNECT_TEXT = "Camera unavailable - reconnecting ..."
WIFI_TEXT = "Preview unavailable (wifi profile)"
WAITING_TEXT = "Waiting for camera ..."


class SourceLost(Exception):
    """The camera went away (cable, power, heartbeat timeout): reconnect."""


class GrabFailed(RuntimeError):
    """A capture grab failed but the camera is still there (no reconnect).

    timed_out: every attempt timed out (no image arrived at all). False means
    at least one image came back incomplete, which the one-shot path may still
    grab cleanly (see LiveCamera._one_shot_capture)."""

    def __init__(self, message: str, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


class TooManyClients(Exception):
    """PREVIEW_MAX_CLIENTS viewers are already connected."""


@dataclass(frozen=True)
class Preview:
    """One encoded preview JPEG. Published by a single attribute swap."""

    seq: int
    jpeg: bytes
    t: float  # time.monotonic() at publish
    w: int
    h: int


def _source_info(
    source: str, serial: Optional[str] = None, model: Optional[str] = None
) -> dict[str, Any]:
    """Same shape as CameraService.capture_frame's source_info."""
    return {"source": source, "camera_serial": serial, "camera_model": model}


# ---- Frame sources ----------------------------------------------------------
#
# Interface used by the owner thread (duck-typed): name, streamable, is_open,
# camera_model, camera_serial, profile, frames_incomplete, last_error,
# preview_label, open(), close(), start_stream(), stop_stream(),
# read_stream(timeout_s) -> frame | None, grab_fresh() -> (frame, info),
# optional check_alive(). Only the owner thread calls them.


class SampleSource:
    """sample.jpg posing as a camera (USE_CAMERA=false or no pypylon)."""

    name = "sample"
    preview_label = "SAMPLE"  # drawn on preview frames: never mistaken for live
    streamable = True
    profile: Optional[str] = None
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._next_t = 0.0
        self.frames_incomplete = 0
        self.last_error: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self._frame is not None

    def open(self) -> None:
        if self._frame is None:
            # File I/O only; loaded once. Same file and rules as the legacy
            # fallback (a test image is generated if sample.jpg is missing).
            self._frame, _ = camera_service.load_fallback()

    def close(self) -> None:
        self._frame = None

    def start_stream(self) -> None:
        self._next_t = time.monotonic()

    def stop_stream(self) -> None:
        pass

    def read_stream(self, timeout_s: float) -> Optional[np.ndarray]:
        """Returns the (shared, read-only) sample frame at PREVIEW_FPS."""
        wait = self._next_t - time.monotonic()
        if wait > timeout_s:
            time.sleep(timeout_s)
            return None
        if wait > 0:
            time.sleep(wait)
        self._next_t = max(self._next_t, time.monotonic()) + 1.0 / float(settings.preview_fps)
        return self._frame

    def grab_fresh(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self._frame is None:
            raise SourceLost("sample source is closed")
        return self._frame.copy(), _source_info("fallback")


class BaslerSource:
    """The Basler GigE camera via pypylon, on a PRIVATE CameraService."""

    name = "basler"
    preview_label: Optional[str] = None

    def __init__(self) -> None:
        # Never the module-level camera_service singleton (the legacy path):
        # this instance holds its own handle and reuses _init_pylon's
        # resolution, pixel format and transport-profile setup.
        self._svc = CameraService()
        self._camera: Any = None
        self._converter: Any = None
        self.streamable = True
        self.profile: Optional[str] = None
        self.camera_model: Optional[str] = None
        self.camera_serial: Optional[str] = None
        self.frames_incomplete = 0
        self.last_error: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self._camera is not None

    def open(self) -> None:
        camera, converter = self._svc.open_pylon()
        try:
            self._cap_frame_rate(camera, float(settings.preview_fps))
            self.profile = self._svc.profile_label
            # The WiFi profile caps the link at 2 Mbit/s: a stream would
            # starve the captures, so the preview is unavailable there, and a
            # capture gets the one-shot path's budget (see grab_fresh).
            self.streamable = self.profile != "wifi"
            try:
                info = camera.GetDeviceInfo()
                self.camera_model = info.GetModelName()
                self.camera_serial = info.GetSerialNumber()
            except Exception as e:
                logger.debug(f"GetDeviceInfo failed: {e}")
        except Exception:
            self._svc.reset_pylon()
            raise
        self._camera, self._converter = camera, converter

    @staticmethod
    def _cap_frame_rate(camera: Any, fps: float) -> None:
        """
        Let the camera itself run at the preview rate (saves link bandwidth
        and host CPU). Newer cameras call the node AcquisitionFrameRate, older
        ace GigE models AcquisitionFrameRateAbs. The owner thread paces reads
        anyway, so a camera without either node still previews at PREVIEW_FPS.
        """
        CameraService._try_set(
            lambda: camera.AcquisitionFrameRateEnable.SetValue(True),
            "AcquisitionFrameRateEnable=True",
        )
        errors = []
        for node_name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
            try:
                node = getattr(camera, node_name)  # pypylon raises for a missing node
                value = fps
                try:
                    value = min(fps, float(node.GetMax()))
                except Exception:
                    pass
                node.SetValue(value)
                logger.info(f"Set {node_name}={value:g} (preview frame-rate cap)")
                return
            except Exception as e:
                errors.append(f"{node_name}: {e}")
        logger.warning(
            "Could not cap the camera frame rate (" + "; ".join(errors) + "); "
            "the owner thread still paces the preview"
        )

    def close(self) -> None:
        self._camera = None
        self._converter = None
        self._svc.reset_pylon()

    def check_alive(self) -> None:
        self._raise_if_removed()

    def _raise_if_removed(self) -> None:
        camera = self._camera
        if camera is None:
            raise SourceLost("camera is not open")
        try:
            removed = bool(camera.IsCameraDeviceRemoved())
        except Exception as e:
            raise SourceLost(f"device state unknown: {e}") from e
        if removed:
            raise SourceLost("camera device removed")

    def start_stream(self) -> None:
        try:
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            raise SourceLost(f"StartGrabbing failed: {e}") from e

    def stop_stream(self) -> None:
        camera = self._camera
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception as e:  # documented as non-throwing; be safe anyway
            raise SourceLost(f"StopGrabbing failed: {e}") from e

    def read_stream(self, timeout_s: float) -> Optional[np.ndarray]:
        camera = self._camera
        try:
            result = camera.RetrieveResult(
                max(1, int(timeout_s * 1000)), pylon.TimeoutHandling_Return
            )
        except Exception as e:
            raise SourceLost(f"RetrieveResult failed: {e}") from e
        try:
            if result is None or not result.IsValid():
                # Timed out. A removed device or a stopped grab engine would
                # otherwise return "no frame" forever.
                self._raise_if_removed()
                if not camera.IsGrabbing():
                    raise SourceLost("grabbing stopped unexpectedly")
                self.frames_incomplete += 1
                self.last_error = f"no frame within {timeout_s:.1f} s"
                return None
            if not result.GrabSucceeded():
                self._raise_if_removed()
                self.frames_incomplete += 1
                self.last_error = f"incomplete frame: {result.ErrorDescription}"
                return None
            return self._converter.Convert(result).GetArray()
        finally:
            if result is not None:
                try:
                    result.Release()
                except Exception:
                    pass

    def grab_fresh(
        self, timeout_ms: Optional[int] = None, attempts: Optional[int] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        camera = self._camera
        if camera is None:
            raise SourceLost("camera is not open")
        if self.streamable:
            timeout_ms = int(timeout_ms or settings.live_grab_timeout_ms)
            attempts = int(attempts or LIVE_GRAB_ATTEMPTS)
        else:
            # WiFi profile: a full frame needs ~8 s+ over the 2 Mbit/s link, so
            # every LIVE_GRAB_TIMEOUT_MS (5 s) attempt timed out and each capture
            # fell back to sample.jpg. Use the one-shot path's budget instead:
            # ONE attempt of 30 s (a timeout means the stream is not arriving).
            timeout_ms = int(timeout_ms or ONE_SHOT_GRAB_TIMEOUT_MS)
            attempts = int(attempts or 1)
        # GrabOne refuses to run while the stream grabs, and stopping flushes
        # the queue: the exposure starts after this call, so the capture is
        # the first frame after the trigger.
        self.stop_stream()
        last_error: Optional[str] = None
        incomplete = False  # an image arrived, but not all of it
        for attempt in range(1, attempts + 1):
            result = None
            try:
                result = camera.GrabOne(timeout_ms, pylon.TimeoutHandling_ThrowException)
                if result.GrabSucceeded():
                    frame = self._converter.Convert(result).GetArray()
                    if attempt > 1:
                        logger.info(f"Live GrabOne succeeded on attempt {attempt}")
                    return frame, _source_info("basler", self.camera_serial, self.camera_model)
                incomplete = True
                last_error = str(result.ErrorDescription)
                logger.warning(f"Live GrabOne attempt {attempt}/{attempts} failed: {last_error}")
            except pylon.TimeoutException:
                last_error = f"timeout after {timeout_ms} ms"
                logger.warning(f"Live GrabOne attempt {attempt}/{attempts} timed out ({timeout_ms} ms)")
            except Exception as e:
                self._raise_if_removed()
                raise SourceLost(f"GrabOne failed: {e}") from e
            finally:
                if result is not None:
                    try:
                        result.Release()
                    except Exception:
                        pass
            self._raise_if_removed()
        CameraService.log_stream_statistics(camera, "Live grab failed")
        raise GrabFailed(
            f"Basler grab failed after {attempts} attempts: {last_error}", timed_out=not incomplete
        )


def default_source_factory() -> Any:
    """Basler when USE_CAMERA=true and pypylon is installed, else sample.jpg."""
    if settings.use_camera and PYPYLON_AVAILABLE:
        return BaslerSource()
    if settings.use_camera:
        logger.warning(
            "USE_CAMERA=true but pypylon is not installed: the live preview "
            "streams sample.jpg (live mode does not drive OpenCV webcams)"
        )
    return SampleSource()


# ---- Rendering (owner thread; placeholder_jpeg also from the event loop) ----


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _draw_label(image: np.ndarray, text: str) -> None:
    """Black box + yellow text in the top-left corner (in place)."""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(1.5, w / 800.0))
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = max(3, int(round(6 * scale)))
    cv2.rectangle(image, (0, 0), (min(w, tw + 2 * pad), min(h, th + baseline + 2 * pad)), (0, 0, 0), -1)
    cv2.putText(image, text, (pad, pad + th), font, scale, (0, 215, 255), thickness, cv2.LINE_AA)


def _render_preview(frame: np.ndarray, label: Optional[str]) -> tuple[bytes, int, int]:
    """Downscale to PREVIEW_MAX_WIDTH (INTER_AREA) and encode once."""
    h, w = frame.shape[:2]
    max_w = int(settings.preview_max_width)
    if w > max_w:
        new_h = max(1, int(round(h * max_w / w)))
        image = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # The source frame may be shared (SampleSource): never draw on it.
        image = frame.copy() if label else frame
    if label:
        _draw_label(image, label)
    return _encode_jpeg(image, settings.preview_jpeg_quality), image.shape[1], image.shape[0]


def _render_placeholder(text: str, w: int, h: int, stamp: bool = True) -> bytes:
    """A grey card with centred text (and the time, so it visibly refreshes)."""
    image = np.full((h, w, 3), 64, dtype=np.uint8)
    lines = [text] + ([time.strftime("%H:%M:%S")] if stamp else [])
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, w // 480)
    widest = max(cv2.getTextSize(line, font, 1.0, thickness)[0][0] for line in lines)
    scale = max(0.3, min(2.0, 0.85 * w / max(1, widest)))
    line_h = int(cv2.getTextSize("Ag", font, scale, thickness)[0][1] * 2.0)
    y = h // 2 - line_h * (len(lines) - 1) // 2
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.putText(image, line, ((w - tw) // 2, y + th // 2), font, scale, (225, 225, 225), thickness, cv2.LINE_AA)
        y += line_h
    return _encode_jpeg(image, settings.preview_jpeg_quality)


_placeholder_cache: dict[tuple[str, int, int], bytes] = {}


def placeholder_jpeg(text: str) -> bytes:
    """Static placeholder (cached). The stream sends it until a frame exists."""
    w = min(640, int(settings.preview_max_width))
    h = w * 3 // 4
    key = (text, w, h)
    jpeg = _placeholder_cache.get(key)
    if jpeg is None:
        jpeg = _render_placeholder(text, w, h, stamp=False)
        _placeholder_cache[key] = jpeg
    return jpeg


# ---- The owner --------------------------------------------------------------


class _CaptureRequest:
    __slots__ = ("done", "frame", "info", "error", "abandoned", "one_shot")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.frame: Optional[np.ndarray] = None
        self.info: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.abandoned = False
        self.one_shot = False  # the error may be fixed by the one-shot path


class LiveCamera:
    """Owns the camera while the live preview is enabled (see module doc)."""

    def __init__(
        self,
        source_factory: Optional[Callable[[], Any]] = None,
        one_shot: Any = None,
    ) -> None:
        self._source_factory = source_factory or default_source_factory
        # Takes a photo the live grabs could not (capture_frame + release).
        self._one_shot = one_shot if one_shot is not None else camera_service
        self._one_shot_captures = 0
        self._lock = threading.Lock()   # guards _requests, _clients, _last_client_left
        self._wake = threading.Event()  # capture queued / viewer change / stop
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._requests: Deque[_CaptureRequest] = deque()
        self._clients = 0
        self._last_client_left = float("-inf")
        # Written by the owner thread only; read without a lock by status()
        # and the HTTP layer (single attribute reads, fine for monitoring).
        self._preview: Optional[Preview] = None
        self._seq = 0
        self._state = "stopped"
        self._source_name: Optional[str] = None
        self._camera_model: Optional[str] = None
        self._camera_serial: Optional[str] = None
        self._profile: Optional[str] = None
        # The open source cannot stream (WiFi profile): captures get the
        # one-shot budget, so callers wait longer (see _capture_wait_s).
        self._slow_link = False
        self._fps_ema = 0.0
        self._last_frame_t: Optional[float] = None
        # When the running stream started: with _last_frame_t still None it
        # tells "no frame yet" (e.g. the Windows firewall drops the camera's
        # UDP stream) from "just started" (status "stream_age_s").
        self._stream_started_t: Optional[float] = None
        self._frame_w: Optional[int] = None
        self._frame_h: Optional[int] = None
        self._frames_incomplete = 0
        self._reconnects = 0
        self._last_error: Optional[str] = None
        self._placeholder_text: Optional[str] = None
        self._placeholder_t = float("-inf")
        self.boot_id = uuid.uuid4().hex  # changes on every camerapi start

    # ---- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        grab_budget_s = LIVE_GRAB_ATTEMPTS * settings.live_grab_timeout_ms / 1000.0
        if grab_budget_s >= settings.live_capture_timeout_s:
            logger.warning(
                f"LIVE_CAPTURE_TIMEOUT_S={settings.live_capture_timeout_s:g} is not above "
                f"{LIVE_GRAB_ATTEMPTS} x LIVE_GRAB_TIMEOUT_MS ({grab_budget_s:g} s): "
                "slow grabs fall back to sample.jpg"
            )
        self._stop.clear()
        self._wake.clear()
        self._state = "starting"
        logger.info(
            f"Live camera starting (preview {settings.preview_fps:g} fps, "
            f"max width {settings.preview_max_width} px)"
        )
        self._thread = threading.Thread(target=self._run, name="camera-owner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return  # never started (preview disabled)
        self._stop.set()
        self._wake.set()
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("camera-owner thread did not stop in time; it ends with the process")
        self._fail_pending("live camera stopped")
        self._state = "stopped"
        logger.info("Live camera stopped")

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop.is_set()

    @property
    def state(self) -> str:
        return self._state

    # ---- Viewers ----------------------------------------------------------

    def add_client(self) -> None:
        with self._lock:
            if self._clients >= settings.preview_max_clients:
                raise TooManyClients(
                    f"{self._clients} preview viewers connected (PREVIEW_MAX_CLIENTS)"
                )
            self._clients += 1
        self._wake.set()

    def remove_client(self) -> None:
        with self._lock:
            if self._clients > 0:
                self._clients -= 1
            if self._clients == 0:
                self._last_client_left = time.monotonic()
        self._wake.set()

    def has_capacity(self) -> bool:
        with self._lock:
            return self._clients < settings.preview_max_clients

    def _client_count(self) -> int:
        with self._lock:
            return self._clients

    def latest_preview(self) -> Optional[Preview]:
        return self._preview

    # ---- Captures ---------------------------------------------------------

    def capture_frame(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Same contract as CameraService.capture_frame: (frame, source_info),
        falling back to sample.jpg (source "fallback") when the camera cannot
        deliver in time. Blocks the calling (threadpool) thread.
        """
        if not self.running:
            logger.warning("Live camera is not running; falling back to sample image")
            return camera_service.load_fallback()
        request = _CaptureRequest()
        started = time.monotonic()
        with self._lock:
            self._requests.append(request)
        self._wake.set()

        timeout_s = self._capture_wait_s()
        done = request.done.wait(timeout_s)
        while not done:
            # The camera may have been opened (or re-opened) onto the WiFi
            # profile while this request waited: its longer budget applies,
            # counted from the request.
            longer = self._capture_wait_s()
            if longer <= timeout_s:
                break
            timeout_s = longer
            done = request.done.wait(max(0.0, started + timeout_s - time.monotonic()))
        if not done:
            with self._lock:
                if not request.done.is_set():
                    request.abandoned = True
                    try:
                        self._requests.remove(request)
                    except ValueError:
                        pass  # the owner is grabbing it; it drops the late result
            if request.abandoned:
                logger.warning(
                    f"Live capture got no frame within {timeout_s:g} s; "
                    "falling back to sample image"
                )
                return camera_service.load_fallback()
        if request.error is not None:
            if request.one_shot:
                return self._one_shot_capture(request.error)
            logger.warning(
                f"Live camera capture failed ({request.error}); falling back to sample image"
            )
            return camera_service.load_fallback()
        return request.frame, request.info

    def _one_shot_capture(self, reason: str) -> tuple[np.ndarray, dict[str, Any]]:
        """
        The live grabs came back incomplete: take this photo with the one-shot
        path, exactly as with the preview off, then resume the owner. The
        owner thread closes the camera before the one-shot path opens it, and
        the one-shot handle is released before the owner opens it again (one
        owner at a time). The caller holds live_mode.lock, so no switch and
        no second capture runs meanwhile.
        """
        logger.warning(f"Live capture failed ({reason}); taking this photo with the one-shot path")
        self.stop()
        try:
            frame, info = self._one_shot.capture_frame()  # its own retries + sample fallback
        finally:
            try:
                self._one_shot.release()
            finally:
                self.start()
        self._one_shot_captures += 1
        self._last_error = f"live grab failed ({reason}); photo taken with the one-shot path"
        logger.info(f"One-shot capture after a failed live grab: source={info.get('source')}")
        return frame, info

    def _capture_wait_s(self) -> float:
        """How long a capture caller waits for the owner: LIVE_CAPTURE_TIMEOUT_S,
        and at least SLOW_LINK_CAPTURE_WAIT_S on the WiFi profile, where the
        owner makes one 30 s GrabOne (BaslerSource.grab_fresh). Both stay below
        the keypad listener's 40 s request timeout."""
        wait = float(settings.live_capture_timeout_s)
        if self._slow_link:
            wait = max(wait, SLOW_LINK_CAPTURE_WAIT_S)
        return wait

    def _has_requests(self) -> bool:
        with self._lock:
            return bool(self._requests)

    def _pop_request(self) -> Optional[_CaptureRequest]:
        with self._lock:
            while self._requests:
                request = self._requests.popleft()
                if not request.abandoned:
                    return request
        return None

    def _finish(
        self,
        request: _CaptureRequest,
        frame: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        one_shot: bool = False,
    ) -> None:
        with self._lock:
            if request.abandoned:
                logger.warning("Capture finished after the caller gave up; result discarded")
                return
            request.frame, request.info, request.error = frame, info, error
            request.one_shot = one_shot
            request.done.set()

    def _fail_pending(self, reason: str) -> None:
        with self._lock:
            pending = list(self._requests)
            self._requests.clear()
        for request in pending:
            self._finish(request, error=reason)

    # ---- Status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The CONTRACTS section 4 preview status (also /status "live")."""
        preview = self._preview
        now = time.monotonic()
        last = self._last_frame_t
        age = None if last is None else round(now - last, 2)
        live = self._state == "streaming" and age is not None and age < 2.0
        started = self._stream_started_t
        streaming = self._state == "streaming" and started is not None
        return {
            "enabled": True,
            "state": self._state,
            "source": self._source_name,
            "camera_model": self._camera_model,
            "camera_serial": self._camera_serial,
            "profile": self._profile,
            "fps_target": float(settings.preview_fps),
            "fps_measured": round(self._fps_ema, 2) if live else 0.0,
            "clients": self._client_count(),
            "frame_seq": preview.seq if preview is not None else 0,
            "last_frame_age_s": age,
            "stream_age_s": round(now - started, 2) if streaming else None,
            "frames_incomplete": self._frames_incomplete,
            "reconnects": self._reconnects,
            "one_shot_captures": self._one_shot_captures,
            "last_error": self._last_error,
            "preview_width": self._frame_w,
            "preview_height": self._frame_h,
            "boot_id": self.boot_id,
        }

    def disabled_status(self) -> dict[str, Any]:
        """Same keys as status(), while the live preview is off."""
        return {
            "enabled": False,
            "state": "disabled",
            "source": None,
            "camera_model": None,
            "camera_serial": None,
            "profile": None,
            "fps_target": float(settings.preview_fps),
            "fps_measured": 0.0,
            "clients": 0,
            "frame_seq": 0,
            "last_frame_age_s": None,
            "stream_age_s": None,
            "frames_incomplete": 0,
            "reconnects": 0,
            "one_shot_captures": 0,
            "last_error": None,
            "preview_width": None,
            "preview_height": None,
            "boot_id": self.boot_id,
        }

    def camera_status(self) -> dict[str, Any]:
        """/status "camera" block in live mode: the owner's view, no probing
        (only the owner thread may call pylon)."""
        return {
            "use_camera": settings.use_camera,
            "camera_backend": "pypylon" if PYPYLON_AVAILABLE else "opencv",
            "camera_index": settings.camera_index,
            "fallback_available": Path(settings.fallback_image_path).exists(),
            "camera_available": self._source_name == "basler"
            and self._state in ("idle", "streaming", "unavailable"),
            "camera_model": self._camera_model,
            "camera_serial": self._camera_serial,
            "owner": "live",
        }

    # ---- Owner thread -----------------------------------------------------

    def _set_state(self, state: str) -> None:
        if state != self._state:
            logger.info(f"Live camera state: {self._state} -> {state}")
            self._state = state

    def _want_stream(self) -> bool:
        with self._lock:
            if self._clients > 0:
                return True
            idle_s = time.monotonic() - self._last_client_left
        return idle_s < float(settings.preview_idle_stop_s)

    def _run(self) -> None:
        max_backoff = float(settings.camera_reconnect_max_backoff_s)
        initial_backoff = min(1.0, max_backoff)
        backoff = initial_backoff
        next_open_t = 0.0
        healthy = False  # a frame or a capture succeeded since the last open
        opened_at = 0.0
        streaming = False
        last_read_t = float("-inf")
        source: Any = None
        try:
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    # 1. Make sure the camera is open. A queued capture cuts
                    #    the backoff short: try once more right now.
                    if source is None or not source.is_open:
                        streaming = False
                        now = time.monotonic()
                        if now < next_open_t and not self._has_requests():
                            self._set_state("reconnecting")
                            self._publish_placeholder(RECONNECT_TEXT)
                            self._wake.wait(min(next_open_t - now, PLACEHOLDER_REFRESH_S))
                            continue
                        try:
                            if source is None:
                                source = self._source_factory()
                                self._source_name = getattr(source, "name", None)
                            source.open()
                        except Exception as e:
                            self._open_failed(e, backoff)
                            next_open_t = time.monotonic() + backoff
                            backoff = min(backoff * 2.0, max_backoff)
                            continue
                        healthy, opened_at = False, time.monotonic()
                        self._opened(source)

                    # 2. Captures first.
                    request = self._pop_request()
                    if request is not None:
                        if streaming:
                            source.stop_stream()
                            streaming = False
                        if self._serve_capture(source, request):
                            healthy, backoff = True, initial_backoff
                        continue

                    # 3. Stream only while someone watches (+ idle grace).
                    want = bool(source.streamable) and self._want_stream()
                    if want and not streaming:
                        source.start_stream()
                        streaming = True
                        self._fps_ema, self._last_frame_t = 0.0, None
                        self._stream_started_t = time.monotonic()
                    elif streaming and not want:
                        source.stop_stream()
                        streaming = False

                    # 4. Work.
                    if streaming:
                        self._set_state("streaming")
                        fps = float(settings.preview_fps)
                        pause = last_read_t + 1.0 / fps - time.monotonic()
                        if pause > 0 and self._wake.wait(pause):
                            continue  # capture / viewer change / stop first
                        frame = source.read_stream(max(1.0, 2.0 / fps))
                        last_read_t = time.monotonic()
                        self._frames_incomplete = int(getattr(source, "frames_incomplete", 0) or 0)
                        if frame is None:
                            self._last_error = getattr(source, "last_error", None) or self._last_error
                        else:
                            if not healthy:
                                healthy, backoff = True, initial_backoff
                            self._on_frame(frame, getattr(source, "preview_label", None))
                    else:
                        check_alive = getattr(source, "check_alive", None)
                        if check_alive is not None:
                            check_alive()  # notices an unplug while idle
                        if source.streamable:
                            self._set_state("idle")
                        else:
                            self._set_state("unavailable")
                            self._publish_placeholder(WIFI_TEXT)
                        self._wake.wait(IDLE_POLL_S)
                except Exception as e:
                    streaming = False
                    if not isinstance(e, SourceLost):
                        logger.error(f"Live camera loop error: {type(e).__name__}: {e}", exc_info=True)
                    self._lost(source, e)
                    if healthy or time.monotonic() - opened_at > STABLE_OPEN_S:
                        next_open_t, backoff = 0.0, initial_backoff  # it worked: retry at once
                    else:  # flapping (lost right after an open): back off
                        next_open_t = time.monotonic() + backoff
                        backoff = min(backoff * 2.0, max_backoff)
                    healthy = False
        finally:
            self._fail_pending("live camera stopped")
            if source is not None:
                self._close_quietly(source)
            self._set_state("stopped")

    def _opened(self, source: Any) -> None:
        self._source_name = getattr(source, "name", None)
        self._camera_model = getattr(source, "camera_model", None)
        self._camera_serial = getattr(source, "camera_serial", None)
        self._profile = getattr(source, "profile", None)
        self._slow_link = not source.streamable
        logger.info(
            f"Live camera open: source={self._source_name} model={self._camera_model} "
            f"serial={self._camera_serial} profile={self._profile}"
        )
        if not source.streamable:
            logger.warning(
                "WiFi transport profile: live preview unavailable; each capture "
                f"makes one {ONE_SHOT_GRAB_TIMEOUT_MS // 1000} s GrabOne, as with the preview off"
            )
        self._set_state("idle" if source.streamable else "unavailable")

    def _open_failed(self, error: Exception, retry_in: float) -> None:
        message = f"open failed: {error}"
        level = logger.info if message == self._last_error else logger.warning
        self._last_error = message
        self._set_state("reconnecting")
        level(f"Camera open failed ({error}); retrying in {retry_in:.1f} s")
        self._fail_pending(f"camera unavailable: {error}")
        self._publish_placeholder(RECONNECT_TEXT, force=True)

    def _lost(self, source: Any, error: Exception) -> None:
        self._reconnects += 1
        self._last_error = f"camera lost: {error}"
        self._set_state("reconnecting")
        logger.warning(f"Camera lost ({error}); reconnecting")
        if source is not None:
            self._close_quietly(source)
        self._publish_placeholder(RECONNECT_TEXT, force=True)

    @staticmethod
    def _close_quietly(source: Any) -> None:
        try:
            source.stop_stream()
        except Exception:
            pass
        try:
            source.close()
        except Exception as e:
            logger.debug(f"source close failed: {e}")

    def _serve_capture(self, source: Any, request: _CaptureRequest) -> bool:
        started = time.monotonic()
        try:
            frame, info = source.grab_fresh()
        except GrabFailed as e:
            self._last_error = f"capture failed: {e}"
            self._finish(request, error=str(e), one_shot=not e.timed_out)
            return False
        except SourceLost as e:
            self._finish(request, error=f"camera lost: {e}")
            raise
        except Exception as e:
            self._finish(request, error=f"{type(e).__name__}: {e}")
            raise SourceLost(f"capture error: {e}") from e
        self._finish(request, frame=frame, info=info)
        logger.info(
            f"Live capture grabbed in {time.monotonic() - started:.2f} s "
            f"(source={info.get('source')})"
        )
        return True

    def _on_frame(self, frame: np.ndarray, label: Optional[str]) -> None:
        now = time.monotonic()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                rate = 1.0 / dt
                self._fps_ema = rate if self._fps_ema <= 0 else 0.8 * self._fps_ema + 0.2 * rate
        self._last_frame_t = now
        if self._client_count() == 0:
            return  # idle grace period: keep the stream warm, encode nothing
        text = f"{label} {time.strftime('%H:%M:%S')}" if label else None
        try:
            jpeg, w, h = _render_preview(frame, text)
        except Exception as e:
            logger.warning(f"Preview encode failed: {e}")
            return
        self._frame_w, self._frame_h = w, h
        self._publish(jpeg, w, h)

    def _publish(self, jpeg: bytes, w: int, h: int) -> None:
        self._seq += 1
        self._preview = Preview(self._seq, jpeg, time.monotonic(), w, h)

    def _publish_placeholder(self, text: str, force: bool = False) -> None:
        if self._client_count() == 0:
            return
        now = time.monotonic()
        if (
            not force
            and text == self._placeholder_text
            and now - self._placeholder_t < PLACEHOLDER_REFRESH_S
        ):
            return
        # Same size as the live frames, so the viewer's layout does not jump.
        w = self._frame_w or min(640, int(settings.preview_max_width))
        h = self._frame_h or w * 3 // 4
        try:
            jpeg = _render_placeholder(text, w, h)
        except Exception as e:
            logger.warning(f"Placeholder render failed: {e}")
            return
        self._placeholder_text, self._placeholder_t = text, now
        self._publish(jpeg, w, h)


# Global instance. Its thread is started and stopped by services/live_mode.py
# (at startup and by the dashboard's live view switch).
live_camera = LiveCamera()
