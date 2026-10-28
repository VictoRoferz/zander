"""
Live preview endpoints (docs/CONTRACTS.md section 4).

  GET /api/v1/preview/stream.mjpg   multipart/x-mixed-replace MJPEG stream
  GET /api/v1/preview/frame.jpg     one fresh preview JPEG
  GET /api/v1/preview/status        live camera state (200 even when disabled)
  POST /api/v1/preview/mode         switch the preview on/off at runtime
                                    ({"enabled": bool, "changed_by"?}; saved)

All of them sit behind require_preview_access. Production sets PREVIEW_TOKEN
and the dashboard relays the stream with X-Preview-Token, so the browser never
sees the token. Stream and frame answer 404 while the preview is off
(services/live_mode.py: LIVE_PREVIEW_ENABLED is only the startup default).

The handlers only read what the camera-owner thread published (one JPEG per
frame, shared by all viewers); they never touch the camera.
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import time
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, StrictBool

from config.settings import settings
from services.live_camera import (
    WAITING_TEXT,
    LiveCamera,
    Preview,
    TooManyClients,
    live_camera,
    placeholder_jpeg,
)
from services.live_mode import live_mode
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

BOUNDARY = "frame"
KEEPALIVE_S = 2.0   # the last frame (or a placeholder) is re-sent at least this often
FRAME_WAIT_S = 3.0  # frame.jpg waits this long for a fresh frame
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})

_closing = False  # set once when the server shuts down


def close_streams() -> None:
    """
    Server shutdown: finish every open MJPEG stream within one poll period,
    so uvicorn's graceful shutdown does not have to cancel them (which logs
    a CancelledError traceback). Captures keep working until the lifespan
    stops the camera.
    """
    global _closing
    _closing = True


# ---- Access rule --------------------------------------------------------------


def _host_name(host: str) -> str:
    """'localhost:8001' -> 'localhost', '[::1]:8001' -> '[::1]' (lower-case)."""
    host = host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.split(":", 1)[0]


def _is_loopback(address: Optional[str]) -> bool:
    if not address:
        return False
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)  # ::ffff:127.0.0.1
    return (mapped or ip).is_loopback


def require_preview_access(
    request: Request,
    x_preview_token: Optional[str] = Header(default=None),
) -> None:
    """
    Allowed when PREVIEW_TOKEN is set and X-Preview-Token equals it, or when
    PREVIEW_TOKEN is empty, the peer is loopback and the Host header is
    localhost / 127.0.0.1 / [::1] (the Host check defeats DNS rebinding).
    401 = token missing or wrong, 403 = not local and no token configured.
    """
    expected = settings.preview_token
    if expected:
        if not x_preview_token:
            raise HTTPException(status_code=401, detail="preview token missing")
        if not hmac.compare_digest(x_preview_token.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="preview token invalid")
        return

    peer = request.client.host if request.client else None
    if not (_is_loopback(peer) and _host_name(request.headers.get("host", "")) in _LOCAL_HOSTS):
        raise HTTPException(
            status_code=403,
            detail="live preview is local-only; set PREVIEW_TOKEN for remote access",
        )
    # Local but without a token: also refuse cross-site browser requests.
    # camerapi allows CORS from "*", so any web page open on this machine
    # could otherwise read the preview with fetch().
    origin = request.headers.get("origin")
    if origin and _host_name(urlsplit(origin).netloc) not in _LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="cross-site preview request refused")


router = APIRouter(
    prefix="/api/v1/preview",
    tags=["preview"],
    dependencies=[Depends(require_preview_access)],
)


def _require_enabled() -> None:
    if not live_mode.enabled:
        raise HTTPException(
            status_code=404, detail="live preview is off (switch it on in the dashboard)"
        )


# ---- MJPEG stream --------------------------------------------------------------


def _part(jpeg: bytes) -> bytes:
    return (
        b"--" + BOUNDARY.encode("ascii") + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
        + jpeg + b"\r\n"
    )


async def mjpeg_parts(cam: LiveCamera) -> AsyncIterator[bytes]:
    """
    One multipart part per new preview frame; the last frame (or a
    placeholder) is repeated every KEEPALIVE_S so relays and browsers never
    time out. Registers the viewer INSIDE the generator, so a response that
    never starts cannot leak a slot, and releases it in `finally`
    synchronously, which also runs on cancellation (client gone, shutdown).
    """
    try:
        cam.add_client()
    except TooManyClients:
        logger.warning("Preview stream refused: too many viewers")
        return
    try:
        poll_s = min(0.05, 0.5 / float(settings.preview_fps))
        last_seq: Optional[int] = None
        last_sent = float("-inf")
        while not (cam.stopping or _closing):
            preview = cam.latest_preview()
            now = time.monotonic()
            if preview is not None:
                if preview.seq != last_seq or now - last_sent >= KEEPALIVE_S:
                    last_seq, last_sent = preview.seq, now
                    yield _part(preview.jpeg)
            elif now - last_sent >= KEEPALIVE_S:
                last_sent = now
                yield _part(placeholder_jpeg(WAITING_TEXT))
            await asyncio.sleep(poll_s)
    finally:
        cam.remove_client()


class _ClosingStreamingResponse(StreamingResponse):
    """
    Closes the body generator as soon as the response ends. When the client
    disconnects while a part is being sent, the generator is parked at a
    `yield`; without this its `finally` (remove_client) would only run when
    the garbage collector finalizes it.
    """

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


@router.get("/stream.mjpg")
async def preview_stream() -> StreamingResponse:
    _require_enabled()
    if not live_camera.has_capacity():
        raise HTTPException(status_code=503, detail="too many preview viewers")
    return _ClosingStreamingResponse(
        mjpeg_parts(live_camera),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---- Single frame ----------------------------------------------------------------


async def _fresh_preview(cam: LiveCamera) -> Optional[Preview]:
    fps = float(settings.preview_fps)
    poll_s = min(0.05, 0.5 / fps)
    start = time.monotonic()
    current = cam.latest_preview()
    # A frame of the running stream is fresh enough (at most ~2 periods old).
    if current is not None and cam.state == "streaming" and start - current.t <= max(0.5, 2.0 / fps):
        return current
    deadline = start + FRAME_WAIT_S
    while time.monotonic() < deadline and not (cam.stopping or _closing):
        await asyncio.sleep(poll_s)
        latest = cam.latest_preview()
        if latest is not None and (current is None or latest.seq != current.seq):
            return latest
    return None


@router.get("/frame.jpg")
async def preview_frame() -> Response:
    _require_enabled()
    try:
        live_camera.add_client()  # starts the stream if it is idle
    except TooManyClients:
        raise HTTPException(status_code=503, detail="too many preview viewers")
    try:
        preview = await _fresh_preview(live_camera)
    finally:
        live_camera.remove_client()
    if preview is None:
        raise HTTPException(
            status_code=503, detail=f"no preview frame within {FRAME_WAIT_S:g} s"
        )
    return Response(
        content=preview.jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


# ---- Status ------------------------------------------------------------------------


@router.get("/status")
def preview_status() -> dict:
    if not live_mode.enabled:
        return live_camera.disabled_status()
    return live_camera.status()


# ---- Runtime switch ---------------------------------------------------------------


class ModeRequest(BaseModel):
    enabled: StrictBool
    changed_by: Optional[str] = Field(default=None, max_length=320)


@router.post("/mode")
def set_mode(body: ModeRequest) -> dict:
    """
    Turn the live preview on or off now (and keep it after a restart).
    Plain `def`: it waits for a capture in progress and for the owner
    thread to stop, so it runs in the threadpool. Answers the new
    preview status plus "saved" (false = the choice lasts until restart).
    """
    saved = live_mode.set(body.enabled, body.changed_by)
    return {**preview_status(), "saved": saved}
