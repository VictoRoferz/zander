"""
Dashboard service (Mac/Windows).

Standalone UI for the camera pipeline:
  - Login (own SQLite users; bcrypt passwords)
  - Counts + thumbnails of unlabeled and labeled captures
  - "Capture" button (proxies to camerapi with X-Triggered-By)
  - "Open Label Studio" button (link)
  - Live camera panel: login-protected relay of camerapi's MJPEG preview
    (GET /api/live.mjpg, GET /api/live/status)
  - "Latest result" panel: the newest capture with its prediction
    (GET /api/latest, GET /api/captures/{id}; docs/CONTRACTS.md section 5)

The "labeled" view is read LIVE from the Label Studio REST API (LS is the
single source of truth), so the dashboard always matches LS. "unlabeled" =
images on disk that LS does not yet report as labeled. Thumbnails are served
from the local image folder (the same files LS serves via local-files).
Predictions are read-only here: the receiver writes
DATA_ROOT/predictions/<id>.pred.json, and "pending" is derived, never stored.

Concurrency: the Label Studio client uses blocking `requests`, and the data
folder can be a slow bind mount (Docker on Windows), so every endpoint that
touches Label Studio, SQLite or the data folder is a plain `def` and runs in
the threadpool. The event loop stays free for the long-lived MJPEG relays,
which share one httpx.AsyncClient.

Cross-platform paths via pathlib. Reads ~/zander-data/ (override DATA_ROOT).
The page polls; no SSE/WebSocket.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anyio
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import auth
import db
from ls_client import LabelStudioUnavailable, LsClient

SERVICE_VERSION = "2.2.0"

# ---- Setup ----------------------------------------------------------------

# Load this service's .env so it can be configured like the others. Existing
# environment variables (e.g. those injected by scripts/launch.py) take
# precedence over the file.
load_dotenv()

LOG = logging.getLogger("dashboard")


def _log_level(name: str) -> Optional[int]:
    """LOG_LEVEL as a logging level, case-insensitively; None for a name that
    is no level. logging.basicConfig raises ValueError for "debug", which
    crashed the dashboard at import (a restart loop in Docker)."""
    level = getattr(logging, str(name).strip().upper(), None)
    return level if isinstance(level, int) else None


_LOG_LEVEL_ENV = os.environ.get("LOG_LEVEL", "INFO")
LOG_LEVEL = _log_level(_LOG_LEVEL_ENV)
logging.basicConfig(
    level=logging.INFO if LOG_LEVEL is None else LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
if LOG_LEVEL is None:
    LOG.warning(f"LOG_LEVEL={_LOG_LEVEL_ENV!a} is not a log level; using INFO")


def _data_root() -> Path:
    override = os.environ.get("DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "zander-data"


def _env_float(name: str, default: float) -> float:
    """A non-negative float from the environment; the default if unset or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        LOG.warning(f"{name}={raw!r} is not a non-negative number; using {default}")
        return default
    return value


DATA_ROOT = _data_root()
UNLABELED_DIR = DATA_ROOT / "unlabeled"
LABELED_DIR = DATA_ROOT / "labeled"
UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)

# Prediction files are written by the receiver only (CONTRACTS section 2).
# Read-only here, so the folder is never created by the dashboard.
_PREDICTIONS_OVERRIDE = os.environ.get("PREDICTIONS_DIR", "").strip()
PREDICTIONS_DIR = (
    Path(_PREDICTIONS_OVERRIDE).expanduser()
    if _PREDICTIONS_OVERRIDE
    else DATA_ROOT / "predictions"
)
# A capture without a prediction file shows "pending" for this long after its
# image landed in unlabeled/, then "none" (inspector off, down or too slow).
PREDICTION_PENDING_TIMEOUT_S = _env_float("PREDICTION_PENDING_TIMEOUT_S", 60.0)

LABELSTUDIO_URL = os.environ.get("LABELSTUDIO_URL", "http://localhost:8081")
# Browser-facing LS URL for the "Open Label Studio" link. Differs from
# LABELSTUDIO_URL under Docker, where the API is reached at the internal
# service name (http://labelstudio:8080) but the link must be host-reachable
# (http://localhost:8081). Defaults to LABELSTUDIO_URL for the native setup.
LABELSTUDIO_PUBLIC_URL = os.environ.get("LABELSTUDIO_PUBLIC_URL", LABELSTUDIO_URL)
CAMERAPI_URL = os.environ.get("CAMERAPI_URL", "http://localhost:8001")
# Shared secret for camerapi's live preview (sent as X-Preview-Token). Must be
# the same value as camerapi's PREVIEW_TOKEN. Empty = rely on camerapi's
# loopback rule, which only works when the dashboard runs natively on the
# camera PC. The browser never sees it: the dashboard proxies the stream.
PREVIEW_TOKEN = os.environ.get("PREVIEW_TOKEN", "").strip()

# Label Studio API access (read-only) for the live labeled view.
LABELSTUDIO_API_KEY = os.environ.get("LABELSTUDIO_API_KEY", "")
LABELSTUDIO_PROJECT_NAME = os.environ.get("LABELSTUDIO_PROJECT_NAME", "PCB Defect Inspection")
LABELSTUDIO_PROJECT_ID = os.environ.get("LABELSTUDIO_PROJECT_ID")  # optional override

ls_client = LsClient(
    base_url=LABELSTUDIO_URL,
    api_key=LABELSTUDIO_API_KEY,
    project_name=LABELSTUDIO_PROJECT_NAME,
    project_id=int(LABELSTUDIO_PROJECT_ID) if LABELSTUDIO_PROJECT_ID else None,
)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8003"))

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Outbound HTTP (camerapi). One shared client; per-request timeouts below.
HTTP_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
CAPTURE_TIMEOUT = httpx.Timeout(60.0, connect=3.0)      # a cold Basler grab is slow
LIVE_STATUS_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
# camerapi re-sends a frame (or a placeholder) at least every 2 s, so 15 s of
# silence means the upstream is stuck and the relay ends.
LIVE_STREAM_TIMEOUT = httpx.Timeout(15.0, connect=3.0)

# Collapse the 3 near-simultaneous poll endpoints (stats/unlabeled/labeled)
# into a single LS pass per ~2s window. The lock makes it one pass even when
# the threadpool runs the three requests concurrently; the snapshot dict is
# replaced as a whole (never mutated), so readers always see one generation.
_LABELED_TTL = 2.0
_labeled_lock = threading.Lock()
_labeled_cache: dict[str, Any] = {"ts": -1e9, "items": [], "ids": set(), "ok": False}

# Capture ids accepted by the capture endpoints (CONTRACTS section 5).
_CAPTURE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_PRED_STATUSES = ("ok", "no_model", "error")
_EMPTY_LATEST: dict[str, Any] = {"capture": None, "state": "none", "prediction": None}


def _new_http_client() -> httpx.AsyncClient:
    """The shared outbound client (created in the lifespan; tests swap this factory)."""
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT)


def _is_local_url(url: str) -> bool:
    try:
        host = httpx.URL(url).host
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "::1")


# ---- App ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize()
    if not LABELSTUDIO_API_KEY:
        LOG.warning(
            "LABELSTUDIO_API_KEY not set — the labeled view will stay empty. "
            "Add it to the dashboard .env."
        )
    if PREVIEW_TOKEN and not (PREVIEW_TOKEN.isascii() and PREVIEW_TOKEN.isprintable()):
        LOG.warning(
            "PREVIEW_TOKEN contains non-ASCII or control characters and cannot "
            "be sent as a header; the live preview will fail. Use e.g. "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    if not PREVIEW_TOKEN and not _is_local_url(CAMERAPI_URL):
        LOG.warning(
            "PREVIEW_TOKEN is empty but camerapi is not on localhost "
            f"({CAMERAPI_URL}): camerapi will refuse the live preview. Set the "
            "same PREVIEW_TOKEN for camerapi and the dashboard (deploy/.env)."
        )
    client = _new_http_client()
    app.state.http = client
    LOG.info(
        f"dashboard v{SERVICE_VERSION} ready: data_root={DATA_ROOT} "
        f"predictions={PREDICTIONS_DIR} ls={LABELSTUDIO_URL} camerapi={CAMERAPI_URL} "
        f"preview_token={'set' if PREVIEW_TOKEN else 'unset'}"
    )
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="zander-dashboard", version=SERVICE_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---- Auth flow ------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None) -> Any:
    return TEMPLATES.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> Any:
    # Plain def: bcrypt takes ~0.25 s of CPU and must not stall the event loop.
    if not auth.verify_password(email, password):
        return TEMPLATES.TemplateResponse(
            request,
            "login.html",
            {"error": "Wrong email or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    session_id, expires_at = auth.login(email)
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    # SameSite=Lax = standard CSRF mitigation for browser logins.
    # secure=False because we run over plain HTTP on the local network.
    response.set_cookie(
        auth.SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        expires=int(expires_at.timestamp()),
    )
    LOG.info(f"login: {email}")
    return response


@app.post("/logout")
def logout(dashboard_session: Optional[str] = Cookie(default=None)) -> Any:
    if dashboard_session:
        auth.logout(dashboard_session)
    redirect = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    redirect.delete_cookie(auth.SESSION_COOKIE)
    return redirect


# ---- Dashboard home -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    dashboard_session: Optional[str] = Cookie(default=None),
) -> Any:
    email = auth.session_to_email(dashboard_session)
    if not email:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "user_email": email,
            "labelstudio_url": LABELSTUDIO_PUBLIC_URL,
            # Cache-buster for app.js/style.css after an upgrade.
            "asset_version": SERVICE_VERSION,
        },
    )


# ---- Read-only data API (used by app.js) ---------------------------------
# Plain `def` on purpose: Label Studio is queried with blocking `requests`.

@app.get("/api/stats")
def stats(_: str = Depends(auth.current_user_email)) -> dict:
    cache = _get_labeled_cached()
    labeled = len(cache["items"])
    unlabeled = len(_unlabeled_from_folder(cache["ids"]))
    return {"unlabeled": unlabeled, "labeled": labeled, "total": labeled + unlabeled}


@app.get("/api/unlabeled")
def api_unlabeled(_: str = Depends(auth.current_user_email)) -> list[dict]:
    cache = _get_labeled_cached()
    return _unlabeled_from_folder(cache["ids"])


@app.get("/api/labeled")
def api_labeled(_: str = Depends(auth.current_user_email)) -> list[dict]:
    return _get_labeled_cached()["items"]


@app.get("/thumb/{kind}/{filename}")
def thumb(
    kind: str,
    filename: str,
    _: str = Depends(auth.current_user_email),
) -> FileResponse:
    # Reject anything that would escape the directory or isn't a .jpg.
    if "/" in filename or "\\" in filename or not filename.endswith(".jpg"):
        raise HTTPException(status_code=400)
    if kind == "labeled":
        # The labeled export may lag the LS annotation; the unlabeled copy is
        # the same physical file, so fall back to it.
        candidates = [LABELED_DIR / filename, UNLABELED_DIR / filename]
    elif kind == "unlabeled":
        candidates = [UNLABELED_DIR / filename]
    else:
        raise HTTPException(status_code=404)
    for path in candidates:
        if path.exists():
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404)


# ---- Latest capture + prediction (CONTRACTS section 5) --------------------

@app.get("/api/latest")
def api_latest(_: str = Depends(auth.current_user_email)) -> dict:
    """The newest capture (by capture time, see _CaptureIndex) with its prediction state."""
    for _attempt in range(2):
        capture_id = _newest_capture()
        if capture_id is None:
            break
        try:
            return _capture_view(capture_id)
        except HTTPException as e:
            if e.status_code != 404:
                raise
            # Deleted between the scan and the read: rescan once.
            _capture_index.invalidate()
    return dict(_EMPTY_LATEST)


@app.get("/api/captures/{capture_id}")
def api_capture(capture_id: str, _: str = Depends(auth.current_user_email)) -> dict:
    return _capture_view(capture_id)


# ---- Live preview relay (camerapi -> browser) ----------------------------

@app.get("/api/live.mjpg")
async def live_mjpg(request: Request, _: str = Depends(auth.current_user_email)) -> Any:
    """
    Relay camerapi's MJPEG preview to a logged-in browser, adding the token.

    Errors: 404 preview disabled, 503 too many viewers, 502 camerapi
    unreachable or token rejected (logged).
    """
    client: httpx.AsyncClient = request.app.state.http
    url = _camerapi_url("/api/v1/preview/stream.mjpg")
    # The bytes are relayed raw, so they must not arrive content-encoded.
    headers = {**_preview_headers(), "Accept-Encoding": "identity"}
    try:
        upstream = await client.send(
            client.build_request("GET", url, headers=headers, timeout=LIVE_STREAM_TIMEOUT),
            stream=True,
        )
    except (httpx.HTTPError, httpx.InvalidURL, UnicodeEncodeError) as e:
        LOG.warning(f"live relay: camerapi unreachable at {url}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="camerapi unreachable")

    code = upstream.status_code
    content_type = upstream.headers.get("content-type", "")
    if code != 200 or not content_type.lower().startswith("multipart/x-mixed-replace"):
        await upstream.aclose()
        if code == 404:
            LOG.info("live relay: camerapi preview is disabled (404)")
            raise HTTPException(status_code=404, detail="live preview disabled")
        if code == 503:
            LOG.info("live relay: camerapi has too many preview viewers (503)")
            raise HTTPException(status_code=503, detail="too many preview viewers")
        if code in (401, 403):
            LOG.warning(
                f"live relay: camerapi refused the preview (HTTP {code}); "
                "PREVIEW_TOKEN must be the same for camerapi and the dashboard"
            )
            raise HTTPException(status_code=502, detail="camerapi refused the preview")
        LOG.warning(f"live relay: unexpected camerapi answer HTTP {code} ({content_type!r})")
        raise HTTPException(status_code=502, detail=f"camerapi answered HTTP {code}")

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError as e:
            # Read timeout / camerapi restart: end this response; the page
            # reconnects via its status poll.
            LOG.info(f"live relay: upstream ended: {type(e).__name__}")
        finally:
            # Starlette cancels this generator when the browser disconnects.
            # Unshielded, the close would be cancelled halfway (httpx marks the
            # response closed first, so no later close retries it) and the
            # upstream connection would leak.
            with anyio.CancelScope(shield=True):
                await upstream.aclose()

    return _RelayResponse(
        upstream,
        relay(),
        media_type=content_type,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/live/status")
async def live_status(request: Request, _: str = Depends(auth.current_user_email)) -> dict:
    """{"reachable", "error", ...camerapi preview status}; see CONTRACTS section 5."""
    client: httpx.AsyncClient = request.app.state.http
    url = _camerapi_url("/api/v1/preview/status")
    try:
        resp = await client.get(url, headers=_preview_headers(), timeout=LIVE_STATUS_TIMEOUT)
    except (httpx.HTTPError, httpx.InvalidURL, UnicodeEncodeError) as e:
        error = f"camerapi unreachable at {CAMERAPI_URL} ({type(e).__name__})"
        _note_live_problem(error)
        return {"reachable": False, "error": error, "state": "unreachable"}

    code = resp.status_code
    if code == 200:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            _note_live_problem(None)
            return {**body, "reachable": True, "error": None}
        error = "camerapi sent an invalid preview status"
    elif code in (401, 403):
        error = (
            f"camerapi refused the preview (HTTP {code}): set the same "
            "PREVIEW_TOKEN for camerapi and the dashboard"
        )
        _note_live_problem(error)
        return {"reachable": True, "error": error, "state": "auth_error"}
    elif code == 404:
        error = "this camerapi version has no live preview"
        _note_live_problem(error)
        return {"reachable": True, "error": error, "state": "unsupported"}
    else:
        error = f"camerapi preview status answered HTTP {code}"
    _note_live_problem(error)
    return {"reachable": True, "error": error, "state": "error"}


# ---- Identity endpoint (used by camerapi for USB-button attribution) -----

@app.get("/api/current-user")
def current_user_endpoint() -> dict:
    """
    For camerapi to call when the USB button fires (no header to read).
    Returns the email of whoever owns the most recently-active session,
    or null if nobody is logged in.
    """
    return {"email": db.latest_active_email()}


# ---- Capture proxy (dashboard "Capture" button) --------------------------

@app.post("/api/capture")
async def capture_proxy(
    request: Request,
    email: str = Depends(auth.current_user_email),
) -> Any:
    """
    Forward to camerapi with the user's email as X-Triggered-By.
    The dashboard owns the session, camerapi never sees credentials.
    """
    client: httpx.AsyncClient = request.app.state.http
    url = _camerapi_url("/api/v1/capture")
    try:
        resp = await client.post(
            url, headers={"X-Triggered-By": email}, timeout=CAPTURE_TIMEOUT
        )
    except httpx.RequestError as e:
        LOG.warning(f"capture proxy: connection error {e}")
        raise HTTPException(status_code=502, detail=f"camerapi unreachable: {e}")
    if resp.status_code >= 400:
        return JSONResponse(status_code=resp.status_code, content=_safe_json(resp))
    return JSONResponse(content=_safe_json(resp))


# ---- Label Studio → dashboard data ---------------------------------------

def _get_labeled_cached() -> dict:
    """
    Build (and briefly cache) the labeled list + the set of labeled capture_ids
    from Label Studio. On LS failure, degrade gracefully: empty labeled list and
    empty labeled set (so everything on disk shows as unlabeled).

    Single-flight: concurrent callers wait for one refresh instead of each
    querying LS. The TTL window starts when the refresh ends, so a slow LS is
    still asked at most once per window.
    """
    global _labeled_cache
    snapshot = _labeled_cache
    if time.monotonic() - snapshot["ts"] < _LABELED_TTL:
        return snapshot
    with _labeled_lock:
        snapshot = _labeled_cache  # another thread may have refreshed meanwhile
        if time.monotonic() - snapshot["ts"] < _LABELED_TTL:
            return snapshot
        items: list[dict] = []
        ids: set[str] = set()
        ok = False
        try:
            items, ids = _labeled_from_ls()
            ok = True
        except LabelStudioUnavailable as e:
            LOG.warning(f"Label Studio unavailable: {e}")
        except Exception as e:
            LOG.error(f"labeled-from-LS failed: {e}", exc_info=True)
        snapshot = {"ts": time.monotonic(), "items": items, "ids": ids, "ok": ok}
        _labeled_cache = snapshot
        return snapshot


def _labeled_from_ls() -> tuple[list[dict], set[str]]:
    pid = ls_client.project_id()
    tasks = ls_client.list_tasks(pid)
    items: list[dict] = []
    labeled_ids: set[str] = set()

    for t in tasks:
        capture_id = _task_capture_id(t)
        if not capture_id:
            continue
        total = t.get("total_annotations") or 0
        is_labeled = bool(t.get("is_labeled")) or total > 0 or bool(t.get("annotations"))
        if not is_labeled:
            continue
        labeled_ids.add(capture_id)

        full = ls_client.get_task(t["id"], t.get("updated_at"))
        ann = _first_annotation(full)
        meta = full.get("meta") or t.get("meta") or {}
        items.append(
            {
                "capture_id": capture_id,
                "captured_at": meta.get("captured_at"),
                "triggered_by": meta.get("triggered_by"),
                "completed_by": _completed_by_email(ann) if ann else None,
                "overall_quality": _extract_overall_quality(
                    ann.get("result") if ann else []
                ),
            }
        )

    items.sort(key=lambda i: i.get("captured_at") or "", reverse=True)
    return items, labeled_ids


def _unlabeled_from_folder(labeled_ids: set[str]) -> list[dict]:
    """Unlabeled = images on disk whose capture_id LS does not report labeled."""
    items: list[dict] = []
    try:
        entries = list(UNLABELED_DIR.glob("*.jpg"))
    except FileNotFoundError:
        return []
    for jpg in entries:
        capture_id = jpg.stem
        if capture_id in labeled_ids:
            continue
        meta = _read_sidecar(UNLABELED_DIR / f"{capture_id}.json")
        items.append(
            {
                "capture_id": capture_id,
                "captured_at": meta.get("captured_at"),
                "triggered_by": meta.get("triggered_by"),
            }
        )
    items.sort(key=lambda i: i.get("captured_at") or "", reverse=True)
    return items


# ---- Latest capture: index + view ------------------------------------------

class _CaptureIndex:
    """
    Finds the newest capture in unlabeled/ cheaply enough to be polled every
    750 ms. Newest = the latest CAPTURE time (the sidecar's captured_at), not
    the latest arrival: after a spool backlog camerapi uploads in uuid order,
    so image mtimes do not follow capture order. A capture without a usable
    captured_at (no sidecar yet, or an invalid one) ranks by its image mtime.

    A rescan happens when the directory's mtime_ns changes (a file was added,
    replaced or removed) or after `ttl_s`, because a bind mount (Docker on
    Windows) may not report directory mtimes reliably. Stats and reads are
    slow on such mounts, so a rescan only stats files it has not seen; known
    files keep their first-seen mtime (a jpg is only rewritten when an ingest
    is retried) until the full re-stat every `full_restat_s`. A sidecar is
    read once per capture (captured_at never changes): a missing one is
    retried on the next rescan while its image is younger than
    `full_restat_s` (the receiver writes the image first), an unusable one on
    the next full re-stat. The pending age of a capture never comes from here:
    _capture_view stats its image itself.
    """

    def __init__(self, directory: Path, ttl_s: float = 5.0, full_restat_s: float = 60.0) -> None:
        self.directory = directory
        self.ttl_s = ttl_s
        self.full_restat_s = full_restat_s
        self._lock = threading.Lock()
        self._mtimes: dict[str, float] = {}
        # capture_id -> captured_at (POSIX s), or None = no usable time (mtime ranks)
        self._captured: dict[str, Optional[float]] = {}
        self._newest: Optional[str] = None
        self._dir_mtime_ns: Optional[int] = None
        self._scanned_at = -math.inf
        self._full_at = -math.inf

    def invalidate(self) -> None:
        with self._lock:
            self._dir_mtime_ns = None

    def newest(self) -> Optional[str]:
        try:
            # Read before scanning: a file added during the scan changes the
            # mtime again and triggers the next rescan.
            dir_mtime_ns = os.stat(self.directory).st_mtime_ns
        except OSError:
            with self._lock:
                self._mtimes, self._newest, self._dir_mtime_ns = {}, None, None
                self._captured = {}
            return None
        with self._lock:
            now = time.monotonic()
            if dir_mtime_ns == self._dir_mtime_ns and now - self._scanned_at < self.ttl_s:
                return self._newest
            full = now - self._full_at >= self.full_restat_s
            try:
                self._rescan(full)
            except OSError as e:
                LOG.warning(f"scanning {self.directory} failed: {e}")
                return self._newest
            self._dir_mtime_ns = dir_mtime_ns
            self._scanned_at = now
            if full:
                self._full_at = now
            return self._newest

    def _rescan(self, full: bool) -> None:
        known = {} if full else self._mtimes
        fresh: dict[str, float] = {}
        with os.scandir(self.directory) as entries:
            for entry in entries:
                name = entry.name
                # Skip temp files (".<name>.XXXX.tmp") and other artifacts.
                if name.startswith(".") or not name.endswith(".jpg"):
                    continue
                capture_id = name[:-4]
                if not _is_valid_capture_id(capture_id):
                    continue
                mtime = known.get(capture_id)
                if mtime is None:
                    try:
                        st = entry.stat()
                    except OSError:
                        continue  # vanished between listing and stat
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    mtime = st.st_mtime
                fresh[capture_id] = mtime

        now = time.time()
        captured: dict[str, Optional[float]] = {}
        for capture_id, mtime in fresh.items():
            if capture_id in self._captured:
                value = self._captured[capture_id]
                if value is not None or not full:
                    captured[capture_id] = value
                    continue
            value = _sidecar_captured_at(self.directory / f"{capture_id}.json")
            if value is _NO_SIDECAR:
                if abs(now - mtime) < self.full_restat_s:
                    continue  # probably not written yet: try again next rescan
                value = None
            captured[capture_id] = value
        self._mtimes, self._captured = fresh, captured

        def rank(capture_id: str) -> tuple[float, float, str]:
            # Ties (coarse clocks) are broken by mtime, then id: a stable answer.
            mtime = fresh[capture_id]
            at = captured.get(capture_id)
            return (mtime if at is None else at, mtime, capture_id)

        self._newest = max(fresh, key=rank) if fresh else None


_NO_SIDECAR = object()  # _sidecar_captured_at: the file cannot be read (yet)


def _sidecar_captured_at(path: Path) -> Any:
    """A capture sidecar's captured_at as a POSIX timestamp; None when the file
    has no usable one; _NO_SIDECAR when it cannot be read."""
    try:
        raw = path.read_bytes()
    except OSError:
        return _NO_SIDECAR
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=lambda _name: None)
    except (UnicodeDecodeError, ValueError):
        return None
    return _iso_timestamp(data.get("captured_at")) if isinstance(data, dict) else None


def _iso_timestamp(value: Any) -> Optional[float]:
    """An ISO-8601 time as a POSIX timestamp (no offset = UTC); None if invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text[-1] in "Zz":
        text = text[:-1] + "+00:00"  # datetime.fromisoformat takes "Z" only from 3.11
    try:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        stamp = moment.timestamp()
    except (ValueError, OverflowError, OSError):
        return None
    return stamp if math.isfinite(stamp) else None


_capture_index = _CaptureIndex(UNLABELED_DIR)


def _newest_capture() -> Optional[str]:
    """capture_id of the newest capture in unlabeled/, or None if there is none."""
    return _capture_index.newest()


def _is_valid_capture_id(value: str) -> bool:
    # fullmatch, not ^...$: "$" would also accept a trailing newline.
    return bool(_CAPTURE_ID_RE.fullmatch(value))


def _capture_view(capture_id: str) -> dict:
    """{capture, state, prediction} for one capture (CONTRACTS section 5)."""
    if not _is_valid_capture_id(capture_id):
        raise HTTPException(status_code=400, detail="invalid capture_id")
    image = UNLABELED_DIR / f"{capture_id}.jpg"
    try:
        st = image.stat()
    except OSError:
        raise HTTPException(status_code=404, detail="unknown capture")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=404, detail="unknown capture")

    sidecar = _read_sidecar(UNLABELED_DIR / f"{capture_id}.json")
    state, prediction, trusted = _load_prediction(capture_id, st.st_mtime)
    width, height = _frame_size(prediction if trusted else None, sidecar)
    return {
        "capture": {
            "capture_id": capture_id,
            "captured_at": sidecar.get("captured_at"),
            "triggered_by": sidecar.get("triggered_by"),
            "source": sidecar.get("source"),
            "image_url": f"/thumb/unlabeled/{capture_id}.jpg",
            "width": width,
            "height": height,
        },
        "state": state,
        "prediction": prediction,
    }


def _load_prediction(capture_id: str, image_mtime: float) -> tuple[str, Optional[dict], bool]:
    """
    (state, prediction as stored or None, trusted) for one capture.

    trusted = a readable schema-v1 document with a known status; only then
    does the dashboard use its fields (image size). Anything else is state
    "error": the page derives the message from the stored document
    (schema_version > 1 = "unsupported"; null = unreadable).
    """
    path = PREDICTIONS_DIR / f"{capture_id}.pred.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        # No prediction (yet). abs(): an image "from the future" (clock skew
        # between the writer and this process) must not stay pending forever.
        age = time.time() - image_mtime
        if abs(age) < PREDICTION_PENDING_TIMEOUT_S:
            return "pending", None, False
        return "none", None, False
    except OSError as e:
        _warn_once(f"{capture_id}:read", f"[{capture_id}] prediction file unreadable: {e}")
        return "error", None, False

    try:
        # NaN/Infinity are not JSON; map them to null so the answer can be
        # serialized (Starlette rejects NaN).
        doc = json.loads(raw.decode("utf-8"), parse_constant=lambda _name: None)
    except (UnicodeDecodeError, ValueError) as e:
        _warn_once(f"{capture_id}:json", f"[{capture_id}] prediction file is not valid JSON: {e}")
        return "error", None, False
    if not isinstance(doc, dict):
        _warn_once(f"{capture_id}:shape", f"[{capture_id}] prediction file is not a JSON object")
        return "error", None, False

    version = doc.get("schema_version")
    if not _is_int(version) or version < 1:
        _warn_once(f"{capture_id}:version", f"[{capture_id}] prediction has no valid schema_version")
        return "error", doc, False
    if version > 1:
        _warn_once(
            f"{capture_id}:unsupported",
            f"[{capture_id}] unsupported prediction schema_version {version} "
            "(this dashboard reads version 1)",
        )
        return "error", doc, False
    status_value = doc.get("status")
    if status_value not in _PRED_STATUSES:
        _warn_once(
            f"{capture_id}:status",
            f"[{capture_id}] prediction has an invalid status {str(status_value)[:40]!r}",
        )
        return "error", doc, False
    return status_value, doc, True


def _frame_size(prediction: Optional[dict], sidecar: dict) -> tuple[Optional[int], Optional[int]]:
    """
    The photo's pixel frame: the prediction's image_width/image_height (the
    frame its boxes are in), else the capture sidecar. Taken as a pair so
    width and height never come from different sources.
    """
    pairs = []
    if prediction is not None:
        pairs.append((prediction.get("image_width"), prediction.get("image_height")))
    pairs.append((sidecar.get("width"), sidecar.get("height")))
    for w, h in pairs:
        w, h = _positive_int(w), _positive_int(h)
        if w and h:
            return w, h
    return None, None


# ---- Live preview helpers ---------------------------------------------------

class _RelayResponse(StreamingResponse):
    """
    A StreamingResponse that always closes the upstream camerapi response:
    when the stream ends, when the browser disconnects, and also when the
    relay generator never started (browser gone before the first byte, or an
    ASGI >= 2.4 server raising ClientDisconnect out of __call__, where
    Starlette neither runs background tasks nor closes the generator).
    A second aclose() is a no-op.
    """

    def __init__(self, upstream: httpx.Response, content: Any, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._upstream = upstream

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._upstream.aclose()


def _camerapi_url(path: str) -> str:
    return f"{CAMERAPI_URL.rstrip('/')}{path}"


def _preview_headers() -> dict[str, str]:
    return {"X-Preview-Token": PREVIEW_TOKEN} if PREVIEW_TOKEN else {}


_live_problem: Optional[str] = None  # last logged preview problem (event loop only)


def _note_live_problem(problem: Optional[str]) -> None:
    """Log a live-preview problem when it starts or changes, not on every poll."""
    global _live_problem
    if problem == _live_problem:
        return
    if problem is None:
        LOG.info("live preview: camerapi preview status OK")
    else:
        LOG.warning(f"live preview: {problem}")
    _live_problem = problem


# ---- Helpers --------------------------------------------------------------

def _task_capture_id(task: dict) -> Optional[str]:
    """capture_id from task.meta, else parsed from the local-files image URL."""
    meta = task.get("meta") or {}
    cap = meta.get("capture_id")
    if isinstance(cap, str) and cap:
        return cap
    image_url = (task.get("data") or {}).get("image")
    if isinstance(image_url, str) and image_url:
        tail = image_url.rsplit("/", 1)[-1]      # ...unlabeled/<id>.jpg
        stem = tail.split("?", 1)[0]
        if stem.lower().endswith(".jpg"):
            return stem[:-4]
    return None


def _first_annotation(full_task: dict) -> Optional[dict]:
    anns = full_task.get("annotations") or []
    for a in anns:
        if isinstance(a, dict) and not a.get("was_cancelled"):
            return a
    return anns[0] if anns and isinstance(anns[0], dict) else None


def _completed_by_email(ann: dict) -> Optional[str]:
    cb = ann.get("completed_by")
    if isinstance(cb, dict):
        return cb.get("email")
    return ls_client.user_email(cb)


def _extract_overall_quality(results: Any) -> Optional[str]:
    """Pull the iO/NiO/Weitere-Überprüfung choice out of an LS annotation."""
    for r in results or []:
        if r.get("from_name") == "overall_quality":
            value = r.get("value") or {}
            choices = value.get("choices") or []
            if choices:
                return choices[0]
    return None


def _read_sidecar(path: Path) -> dict:
    try:
        data = json.loads(path.read_text("utf-8"), parse_constant=lambda _name: None)
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any) -> Optional[int]:
    if _is_int(value):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


_warned: dict[str, None] = {}  # insertion-ordered set of logged keys (bounded)
_warned_lock = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    """Log a per-capture problem once, not on every poll of the page."""
    with _warned_lock:
        if key in _warned:
            return
        _warned[key] = None
        if len(_warned) > 512:
            _warned.pop(next(iter(_warned)))
    LOG.warning(message)


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"error": resp.text[:300]}


# ---- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    LOG.info(f"Starting dashboard v{SERVICE_VERSION} on {HOST}:{PORT}")
    # A short graceful-shutdown timeout: open MJPEG relays never finish on
    # their own, so without it Ctrl-C / docker stop would hang on them.
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
        timeout_graceful_shutdown=3,
    )
