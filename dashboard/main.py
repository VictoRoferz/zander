"""
Dashboard service (Mac/Windows).

Standalone read-only UI for the camera pipeline:
  - Login (own SQLite users; bcrypt passwords)
  - Counts + thumbnails of unlabeled and labeled captures
  - "Capture" button (proxies to camerapi with X-Triggered-By)
  - "Open Label Studio" button (link)

The "labeled" view is read LIVE from the Label Studio REST API (LS is the
single source of truth), so the dashboard always matches LS. "unlabeled" =
images on disk that LS does not yet report as labeled. Thumbnails are served
from the local image folder (the same files LS serves via local-files).

Cross-platform paths via pathlib. Reads ~/zander-data/ (override DATA_ROOT).
Polls every 3s; no SSE in v1.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

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
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import auth
import db
from ls_client import LabelStudioUnavailable, LsClient

# ---- Setup ----------------------------------------------------------------

# Load this service's .env so it can be configured like the others. Existing
# environment variables (e.g. those injected by scripts/launch.py) take
# precedence over the file.
load_dotenv()

LOG = logging.getLogger("dashboard")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _data_root() -> Path:
    override = os.environ.get("DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "zander-data"


DATA_ROOT = _data_root()
UNLABELED_DIR = DATA_ROOT / "unlabeled"
LABELED_DIR = DATA_ROOT / "labeled"
UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)

LABELSTUDIO_URL = os.environ.get("LABELSTUDIO_URL", "http://localhost:8081")
CAMERAPI_URL = os.environ.get("CAMERAPI_URL", "http://192.168.0.115:8001")

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

# Collapse the 3 near-simultaneous poll endpoints (stats/unlabeled/labeled)
# into a single LS pass per ~2s window.
_LABELED_TTL = 2.0
_labeled_cache: dict[str, Any] = {"ts": -1e9, "items": [], "ids": set(), "ok": False}


# ---- App ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize()
    if not LABELSTUDIO_API_KEY:
        LOG.warning(
            "LABELSTUDIO_API_KEY not set — the labeled view will stay empty. "
            "Add it to the dashboard .env."
        )
    LOG.info(
        f"dashboard ready: data_root={DATA_ROOT} ls={LABELSTUDIO_URL} "
        f"camerapi={CAMERAPI_URL}"
    )
    yield


app = FastAPI(title="zander-dashboard", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---- Auth flow ------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None) -> Any:
    return TEMPLATES.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> Any:
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
async def logout(dashboard_session: Optional[str] = Cookie(default=None)) -> Any:
    if dashboard_session:
        auth.logout(dashboard_session)
    redirect = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    redirect.delete_cookie(auth.SESSION_COOKIE)
    return redirect


# ---- Dashboard home -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    dashboard_session: Optional[str] = Cookie(default=None),
) -> Any:
    email = auth.session_to_email(dashboard_session)
    if not email:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {"user_email": email, "labelstudio_url": LABELSTUDIO_URL},
    )


# ---- Read-only data API (used by app.js) ---------------------------------

@app.get("/api/stats")
async def stats(_: str = Depends(auth.current_user_email)) -> dict:
    cache = _get_labeled_cached()
    labeled = len(cache["items"])
    unlabeled = len(_unlabeled_from_folder(cache["ids"]))
    return {"unlabeled": unlabeled, "labeled": labeled, "total": labeled + unlabeled}


@app.get("/api/unlabeled")
async def api_unlabeled(_: str = Depends(auth.current_user_email)) -> list[dict]:
    cache = _get_labeled_cached()
    return _unlabeled_from_folder(cache["ids"])


@app.get("/api/labeled")
async def api_labeled(_: str = Depends(auth.current_user_email)) -> list[dict]:
    return _get_labeled_cached()["items"]


@app.get("/thumb/{kind}/{filename}")
async def thumb(
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


# ---- Identity endpoint (used by camerapi for GPIO-button attribution) ----

@app.get("/api/current-user")
async def current_user_endpoint() -> dict:
    """
    For camerapi to call when the GPIO button fires (no header to read).
    Returns the email of whoever owns the most recently-active session,
    or null if nobody is logged in.
    """
    return {"email": db.latest_active_email()}


# ---- Capture proxy (dashboard "Capture" button) --------------------------

@app.post("/api/capture")
async def capture_proxy(email: str = Depends(auth.current_user_email)) -> Any:
    """
    Forward to camerapi with the user's email as X-Triggered-By.
    The dashboard owns the session, camerapi never sees credentials.
    """
    url = f"{CAMERAPI_URL.rstrip('/')}/api/v1/capture"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers={"X-Triggered-By": email})
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
    """
    now = time.monotonic()
    if now - _labeled_cache["ts"] < _LABELED_TTL:
        return _labeled_cache
    try:
        items, ids = _labeled_from_ls()
        _labeled_cache.update(ts=now, items=items, ids=ids, ok=True)
    except LabelStudioUnavailable as e:
        LOG.warning(f"Label Studio unavailable: {e}")
        _labeled_cache.update(ts=now, items=[], ids=set(), ok=False)
    except Exception as e:
        LOG.error(f"labeled-from-LS failed: {e}", exc_info=True)
        _labeled_cache.update(ts=now, items=[], ids=set(), ok=False)
    return _labeled_cache


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
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"error": resp.text[:300]}


# ---- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    LOG.info(f"Starting dashboard on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
