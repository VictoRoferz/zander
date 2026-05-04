"""
Dashboard service (Mac/Windows).

Standalone read-only UI for the camera pipeline:
  - Login (own SQLite users; bcrypt passwords)
  - Counts + thumbnails of unlabeled and labeled captures
  - "Capture" button (proxies to camerapi with X-Triggered-By)
  - "Open Label Studio" button (link)

Cross-platform paths via pathlib. Reads ~/zander-data/ (override with DATA_ROOT).
Polls every 3s; no SSE in v1.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
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

# ---- Setup ----------------------------------------------------------------

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

LABELSTUDIO_URL = os.environ.get("LABELSTUDIO_URL", "http://192.168.0.115:8081")
CAMERAPI_URL = os.environ.get("CAMERAPI_URL", "http://192.168.0.115:8001")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8003"))

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---- App ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize()
    LOG.info(
        f"dashboard ready: data_root={DATA_ROOT} ls={LABELSTUDIO_URL} "
        f"camerapi={CAMERAPI_URL}"
    )
    yield


app = FastAPI(title="zander-dashboard", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---- Auth flow ------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None) -> Any:
    return TEMPLATES.TemplateResponse(
        request, "login.html", {"error": error}
    )


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
async def logout(
    dashboard_session: Optional[str] = Cookie(default=None),
) -> Any:
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
        {
            "user_email": email,
            "labelstudio_url": LABELSTUDIO_URL,
        },
    )


# ---- Read-only data API (used by app.js) ---------------------------------

@app.get("/api/stats")
async def stats(_: str = Depends(auth.current_user_email)) -> dict:
    u = _count_jpgs(UNLABELED_DIR)
    l = _count_jpgs(LABELED_DIR)
    return {"unlabeled": u, "labeled": l, "total": u + l}


@app.get("/api/unlabeled")
async def api_unlabeled(_: str = Depends(auth.current_user_email)) -> list[dict]:
    return _list_captures(UNLABELED_DIR, kind="unlabeled")


@app.get("/api/labeled")
async def api_labeled(_: str = Depends(auth.current_user_email)) -> list[dict]:
    return _list_captures(LABELED_DIR, kind="labeled")


@app.get("/thumb/{kind}/{filename}")
async def thumb(
    kind: str,
    filename: str,
    _: str = Depends(auth.current_user_email),
) -> FileResponse:
    if kind == "unlabeled":
        directory = UNLABELED_DIR
    elif kind == "labeled":
        directory = LABELED_DIR
    else:
        raise HTTPException(status_code=404)
    # Reject anything that would escape the directory or isn't a .jpg.
    if "/" in filename or "\\" in filename or not filename.endswith(".jpg"):
        raise HTTPException(status_code=400)
    path = directory / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# ---- Identity endpoint (used by camerapi in Phase C) --------------------

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
        return JSONResponse(
            status_code=resp.status_code,
            content=_safe_json(resp),
        )
    return JSONResponse(content=resp.json())


# ---- Helpers --------------------------------------------------------------

def _count_jpgs(directory: Path) -> int:
    try:
        return sum(1 for p in directory.iterdir() if p.suffix.lower() == ".jpg")
    except FileNotFoundError:
        return 0


def _list_captures(directory: Path, kind: str) -> list[dict]:
    """
    Build a sortable list of captures by reading the JSON sidecars.
    Newest first. Tolerates files without sidecars.
    """
    items: list[dict] = []
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return []

    for jpg in entries:
        if jpg.suffix.lower() != ".jpg":
            continue
        capture_id = jpg.stem
        json_path = directory / f"{capture_id}.json"
        item: dict = {
            "capture_id": capture_id,
            "captured_at": None,
            "triggered_by": None,
        }
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text("utf-8"))
                if kind == "unlabeled":
                    item["captured_at"] = payload.get("captured_at")
                    item["triggered_by"] = payload.get("triggered_by")
                else:  # labeled
                    cap = payload.get("capture_metadata") or {}
                    item["captured_at"] = cap.get("captured_at")
                    item["triggered_by"] = cap.get("triggered_by")
                    annotation = payload.get("annotation") or {}
                    completed = annotation.get("completed_by") or {}
                    item["completed_by"] = (
                        completed.get("email") if isinstance(completed, dict) else None
                    )
                    item["overall_quality"] = _extract_overall_quality(
                        annotation.get("result") or []
                    )
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                LOG.warning(f"bad sidecar {json_path}: {e}")
        items.append(item)

    items.sort(key=lambda i: i.get("captured_at") or "", reverse=True)
    return items


def _extract_overall_quality(results: list[dict]) -> Optional[str]:
    """Pull the Pass/Fail/Needs-Review choice out of a Label Studio annotation."""
    for r in results:
        if r.get("from_name") == "overall_quality":
            value = r.get("value") or {}
            choices = value.get("choices") or []
            if choices:
                return choices[0]
    return None


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"error": resp.text[:300]}


# ---- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    LOG.info(f"Starting dashboard on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
