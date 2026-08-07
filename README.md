# Zander — PCB Solder-Joint Defect Inspection Pipeline

Zander is a data pipeline for collecting and labeling images of PCB solder joints,
producing a training dataset for a defect-detection model.

A **Basler GigE camera** direct-attached to a **Windows PC** captures images of
boards (triggered by a **USB button** or the dashboard); the images are annotated in
**Label Studio** (bounding boxes: defect classes *Brücke*, *Nicht ausreichend Lot*,
*Fahne* + overall quality *iO* / *NiO*) and reviewed in a custom **dashboard**.

---

## Architecture

The whole system runs on one PC: three hub services in Docker plus the native
camera stack, talking plain HTTP on localhost — no message queue, no Pi, no LAN
dependency.

```
USB keypad (F13) / dashboard "Capture"
        │  POST /api/v1/capture
        ▼
   camerapi (native on the PC, :8001)   [Basler GigE camera direct-attached]
        │  grab image → write to local SPOOL → respond 202
        │  background spool_uploader: POST /api/v1/ingest (retries until ACK)
        ▼
   receiver (Docker, :8002)
        │  store DATA_ROOT/unlabeled/ + create Label Studio task
        │                                ▲ webhook on annotation
        │                                │ → write DATA_ROOT/labeled/
        ▼                                │
   Label Studio (Docker, :8081) ─────────┘   serves images straight off disk
        ▲  live REST queries (tasks + annotations)
        │
   dashboard (Docker, :8003)   login, live "labeled/unlabeled" views,
                               thumbnails, "Capture" button → camerapi
```

camerapi runs natively (not in a container) because Docker Desktop can neither
discover a GigE camera on a host NIC nor hook USB keyboard input.

### Design principles

- **The Docker data folder is the canonical store.** Every image and label ends up
  in one folder. camerapi only holds a short-lived retry spool.
- **A capture is never lost.** Each capture is written to a local spool first; a
  background uploader ships it to the receiver and retries with backoff until it is
  acknowledged. If the Docker side is down or still booting, captures simply wait.
- **Ingest is idempotent.** The uploader retries until ACK, so the receiver
  recognizes a re-sent capture (per-capture `.task` marker) and never creates a
  duplicate task.
- **Label Studio reads images from disk** (local-files serving), not via upload —
  the receiver registers the storage and creates one task per capture via the API.
- **The dashboard never caches labels.** Its "labeled" view live-queries the Label
  Studio REST API, so it always matches what annotators actually did.
- **All disk writes are atomic** (temp file → fsync → `os.replace`), so half-written
  files are never visible to Label Studio or the dashboard.

---

## The services

### `camerapi/` — native capture service (PC)

Thin FastAPI app wired to the Basler camera (pypylon, with OpenCV-webcam and
`sample.jpg` fallbacks for dev machines).

| Piece | Function |
|---|---|
| `api/routes.py` | `POST /api/v1/capture` (grab → spool → 202), `POST /api/v1/test-camera`, `GET /api/v1/status` (camera + spool depth) |
| `services/camera_service.py` | Camera access; picks a GigE transport profile: ethernet by default on Windows/macOS (direct-attach), route-detected on Linux, forced with `CAMERA_TRANSPORT` |
| `services/storage_service.py` | The local spool: atomic writes under `DATA_ROOT/spool/` |
| `services/spool_uploader.py` | Daemon thread: ships each spooled capture to the receiver's `/api/v1/ingest`. ACK contract: `2xx`/`409` → delete; other `4xx` → move to `spool/failed/`; `5xx`/connection error → keep + exponential backoff |
| `scripts/usb_button_listener.py` | USB macro-keypad trigger (pynput global keyboard hook; F13 → capture, F14 → test shot; `BUTTON_MODE=stdin` for buttonless dev) |

Identity: each capture gets a UUID4 `capture_id`; artifacts are `<capture_id>.jpg`
plus a `.json` metadata sidecar (timestamp, camera serial, SHA256, `triggered_by`).

### `receiver/` — ingestion hub (Windows PC)

The owner of all Label Studio setup (project creation, labeling config, local-storage
registration) and of the on-disk store.

| Piece | Function |
|---|---|
| `api/ingest.py` | `POST /api/v1/ingest`: verify SHA256 (`422` on mismatch), store `unlabeled/<id>.jpg` + sidecar, create the LS task, write the idempotency marker. Returns `503` while LS is unreachable (the camerapi spool retries) |
| `api/webhooks.py` | `POST /api/v1/webhook/annotation-created`: on each submitted annotation, write the durable training export `labeled/<id>.jpg` + merged `.json` |
| `services/labelstudio_service.py` | LS client: get-or-create the project (labeling config applied **only at creation**), register the local-files storage at `unlabeled/`, create one task per capture (`data.image = /data/local-files/?d=<relpath>` — kept POSIX via `as_posix()` on all platforms) |
| `services/storage_service.py` | Atomic writes; files are written world-readable (0644) so the non-root Label Studio container can serve them |

### `dashboard/` — review UI (Windows PC)

Single-page FastAPI app with its own SQLite user database (bcrypt + session cookies).

| Piece | Function |
|---|---|
| `main.py` | Pages + API: unlabeled/labeled views, thumbnails from `DATA_ROOT`, "Capture" proxy to camerapi (adds `X-Triggered-By` so captures are attributed to the logged-in user) |
| `ls_client.py` | Read-only Label Studio REST client — the "labeled" view is computed live from LS tasks/annotations |
| `manage_users.py` | CLI: `add` / `list` / `remove` / `reset` dashboard users |

### Label Studio (`:8081`)

Stock Label Studio (Docker image), configured for local-files serving. Labeling
interface: `RectangleLabels` bounding boxes (so the dataset can be exported in
**YOLO** format directly from LS) + an iO/NiO quality choice + free-text notes.

---

## Data layout (on the hub)

Everything lives under one folder — `deploy/data/` in Docker (`DATA_DIR` in
`deploy/.env`), mounted as `/data` in all three containers:

```
data/
├── unlabeled/   <id>.jpg + <id>.json (capture metadata) + <id>.task (LS task id marker)
└── labeled/     <id>.jpg + <id>.json (annotation + metadata) ← the training dataset
```

`labeled/` is self-contained image/label pairs: zip it and hand it to whoever trains
the model. Label Studio's own database (projects, accounts, annotations) lives in the
Docker volume `zander_ls_internal`. camerapi's own `DATA_ROOT` (native, default
`~/zander-data`) holds only the transient `spool/` — it is deliberately a different
folder from the Docker data dir; images move by HTTP, never a shared path.

---

## Running it

### Production (Windows PC): Docker hub + native camera stack

Everything lives in `deploy/`. Full instructions: [`deploy/README.md`](deploy/README.md).

```
cd deploy
copy .env.example .env        # token comes later
setup-camera.bat              # one-time: venv + camerapi deps (pypylon, pynput)
docker compose up -d --build
```

Three containers: `zander-labelstudio` (`:8081`), `zander-receiver` (`:8002`),
`zander-dashboard` (`:8003`), plus the native camera stack (camerapi `:8001` + USB
button listener). One-time bootstrap after the first start:

1. Log into LS (`http://localhost:8081`, admin from `.env`), copy the **legacy**
   API token (~40 chars — not the `eyJ…` JWT) into `deploy/.env` →
   `LABELSTUDIO_API_KEY`, then `docker compose up -d` again.
2. Add the webhook in the LS project: Settings → Webhooks →
   `http://localhost:8002/api/v1/webhook/annotation-created`.
3. Create a dashboard login:
   `docker compose exec dashboard python manage_users.py add you@example.com "You"`.
4. Program the USB keypad: key 1 → **F13** (capture), key 2 → **F14** (test shot).
5. Give the camera NIC a static IP on the camera subnet (`192.168.177.x`).

Daily use: double-click the **Zander** desktop icon (`start-zander.bat` →
`docker compose up -d` + the minimized **"Zander Camera"** window + opens the
dashboard). `install-shortcut.ps1` creates the icon. `stop-zander.bat` stops both.
Camera-stack logs: the minimized window, or `deploy\camera-stack.log`.

### Dev — everything native (macOS/Windows, no Docker)

```
python scripts/launch.py                # hub only: LS + receiver + dashboard
python scripts/launch.py --with-camera  # + camerapi + USB button listener
python scripts/launch.py --camera-only  # camera stack only (alongside Docker)
```

Starts the hub services with a shared `DATA_ROOT` (default `~/zander-data`) and
tears everything down on Ctrl-C. Each service reads its own `.env`. On a dev machine
without the camera set `USE_CAMERA=false` in `camerapi/.env` (sample.jpg fallback),
and without the keypad run the listener in stdin mode in its own terminal:

```
BUTTON_MODE=stdin python camerapi/scripts/usb_button_listener.py   # Enter = capture
```

---

## Docker details

Defined in [`deploy/docker-compose.yml`](deploy/docker-compose.yml):

| Container | Image | Port (host) | Role |
|---|---|---|---|
| `zander-labelstudio` | `heartexlabs/label-studio` | 8081 → 8080 | Annotation UI; serves images from `/data` (`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`, document root `/data`); legacy API tokens enabled |
| `zander-receiver` | built from `receiver/` (`python:3.11-slim`) | 8002 | Ingest + LS task creation + labeled export |
| `zander-dashboard` | built from `dashboard/` (`python:3.11-slim`) | 8003 | Review UI + login + capture proxy |

Key invariants:

- All three mount the same host folder as `/data`; Label Studio's
  `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` **must equal** the services' `DATA_ROOT`.
- The receiver talks to LS at the in-network URL `http://labelstudio:8080`; browsers
  use `http://<host>:8081` (`LABELSTUDIO_PUBLIC_URL` for the dashboard's link).
- After editing `deploy/.env`, run `docker compose up -d` again (recreate —
  a plain `restart` does not pick up env changes). After pulling code changes,
  `docker compose up -d --build`.
- Label Studio runs as a non-root user: files in `/data` must be world-readable
  (the receiver writes 0644 for exactly this reason).

---

## Configuration

All settings are env-driven, one `.env` per service (gitignored):

| Service | Mechanism | Key variables |
|---|---|---|
| camerapi | pydantic `Settings` | `INGEST_URL`, `DASHBOARD_URL`, `DATA_ROOT`, `USE_CAMERA`, `CAMERA_TRANSPORT` |
| USB listener | `os.environ` + python-dotenv (reads `camerapi/.env`) | `CAMERAPI_URL`, `BUTTON_MODE`, `BUTTON_CAPTURE_KEY`, `BUTTON_SECONDARY_KEY`, `BUTTON_COOLDOWN` |
| receiver | pydantic `Settings` | `DATA_ROOT`, `LABELSTUDIO_URL`, `LABELSTUDIO_API_KEY`, `LABELSTUDIO_PROJECT_NAME` |
| dashboard | `os.environ` + python-dotenv | `DATA_ROOT`, `LABELSTUDIO_URL`, `LABELSTUDIO_PUBLIC_URL`, `LABELSTUDIO_API_KEY`, `CAMERAPI_URL` |
| Docker (all) | `deploy/.env` | `DATA_DIR`, `LAPTOP_HOST`, `LABELSTUDIO_API_KEY`, LS admin credentials |

`.env` files are read **once at startup** — restart the service after any change.

---

## Tests

```
python tests/test_parsers.py
python tests/test_button_listener.py   # key parsing for the USB listener
```

Covers the load-bearing capture-id/quality parsers and the USB-listener key
parsing. There is no build step: "build" = install each service's
`requirements.txt` and run.

---

## Repository layout

```
zander/
├── camerapi/     native capture service (FastAPI + pypylon) + USB button listener
├── receiver/     hub ingestion service (FastAPI)
├── dashboard/    hub review UI (FastAPI + SQLite)
├── deploy/       Windows production kit (docker-compose, .bat scripts, icon)
├── scripts/      launch.py — native launcher (hub / +camera / camera-only)
├── tests/        self-contained unit tests
├── picreceiver/  minimal standalone upload sink (utility, not wired in)
└── _grave/       archived dead code — ignore
```
