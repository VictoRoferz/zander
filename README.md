# Zander — PCB Solder-Joint Defect Inspection Pipeline

Zander is a data pipeline for collecting and labeling images of PCB solder joints,
producing a training dataset for a defect-detection model.

A Basler GigE camera on a **Raspberry Pi 3** captures images of boards; the images
are sent to a **Windows PC (the hub)**, where they are annotated in **Label Studio**
(bounding boxes: defect classes *Brücke*, *Nicht ausreichend Lot*, *Fahne* + overall
quality *iO* / *NiO*) and reviewed in a custom **dashboard**.

---

## Architecture

The system is two machines and four small services talking plain HTTP on the LAN —
no message queue, no shared filesystem between machines.

```
GPIO button / dashboard "Capture"
        │  POST /api/v1/capture
        ▼
   camerapi (Raspberry Pi 3, :8001)
        │  grab image → write to local SPOOL → respond 202
        │  background spool_uploader: POST /api/v1/ingest (retries until ACK)
        ▼
   receiver (Windows PC, :8002)
        │  store DATA_ROOT/unlabeled/ + create Label Studio task
        │                                ▲ webhook on annotation
        │                                │ → write DATA_ROOT/labeled/
        ▼                                │
   Label Studio (Windows PC, :8081) ─────┘   serves images straight off disk
        ▲  live REST queries (tasks + annotations)
        │
   dashboard (Windows PC, :8003)   login, live "labeled/unlabeled" views,
                                   thumbnails, "Capture" button → Pi
```

### Design principles

- **The hub (Windows PC) is the canonical store.** Every image and label ends up in
  one folder there. The Pi only holds a short-lived retry spool.
- **The Pi never loses a capture.** Each capture is written to a local spool first;
  a background uploader ships it to the hub and retries with backoff until the hub
  acknowledges. If the PC is off, captures simply wait.
- **Ingest is idempotent.** The Pi retries until ACK, so the receiver recognizes a
  re-sent capture (per-capture `.task` marker) and never creates a duplicate task.
- **Label Studio reads images from disk** (local-files serving), not via upload —
  the receiver registers the storage and creates one task per capture via the API.
- **The dashboard never caches labels.** Its "labeled" view live-queries the Label
  Studio REST API, so it always matches what annotators actually did.
- **All disk writes are atomic** (temp file → fsync → `os.replace`), so half-written
  files are never visible to Label Studio or the dashboard.

---

## The services

### `camerapi/` — capture node (Raspberry Pi 3)

Thin FastAPI app wired to the Basler camera (pypylon, with OpenCV-webcam and
`sample.jpg` fallbacks for dev machines).

| Piece | Function |
|---|---|
| `api/routes.py` | `POST /api/v1/capture` (grab → spool → 202), `POST /api/v1/test-camera`, `GET /api/v1/status` (camera + spool depth) |
| `services/camera_service.py` | Camera access; picks an Ethernet vs WiFi transport profile for the GigE camera (auto-detected via `ip route`, or forced with `CAMERA_TRANSPORT`) |
| `services/storage_service.py` | The local spool: atomic writes under `DATA_ROOT/spool/` |
| `services/spool_uploader.py` | Daemon thread: ships each spooled capture to the hub's `/api/v1/ingest`. ACK contract: `2xx`/`409` → delete; other `4xx` → move to `spool/failed/`; `5xx`/connection error → keep + exponential backoff |
| `scripts/button_listener.py` | GPIO button → `POST /api/v1/capture` |

Identity: each capture gets a UUID4 `capture_id`; artifacts are `<capture_id>.jpg`
plus a `.json` metadata sidecar (timestamp, camera serial, SHA256, `triggered_by`).

### `receiver/` — ingestion hub (Windows PC)

The owner of all Label Studio setup (project creation, labeling config, local-storage
registration) and of the on-disk store.

| Piece | Function |
|---|---|
| `api/ingest.py` | `POST /api/v1/ingest`: verify SHA256 (`422` on mismatch), store `unlabeled/<id>.jpg` + sidecar, create the LS task, write the idempotency marker. Returns `503` while LS is unreachable (the Pi spool retries) |
| `api/webhooks.py` | `POST /api/v1/webhook/annotation-created`: on each submitted annotation, write the durable training export `labeled/<id>.jpg` + merged `.json` |
| `services/labelstudio_service.py` | LS client: get-or-create the project (labeling config applied **only at creation**), register the local-files storage at `unlabeled/`, create one task per capture (`data.image = /data/local-files/?d=<relpath>` — kept POSIX via `as_posix()` on all platforms) |
| `services/storage_service.py` | Atomic writes; files are written world-readable (0644) so the non-root Label Studio container can serve them |

### `dashboard/` — review UI (Windows PC)

Single-page FastAPI app with its own SQLite user database (bcrypt + session cookies).

| Piece | Function |
|---|---|
| `main.py` | Pages + API: unlabeled/labeled views, thumbnails from `DATA_ROOT`, "Capture" proxy to the Pi (adds `X-Triggered-By` so captures are attributed to the logged-in user) |
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
Docker volume `zander_ls_internal`. On the Pi, `DATA_ROOT` (default
`/home/pi/zander-data`) holds only the transient `spool/`.

---

## Running it

### Hub — Docker (production, Windows PC)

Everything lives in `deploy/`. Full instructions: [`deploy/README.md`](deploy/README.md).

```
cd deploy
copy .env.example .env        # set PI_IP, LAPTOP_HOST; token comes later
docker compose up -d --build
```

Three containers: `zander-labelstudio` (`:8081`), `zander-receiver` (`:8002`),
`zander-dashboard` (`:8003`). One-time bootstrap after the first start:

1. Log into LS (`http://localhost:8081`, admin from `.env`), copy the **legacy**
   API token (~40 chars — not the `eyJ…` JWT) into `deploy/.env` →
   `LABELSTUDIO_API_KEY`, then `docker compose up -d` again.
2. Add the webhook in the LS project: Settings → Webhooks →
   `http://localhost:8002/api/v1/webhook/annotation-created`.
3. Create a dashboard login:
   `docker compose exec dashboard python manage_users.py add you@example.com "You"`.

Daily use: double-click the **Zander** desktop icon (`start-zander.bat` →
`docker compose up -d` + opens the dashboard). `install-shortcut.ps1` creates the
icon. Containers restart automatically with Docker (`restart: unless-stopped`).

### Hub — native (no Docker, dev on macOS/Windows)

```
python scripts/launch.py
```

Starts Label Studio + receiver + dashboard together with a shared `DATA_ROOT`
(default `~/zander-data`) and tears all three down on Ctrl-C. Each service reads its
own `.env` (`receiver/.env`, `dashboard/.env`).

### Pi

```
cd camerapi
cp .env.example .env     # INGEST_URL / DASHBOARD_URL = http://<hub-ip>:8002 / :8003
pip install -r requirements.txt      # plus pypylon for the real camera
python main.py
python scripts/button_listener.py    # GPIO trigger (separate terminal)
```

The Pi and the hub must be on the same subnet as the camera (`192.168.177.x`).
Sanity check from the Pi before debugging anything else:
`curl http://<hub-ip>:8002/api/v1/health`.

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
| receiver | pydantic `Settings` | `DATA_ROOT`, `LABELSTUDIO_URL`, `LABELSTUDIO_API_KEY`, `LABELSTUDIO_PROJECT_NAME` |
| dashboard | `os.environ` + python-dotenv | `DATA_ROOT`, `LABELSTUDIO_URL`, `LABELSTUDIO_PUBLIC_URL`, `LABELSTUDIO_API_KEY`, `CAMERAPI_URL` |
| Docker (all) | `deploy/.env` | `DATA_DIR`, `PI_IP`, `LAPTOP_HOST`, `LABELSTUDIO_API_KEY`, LS admin credentials |

`.env` files are read **once at startup** — restart the service after any change.

---

## Tests

```
python tests/test_parsers.py
```

Covers the load-bearing capture-id and quality parsers. There is no build step:
"build" = install each service's `requirements.txt` and run.

---

## Repository layout

```
zander/
├── camerapi/     Pi capture node (FastAPI + pypylon)
├── receiver/     hub ingestion service (FastAPI)
├── dashboard/    hub review UI (FastAPI + SQLite)
├── deploy/       Windows production kit (docker-compose, .bat scripts, icon)
├── scripts/      launch.py — native all-in-one launcher
├── tests/        parser unit test
├── picreceiver/  minimal standalone upload sink (utility, not wired in)
└── _grave/       archived dead code — ignore
```
