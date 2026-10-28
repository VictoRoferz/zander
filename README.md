# Zander — PCB Solder-Joint Defect Inspection Pipeline

Zander is a data pipeline for collecting and labeling images of PCB solder joints,
producing a training dataset for a defect-detection model — and, once a model is
trained, classifying every new capture and showing the result on the dashboard.

A **Basler GigE camera** direct-attached to a **Windows PC** captures images of
boards (triggered by a **USB button** or the dashboard); the images are annotated in
**Label Studio** (bounding boxes: defect classes *Brücke*, *Nicht ausreichend Lot*,
*Fahne* + overall quality *iO* / *NiO*) and reviewed in a custom **dashboard**, which
also shows the **live camera** (opt-in) and the latest capture **classified by the
inspector** (verdict iO/NiO, defect classes, boxes on the board). The model toolchain
lives in [`ml/`](ml/README.md); the shared interfaces are in
[`docs/CONTRACTS.md`](docs/CONTRACTS.md).

---

## Architecture

The whole system runs on one PC: four hub services in Docker plus the native
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
                               thumbnails, "Capture" button → camerapi,
                               live camera panel, latest-result panel

Prediction path (after every ingest, never delaying the ACK):
   receiver ──POST /api/v1/inspect──► inspector (Docker, :8004, detector.onnx + detector.json)
       └─ writes DATA_ROOT/predictions/<id>.pred.json ──► dashboard /api/latest (boxes, verdict, classes)

Live path (opt-in, LIVE_PREVIEW_ENABLED=true in camerapi/.env):
   camera ──► camerapi owner thread ──GET /api/v1/preview/stream.mjpg (X-Preview-Token)──►
       dashboard /api/live.mjpg (login required) ──► browser
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
- **Predictions never block ingest.** The receiver queues each new capture for the
  inspector after acknowledging it; a missing model or a crashed inspector only
  shows up as "No model loaded" / an error on the dashboard.
- **The model travels with its card.** `detector.json` next to `detector.onnx` holds
  the classes, the frozen threshold and provenance; in production the inspector
  refuses a model without a valid card (`REQUIRE_CARD=true`). Several models can sit
  in the models folder (`<name>.onnx` + `<name>.json`); the dashboard's Model chooser
  picks the active one, and every prediction records the model that made it.
- **The live preview is opt-in and owns the camera.** With `LIVE_PREVIEW_ENABLED`
  off (default) captures work exactly as before. With it on, one thread owns the
  camera; a capture stops the stream and grabs the first frame after the key press.

---

## The services

### `camerapi/` — native capture service (PC)

Thin FastAPI app wired to the Basler camera (pypylon, with OpenCV-webcam and
`sample.jpg` fallbacks for dev machines).

| Piece | Function |
|---|---|
| `api/routes.py` | `POST /api/v1/capture` (grab → spool → 202), `POST /api/v1/test-camera`, `GET /api/v1/status` (camera + spool depth + `live` block) |
| `api/preview.py` | Live preview (only with `LIVE_PREVIEW_ENABLED=true`): `GET /api/v1/preview/stream.mjpg`, `/frame.jpg`, `/status`; access by `X-Preview-Token`, or local-only when the token is empty |
| `services/camera_service.py` | Camera access (one-shot `GrabOne` path, serialized by a lock); picks a GigE transport profile: ethernet by default on Windows/macOS (direct-attach), route-detected on Linux, forced with `CAMERA_TRANSPORT` |
| `services/live_camera.py` | The camera-owner thread for the live preview: continuous grabbing at `PREVIEW_FPS`, one JPEG encode per frame shared by all viewers, captures = stop stream + `GrabOne`, reconnect with backoff |
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
| `services/storage_service.py` | Atomic writes; files are written world-readable (0644) so the non-root Label Studio container can serve them; prediction files under `predictions/` |
| `services/prediction_service.py` | One worker thread + bounded queue: asks the inspector about each new capture and writes `predictions/<id>.pred.json` (`ok` / `no_model` / `error`; never downgrades an `ok`). Retries while the inspector restarts or fails; at startup re-queues captures still without a result (newest first) |
| `api/predictions.py` | `POST /api/v1/predict/rescan` — queue every capture still without a result (e.g. after deploying a model); `POST /api/v1/predict/{capture_id}?force=false` — re-queue one capture |

### `inspector/` — defect detector (Windows PC, Docker `:8004`)

Serves the exported detector torch-free (ONNX Runtime) through the same `ml/infer`
code the evaluation used: 20 native-resolution tiles per photo, cross-tile merge,
then the photo decision (`infer/verdict.py`: NiO if any detection reaches the
threshold, plus the defect classes). `POST /api/v1/inspect` (file or `capture_id`),
`GET /api/v1/status` (model, card, threshold, load error). Loads
`DATA_ROOT/models/detector.onnx` + `detector.json`; answers `503` until a model is
there.

### `dashboard/` — review UI (Windows PC)

Single-page FastAPI app with its own SQLite user database (bcrypt + session cookies).

| Piece | Function |
|---|---|
| `main.py` | Pages + API: unlabeled/labeled views, thumbnails from `DATA_ROOT`, "Capture" proxy to camerapi (adds `X-Triggered-By` so captures are attributed to the logged-in user), `GET /api/latest` + `/api/captures/{id}` (capture + prediction state), `GET /api/live.mjpg` + `/api/live/status` (login-protected relay of camerapi's preview) |
| `static/app.js` | Live panel and latest-result panel: the photo as an SVG with boxes in photo pixels, verdict badge, class chips, one zoomed crop per defect |
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
├── unlabeled/    <id>.jpg + <id>.json (capture metadata) + <id>.task (LS task id marker)
├── labeled/      <id>.jpg + <id>.json (annotation + metadata) ← the training dataset
├── predictions/  <id>.pred.json (inspector result, written by the receiver)
└── models/       detector.onnx + detector.json (the deployed model and its card)
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
6. Deploy a model (once one is trained): copy `detector.onnx` + `detector.json`
   (from `ml/`'s `cli.py export`) into `deploy/data/models/`, then
   `docker compose restart inspector` and
   `curl -X POST http://localhost:8002/api/v1/predict/rescan` (classifies the older
   captures).
7. Live view (optional): put the same random `PREVIEW_TOKEN` in `deploy/.env` and
   `camerapi/.env`, set `LIVE_PREVIEW_ENABLED=true` in `camerapi/.env`, run
   `docker compose up -d` and restart the camera stack.

Daily use: double-click the **Zander** desktop icon (`start-zander.bat` →
`docker compose up -d --build`, falling back to the existing images when a build fails,
+ the minimized **"Zander Camera"** window, which starts even when Docker has a
problem + opens the dashboard). `install-shortcut.ps1` creates the icon.
`stop-zander.bat` stops both; `update-zander.bat` pulls, rebuilds and restarts after
code changes.
Camera-stack logs: the minimized window, or `deploy\camera-stack.log`.

### Dev — everything native (macOS/Windows, no Docker)

```
python scripts/launch.py                # hub only: LS + receiver + dashboard
python scripts/launch.py --with-camera  # + camerapi + USB button listener
python scripts/launch.py --camera-only  # camera stack only (alongside Docker)
```

Starts the hub services with a shared `DATA_ROOT` (default `~/zander-data`) and
tears everything down on Ctrl-C. In hub modes it also starts the inspector as an
optional child (interpreter `INSPECTOR_PYTHON` or `ml/.venv`; `--no-inspector` to skip);
if the inspector crashes, the hub keeps running. Each service reads its own `.env`. On a dev machine
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
| `zander-dashboard` | built from `dashboard/` (`python:3.11-slim`) | 8003 | Review UI + login + capture proxy + live/result panels |
| `zander-inspector` | built from the repo root (`inspector/Dockerfile` + `ml/infer`) | 127.0.0.1:8004 | Defect detector; reads `/data` read-only (writes only `models/active_model.json`); `REQUIRE_CARD=true` |

Key invariants:

- All four mount the same host folder as `/data` (the inspector read-only); Label Studio's
  `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` **must equal** the services' `DATA_ROOT`.
- The receiver talks to LS at the in-network URL `http://labelstudio:8080`; browsers
  use `http://<host>:8081` (`LABELSTUDIO_PUBLIC_URL` for the dashboard's link).
- After editing `deploy/.env`, run `docker compose up -d` again (recreate —
  a plain `restart` does not pick up env changes). After code changes, run
  `deploy\update-zander.bat` (pull, build, restart incl. the camera stack).
- Label Studio runs as a non-root user: files in `/data` must be world-readable
  (the receiver writes 0644 for exactly this reason).

---

## Configuration

All settings are env-driven, one `.env` per service (gitignored):

| Service | Mechanism | Key variables |
|---|---|---|
| camerapi | pydantic `Settings` | `INGEST_URL`, `DASHBOARD_URL`, `DATA_ROOT`, `USE_CAMERA`, `CAMERA_TRANSPORT`, `LIVE_PREVIEW_ENABLED`, `PREVIEW_TOKEN`, `PREVIEW_FPS` (+ preview tuning, see `.env.example`) |
| USB listener | `os.environ` + python-dotenv (reads `camerapi/.env`) | `CAMERAPI_URL`, `BUTTON_MODE`, `BUTTON_CAPTURE_KEY`, `BUTTON_SECONDARY_KEY`, `BUTTON_COOLDOWN` |
| receiver | pydantic `Settings` | `DATA_ROOT`, `LABELSTUDIO_URL`, `LABELSTUDIO_API_KEY`, `LABELSTUDIO_PROJECT_NAME`, `INSPECTOR_URL` (empty = no predictions), `INSPECTOR_TIMEOUT` |
| dashboard | `os.environ` + python-dotenv | `DATA_ROOT`, `LABELSTUDIO_URL`, `LABELSTUDIO_PUBLIC_URL`, `LABELSTUDIO_API_KEY`, `CAMERAPI_URL`, `PREVIEW_TOKEN`, `PREDICTION_PENDING_TIMEOUT_S`, `INSPECTOR_URL` + `RECEIVER_URL` (model chooser) |
| inspector | pydantic `Settings` | `MODEL_PATH` (card = same name `.json`; the default model), `MODELS_DIR` (every `*.onnx` there loads; default the folder of `MODEL_PATH`), `REQUIRE_CARD`, `DATA_ROOT`, `ORT_THREADS`; legacy `CONFIDENCE_THRESHOLD`/`ADAPTER`/`CLASS_NAMES` only without a card |
| Docker (all) | `deploy/.env` | `DATA_DIR`, `LAPTOP_HOST`, `LABELSTUDIO_API_KEY`, LS admin credentials, `PREVIEW_TOKEN`, `INSPECTOR_REQUIRE_CARD`, `INSPECTOR_ORT_THREADS` |

`.env` files are read **once at startup** — restart the service after any change.

---

## Tests

Plain assert scripts (no pytest), run from the repo root; each prints PASS/FAIL per
test and exits non-zero on failure:

```
# services (the services venv: ../venv)
../venv/bin/python -B tests/test_live_camera.py        # camera owner thread, fake pylon
../venv/bin/python -B tests/test_camerapi_http.py      # preview access rules, capture 202
../venv/bin/python -B tests/test_prediction_hook.py    # receiver -> inspector -> pred.json
../venv/bin/python -B tests/test_launch_helpers.py     # optional inspector never stops the hub
../venv/bin/python -B tests/test_dashboard_latest.py   # /api/latest, live relay, auth
../venv/bin/python -B tests/test_parsers.py            # LS capture-id/quality parsers
../venv/bin/python -B tests/test_button_listener.py    # USB listener key parsing (needs pynput)

# ML toolchain + inspector (ml/.venv)
ml/.venv/bin/python -B tests/test_inspector_service.py
ml/.venv/bin/python -B tests/test_{scorer,split,tiler,ingest,stitch,verdict,detector_card}.py
ml/.venv/bin/python -B tests/test_{eval_guards,compare,export,registry}.py
ml/.venv/bin/python -B tests/test_{train_config,train_matrix,notebook}.py
```

Each test process loads one service (they share top-level module names). There is no
build step: "build" = install each service's `requirements.txt` and run.

---

## Repository layout

```
zander/
├── camerapi/     native capture service (FastAPI + pypylon) + USB button listener
├── receiver/     hub ingestion service (FastAPI)
├── dashboard/    hub review UI (FastAPI + SQLite)
├── inspector/    defect detector service (FastAPI + ONNX Runtime, uses ml/infer)
├── ml/           model toolchain: dataset, Colab training, evaluation, export, verifier
├── docs/         CONTRACTS.md — the interfaces between the services and ml/
├── deploy/       Windows production kit (docker-compose, .bat scripts, icon)
├── scripts/      launch.py — native launcher (hub / +camera / camera-only)
├── tests/        self-contained unit tests
├── picreceiver/  minimal standalone upload sink (utility, not wired in)
└── _grave/       archived dead code — ignore
```
