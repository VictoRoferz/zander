# Zander — laptop hub deployment

The **laptop hub** = Label Studio + `receiver` + `dashboard`, all on one machine.
The **Raspberry Pi** stays the camera node (`camerapi`), wired to the Basler camera,
and uploads captures to this laptop over the LAN.

```
🍓 Pi: camerapi (:8001)  ──upload──▶  🖥️ laptop:  receiver (:8002) ──creates tasks──▶ Label Studio (:8081)
                                                   dashboard (:8003) ──live-reads──────▶ Label Studio
```

Two ways to run the hub: **Docker (recommended)** or **native (no Docker)**.

---

## A. Docker (recommended)

### One-time setup on the Windows PC
1. Install **Docker Desktop** and start it (wait until it says *running*). In
   Settings → Resources → File Sharing, make sure the drive holding this repo is shared.
2. Install **Git**, then:
   ```
   git clone <repo-url>
   cd zander
   git checkout cv-project-on-pc
   ```
3. Configure — **the only file you edit**:
   ```
   cd deploy
   copy .env.example .env
   ```
   Open `deploy\.env` and set:
   - `PI_IP` = the Raspberry Pi's IP (for the dashboard "Capture" button)
   - `LAPTOP_HOST` = `localhost` (or this PC's LAN IP if others open the dashboard remotely)
   - leave `LABELSTUDIO_API_KEY` blank for now (filled in step 5)

### First start (gets the token)
4. Start everything:
   ```
   start-zander.bat        (or:  docker compose up -d)
   ```
   Label Studio comes up on **http://localhost:8081**. The receiver/dashboard start too,
   but log "label studio not ready" until the token is set — that's expected.
5. Open **http://localhost:8081**, log in with the admin from `.env`
   (`LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD`). Go to **Account & Settings**,
   ensure **Legacy Token** is available (it's enabled headlessly), and **copy the legacy token**
   (a ~40-char string — *not* the long `eyJ…` one).
6. Paste it into `deploy\.env` → `LABELSTUDIO_API_KEY=<token>`, then re-apply:
   ```
   docker compose up -d
   ```
   The receiver now logs `Label Studio ready: project 'PCB Defect Inspection'`.
7. Create a dashboard login:
   ```
   docker compose exec dashboard python manage_users.py add you@example.com "You"
   ```
8. In Label Studio → project **PCB Defect Inspection** → **Settings → Webhooks → Add Webhook**:
   `http://localhost:8002/api/v1/webhook/annotation-created`

### Make a one-click desktop icon
```
powershell -ExecutionPolicy Bypass -File install-shortcut.ps1
```
This drops a **Zander** icon on the Desktop. Double-click it any time to start the hub
and open the dashboard. (Uses `deploy\zander.ico`.)

### Daily use
- **Start:** double-click **Zander** (or `start-zander.bat`).
- **Stop:** `stop-zander.bat` (data is kept).
- **Logs:** `docker compose logs -f receiver` (or `dashboard`, `labelstudio`).
- **Status:** `docker compose ps`.

### Data & persistence
- Images + the dashboard's user DB live in `deploy\data\` (host folder, `DATA_DIR` in `.env`).
- Label Studio's projects/annotations live in the `zander_ls_internal` Docker volume.
- Both survive `docker compose down`.

---

## B. Native (no Docker)

For a machine that can't run Docker. Needs **Python 3.11** on PATH.
```
cd deploy\windows-native
setup.bat        REM one-time: makes .venv and installs Label Studio + deps
```
Put your legacy token in **`receiver\.env`** and **`dashboard\.env`**
(`LABELSTUDIO_API_KEY=…`), create a user (`python dashboard\manage_users.py add …`
from an activated `.venv`), then:
```
start.bat        REM runs scripts\launch.py — LS 8081 + receiver 8002 + dashboard 8003
```
> Note: the native path uses the **per-service** `.env` files (`receiver\.env`,
> `dashboard\.env`), not `deploy\.env` (which is Docker-only).

---

## Configure the Raspberry Pi (either path)

On the Pi, point `camerapi` at this Windows PC. Edit `~/zander/camerapi/.env`:
```
INGEST_URL=http://<WINDOWS_PC_IP>:8002
DASHBOARD_URL=http://<WINDOWS_PC_IP>:8003
DATA_ROOT=/home/pi/zander-data
USE_CAMERA=true
```
Then run `python main.py` (and `scripts/button_listener.py` for the GPIO button).
Find the Windows IP with `ipconfig` (IPv4 on the LAN the Pi shares).

---

## Changing IPs later
- **Laptop:** edit `deploy\.env` → `PI_IP` (and `LAPTOP_HOST`), then `docker compose up -d`.
- **Pi:** edit `camerapi\.env` → `INGEST_URL` / `DASHBOARD_URL`, restart camerapi.

## End-to-end check
1. `docker compose ps` → all up; `curl http://localhost:8002/api/v1/status` → `label_studio.ready: true`.
2. Press the Pi button (or dashboard **Capture**) → camerapi logs `ingested ok` → tile appears under
   "Unlabeled" → it shows as a task in Label Studio.
3. Brush-label it in LS, pick `iO`/`NiO`, **Submit** → tile moves to "Labeled" with the quality badge.

## Troubleshooting
- **receiver logs `503 label studio not ready`** → token missing/wrong in `deploy\.env`; set it and `docker compose up -d`. Must be the **legacy** token.
- **dashboard `401` from LS** → same token issue.
- **Pi `connection timed out` / `no route to host`** → wrong `INGEST_URL` IP, different subnet, or Windows Firewall blocking 8002/8003 (allow them for Docker/Python).
- **Images don't render in LS** → the `DATA_DIR` bind mount isn't shared in Docker Desktop, or `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` ≠ the mounted `/data` (don't change these in the compose file).
- **Label Studio version** → pin `LS_IMAGE_TAG` in `.env` to the version you validated; avoid `latest` in production.

## macOS dev
Same Docker flow (`docker compose up -d` from `deploy/`), or the native launcher:
`python scripts/launch.py`. camerapi can run on the Mac with `USE_CAMERA=false` (sample image).
