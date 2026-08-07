# Zander — single-PC deployment

The whole system runs on **one Windows PC**: Label Studio + `receiver` +
`dashboard` in Docker, plus the **native camera stack** — `camerapi` with the
Basler GigE camera direct-attached, and the USB button listener. (Native because a
container can neither discover a GigE camera on a host NIC nor hook USB keyboard
input.)

```
USB keypad (F13) ─▶ camerapi (native, :8001) ──upload──▶ receiver (Docker, :8002) ──creates tasks──▶ Label Studio (Docker, :8081)
dashboard "Capture" ─┘                                    dashboard (Docker, :8003) ──live-reads─────▶ Label Studio
```

---

## One-time setup on the Windows PC

1. Install **Docker Desktop** and start it (wait until it says *running*). In
   Settings → Resources → File Sharing, make sure the drive holding this repo is shared.
2. Install **Python 3.11** (check "Add to PATH") and **Git**, then:
   ```
   git clone <repo-url>
   cd zander
   git checkout cv-project-on-pc
   ```
3. Configure Docker — usually nothing to change:
   ```
   cd deploy
   copy .env.example .env
   ```
   Open `deploy\.env`: leave `LABELSTUDIO_API_KEY` blank for now (filled in the
   first-start step); `LAPTOP_HOST=localhost` unless others open the dashboard
   from another machine.
4. Set up the native camera stack:
   ```
   setup-camera.bat
   ```
   This creates the repo venv and installs camerapi's deps (incl. **pypylon** —
   the PyPI wheel bundles the pylon runtime — and **pynput** for the USB button),
   and creates `camerapi\.env` from the example (defaults suit this setup).
   Safety net: if this step is skipped or half-finished, `start-zander.bat`
   detects it (missing `.venv\camera-stack-ready` marker) and runs the setup
   itself on the next start.
5. **Camera NIC:** give the Ethernet port that the Basler camera is plugged into a
   static IP on the camera subnet (e.g. `192.168.177.90`, mask `255.255.255.0`, no
   gateway). The camera sits at `192.168.177.95`. Verify: `ping 192.168.177.95`.
   For commissioning tools (pylon IP Configurator / Viewer), install the free
   Basler **pylon** suite — capture itself doesn't need it.
6. **USB keypad:** with the vendor tool, program key 1 → **F13** (capture) and
   key 2 → **F14** (test shot). F13/F14 don't exist on normal keyboards, so
   ordinary typing can never trigger the camera.
7. **Firewall:** on the first camerapi start Windows will ask — allow
   `python.exe` on **Private** networks (the Docker→host "Capture" call and the
   camera's UDP stream both arrive as inbound traffic). Inbound 8002/8003 from
   the LAN are no longer needed (there is no Pi) unless people browse the
   dashboard from other machines.

### First start (gets the token)

8. Start everything:
   ```
   start-zander.bat        (or double-click the Zander icon, step 11)
   ```
   Label Studio comes up on **http://localhost:8081**. The receiver/dashboard
   start too, but log "label studio not ready" until the token is set — expected.
9. Open **http://localhost:8081**, log in with the admin from `.env`
   (`LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD`). Go to **Account & Settings**
   and **copy the legacy token** (a ~40-char string — *not* the long `eyJ…` one).
   Paste it into `deploy\.env` → `LABELSTUDIO_API_KEY=<token>`, then re-apply:
   ```
   docker compose up -d
   ```
   The receiver now logs `Label Studio ready: project 'PCB Defect Inspection'`.
10. Finish the bootstrap:
    - dashboard login: `docker compose exec dashboard python manage_users.py add you@example.com "You"`
    - LS webhook: project → **Settings → Webhooks → Add Webhook** →
      `http://localhost:8002/api/v1/webhook/annotation-created`

### Make the one-click desktop icon

11. ```
    powershell -ExecutionPolicy Bypass -File install-shortcut.ps1
    ```
    Drops a **Zander** icon on the Desktop. Double-clicking it starts the three
    containers **and** the camera stack, then opens the dashboard.

---

## Daily use

- **Start:** double-click **Zander** (or `start-zander.bat`). The camera stack runs
  in a minimized console window titled **"Zander Camera"**.
- **Capture:** press the keypad button (F13) or the dashboard's **Capture** button.
- **Stop:** `stop-zander.bat` (kills the camera stack via `camera-stack.pid`, then
  stops the containers; data is kept).
- **Logs:** containers → `docker compose logs -f receiver` (or `dashboard`,
  `labelstudio`); camera stack → the "Zander Camera" window or `deploy\camera-stack.log`.
- **Status:** `docker compose ps`; camera + spool depth →
  `curl http://localhost:8001/api/v1/status`.

### Data & persistence

- Images + the dashboard's user DB live in `deploy\data\` (host folder, `DATA_DIR`
  in `.env`) — the canonical store.
- camerapi's `%USERPROFILE%\zander-data\spool\` holds only not-yet-acknowledged
  captures (normally empty; it drains the moment the receiver is up).
- Label Studio's projects/annotations live in the `zander_ls_internal` Docker
  volume. Everything survives `docker compose down`.

---

## Native fallback (no Docker)

For a machine that can't run Docker. Needs **Python 3.11** on PATH.
```
cd deploy\windows-native
setup.bat        REM one-time: makes .venv and installs Label Studio + hub deps
```
Put your legacy token in **`receiver\.env`** and **`dashboard\.env`**
(`LABELSTUDIO_API_KEY=…`), create a user (`python dashboard\manage_users.py add …`
from an activated `.venv`), run `deploy\setup-camera.bat` for the camera deps, then:
```
..\..\.venv\Scripts\python ..\..\scripts\launch.py --with-camera
```
> The native path uses the **per-service** `.env` files (`receiver\.env`,
> `dashboard\.env`, `camerapi\.env`), not `deploy\.env` (which is Docker-only).

---

## Configuration reference

- `deploy\.env` — Docker side: `LABELSTUDIO_API_KEY`, `DATA_DIR`, `LAPTOP_HOST`,
  LS admin credentials, `LS_IMAGE_TAG`. After editing: `docker compose up -d` again.
- `camerapi\.env` — camera stack: `CAMERA_TRANSPORT=ethernet`, `USE_CAMERA`,
  `INGEST_URL`/`DASHBOARD_URL` (localhost defaults are right for this setup),
  button keys (`BUTTON_CAPTURE_KEY`, …). Restart the camera stack after editing.
- The dashboard container reaches native camerapi at `host.docker.internal:8001`
  (Docker Desktop built-in; on plain Linux Docker you would add
  `extra_hosts: host.docker.internal:host-gateway`).

## End-to-end check

1. `docker compose ps` → all up; `curl http://localhost:8002/api/v1/status` → `label_studio.ready: true`.
2. Press the keypad button (or dashboard **Capture**) → the "Zander Camera" window
   logs `ingested ok` → tile appears under "Unlabeled" → task in Label Studio.
3. Label it in LS (bounding boxes), pick `iO`/`NiO`, **Submit** → tile moves to
   "Labeled" with the quality badge.

## Troubleshooting

- **receiver logs `503 label studio not ready`** → token missing/wrong in `deploy\.env`; set it and `docker compose up -d`. Must be the **legacy** token.
- **dashboard `401` from LS** → same token issue.
- **Dashboard Capture button → `502 camerapi unreachable`** → the camera stack isn't
  running (no "Zander Camera" window). Re-run `start-zander.bat`; check `deploy\camera-stack.log`.
- **Keypad press does nothing** → is the "Zander Camera" window up? Run the listener
  with `LOG_LEVEL=DEBUG` to see what the keypad actually emits (re-program to F13 if
  it sends something else). Note: global hooks receive nothing while Windows is
  **locked** — disable auto-lock on the station or use the dashboard button.
- **Camera not found** → NIC static IP wrong/missing (`ping 192.168.177.95`), cable,
  or another app holds the camera (pylon Viewer open?). After a hard kill the camera
  can stay claimed a few seconds (GigE heartbeat) — retry.
- **Slow captures (~8 s)** → the WiFi transport profile got forced somehow; ensure
  `CAMERA_TRANSPORT=ethernet` in `camerapi\.env` (auto also picks ethernet on Windows).
- **Images don't render in LS** → the `DATA_DIR` bind mount isn't shared in Docker
  Desktop, or `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` ≠ the mounted `/data`.
- **Label Studio version** → pin `LS_IMAGE_TAG` in `.env` to the version you
  validated; avoid `latest` in production.

## macOS dev

Docker flow (`docker compose up -d` from `deploy/`) plus
`python scripts/launch.py --camera-only`, or everything native:
`python scripts/launch.py --with-camera`. Without the camera set `USE_CAMERA=false`
in `camerapi/.env`; without the keypad use
`BUTTON_MODE=stdin python camerapi/scripts/usb_button_listener.py` (keyboard mode
needs Accessibility permission on macOS).
