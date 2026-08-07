#!/usr/bin/env python3
"""
One-command launcher (Windows + macOS).

Modes (all tear everything down together when any child exits or on Ctrl-C):
  python scripts/launch.py                # hub trio: Label Studio :8081,
                                          # receiver :8002, dashboard :8003
  python scripts/launch.py --with-camera  # hub trio + camerapi :8001 +
                                          # USB button listener (macOS dev:
                                          # everything native in one command)
  python scripts/launch.py --camera-only  # camerapi + listener only — the
                                          # Windows production mode, run
                                          # alongside the Docker hub by
                                          # deploy/start-zander.bat
  --no-listener                           # skip the USB button listener
                                          # (dashboard-button-only, or run a
                                          # stdin-mode listener separately)
  --pid-file PATH                         # write this process's PID (used by
                                          # deploy/stop-zander.bat)
  --log-file PATH                         # additionally append all child
                                          # output here (diagnosable when the
                                          # console window is minimized/lost)

DATA_ROOT defaults to ~/zander-data (override via the DATA_ROOT env var). It
is forced to be identical for Label Studio's local-files document root and
for the hub services, so LS, the receiver, and the dashboard all agree on
where images live. camerapi deliberately does NOT get this injection — its
spool location comes from camerapi/.env (real env vars would override it
under pydantic-settings).

First-run bootstrap (manual, once):
  1. Start this launcher; open http://localhost:8081 and create an LS account.
  2. Copy your token (LS → Account & Settings → Access Token) into
     receiver/.env (LABELSTUDIO_API_KEY) and dashboard/.env (LABELSTUDIO_API_KEY).
  3. Create a dashboard login:  python dashboard/manage_users.py add you@x "You"
  4. Restart this launcher.

Each service reads its own .env (receiver/camerapi via pydantic-settings,
dashboard via python-dotenv); the launcher only injects DATA_ROOT + the LS
local-files vars for the hub services.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # zander/
RECEIVER_DIR = ROOT / "receiver"
DASHBOARD_DIR = ROOT / "dashboard"
CAMERAPI_DIR = ROOT / "camerapi"

# (name, started Popen) — populated as services come up.
PROCS: list[tuple[str, subprocess.Popen]] = []

# Optional tee target for all child output (--log-file).
LOG_FH = None
LOG_LOCK = threading.Lock()


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(Path.home() / "zander-data"))).expanduser()


def label_studio_args() -> list[str]:
    """Prefer the label-studio console script; fall back to `python -m`."""
    exe = shutil.which("label-studio")
    if exe:
        return [exe, "start", "--host", "0.0.0.0", "--port", "8081"]
    return [sys.executable, "-m", "label_studio", "start", "--host", "0.0.0.0", "--port", "8081"]


def _pump(name: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if line:
            out = f"[{name}] {line}"
            sys.stdout.write(out)
            sys.stdout.flush()
            if LOG_FH is not None:
                with LOG_LOCK:
                    LOG_FH.write(out)
                    LOG_FH.flush()
    proc.stdout.close()


def start(name: str, args: list[str], cwd: Path, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    PROCS.append((name, proc))
    threading.Thread(target=_pump, args=(name, proc), daemon=True).start()
    return proc


def shutdown() -> None:
    for name, proc in PROCS:
        if proc.poll() is None:
            print(f"[launch] stopping {name}…")
            proc.terminate()
    deadline = time.time() + 8
    for name, proc in PROCS:
        try:
            proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            print(f"[launch] killing {name}…")
            proc.kill()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Start the zander services together.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--with-camera", action="store_true",
                      help="hub trio + camerapi + USB button listener")
    mode.add_argument("--camera-only", action="store_true",
                      help="camerapi + USB button listener only (alongside the Docker hub)")
    ap.add_argument("--no-listener", action="store_true",
                    help="skip the USB button listener in the camera modes")
    ap.add_argument("--pid-file", type=Path, default=None,
                    help="write this launcher's PID here (deleted on exit)")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="also append all child output to this file")
    return ap.parse_args()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    global LOG_FH
    args = parse_args()
    hub = not args.camera_only
    camera = args.with_camera or args.camera_only

    # A plain `kill` must tear the children down too (Python's default SIGTERM
    # handling skips `finally`, which would orphan them). Windows production
    # uses `taskkill /T` (whole tree), so this is for POSIX/dev.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        LOG_FH = open(args.log_file, "a", encoding="utf-8", buffering=1)
    if args.pid_file is not None:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    # Windows consoles/pipes default to cp1252, which can't encode the unicode
    # arrows/dashes in our log messages ("Logging error" tracebacks, harmless
    # but alarming). Force UTF-8 for all children.
    base_env["PYTHONUTF8"] = "1"

    try:
        starting = []
        if hub:
            root = data_root()
            (root / "unlabeled").mkdir(parents=True, exist_ok=True)
            (root / "labeled").mkdir(parents=True, exist_ok=True)

            hub_env = base_env.copy()
            hub_env["DATA_ROOT"] = str(root)
            ls_env = hub_env.copy()
            ls_env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
            ls_env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(root)

            print(f"[launch] DATA_ROOT = {root}")
            starting.append("Label Studio (8081), receiver (8002), dashboard (8003)")
        if camera:
            starting.append("camerapi (8001)" + ("" if args.no_listener else " + USB button listener"))
        print(f"[launch] starting {'; '.join(starting)}…")
        print("[launch] press Ctrl-C to stop everything.\n")

        if hub:
            start("label-studio", label_studio_args(), ROOT, ls_env)
            start("receiver", [sys.executable, "main.py"], RECEIVER_DIR, hub_env)
            start("dashboard", [sys.executable, "main.py"], DASHBOARD_DIR, hub_env)
        if camera:
            # No DATA_ROOT injection: camerapi's transient spool root comes from
            # camerapi/.env and must not be relocated to the hub's image store.
            start("camerapi", [sys.executable, "main.py"], CAMERAPI_DIR, base_env)
            if not args.no_listener:
                start(
                    "usb-button",
                    [sys.executable, str(CAMERAPI_DIR / "scripts" / "usb_button_listener.py")],
                    CAMERAPI_DIR,
                    base_env,
                )

        try:
            while True:
                for name, proc in PROCS:
                    code = proc.poll()
                    if code is not None:
                        print(f"\n[launch] {name} exited with code {code}; stopping the rest.")
                        return 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[launch] Ctrl-C received.")
            return 0
        finally:
            shutdown()
    finally:
        if args.pid_file is not None:
            try:
                args.pid_file.unlink()
            except OSError:
                pass
        if LOG_FH is not None:
            LOG_FH.close()


if __name__ == "__main__":
    raise SystemExit(main())
