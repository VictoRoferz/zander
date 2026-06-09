#!/usr/bin/env python3
"""
One-command laptop launcher (Windows + macOS).

Starts the three laptop services together and tears them all down on Ctrl-C:
  - Label Studio            :8081  (with local-files serving from DATA_ROOT)
  - receiver (ingestion hub):8002
  - dashboard               :8003

Run from anywhere:  python scripts/launch.py

DATA_ROOT defaults to ~/zander-data (override via the DATA_ROOT env var). It is
forced to be identical for Label Studio's local-files document root and for
both services, so LS, the receiver, and the dashboard all agree on where images
live.

First-run bootstrap (manual, once):
  1. Start this launcher; open http://localhost:8081 and create an LS account.
  2. Copy your token (LS → Account & Settings → Access Token) into
     receiver/.env (LABELSTUDIO_API_KEY) and dashboard/.env (LABELSTUDIO_API_KEY).
  3. Create a dashboard login:  python dashboard/manage_users.py add you@x "You"
  4. Restart this launcher.

Each service reads its own .env (receiver via pydantic-settings, dashboard via
python-dotenv); the launcher only injects DATA_ROOT + the LS local-files vars.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # zander/
RECEIVER_DIR = ROOT / "receiver"
DASHBOARD_DIR = ROOT / "dashboard"

# (name, started Popen) — populated as services come up.
PROCS: list[tuple[str, subprocess.Popen]] = []


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
            sys.stdout.write(f"[{name}] {line}")
            sys.stdout.flush()
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


def main() -> int:
    root = data_root()
    (root / "unlabeled").mkdir(parents=True, exist_ok=True)
    (root / "labeled").mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["DATA_ROOT"] = str(root)
    base_env["PYTHONUNBUFFERED"] = "1"

    ls_env = base_env.copy()
    ls_env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    ls_env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(root)

    print(f"[launch] DATA_ROOT = {root}")
    print("[launch] starting Label Studio (8081), receiver (8002), dashboard (8003)…")
    print("[launch] press Ctrl-C to stop everything.\n")

    start("label-studio", label_studio_args(), ROOT, ls_env)
    start("receiver", [sys.executable, "main.py"], RECEIVER_DIR, base_env)
    start("dashboard", [sys.executable, "main.py"], DASHBOARD_DIR, base_env)

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


if __name__ == "__main__":
    raise SystemExit(main())
