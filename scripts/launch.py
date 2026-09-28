#!/usr/bin/env python3
"""
One-command launcher (Windows + macOS).

Modes (all tear everything down together when a critical child exits or on
Ctrl-C; the optional inspector never does, see below):
  python scripts/launch.py                # hub: Label Studio :8081,
                                          # receiver :8002, dashboard :8003,
                                          # + the inspector :8004 (optional)
  python scripts/launch.py --with-camera  # hub + camerapi :8001 +
                                          # USB button listener (macOS dev:
                                          # everything native in one command)
  python scripts/launch.py --camera-only  # camerapi + listener only — the
                                          # Windows production mode, run
                                          # alongside the Docker hub by
                                          # deploy/start-zander.bat
  --no-listener                           # skip the USB button listener
                                          # (dashboard-button-only, or run a
                                          # stdin-mode listener separately)
  --no-inspector                          # hub modes: don't start the inspector
  --pid-file PATH                         # write this process's PID (used by
                                          # deploy/stop-zander.bat)
  --log-file PATH                         # additionally append all child
                                          # output here (diagnosable when the
                                          # console window is minimized/lost)

The inspector (defect detection, docs/CONTRACTS.md section 7) is a
NON-critical child: it needs the ML inference stack (onnxruntime), which the
service venv does not carry, so it runs under INSPECTOR_PYTHON or else
ml/.venv's interpreter, verified with `-c "import onnxruntime, PIL, fastapi"`.
If neither works it is skipped ("inspector skipped: ..."); if it exits later
that is logged once and the hub keeps running (new captures then get an
"error" prediction instead of a verdict). Only when the launcher started it
does the receiver get INSPECTOR_URL=http://localhost:8004, and only if
INSPECTOR_URL is not already set (environment or receiver/.env), so an
explicit empty value still turns predictions off.

DATA_ROOT defaults to ~/zander-data (override via the DATA_ROOT env var). It
is forced to be identical for Label Studio's local-files document root and
for the hub services (receiver, dashboard, inspector), so they all agree on
where images live. camerapi deliberately does NOT get this injection — its
spool location comes from camerapi/.env (real env vars would override it
under pydantic-settings).

First-run bootstrap (manual, once):
  1. Start this launcher; open http://localhost:8081 and create an LS account.
  2. Copy your token (LS -> Account & Settings -> Access Token) into
     receiver/.env (LABELSTUDIO_API_KEY) and dashboard/.env (LABELSTUDIO_API_KEY).
  3. Create a dashboard login:  python dashboard/manage_users.py add you@x "You"
  4. Restart this launcher.

Each service reads its own .env (receiver/camerapi/inspector via
pydantic-settings, dashboard via python-dotenv); the launcher only injects
DATA_ROOT + the LS local-files vars for the hub services, and INSPECTOR_URL
for the receiver as described above.

Log lines stay ASCII: Windows consoles default to cp1252.
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
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # zander/
RECEIVER_DIR = ROOT / "receiver"
DASHBOARD_DIR = ROOT / "dashboard"
CAMERAPI_DIR = ROOT / "camerapi"
INSPECTOR_DIR = ROOT / "inspector"

INSPECTOR_URL = "http://localhost:8004"
INSPECTOR_IMPORT_CHECK = "import onnxruntime, PIL, fastapi"

# (name, started Popen, critical) — populated as services come up. A critical
# child exiting stops everything; a non-critical one is only logged.
PROCS: list[tuple[str, subprocess.Popen, bool]] = []

# Optional tee target for all child output (--log-file).
LOG_FH = None
LOG_LOCK = threading.Lock()


def _ascii(text: object) -> str:
    """Console-safe text (paths may hold characters cp1252 cannot print)."""
    return str(text).encode("ascii", "replace").decode("ascii")


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(Path.home() / "zander-data"))).expanduser()


def label_studio_args() -> list[str]:
    """Prefer the label-studio console script; fall back to `python -m`."""
    exe = shutil.which("label-studio")
    if exe:
        return [exe, "start", "--host", "0.0.0.0", "--port", "8081"]
    return [sys.executable, "-m", "label_studio", "start", "--host", "0.0.0.0", "--port", "8081"]


def ml_venv_python() -> Path:
    if sys.platform == "win32":
        return ROOT / "ml" / ".venv" / "Scripts" / "python.exe"
    return ROOT / "ml" / ".venv" / "bin" / "python"


def find_inspector_python(env: dict) -> tuple[Optional[str], str]:
    """(interpreter, description) for the inspector; interpreter None = skip it.

    INSPECTOR_PYTHON wins when set (no silent fallback if it is broken);
    otherwise ml/.venv. Either must import the inference stack.
    """
    explicit = (env.get("INSPECTOR_PYTHON") or "").strip()
    if explicit:
        candidate, origin = explicit, "INSPECTOR_PYTHON"
    else:
        candidate, origin = str(ml_venv_python()), "ml/.venv"
        if not Path(candidate).exists():
            return None, f"no ml/.venv interpreter at {candidate} (or set INSPECTOR_PYTHON)"
    try:
        probe = subprocess.run(
            [candidate, "-c", INSPECTOR_IMPORT_CHECK],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{origin} interpreter {candidate} is not runnable ({type(e).__name__})"
    if probe.returncode != 0:
        lines = (probe.stderr or "").strip().splitlines()
        reason = lines[-1] if lines else f"exit code {probe.returncode}"
        return None, f"{origin} interpreter {candidate} lacks the inspector deps ({reason})"
    return candidate, f"{origin} interpreter {candidate}"


def dotenv_defines(path: Path, key: str) -> bool:
    """True if the .env file at `path` assigns `key` (uncommented, any case)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, _ = line.partition("=")
        if sep and name.strip().upper() == key.upper():
            return True
    return False


def inspector_url_configured(env: dict, receiver_env_file: Optional[Path] = None) -> bool:
    """INSPECTOR_URL already set where the receiver would read it (env or .env)."""
    env_file = receiver_env_file if receiver_env_file is not None else RECEIVER_DIR / ".env"
    return any(k.upper() == "INSPECTOR_URL" for k in env) or dotenv_defines(
        env_file, "INSPECTOR_URL"
    )


def first_critical_exit(
    procs: Iterable[tuple[str, subprocess.Popen, bool]], reported: set[str]
) -> Optional[tuple[str, int]]:
    """(name, exit code) of the first critical child that has exited, else None.

    Non-critical children that exited are logged once (their names collect in
    `reported`) and never stop the rest.
    """
    for name, proc, critical in procs:
        code = proc.poll()
        if code is None:
            continue
        if critical:
            return name, code
        if name not in reported:
            reported.add(name)
            print(f"\n[launch] {name} exited with code {code}; it is optional, "
                  "the rest keeps running.")
    return None


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


# Set by --ignore-ctrl-c: children start in their own process group so a
# console Ctrl+C (e.g. a keypad programmed to ctrl+c!) cannot stop them.
SHIELD_CHILDREN = False


def start(name: str, args: list[str], cwd: Path, env: dict, critical: bool = True) -> subprocess.Popen:
    kwargs = {}
    if SHIELD_CHILDREN and sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # Children run with PYTHONUTF8=1, so decode their output as UTF-8
        # regardless of the console's locale (else mojibake).
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )
    PROCS.append((name, proc, critical))
    threading.Thread(target=_pump, args=(name, proc), daemon=True).start()
    return proc


def shutdown() -> None:
    for name, proc, _critical in PROCS:
        if proc.poll() is None:
            print(f"[launch] stopping {name}...")
            proc.terminate()
    deadline = time.time() + 8
    for name, proc, _critical in PROCS:
        try:
            proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            print(f"[launch] killing {name}...")
            proc.kill()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Start the zander services together.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--with-camera", action="store_true",
                      help="hub + camerapi + USB button listener")
    mode.add_argument("--camera-only", action="store_true",
                      help="camerapi + USB button listener only (alongside the Docker hub)")
    ap.add_argument("--no-listener", action="store_true",
                    help="skip the USB button listener in the camera modes")
    ap.add_argument("--no-inspector", action="store_true",
                    help="hub modes: do not start the (optional) inspector")
    ap.add_argument("--pid-file", type=Path, default=None,
                    help="write this launcher's PID here (deleted on exit)")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="also append all child output to this file")
    ap.add_argument("--ignore-ctrl-c", action="store_true",
                    help="production mode (Windows): ignore console Ctrl+C so a "
                         "keypad bound to ctrl+c cannot stop the stack; stop via "
                         "stop-zander.bat (taskkill) instead")
    return ap.parse_args()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    global LOG_FH, SHIELD_CHILDREN
    args = parse_args()
    hub = not args.camera_only
    camera = args.with_camera or args.camera_only

    # A plain `kill` must tear the children down too (Python's default SIGTERM
    # handling skips `finally`, which would orphan them). Windows production
    # uses `taskkill /T` (whole tree), so this is for POSIX/dev.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    if args.ignore_ctrl_c and sys.platform == "win32":
        # A USB keypad bound to ctrl+c types a REAL Ctrl+C: if this console has
        # focus, that would shut the whole camera stack down mid-capture. In
        # production the stack is stopped by stop-zander.bat (taskkill /T), so
        # console interrupts can be ignored entirely — here and, via
        # CREATE_NEW_PROCESS_GROUP (see start()), in every child.
        SHIELD_CHILDREN = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)

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
        inspector_python: Optional[str] = None
        if hub:
            root = data_root()
            for sub in ("unlabeled", "labeled", "predictions"):
                (root / sub).mkdir(parents=True, exist_ok=True)

            hub_env = base_env.copy()
            hub_env["DATA_ROOT"] = str(root)
            ls_env = hub_env.copy()
            ls_env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
            ls_env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(root)

            print(f"[launch] DATA_ROOT = {_ascii(root)}")
            starting.append("Label Studio (8081), receiver (8002), dashboard (8003)")
            if args.no_inspector:
                print("[launch] inspector disabled (--no-inspector)")
            else:
                inspector_python, how = find_inspector_python(base_env)
                if inspector_python is None:
                    print(f"[launch] inspector skipped: {_ascii(how)}")
                else:
                    print(f"[launch] inspector runs on the {_ascii(how)}")
                    starting.append("inspector (8004, optional)")
        if camera:
            starting.append("camerapi (8001)" + ("" if args.no_listener else " + USB button listener"))
        print(f"[launch] starting {'; '.join(starting)}...")
        print("[launch] press Ctrl-C to stop everything.\n")

        if hub:
            start("label-studio", label_studio_args(), ROOT, ls_env)
            receiver_env = hub_env.copy()
            if inspector_python is not None:
                # Non-critical: without it the hub still ingests and labels.
                start("inspector", [inspector_python, "main.py"], INSPECTOR_DIR, hub_env,
                      critical=False)
                if inspector_url_configured(base_env):
                    print("[launch] receiver keeps its own INSPECTOR_URL setting")
                else:
                    receiver_env["INSPECTOR_URL"] = INSPECTOR_URL
                    print(f"[launch] receiver: INSPECTOR_URL={INSPECTOR_URL}")
            start("receiver", [sys.executable, "main.py"], RECEIVER_DIR, receiver_env)
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

        reported: set[str] = set()
        try:
            while True:
                exited = first_critical_exit(PROCS, reported)
                if exited is not None:
                    name, code = exited
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
