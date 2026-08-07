#!/usr/bin/env python3
"""
USB macro-keypad trigger for camerapi (replaces the Pi's GPIO button_listener).

A 2-key HID keypad (MK321pro-style) enumerates as a USB keyboard; program its
keys to rare codes (F13/F14 recommended — inert in normal apps, so no
suppression is needed and normal typing can never fire a capture). This
script installs a global keyboard hook (pynput) and POSTs to camerapi:

    <capture key>   → POST {CAMERAPI_URL}/api/v1/capture
    <secondary key> → POST {CAMERAPI_URL}/api/v1/test-camera  (configurable)

Like the old GPIO listener it sends no X-Triggered-By header — camerapi
attributes button captures by asking the dashboard who is logged in.

Configuration (env vars, or camerapi/.env which is loaded if present):
    CAMERAPI_URL            default http://localhost:8001
    BUTTON_MODE             keyboard | stdin        (default keyboard)
    BUTTON_CAPTURE_KEY      default f13
    BUTTON_SECONDARY_KEY    default f14; empty = disabled
    BUTTON_SECONDARY_ACTION test-camera | none      (default test-camera)
    BUTTON_COOLDOWN         min seconds between accepted triggers, counted
                            after the request finishes (default 2.0)
    CAPTURE_TIMEOUT         per-request timeout; a cold Basler grab can take
                            >30 s, keep generous (default 40)
    LOG_LEVEL               DEBUG logs every key event — use it to verify
                            what the keypad firmware actually emits

Key name formats for BUTTON_*_KEY: pynput Key names (f1–f20, media_*, …),
f21–f24 (Windows only, via virtual-key codes), vk:<code> (explicit
virtual-key escape hatch), or a single printable character (dev only — it
will also type into the focused app).

BUTTON_MODE=stdin is the no-hardware dev mode: Enter or 'c' = capture,
't' = secondary, 'q' = quit. Run it in its own terminal (under launch.py the
children share stdin, so launch.py only ever starts keyboard mode).

macOS note: keyboard mode needs Accessibility/Input Monitoring permission
for the hosting terminal; without it pynput silently receives nothing.

Windows note: global hooks receive nothing while the session is locked —
the keypad is dead on the lock screen (dashboard Capture still works from
another machine's browser).
"""
from __future__ import annotations

import logging
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    # Shared config with the camerapi service; real env vars take precedence.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

CAMERAPI_URL = os.getenv("CAMERAPI_URL", "http://localhost:8001").rstrip("/")
BUTTON_MODE = os.getenv("BUTTON_MODE", "keyboard").strip().lower()
CAPTURE_KEY = os.getenv("BUTTON_CAPTURE_KEY", "f13")
SECONDARY_KEY = os.getenv("BUTTON_SECONDARY_KEY", "f14")
SECONDARY_ACTION = os.getenv("BUTTON_SECONDARY_ACTION", "test-camera").strip().lower()
COOLDOWN = float(os.getenv("BUTTON_COOLDOWN", "2.0"))
CAPTURE_TIMEOUT = float(os.getenv("CAPTURE_TIMEOUT", "40"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

ENDPOINTS = {
    "capture": f"{CAMERAPI_URL}/api/v1/capture",
    "test-camera": f"{CAMERAPI_URL}/api/v1/test-camera",
}

# Root stays at INFO so LOG_LEVEL=DEBUG doesn't drag urllib3 internals along;
# only this script's logger gets the configured level.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("usb-button")
log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# VK_F13..VK_F24 on Windows.
_VK_F13 = 0x7C


def parse_key(name: str):
    """Turn a config string into a pynput Key/KeyCode (see module docstring)."""
    from pynput.keyboard import Key, KeyCode

    raw = name.strip().lower()
    if not raw:
        raise ValueError("empty key name")
    enum_key = Key.__members__.get(raw)
    if enum_key is not None:
        return enum_key
    m = re.fullmatch(r"f(2[1-4])", raw)
    if m:
        if sys.platform != "win32":
            raise ValueError(f"'{raw}' (F21–F24) needs Windows virtual-key codes")
        return KeyCode.from_vk(_VK_F13 + int(m.group(1)) - 13)
    if raw.startswith("vk:"):
        return KeyCode.from_vk(int(raw[3:], 0))
    if len(raw) == 1:
        log.warning(
            f"'{raw}' is a printable key — every press ALSO types into the focused "
            "app and any typing of it triggers the camera. Dev use only; program "
            "the keypad to F13/F14 for production."
        )
        return KeyCode.from_char(raw)
    raise ValueError(f"unrecognized key name '{name}'")


def _key_matches(pressed, target) -> bool:
    """Compare a hook event against a parsed target key."""
    from pynput.keyboard import Key

    if pressed is None:
        return False
    if isinstance(target, Key):
        return pressed == target
    # target is a KeyCode: match by vk if it has one, else by char.
    if target.vk is not None:
        vk = getattr(pressed, "vk", None)
        if vk is None:  # Key enum members carry their vk on .value
            vk = getattr(getattr(pressed, "value", None), "vk", None)
        return vk == target.vk
    char = getattr(pressed, "char", None)
    return char is not None and char.lower() == target.char


# One request at a time: covers double-taps, keyboard auto-repeat bursts and
# the cooldown window with a single mechanism.
_busy = threading.Event()


def trigger(action: str, source: str) -> None:
    if action == "none":
        return
    if _busy.is_set():
        log.debug(f"{action} ignored ({source}): request in flight or cooling down")
        return
    _busy.set()
    threading.Thread(target=_do_request, args=(action,), daemon=True).start()


def _do_request(action: str) -> None:
    url = ENDPOINTS[action]
    try:
        log.info(f"{action} → POST {url}")
        resp = requests.post(url, timeout=CAPTURE_TIMEOUT)
        body = resp.text[:200].replace("\n", " ")
        log.info(f"{action} ← {resp.status_code} {body}")
    except requests.RequestException as e:
        log.warning(f"{action} failed: {e} — is the camera stack up? Still listening.")
    finally:
        time.sleep(COOLDOWN)
        _busy.clear()


def run_keyboard_mode(stop: threading.Event) -> int:
    try:
        from pynput import keyboard
    except ImportError:
        log.error("pynput is not installed — `pip install pynput`, or use BUTTON_MODE=stdin")
        return 2

    try:
        capture_key = parse_key(CAPTURE_KEY)
        secondary_key = (
            parse_key(SECONDARY_KEY)
            if SECONDARY_KEY.strip() and SECONDARY_ACTION != "none"
            else None
        )
    except ValueError as e:
        log.error(
            f"bad key config: {e}. Supported: pynput Key names (f1–f20, media_*), "
            "f21–f24 (Windows), vk:<code>, or a single character."
        )
        return 2

    bindings = [(capture_key, "capture", CAPTURE_KEY)]
    if secondary_key is not None:
        bindings.append((secondary_key, SECONDARY_ACTION, SECONDARY_KEY))

    held: set[int] = set()  # binding indices physically held (auto-repeat guard)

    def on_press(key):
        log.debug(f"key press: {key!r}")
        for i, (target, action, label) in enumerate(bindings):
            if _key_matches(key, target):
                if i in held:
                    return
                held.add(i)
                trigger(action, f"key '{label}'")

    def on_release(key):
        for i, (target, _action, _label) in enumerate(bindings):
            if _key_matches(key, target):
                held.discard(i)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    summary = f"'{CAPTURE_KEY}' → capture"
    if secondary_key is not None:
        summary += f", '{SECONDARY_KEY}' → {SECONDARY_ACTION}"
    log.info(f"keyboard hook active: {summary}")
    if sys.platform == "darwin":
        log.info(
            "macOS: this needs Accessibility/Input Monitoring permission for the "
            "terminal — if nothing fires, grant it or use BUTTON_MODE=stdin."
        )
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        listener.stop()
        log.info("listener stopped")
    return 0


def run_stdin_mode(stop: threading.Event) -> int:
    print("stdin trigger mode — Enter/'c' = capture, 't' = secondary, 'q' = quit.")
    try:
        for line in sys.stdin:
            if stop.is_set():
                break
            cmd = line.strip().lower()
            if cmd in ("", "c"):
                trigger("capture", "stdin")
            elif cmd == "t":
                trigger(SECONDARY_ACTION, "stdin")
            elif cmd == "q":
                break
            else:
                print("Enter/'c' = capture, 't' = secondary, 'q' = quit.")
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    stop = threading.Event()

    def _handle_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows console close / Ctrl-Break
        signal.signal(signal.SIGBREAK, _handle_signal)

    try:
        resp = requests.get(f"{CAMERAPI_URL}/api/v1/health", timeout=2)
        log.info(f"camerapi healthy at {CAMERAPI_URL} ({resp.status_code})")
    except requests.RequestException:
        log.info(f"camerapi not reachable yet at {CAMERAPI_URL} — will keep listening")

    if BUTTON_MODE == "stdin":
        code = run_stdin_mode(stop)
    elif BUTTON_MODE == "keyboard":
        try:
            code = run_keyboard_mode(stop)
        except KeyboardInterrupt:
            code = 0
    else:
        log.error(f"unknown BUTTON_MODE='{BUTTON_MODE}' (use 'keyboard' or 'stdin')")
        return 2

    # The request worker is a daemon thread — give a just-fired trigger a
    # moment to reach camerapi before exiting (once the request arrives,
    # capture + spool are durable server-side; no need to wait for the grab).
    deadline = time.time() + 2.0
    while _busy.is_set() and time.time() < deadline:
        time.sleep(0.1)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
