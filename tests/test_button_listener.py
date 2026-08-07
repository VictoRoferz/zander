#!/usr/bin/env python3
"""Self-contained checks for usb_button_listener.parse_key (no pytest).

Run:  python tests/test_button_listener.py
Skips (exit 0) if pynput/requests are not installed in this environment.
"""
import importlib.util
import sys
from pathlib import Path

LISTENER = Path(__file__).resolve().parents[1] / "camerapi" / "scripts" / "usb_button_listener.py"

try:
    import pynput  # noqa: F401
    import requests  # noqa: F401
except ImportError as e:
    print(f"SKIP: {e} (install camerapi/requirements.txt to run this test)")
    sys.exit(0)

spec = importlib.util.spec_from_file_location("usb_button_listener", LISTENER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from pynput.keyboard import Key, KeyCode  # noqa: E402

# Named function keys resolve through the Key enum.
assert mod.parse_key("f13") is Key.f13
assert mod.parse_key(" F20 ") is Key.f20  # trimmed + case-insensitive
assert mod.parse_key("esc") is Key.esc

# Enum lookup must not leak non-member attributes.
try:
    mod.parse_key("mro")
    raise AssertionError("'mro' should not resolve to a key")
except ValueError:
    pass

# Explicit virtual-key escape hatch (decimal and hex).
assert mod.parse_key("vk:124") == KeyCode.from_vk(124)
assert mod.parse_key("vk:0x7C") == KeyCode.from_vk(0x7C)

# F21–F24: Windows-only vk mapping, clear error elsewhere.
if sys.platform == "win32":
    assert mod.parse_key("f21") == KeyCode.from_vk(0x84)
    assert mod.parse_key("f24") == KeyCode.from_vk(0x87)
else:
    try:
        mod.parse_key("f21")
        raise AssertionError("f21 should fail off-Windows")
    except ValueError:
        pass

# Single printable char is allowed (dev only, warns).
assert mod.parse_key("c") == KeyCode.from_char("c")

# Garbage is rejected.
for bad in ("", "f99", "not-a-key"):
    try:
        mod.parse_key(bad)
        raise AssertionError(f"{bad!r} should fail")
    except ValueError:
        pass

# parse_binding: singles pass through, "+" strings become pynput hotkey sets.
kind, val = mod.parse_binding("f13")
assert kind == "single" and val is Key.f13
kind, keys = mod.parse_binding("ctrl+c")
assert kind == "combo" and Key.ctrl in keys and KeyCode.from_char("c") in keys
kind, keys = mod.parse_binding("<ctrl>+v")  # pre-bracketed form also accepted
assert kind == "combo" and Key.ctrl in keys and KeyCode.from_char("v") in keys
kind, keys = mod.parse_binding("ctrl+shift+p")
assert kind == "combo" and len(keys) == 3
for bad in ("ctrl+", "nope+alsono"):
    try:
        mod.parse_binding(bad)
        raise AssertionError(f"{bad!r} should fail")
    except ValueError:
        pass
kind, val = mod.parse_binding("+")  # a bare "+" is a plain single-char key
assert kind == "single" and val == KeyCode.from_char("+")

# A HotKey built from a parsed combo activates on the full chord and
# re-arms only after release (auto-repeat of the held chord won't re-fire).
from pynput.keyboard import HotKey  # noqa: E402

fired = []
hk = HotKey(mod.parse_binding("ctrl+c")[1], lambda: fired.append(1))
hk.press(Key.ctrl)
hk.press(KeyCode.from_char("c"))
assert fired == [1]
hk.press(KeyCode.from_char("c"))  # OS auto-repeat while held
assert fired == [1]
hk.release(KeyCode.from_char("c"))
hk.press(KeyCode.from_char("c"))
assert fired == [1, 1]

# _key_matches: enum targets, vk targets (incl. enum-with-value), char targets.
assert mod._key_matches(Key.f13, Key.f13)
assert not mod._key_matches(Key.f14, Key.f13)
assert not mod._key_matches(None, Key.f13)
assert mod._key_matches(KeyCode.from_vk(124), KeyCode.from_vk(124))
if getattr(Key.f13.value, "vk", None) is not None:
    assert mod._key_matches(Key.f13, KeyCode.from_vk(Key.f13.value.vk))
assert mod._key_matches(KeyCode.from_char("C"), KeyCode.from_char("c"))
assert not mod._key_matches(KeyCode.from_char("x"), KeyCode.from_char("c"))

print("test_button_listener: all assertions passed")
