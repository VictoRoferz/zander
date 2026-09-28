"""
Dashboard live panel logic (dashboard/static/app.js), run under node with a
stub DOM: the status chip and the overlay for camerapi's preview states, in
particular "streaming" while no frame ever arrived (the Windows-firewall
case, where the chip used to say LIVE).

Run from the repo root (no pytest needed):
    ../venv/bin/python -B tests/test_dashboard_live_panel.py
Skips (exit 0) when node is not installed. Nothing here talks to a network:
fetch never resolves and timers only run when a test fires them.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "dashboard" / "static" / "app.js"
NODE = shutil.which("node")

# Loads app.js into a vm context with a stub DOM, then runs each step of the
# scenario (JS source) and prints one JSON line of observations per step.
HARNESS = r"""
const vm = require("vm");
const fs = require("fs");
const appPath = process.env.ZANDER_APP_JS;
const scenarioJson = process.env.ZANDER_SCENARIO;

function node(tag) {
  return {
    tag, children: [], attrs: {}, hidden: false, className: "", src: "", style: {},
    _text: "",
    get textContent() {
      return this._text + this.children.map((c) => c.textContent).join("");
    },
    set textContent(v) { this._text = String(v); this.children = []; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    setAttributeNS(ns, k, v) { this.attrs[k] = String(v); },
    appendChild(c) { this.children.push(c); return c; },
    replaceChildren(...cs) { this._text = ""; this.children = cs; },
    addEventListener() {},
    querySelector() { return null; },
    classList: { toggle() {}, add() {}, remove() {} },
  };
}
const byId = {};
const timers = [];
const sandbox = {
  console,
  Date, JSON, Math, Number, String, Boolean, Object, Array, Map, Set, Promise, Error,
  document: {
    hidden: false,
    getElementById(id) { return byId[id] || (byId[id] = node("#" + id)); },
    createElement: node,
    createElementNS: (ns, tag) => node(tag),
    createTextNode: (t) => ({ textContent: String(t) }),
    createDocumentFragment: () => node("#fragment"),
    addEventListener() {},
  },
  window: { location: { assign() {} }, addEventListener() {} },
  fetch: () => new Promise(() => {}),
  setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
  clearTimeout: () => {},
};
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appPath, "utf8"), ctx, { filename: "app.js" });
sandbox.fireConnectTimer = () => {
  const t = timers.filter((x) => x.ms === 4000).pop();   // STREAM_CONNECT_GRACE_MS
  if (t) t.fn();
};
for (const step of JSON.parse(scenarioJson)) {
  vm.runInContext(step, ctx);
  const chip = byId["live-status"];
  const overlay = byId["live-overlay"];
  const img = byId["live-img"];
  console.log(JSON.stringify({
    chip: chip.textContent, chipClass: chip.className,
    overlay: overlay.hidden ? null : overlay.textContent,
    attached: img.src.includes("/api/live.mjpg"),
  }));
}
"""


def status(**fields):
    base = {
        "reachable": True, "error": None, "enabled": True, "state": "streaming",
        "source": "basler", "camera_model": "acA2500-14gc", "profile": "ethernet",
        "fps_target": 5.0, "fps_measured": 0.0, "clients": 1, "frame_seq": 1,
        "last_frame_age_s": None, "stream_age_s": 0.5, "frames_incomplete": 0,
        "reconnects": 0, "last_error": None, "boot_id": "boot-1",
    }
    base.update(fields)
    return base


def run(*steps):
    """Each step: a status dict (-> applyLiveStatus) or raw JS source."""
    js = [s if isinstance(s, str) else f"applyLiveStatus({json.dumps(s)})" for s in steps]
    proc = subprocess.run(
        [NODE, "-e", HARNESS], capture_output=True, text=True, timeout=60,
        env={**os.environ, "ZANDER_APP_JS": str(APP_JS), "ZANDER_SCENARIO": json.dumps(js)},
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


FIREWALL = dict(stream_age_s=6.0, frames_incomplete=6, last_error="no frame within 1.0 s")


def test_live_only_once_a_frame_arrived():
    first, flowing = run(
        status(last_frame_age_s=None, stream_age_s=1.0),       # just started
        status(last_frame_age_s=0.2, fps_measured=4.9, stream_age_s=3.0),
    )
    assert first["chip"] == "starting" and "chip-live" not in first["chipClass"], first
    assert first["attached"], first
    assert flowing["chip"] == "LIVE · 4.9 fps" and "chip-live" in flowing["chipClass"], flowing


def test_streaming_without_frames_says_so_and_keeps_the_stream():
    (step,) = run(status(**FIREWALL))
    assert step["chip"] == "no frames" and "chip-warn" in step["chipClass"], step
    assert step["overlay"] and step["overlay"].startswith("No frames from the camera"), step
    assert "Windows firewall" in step["overlay"] and "no frame within 1.0 s" in step["overlay"]
    assert step["attached"], "the stream must stay attached so the first frame shows at once"


def test_no_frames_message_survives_the_first_mjpeg_part_and_the_connect_timer():
    # The first part of the relay is camerapi's "Waiting for camera" placeholder:
    # neither it nor the connect grace timer may hide the explanation.
    steps = run(status(**FIREWALL), "onLiveLoad()", "fireConnectTimer()")
    assert all(s["overlay"] and s["overlay"].startswith("No frames") for s in steps), steps


def test_no_frames_clears_when_frames_arrive():
    blocked, fixed = run(
        status(**FIREWALL),
        status(last_frame_age_s=0.3, fps_measured=5.0, stream_age_s=40.0, frames_incomplete=30),
    )
    assert blocked["chip"] == "no frames"
    assert fixed["chip"].startswith("LIVE") and fixed["overlay"] is None and fixed["attached"], fixed


def test_old_camerapi_without_stream_age_uses_growing_incomplete_frames():
    old = dict(stream_age_s=None, last_frame_age_s=None, last_error="no frame within 1.0 s")
    first, second, other_boot = run(
        status(frames_incomplete=3, **old),
        status(frames_incomplete=8, **old),
        status(frames_incomplete=9, boot_id="boot-2", **old),   # a camerapi restart
    )
    assert first["chip"] == "starting", first      # nothing to compare with yet
    assert second["chip"] == "no frames" and second["overlay"].startswith("No frames"), second
    assert other_boot["chip"] == "starting", other_boot


def test_a_stall_after_frames_still_detaches():
    (step,) = run(status(last_frame_age_s=12.0, stream_age_s=60.0))
    assert step["chip"] == "stalled" and not step["attached"], step
    assert step["overlay"].startswith("No new frames"), step


def test_unavailable_wifi_profile_is_unchanged():
    (step,) = run(status(state="unavailable", profile="wifi", stream_age_s=None))
    assert not step["attached"] and step["overlay"].startswith("Live preview unavailable"), step


if __name__ == "__main__":
    if NODE is None:
        print("SKIP: node is not installed")
        sys.exit(0)
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
