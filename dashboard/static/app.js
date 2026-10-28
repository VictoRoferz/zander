// Dashboard front-end.
//
// Three independent pollers (setTimeout chains that never overlap):
//   * grids:  /api/stats + /api/unlabeled + /api/labeled every 3 s (10 s while
//             the tab is hidden). A grid is rebuilt only when its data changed.
//   * live:   /api/live/status every 5 s (10 s hidden). #live-img shows the
//             login-protected MJPEG relay /api/live.mjpg only while camerapi
//             reports the preview enabled and reachable. #live-toggle turns
//             the preview on/off at runtime (POST /api/live/mode); photos
//             work either way. The stream is dropped
//             while the tab is hidden: it holds one of the browser's ~6
//             connections per host for as long as it runs. The chip says LIVE
//             only once camerapi has a frame; "streaming" without one after
//             NO_FRAME_GRACE_S shows "no frames" (the Windows-firewall case).
//   * latest: /api/latest every 3 s (10 s hidden), every 750 ms while the
//             prediction is pending, our own capture is on its way or a
//             re-check runs. The result panel is re-rendered only when
//             capture/state/prediction change.
//   * models: /api/models every 10 s (10 s hidden). The chooser switches the
//             inspector's active model for every following photo (PUT
//             /api/models/active); "Re-check with <model>" asks the receiver
//             to classify the shown capture again (POST .../recheck) and shows
//             up only when that capture's result comes from another model.
// The live and result panels live outside the grids, so the grid refresh
// never touches the stream or the result.
//
// Security: server data reaches the DOM only through textContent,
// setAttribute and createElementNS, never through innerHTML.
"use strict";

const GRID_POLL_MS = 3000;
const HIDDEN_POLL_MS = 10000;
const LIVE_STATUS_MS = 5000;
const LATEST_FAST_MS = 750;
const LATEST_SLOW_MS = 3000;
const AWAIT_MAX_MS = 90000;             // fast-poll for our own capture at most this long
const MODELS_POLL_MS = 10000;
const RECHECK_MAX_MS = 60000;           // fast-poll for a re-check result at most this long
const MAX_CROPS = 12;
const CROP_MIN_SIDE = 96;               // photo px
const CROP_BOX_FACTOR = 3;              // crop side = max(96, 3 x the box's longer side)
const STREAM_BACKOFF_MAX_MS = 30000;
const STREAM_CONNECT_GRACE_MS = 4000;   // not every browser fires "load" for MJPEG
const STALL_AGE_S = 10;                 // "streaming" but no new frame for this long
const NO_FRAME_GRACE_S = 5;             // "streaming" but still no first frame after this long
const SVG_NS = "http://www.w3.org/2000/svg";
const XLINK_NS = "http://www.w3.org/1999/xlink";
const BLANK_GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

// ---- DOM helpers ------------------------------------------------------------

function $(id) {
  return document.getElementById(id);
}

function appendChildren(parent, children) {
  for (const c of children) {
    if (c == null || c === false) continue;
    parent.appendChild(
      typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c,
    );
  }
}

// HTML element; attributes via setAttribute, text via text nodes (no innerHTML).
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, String(v));
  }
  appendChildren(e, children);
  return e;
}

// SVG element (createElementNS); same rules as el().
function svgEl(tag, attrs = {}, ...children) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    e.setAttribute(k, String(v));
  }
  appendChildren(e, children);
  return e;
}

function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

function shortId(id) {
  return typeof id === "string" ? id.slice(0, 8) : "";
}

function truncate(s, n) {
  const t = String(s);
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
}

function fmtTimestamp(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

function fmtScore(s) {
  return isNum(s) ? s.toFixed(2) : "?";
}

// ---- API + polling ------------------------------------------------------------

class AuthError extends Error {}

let redirecting = false;

function redirectToLogin() {
  if (redirecting) return;
  redirecting = true;
  window.location.assign("/login");
}

async function apiGet(url) {
  const resp = await fetch(url, { cache: "no-store", credentials: "same-origin" });
  if (resp.status === 401) {
    redirectToLogin();
    throw new AuthError("session expired");
  }
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  return resp.json();
}

// Runs task() now and then again after nextDelay(result) ms. run() while a
// round is in flight queues exactly one immediate follow-up round.
function makePoller(task, nextDelay) {
  let timer = null;
  let running = false;
  let rerun = false;
  async function run() {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (running) {
      rerun = true;
      return;
    }
    running = true;
    let result;
    try {
      result = await task();
    } catch (e) {
      if (!(e instanceof AuthError)) console.warn(e);
    } finally {
      running = false;
    }
    if (redirecting) return;
    const delay = rerun ? 0 : nextDelay(result);
    rerun = false;
    timer = setTimeout(run, delay);
  }
  return { run };
}

// ---- Grids (unlabeled / labeled) --------------------------------------------------

function renderTile(item, kind) {
  const id = String(item.capture_id || "");
  const tile = el("div", { class: "tile" });

  tile.appendChild(el("img", {
    src: `/thumb/${kind}/${encodeURIComponent(id)}.jpg`,
    alt: shortId(id),
    loading: "lazy",
  }));

  const meta = el("div", { class: "meta" });
  meta.appendChild(el("div", { class: "id", title: id }, shortId(id) + "…"));
  meta.appendChild(el("div", { class: "row" },
    el("span", { class: "muted" }, "by"),
    el("span", {}, item.triggered_by || "—"),
  ));
  meta.appendChild(el("div", { class: "row" },
    el("span", { class: "muted" }, "at"),
    el("span", {}, fmtTimestamp(item.captured_at)),
  ));
  if (kind === "labeled") {
    if (item.completed_by) {
      meta.appendChild(el("div", { class: "row" },
        el("span", { class: "muted" }, "labeled by"),
        el("span", {}, item.completed_by),
      ));
    }
    if (item.overall_quality) {
      // Label Studio "overall_quality" choices are German: iO / NiO /
      // Weitere Überprüfung notwendig. Map to the pass/fail/review badges;
      // show the raw German text.
      const q = String(item.overall_quality).trim().toLowerCase();
      const cls =
        q === "io" ? "badge badge-pass" :
        q === "nio" ? "badge badge-fail" :
        "badge badge-review";
      meta.appendChild(el("div", { class: "row" },
        el("span", { class: "muted" }, "quality"),
        el("span", { class: cls }, item.overall_quality),
      ));
    }
  }
  tile.appendChild(meta);
  return tile;
}

const gridSignatures = {};

function renderGrid(containerId, items, kind) {
  const list = Array.isArray(items) ? items : [];
  const signature = JSON.stringify(list);
  if (gridSignatures[containerId] === signature) return;   // unchanged: keep the DOM
  gridSignatures[containerId] = signature;
  const c = $(containerId);
  if (!list.length) {
    c.replaceChildren(el("div", { class: "empty" }, "Nothing yet."));
    return;
  }
  const frag = document.createDocumentFragment();
  for (const item of list) frag.appendChild(renderTile(item, kind));
  c.replaceChildren(frag);
}

async function refreshGrids() {
  const [stats, unlabeled, labeled] = await Promise.all([
    apiGet("/api/stats"),
    apiGet("/api/unlabeled"),
    apiGet("/api/labeled"),
  ]);
  $("count-unlabeled").textContent = String(stats.unlabeled ?? "–");
  $("count-labeled").textContent = String(stats.labeled ?? "–");
  $("count-total").textContent = String(stats.total ?? "–");
  renderGrid("grid-unlabeled", unlabeled, "unlabeled");
  renderGrid("grid-labeled", labeled, "labeled");
}

// ---- Live camera panel ---------------------------------------------------------------

// camerapi/dashboard states in which no stream is requested.
const NO_STREAM_STATES = new Set([
  "disabled", "unavailable", "stopped", "auth_error", "unsupported", "unreachable", "error",
]);

const live = {
  attached: false,     // #live-img points at the relay
  gotFrame: false,     // the current stream delivered its first frame
  usable: false,       // the last status allowed streaming
  bootId: null,        // camerapi process identity: a restart means reconnect
  failures: 0,         // consecutive stream errors (backoff exponent)
  retryAt: 0,          // no reattach before this time (ms since epoch)
  connectTimer: null,
  noFramesShown: false, // the overlay currently shows the no-frames message
  prevIncomplete: null, // {bootId, count} from the last status (old camerapi fallback)
  lastStatus: null,     // the last /api/live/status answer (the toggle needs it)
  switching: false,     // a live view switch is on its way
};

// Frames stopped after the stream had delivered some.
function isStalled(s) {
  return s.state === "streaming" && isNum(s.last_frame_age_s) && s.last_frame_age_s > STALL_AGE_S;
}

// camerapi streams, but not one frame arrived since the stream started. On
// the Windows PC that is the firewall dropping the camera's UDP stream (or a
// cable problem): camerapi then reports last_frame_age_s null and fps 0 while
// "streaming", which used to show a green LIVE chip.
function noFramesYet(s) {
  if (s.state !== "streaming" || isNum(s.last_frame_age_s)) return false;
  if (isNum(s.stream_age_s)) return s.stream_age_s > NO_FRAME_GRACE_S;
  // camerapi without stream_age_s: incomplete frames piling up says the same.
  const prev = live.prevIncomplete;
  return Boolean(prev && prev.bootId === s.boot_id && isNum(s.frames_incomplete)
    && s.frames_incomplete > prev.count);
}

function noFramesMessage(s) {
  return {
    title: "No frames from the camera",
    detail: "Check the camera cable and the Windows firewall rule (allow python.exe inbound, Private)."
      + (s.last_error ? ` ${s.last_error}` : ""),
  };
}

function setLiveOverlay(msg) {
  const overlay = $("live-overlay");
  if (!msg) {
    overlay.hidden = true;
    overlay.replaceChildren();
    return;
  }
  overlay.replaceChildren(el("div", { class: "live-overlay-box" },
    el("strong", {}, msg.title),
    msg.detail ? el("span", { class: "live-overlay-detail" }, truncate(msg.detail, 240)) : null,
  ));
  overlay.hidden = false;
}

// Why the stream is not shown, for the overlay.
function blockerMessage(s, stalled) {
  if (s.state === "dashboard_unreachable") {
    return { title: "Dashboard not reachable", detail: "Retrying…" };
  }
  if (!s.reachable) return { title: "Camera service offline", detail: s.error };
  switch (s.state) {
    case "auth_error":
      return { title: "Live preview refused", detail: "Set the same PREVIEW_TOKEN for camerapi and the dashboard." };
    case "unsupported":
      return { title: "No live preview", detail: "This camerapi version has no preview. Update camerapi." };
    case "unavailable":
      return { title: "Live preview unavailable", detail: s.last_error || "Not available with the Wi-Fi camera profile." };
    case "stopped":
      return { title: "Camera service is stopping" };
    case "error":
      return { title: "Live preview error", detail: s.error || s.last_error };
    default:
      break;
  }
  if (s.enabled !== true || s.state === "disabled") {
    return { title: "Live view is off", detail: "Photos still work. Turn it on with the button above." };
  }
  if (stalled) {
    return {
      title: "No new frames",
      detail: `Last frame ${Math.round(s.last_frame_age_s)} s ago.${s.last_error ? " " + s.last_error : ""}`,
    };
  }
  return { title: "Live preview unavailable", detail: s.error || s.state };
}

function renderLiveChip(s, stalled, noFrames) {
  let text;
  let cls = "chip chip-off";
  if (s.state === "dashboard_unreachable" || !s.reachable) {
    text = "offline";
    cls = "chip chip-error";
  } else if (noFrames) {
    text = "no frames";
    cls = "chip chip-warn";
  } else if (s.state === "streaming" && !stalled && isNum(s.last_frame_age_s)) {
    // LIVE only once a frame has actually arrived.
    text = isNum(s.fps_measured) && s.fps_measured > 0 ? `LIVE · ${s.fps_measured.toFixed(1)} fps` : "LIVE";
    cls = "chip chip-live";
  } else if (stalled) {
    text = "stalled";
    cls = "chip chip-warn";
  } else if (s.state === "reconnecting") {
    text = "reconnecting";
    cls = "chip chip-warn";
  } else if (s.state === "auth_error" || s.state === "error") {
    text = s.state === "auth_error" ? "auth error" : "error";
    cls = "chip chip-error";
  } else if (s.enabled !== true || s.state === "disabled") {
    text = "off";
  } else if (s.state === "starting" || s.state === "idle" || s.state === "streaming") {
    text = "starting";   // "streaming": waiting for the first frame (within the grace)
  } else {
    text = truncate(s.state || "unknown", 24);
  }
  const chip = $("live-status");
  chip.textContent = text;
  chip.className = cls;
}

function renderLiveDetail(s) {
  const parts = [];
  if (s.reachable) {
    if (s.camera_model) parts.push(s.camera_model);
    if (s.profile) parts.push(`${s.profile} profile`);
    if (s.source && s.source !== "basler") parts.push(`source: ${s.source}`);
    if (isNum(s.reconnects) && s.reconnects > 0) parts.push(`${s.reconnects} reconnects`);
    if (isNum(s.frames_incomplete) && s.frames_incomplete > 0) parts.push(`${s.frames_incomplete} incomplete frames`);
    if (s.state === "reconnecting" && s.last_error) parts.push(s.last_error);
  }
  $("live-detail").textContent = truncate(parts.join(" · "), 200);
}

function attachStream() {
  const img = $("live-img");
  live.attached = true;
  live.gotFrame = false;
  img.hidden = false;
  live.noFramesShown = false;
  setLiveOverlay({ title: "Connecting…" });
  clearTimeout(live.connectTimer);
  live.connectTimer = setTimeout(() => {
    if (live.attached && !live.noFramesShown) setLiveOverlay(null);
  }, STREAM_CONNECT_GRACE_MS);
  // A fresh URL forces a new request (and aborts the previous one).
  img.src = `/api/live.mjpg?t=${Date.now()}`;
}

function detachStream() {
  clearTimeout(live.connectTimer);
  if (!live.attached) return;
  live.attached = false;
  const img = $("live-img");
  img.src = BLANK_GIF;   // replacing the source aborts the MJPEG request
  img.hidden = true;
}

function onLiveLoad() {
  // Fires for the first MJPEG part (and, in some browsers, for every part):
  // act on the first one only.
  if (!live.attached || live.gotFrame || !$("live-img").src.includes("/api/live.mjpg")) return;
  live.gotFrame = true;
  live.failures = 0;
  clearTimeout(live.connectTimer);
  // (The first part may be camerapi's "Waiting for camera" placeholder: a
  // no-frames message stays until the status reports a real frame.)
  if (!live.noFramesShown) setLiveOverlay(null);
  livePoller.run();   // frames flow: refresh the chip now, not in up to 5 s
}

function onLiveError() {
  if (!live.attached) return;
  live.failures += 1;
  const delay = Math.min(STREAM_BACKOFF_MAX_MS, 1000 * 2 ** Math.min(live.failures, 5));
  live.retryAt = Date.now() + delay;
  detachStream();
  live.noFramesShown = false;
  setLiveOverlay({ title: "Stream interrupted", detail: `Retrying in ${Math.round(delay / 1000)} s…` });
  setTimeout(() => livePoller.run(), delay + 50);
}

// Shown once camerapi said whether the preview is on (not while it refuses us).
function renderLiveToggle() {
  const s = live.lastStatus;
  const btn = $("live-toggle");
  const known = Boolean(s && s.reachable) && typeof s.enabled === "boolean";
  btn.hidden = !known;
  btn.disabled = live.switching;
  if (!known) return;
  if (live.switching) btn.textContent = "Switching…";
  else btn.textContent = s.enabled ? "Turn live view off" : "Turn live view on";
}

async function onLiveToggle() {
  const s = live.lastStatus;
  if (live.switching || !s || typeof s.enabled !== "boolean") return;
  const want = !s.enabled;
  live.switching = true;
  renderLiveToggle();
  try {
    const answer = await postJson("/api/live/mode", "POST", { enabled: want });
    live.lastStatus = { ...answer, reachable: true };
  } catch (e) {
    if (e instanceof AuthError) return;
    // The status line, not the overlay: the next live poll redraws that at once.
    const statusEl = $("capture-status");
    statusEl.textContent = `Live view switch failed: ${e.message}`;
    clearTimeout(captureStatusTimer);
    captureStatusTimer = setTimeout(() => { statusEl.textContent = ""; }, 8000);
  } finally {
    live.switching = false;
    renderLiveToggle();
    livePoller.run();
  }
}

function applyLiveStatus(s) {
  live.lastStatus = s;
  renderLiveToggle();
  const stalled = isStalled(s);
  const noFrames = noFramesYet(s);   // before prevIncomplete is updated
  live.prevIncomplete = isNum(s.frames_incomplete) && s.boot_id
    ? { bootId: s.boot_id, count: s.frames_incomplete } : null;
  const usable = Boolean(s.reachable) && s.enabled === true && !NO_STREAM_STATES.has(s.state) && !stalled;
  const bootChanged = Boolean(s.boot_id && live.bootId && s.boot_id !== live.bootId);
  if (s.boot_id) live.bootId = s.boot_id;
  const recovered = usable && !live.usable;
  live.usable = usable;
  renderLiveChip(s, stalled, noFrames);
  renderLiveDetail(s);

  if (!usable) {
    live.noFramesShown = false;
    detachStream();
    setLiveOverlay(blockerMessage(s, stalled));
    return;
  }
  if (document.hidden) {
    detachStream();
    return;
  }
  if (bootChanged || recovered) {
    // camerapi restarted or came back: the old connection is dead, start over.
    live.failures = 0;
    live.retryAt = 0;
  }
  // Chromium fires "load" only for the first MJPEG part, so a stream that died
  // silently (e.g. the dashboard restarted between two polls) just freezes.
  // camerapi goes "idle" PREVIEW_IDLE_STOP_S after its last viewer left, so
  // "idle" while we hold a stream that already delivered frames = ours is gone.
  const orphaned = live.attached && live.gotFrame && s.state === "idle";
  if (Date.now() < live.retryAt) return;   // backing off after a stream error
  if (!live.attached || bootChanged || recovered || orphaned) attachStream();
  // No frame yet: unlike a stall, keep the stream attached (camerapi keeps
  // grabbing, and the first real frame shows the moment the cause is fixed)
  // and say why nothing moves.
  if (noFrames) {
    setLiveOverlay(noFramesMessage(s));
    live.noFramesShown = true;
  } else if (live.noFramesShown) {
    live.noFramesShown = false;
    setLiveOverlay(null);
  }
}

async function pollLiveStatus() {
  let s;
  try {
    s = await apiGet("/api/live/status");
  } catch (e) {
    if (e instanceof AuthError) throw e;
    s = { reachable: false, state: "dashboard_unreachable", error: null };
  }
  applyLiveStatus(s);
}

// ---- Latest result panel ---------------------------------------------------------------

const CLASS_KEYS = {
  "nicht ausreichend lot": "nal",
  "nal": "nal",
  "brücke": "bruecke",
  "bruecke": "bruecke",
  "brucke": "bruecke",
  "fahne": "fahne",
};

function classKey(label) {
  const k = String(label || "").normalize("NFC").trim().toLowerCase();
  return CLASS_KEYS[k] || "other";
}

let latestKey = null;
let awaiting = null;   // { id, since } after our own Capture click

function latestKeyOf(data) {
  const cap = (data && data.capture) || null;
  const pred = (data && data.prediction) || null;
  return JSON.stringify([
    cap && cap.capture_id, data && data.state, pred && pred.created_at,
    cap && cap.captured_at, cap && cap.width, cap && cap.height,
  ]);
}

function frameOf(cap) {
  if (!cap) return null;
  const W = cap.width;
  const H = cap.height;
  return isNum(W) && isNum(H) && W > 0 && H > 0 ? { W, H } : null;
}

// Valid detections, clamped to the photo, highest score first.
function detectionsOf(state, pred, frame) {
  if (state !== "ok" || !pred || !Array.isArray(pred.detections) || !frame) return [];
  const out = [];
  for (const d of pred.detections) {
    if (!d || typeof d !== "object") continue;
    if (![d.x1, d.y1, d.x2, d.y2].every(isNum)) continue;
    let x1 = clamp(Math.min(d.x1, d.x2), 0, frame.W);
    let x2 = clamp(Math.max(d.x1, d.x2), 0, frame.W);
    let y1 = clamp(Math.min(d.y1, d.y2), 0, frame.H);
    let y2 = clamp(Math.max(d.y1, d.y2), 0, frame.H);
    // Keep degenerate boxes visible (1 px) instead of silently dropping a defect.
    if (x2 - x1 < 1) { x2 = Math.min(frame.W, x1 + 1); x1 = Math.max(0, x2 - 1); }
    if (y2 - y1 < 1) { y2 = Math.min(frame.H, y1 + 1); y1 = Math.max(0, y2 - 1); }
    const label = typeof d.label === "string" && d.label ? d.label : "unknown";
    out.push({ x1, y1, x2, y2, label, key: classKey(label), score: isNum(d.score) ? d.score : null });
  }
  out.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
  return out;
}

function photoImage(url, W, H) {
  const img = svgEl("image", { x: 0, y: 0, width: W, height: H, preserveAspectRatio: "none" });
  img.setAttribute("href", url);
  img.setAttributeNS(XLINK_NS, "xlink:href", url);   // older WebKit
  return img;
}

function boxRect(d, cls) {
  return svgEl("rect", {
    class: cls,
    x: d.x1,
    y: d.y1,
    width: d.x2 - d.x1,
    height: d.y2 - d.y1,
    "vector-effect": "non-scaling-stroke",
  });
}

// One detection on the full photo: casing + class-coloured box (with a
// <title> tooltip), a halo around small boxes and a number matching its crop.
function detectionGroup(d, i, frame, unit) {
  const w = d.x2 - d.x1;
  const h = d.y2 - d.y1;
  const cx = (d.x1 + d.x2) / 2;
  const cy = (d.y1 + d.y2) / 2;
  const g = svgEl("g", { class: `det cls-${d.key}`, "data-index": i });
  const small = Math.max(w, h) < 4 * unit;
  const r = Math.max(w, h) / 2 + 2 * unit;
  if (small) {
    g.appendChild(svgEl("circle", { class: "det-halo", cx, cy, r, "vector-effect": "non-scaling-stroke" }));
  }
  g.appendChild(boxRect(d, "det-casing"));
  const box = boxRect(d, "det-box");
  box.appendChild(svgEl("title", {}, `#${i + 1} ${d.label} · ${fmtScore(d.score)}`));
  g.appendChild(box);
  if (i < MAX_CROPS) {
    const br = 1.3 * unit;
    const ax = small ? cx + r * 0.72 : d.x2;
    const ay = small ? cy - r * 0.72 : d.y1;
    const bx = clamp(ax, br, frame.W - br);
    const by = clamp(ay, br, frame.H - br);
    g.appendChild(svgEl("g", { class: "det-num" },
      svgEl("circle", { cx: bx, cy: by, r: br, "vector-effect": "non-scaling-stroke" }),
      svgEl("text", {
        x: bx, y: by, "font-size": 1.5 * unit, "text-anchor": "middle", "dominant-baseline": "central",
      }, String(i + 1)),
    ));
  }
  return g;
}

// Square crop window around a box: side max(96, 3 x box), clamped to the photo.
function cropView(d, frame) {
  const side = Math.min(
    Math.max(CROP_MIN_SIDE, CROP_BOX_FACTOR * Math.max(d.x2 - d.x1, d.y2 - d.y1)),
    frame.W,
    frame.H,
  );
  const cx = (d.x1 + d.x2) / 2;
  const cy = (d.y1 + d.y2) / 2;
  return {
    x: clamp(cx - side / 2, 0, frame.W - side),
    y: clamp(cy - side / 2, 0, frame.H - side),
    side,
  };
}

function highlightDetection(i, on) {
  const g = $("result-figure").querySelector(`g.det[data-index="${i}"]`);
  if (g) g.classList.toggle("is-hl", on);
}

// A zoomed crop: the same photo in a small SVG whose viewBox is the crop
// window, so the browser does the cropping (no server-side image work).
function cropFigure(url, frame, d, i) {
  const v = cropView(d, frame);
  const svg = svgEl("svg", {
    viewBox: `${v.x} ${v.y} ${v.side} ${v.side}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": `Defect ${i + 1}: ${d.label}, score ${fmtScore(d.score)}`,
  });
  svg.appendChild(photoImage(url, frame.W, frame.H));
  svg.appendChild(svgEl("g", { class: `det cls-${d.key}` }, boxRect(d, "det-casing"), boxRect(d, "det-box")));
  const w = d.x2 - d.x1;
  const h = d.y2 - d.y1;
  const fig = el("figure", { class: `crop cls-${d.key}`, tabindex: "0" },
    svg,
    el("figcaption", {},
      el("b", {}, `#${i + 1} ${d.label}`),
      el("span", {}, `score ${fmtScore(d.score)}`),
      el("span", {}, `${Math.round(w)} × ${Math.round(h)} px at (${Math.round((d.x1 + d.x2) / 2)}, ${Math.round((d.y1 + d.y2) / 2)})`),
    ),
  );
  fig.addEventListener("mouseenter", () => highlightDetection(i, true));
  fig.addEventListener("mouseleave", () => highlightDetection(i, false));
  fig.addEventListener("focus", () => highlightDetection(i, true));
  fig.addEventListener("blur", () => highlightDetection(i, false));
  return fig;
}

function renderVerdict(state, pred) {
  let text;
  let cls;
  if (state === "ok") {
    const v = pred && pred.verdict;
    if (v === "NiO") { text = "NiO"; cls = "nio"; }
    else if (v === "iO") { text = "iO"; cls = "io"; }
    else { text = "?"; cls = "error"; }
  } else if (state === "pending") { text = "pending"; cls = "pending"; }
  else if (state === "no_model") { text = "no model"; cls = "no-model"; }
  else if (state === "error") { text = "error"; cls = "error"; }
  else { text = "none"; cls = "none"; }
  const badge = $("result-verdict");
  badge.textContent = text;
  badge.className = `verdict verdict-${cls}`;
}

function renderResultChips(cap, state, pred, dets) {
  const chips = [];
  if (state === "ok" && pred) {
    const counts = new Map();
    for (const d of dets) counts.set(d.label, (counts.get(d.label) || 0) + 1);
    const order = [];
    for (const c of Array.isArray(pred.classes) ? pred.classes : []) {
      if (typeof c === "string" && !order.includes(c)) order.push(c);
    }
    for (const label of counts.keys()) if (!order.includes(label)) order.push(label);
    for (const label of order) {
      const n = counts.get(label) || 0;
      chips.push(el("span", { class: `chip cls-${classKey(label)}` }, n ? `${label} · ${n}` : label));
    }
  }
  if (cap && cap.source === "fallback") {
    chips.push(el("span", {
      class: "chip chip-warn",
      title: "The camera failed, so camerapi stored its fallback sample image.",
    }, "Fallback image, not a camera photo"));
  }
  $("result-classes").replaceChildren(...chips);
}

function renderCaptureLine(cap) {
  const parts = [];
  if (cap) {
    parts.push(`capture ${shortId(cap.capture_id)}`);
    if (cap.captured_at) parts.push(fmtTimestamp(cap.captured_at));
    if (cap.triggered_by) parts.push(`by ${cap.triggered_by}`);
  }
  const line = $("result-capture");
  line.textContent = parts.join(" · ");
  line.title = cap ? String(cap.capture_id) : "";
}

// Message for state "error"; the details come from the stored document.
function errorText(pred) {
  if (!pred) return "The prediction file is unreadable (see the dashboard log).";
  const v = pred.schema_version;
  if (isNum(v) && v > 1) return `Unsupported prediction format (schema_version ${v}). Update the dashboard.`;
  if (v !== 1) return "Invalid prediction file (no valid schema_version).";
  if (!["ok", "no_model", "error"].includes(pred.status)) {
    return `Invalid prediction status: ${truncate(pred.status, 40)}`;
  }
  return pred.error ? `Inspection failed: ${truncate(pred.error, 500)}` : "Inspection failed.";
}

// The receiver's texts can already say "no model loaded: <reason>" (the
// inspector's 503 detail); don't say it twice.
function noModelText(pred) {
  const reason = pred && typeof pred.error === "string" ? pred.error.trim() : "";
  if (!reason) return "No model loaded.";
  if (/^(http \d+: )?no model loaded/i.test(reason)) {
    return truncate(reason.charAt(0).toUpperCase() + reason.slice(1), 500);
  }
  return `No model loaded: ${truncate(reason, 500)}`;
}

function renderMessage(cap, state, pred, frame) {
  let text = "";
  let isError = false;
  if (!cap) {
    text = "No captures yet. Press Capture or the F13 key.";
  } else if (state === "pending") {
    text = "Classifying…";
  } else if (state === "no_model") {
    text = noModelText(pred);
  } else if (state === "error") {
    text = errorText(pred);
    isError = true;
  } else if (state === "none") {
    text = "No prediction for this capture.";
  } else if (state === "ok") {
    const raw = Array.isArray(pred && pred.detections) ? pred.detections.length : 0;
    if (!frame && raw) {
      text = "Boxes not drawn: the photo size is unknown.";
      isError = true;
    } else if (pred && pred.verdict === "iO") {
      text = "No defect found above the threshold.";
    } else if (raw) {
      text = `${raw} defect${raw === 1 ? "" : "s"} marked on the photo.`;
    }
  }
  const m = $("result-message");
  m.textContent = text;
  m.classList.toggle("is-error", isError);
}

// The photo URL from the API (a same-origin path), or null.
function imageUrlOf(cap) {
  return cap && typeof cap.image_url === "string" && cap.image_url.startsWith("/") ? cap.image_url : null;
}

function renderFigure(cap, url, frame, dets) {
  const fig = $("result-figure");
  if (!cap || !url) {
    fig.replaceChildren(el("div", { class: "result-empty" }, "No capture to show."));
    return;
  }
  const alt = `Capture ${shortId(cap.capture_id)}`;
  if (!frame) {
    fig.replaceChildren(el("img", { src: url, alt }));
    return;
  }
  const { W, H } = frame;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": dets.length ? `${alt}, ${dets.length} marked defect(s)` : alt,
  });
  svg.style.aspectRatio = `${W} / ${H}`;
  svg.appendChild(photoImage(url, W, H));
  const unit = Math.max(W, H) / 100;   // 1 % of the long side, for markers and labels
  dets.forEach((d, i) => svg.appendChild(detectionGroup(d, i, frame, unit)));
  fig.replaceChildren(svg);
}

function renderCrops(url, frame, dets) {
  const box = $("result-crops");
  if (!url || !frame || !dets.length) {
    box.replaceChildren();
    return;
  }
  const frag = document.createDocumentFragment();
  dets.slice(0, MAX_CROPS).forEach((d, i) => frag.appendChild(cropFigure(url, frame, d, i)));
  if (dets.length > MAX_CROPS) {
    frag.appendChild(el("div", { class: "crop-more" }, `+${dets.length - MAX_CROPS} more`));
  }
  box.replaceChildren(frag);
}

function renderMeta(pred) {
  const parts = [];
  if (pred && pred.schema_version === 1) {
    if (pred.model_version) parts.push(`model ${pred.model_version}`);
    if (isNum(pred.threshold)) {
      parts.push(`t=${pred.threshold.toFixed(3)}${pred.threshold_source ? ` (${pred.threshold_source})` : ""}`);
    }
    if (isNum(pred.latency_ms)) parts.push(`${Math.round(pred.latency_ms)} ms`);
    if (pred.tiler_version) parts.push(`tiler ${pred.tiler_version}`);
  }
  $("result-meta").textContent = parts.join(" · ");
}

function renderResult(data) {
  const cap = (data && data.capture) || null;
  const state = (data && data.state) || "none";
  const pred = (data && data.prediction) || null;
  const url = imageUrlOf(cap);
  const frame = frameOf(cap);
  const dets = detectionsOf(state, pred, frame);
  renderVerdict(state, pred);
  renderResultChips(cap, state, pred, dets);
  renderCaptureLine(cap);
  renderMessage(cap, state, pred, frame);
  renderFigure(cap, url, frame, dets);
  renderCrops(url, frame, dets);
  renderMeta(pred);
}

function renderAwaitingNote(data) {
  const cap = data && data.capture;
  const waiting = awaiting
    && Date.now() - awaiting.since <= AWAIT_MAX_MS
    && !(cap && cap.capture_id === awaiting.id);
  $("result-note").textContent = waiting ? `waiting for capture ${shortId(awaiting.id)}…` : "";
}

async function pollLatest() {
  const data = await apiGet("/api/latest");
  latestData = data;
  const key = latestKeyOf(data);
  if (key !== latestKey) {
    latestKey = key;
    renderResult(data);
  }
  renderAwaitingNote(data);
  updateRechecking(data);
  renderRecheck();
  return data;
}

function latestDelay(data) {
  if (awaiting) {
    const cap = data && data.capture;
    const done = cap && cap.capture_id === awaiting.id && data.state !== "pending";
    if (done || Date.now() - awaiting.since > AWAIT_MAX_MS) awaiting = null;
  }
  if (document.hidden) return HIDDEN_POLL_MS;
  if (awaiting || rechecking || (data && data.state === "pending")) return LATEST_FAST_MS;
  return LATEST_SLOW_MS;
}

// ---- Model chooser -----------------------------------------------------------------

let models = null;        // the last /api/models answer
let modelsKey = null;
let latestData = null;    // the last /api/latest answer (the re-check button needs it)
let rechecking = null;    // { id, version, name, since } after a Re-check click
let modelNoteTimer = null;

function modelList(data) {
  return data && Array.isArray(data.models)
    ? data.models.filter((m) => m && typeof m.name === "string")
    : [];
}

function activeModel() {
  return modelList(models).find((m) => m.active && m.loaded) || null;
}

function modelLabel(m) {
  return m.run_name && m.run_name !== m.name ? `${m.name} (${m.run_name})` : m.name;
}

function setModelNote(text, clearAfterMs) {
  $("model-note").textContent = text;
  clearTimeout(modelNoteTimer);
  if (clearAfterMs) modelNoteTimer = setTimeout(() => { $("model-note").textContent = ""; }, clearAfterMs);
}

function renderModels(data) {
  const list = modelList(data);
  const options = [];
  if (!list.length) {
    let text = "no model";
    if (data && data.enabled === false) text = "predictions off";
    else if (data && data.reachable === false) text = "inspector unreachable";
    options.push(el("option", { value: "" }, text));
  }
  for (const m of list) {
    const tip = [];
    if (m.model_version) tip.push(String(m.model_version));
    if (isNum(m.threshold)) tip.push(`t=${m.threshold.toFixed(3)}`);
    if (m.error) tip.push(truncate(m.error, 200));
    options.push(el("option", {
      value: m.name,
      title: tip.join(" · "),
      disabled: m.loaded ? null : "",
      selected: m.active ? "" : null,
    }, m.loaded ? modelLabel(m) : `${modelLabel(m)} (cannot load)`));
  }
  const select = $("model-select");
  select.replaceChildren(...options);
  select.disabled = !list.some((m) => m.loaded);
  select.title = data && data.error
    ? String(data.error)
    : "The model that checks every new photo (for everyone using this station)";
}

function applyModels(data) {
  models = data;
  const key = JSON.stringify([
    data && data.enabled, data && data.reachable, data && data.error,
    modelList(data).map((m) => [m.name, m.loaded, m.active, m.model_version, m.error]),
  ]);
  if (key !== modelsKey) {
    modelsKey = key;
    renderModels(data);
  }
  renderRecheck();
}

async function pollModels() {
  const data = await apiGet("/api/models");
  applyModels(data);
  return data;
}

// Shown when the latest capture has no result from the active model yet.
function renderRecheck() {
  const btn = $("recheck-btn");
  const cap = latestData && latestData.capture;
  const pred = latestData && latestData.prediction;
  const state = latestData && latestData.state;
  const active = activeModel();
  const show = Boolean(cap && active && state !== "pending"
    && !(state === "ok" && pred && pred.model_version === active.model_version));
  btn.hidden = !show;
  btn.disabled = Boolean(rechecking);
  if (show) btn.textContent = `Re-check with ${active.name}`;
}

function updateRechecking(data) {
  if (!rechecking) return;
  const cap = data && data.capture;
  const pred = data && data.prediction;
  const name = rechecking.name;
  if (!cap || cap.capture_id !== rechecking.id) {
    rechecking = null;   // a newer capture took its place
    setModelNote("");
  } else if (pred && pred.model_version === rechecking.version) {
    rechecking = null;
    setModelNote(`Re-checked with ${name}.`, 5000);
  } else if (Date.now() - rechecking.since > RECHECK_MAX_MS) {
    rechecking = null;
    setModelNote(`No result from ${name} yet (see the receiver log).`, 8000);
  }
}

async function postJson(url, method, body) {
  const resp = await fetch(url, {
    method,
    credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (resp.status === 401) {
    redirectToLogin();
    throw new AuthError("session expired");
  }
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    data = null;
  }
  if (!resp.ok) throw new Error(errorDetail(data) || `HTTP ${resp.status}`);
  return data;
}

async function onModelChange() {
  const select = $("model-select");
  const name = select.value;
  if (!name) return;
  select.disabled = true;
  setModelNote(`Switching to ${name}…`);
  try {
    const data = await postJson("/api/models/active", "PUT", { name });
    modelsKey = null;   // render the answer even if it looks unchanged
    applyModels(data);
    const unsaved = data && data.saved === false ? " Not saved: it resets when the inspector restarts." : "";
    setModelNote(`New photos are checked by ${name}.${unsaved}`, 8000);
  } catch (e) {
    if (e instanceof AuthError) return;
    setModelNote(`Switch failed: ${e.message}`, 8000);
    modelsKey = null;   // put the real active model back into the chooser
    applyModels(models);
    modelsPoller.run();
  }
}

async function onRecheck() {
  const cap = latestData && latestData.capture;
  const active = activeModel();
  if (!cap || !active) return;
  rechecking = { id: cap.capture_id, version: active.model_version, name: active.name, since: Date.now() };
  renderRecheck();
  try {
    const data = await postJson(`/api/captures/${encodeURIComponent(cap.capture_id)}/recheck`, "POST");
    setModelNote(data && data.queued === false
      ? `Already queued, or predictions are off (waiting for ${active.name}…)`
      : `Re-checking with ${active.name}…`);
    latestPoller.run();
  } catch (e) {
    rechecking = null;
    if (!(e instanceof AuthError)) setModelNote(`Re-check failed: ${e.message}`, 8000);
    renderRecheck();
  }
}

// ---- Capture button --------------------------------------------------------------------

let captureStatusTimer = null;

function errorDetail(data) {
  if (!data) return null;
  const d = data.detail ?? data.error;
  if (d == null) return null;
  return truncate(typeof d === "string" ? d : JSON.stringify(d), 200);
}

async function triggerCapture() {
  const statusEl = $("capture-status");
  const buttons = [$("capture-btn"), $("live-capture-btn")].filter(Boolean);
  for (const b of buttons) b.disabled = true;
  statusEl.textContent = "Capturing…";
  try {
    const resp = await fetch("/api/capture", { method: "POST", credentials: "same-origin" });
    if (resp.status === 401) {
      redirectToLogin();
      return;
    }
    let data = null;
    try {
      data = await resp.json();
    } catch (_) {
      data = null;
    }
    if (!resp.ok) throw new Error(errorDetail(data) || `HTTP ${resp.status}`);
    const id = data && typeof data.capture_id === "string" ? data.capture_id : null;
    if (id) {
      // Poll fast until this capture has its result (or 90 s have passed).
      awaiting = { id, since: Date.now() };
      latestPoller.run();
    }
    const source = data && ((data.metadata && data.metadata.source) || data.source);
    const fallback = source === "fallback" ? " (fallback image, the camera was not used)" : "";
    statusEl.textContent = id ? `Captured ${shortId(id)}…${fallback}` : "Captured.";
    setTimeout(() => gridPoller.run(), 200);
  } catch (e) {
    statusEl.textContent = `Capture failed: ${e.message}`;
  } finally {
    for (const b of buttons) b.disabled = false;
    clearTimeout(captureStatusTimer);
    captureStatusTimer = setTimeout(() => { statusEl.textContent = ""; }, 6000);
  }
}

// ---- Wiring --------------------------------------------------------------------------------

const gridPoller = makePoller(refreshGrids, () => (document.hidden ? HIDDEN_POLL_MS : GRID_POLL_MS));
const livePoller = makePoller(pollLiveStatus, () => (document.hidden ? HIDDEN_POLL_MS : LIVE_STATUS_MS));
const latestPoller = makePoller(pollLatest, latestDelay);
const modelsPoller = makePoller(pollModels, () => (document.hidden ? HIDDEN_POLL_MS : MODELS_POLL_MS));

function onVisibilityChange() {
  if (document.hidden) {
    detachStream();
    return;
  }
  // Back in front: refresh now instead of waiting for the slow hidden-tab timers.
  livePoller.run();
  latestPoller.run();
  gridPoller.run();
  modelsPoller.run();
}

function init() {
  $("capture-btn").addEventListener("click", triggerCapture);
  const liveCaptureBtn = $("live-capture-btn");
  if (liveCaptureBtn) liveCaptureBtn.addEventListener("click", triggerCapture);
  $("live-toggle").addEventListener("click", onLiveToggle);
  const img = $("live-img");
  img.addEventListener("load", onLiveLoad);
  img.addEventListener("error", onLiveError);
  document.addEventListener("visibilitychange", onVisibilityChange);
  window.addEventListener("pagehide", detachStream);
  $("model-select").addEventListener("change", onModelChange);
  $("recheck-btn").addEventListener("click", onRecheck);
  gridPoller.run();
  livePoller.run();
  latestPoller.run();
  modelsPoller.run();
}

init();
