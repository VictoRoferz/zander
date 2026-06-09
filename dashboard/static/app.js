// Dashboard front-end logic.
// Polls /api/stats, /api/unlabeled, /api/labeled every 3s and rebuilds the grids.
// Also wires the "Capture" button to POST /api/capture on the dashboard,
// which proxies to camerapi with the X-Triggered-By header.

const POLL_MS = 3000;

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function fmtTimestamp(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch (_) {
    return iso;
  }
}

function renderTile(item, kind) {
  const id = item.capture_id;
  const shortId = id.slice(0, 8);
  const tile = el("div", { class: "tile" });

  tile.appendChild(el("img", {
    src: `/thumb/${kind}/${id}.jpg`,
    alt: shortId,
    loading: "lazy",
  }));

  const meta = el("div", { class: "meta" });
  meta.appendChild(el("div", { class: "id", title: id }, shortId + "…"));
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
      const q = item.overall_quality.trim().toLowerCase();
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

function renderGrid(containerId, items, kind) {
  const c = document.getElementById(containerId);
  c.innerHTML = "";
  if (!items.length) {
    c.appendChild(el("div", { class: "empty" }, "Nothing yet."));
    return;
  }
  for (const item of items) c.appendChild(renderTile(item, kind));
}

async function refresh() {
  try {
    const [stats, unlabeled, labeled] = await Promise.all([
      fetch("/api/stats").then(r => r.json()),
      fetch("/api/unlabeled").then(r => r.json()),
      fetch("/api/labeled").then(r => r.json()),
    ]);
    document.getElementById("count-unlabeled").textContent = stats.unlabeled;
    document.getElementById("count-labeled").textContent = stats.labeled;
    document.getElementById("count-total").textContent = stats.total;
    renderGrid("grid-unlabeled", unlabeled, "unlabeled");
    renderGrid("grid-labeled", labeled, "labeled");
  } catch (e) {
    console.warn("refresh failed", e);
  }
}

async function triggerCapture() {
  const status = document.getElementById("capture-status");
  const btn = document.getElementById("capture-btn");
  btn.disabled = true;
  status.textContent = "Capturing…";
  try {
    const resp = await fetch("/api/capture", { method: "POST" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    status.textContent = `Captured ${data.capture_id?.slice(0, 8) ?? ""}…`;
    setTimeout(refresh, 200);
  } catch (e) {
    status.textContent = `Capture failed: ${e.message}`;
  } finally {
    btn.disabled = false;
    setTimeout(() => { status.textContent = ""; }, 6000);
  }
}

document.getElementById("capture-btn").addEventListener("click", triggerCapture);
refresh();
setInterval(refresh, POLL_MS);
