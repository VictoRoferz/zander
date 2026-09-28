"""
Photo attributes: the committed, anonymised per-photo table every v2 step
joins on (ml/datasets/photo_attributes_v1.csv).

Source: the 2026-09-28 photo audit (ml_data/audit/photo_audit_2026-09-28.csv,
gitignored) — a per-photo visual review of all 566 captures plus image
statistics. It holds things the capture sidecars cannot tell us: what the
photo shows (PCB / empty fixture / placeholder / ...), which board type, the
pose (seated in the fixture vs handheld) and the lighting setup.

Anonymised on purpose — the repo is public:
  - board part numbers are reduced to the letters A / A2 / B
  - `triggered_by` (operator e-mails) and all other audit columns are dropped
  - only a few colour statistics are kept (for the augmentation calibration)

Contract: ingest refuses a photo that has no row here. A new capture session
therefore needs a (quick) audit pass before it can enter the dataset — which
is exactly when a new board type, pose or junk frame should be noticed.

    python cli.py attributes     # (re)build the committed CSV from the audit
"""
from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger("ml.attributes")

COLUMNS = [
    "capture_id", "session", "content", "board", "pose", "lighting",
    "board_L", "board_a", "board_b", "luma_mean", "note",
]

# Audit wording -> stable slug. Unknown wording raises: a new content class is
# a decision (usable or not?), never a silent default.
CONTENT = {
    "PCB": "pcb",
    "empty fixture / test pin / hands (no PCB)": "no_pcb",
    "empty fixture / hands (no PCB)": "no_pcb",
    "PCB setup shot (single board, off-centre)": "setup_shot",
    "placeholder 'TEST IMAGE' (fallback)": "placeholder",
    "dark frame (empty fixture)": "dark_frame",
    "dark frame (board present)": "dark_frame",
    "saturated white frame": "saturated",
}
BOARD = {"": "", "A": "A", "A2 (E50-045-09)": "A2", "B (EN04 4-up panel)": "B"}
LIGHTING = {
    "": "",
    "white panel": "white_panel",
    "magenta": "magenta",
    "magenta (overexposed)": "magenta_overexposed",
    "dim/green": "dim_green",
    "white/green": "white_green",
    "white/teal": "white_teal",
}
POSE = {"": "", "seated": "seated", "handheld": "handheld"}


def _num(v: str) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if x != x else f"{x:.2f}"  # NaN -> empty


def build_rows(audit_csv: Path) -> list[dict]:
    rows = []
    with open(audit_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for field, table in (
                ("content", CONTENT), ("board", BOARD),
                ("lighting", LIGHTING), ("pose", POSE),
            ):
                if r[field] not in table:
                    raise ValueError(
                        f"[{r['capture_id']}] unknown audit {field} {r[field]!r} "
                        "— extend the mapping in dataprep/attributes.py deliberately"
                    )
            rows.append(
                {
                    "capture_id": r["capture_id"],
                    "session": r["session"],
                    "content": CONTENT[r["content"]],
                    "board": BOARD[r["board"]],
                    "pose": POSE[r["pose"]],
                    "lighting": LIGHTING[r["lighting"]],
                    "board_L": _num(r["board_L"]),
                    "board_a": _num(r["board_a"]),
                    "board_b": _num(r["board_b"]),
                    "luma_mean": _num(r["luma_mean"]),
                    "note": r["note"],
                }
            )
    rows.sort(key=lambda r: r["capture_id"])
    ids = [r["capture_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate capture_id in the audit")
    for r in rows:
        if r["content"] == "pcb" and not (r["board"] and r["pose"]):
            raise ValueError(f"[{r['capture_id']}] PCB photo without board/pose")
    return rows


def run() -> Path:
    rows = build_rows(settings.audit_csv)
    out = settings.attributes_csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"attributes written: {out} ({len(rows)} photos)")
    return out


def load_attributes() -> dict[str, dict]:
    path = settings.attributes_csv
    if not path.exists():
        raise FileNotFoundError(f"run `python cli.py attributes` first — missing {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["capture_id"]: r for r in csv.DictReader(f)}


def attributes_hash() -> str:
    """Content hash of the committed table (part of the dataset fingerprint)."""
    return hashlib.sha256(settings.attributes_csv.read_bytes()).hexdigest()
