"""
Group photos that may show the same physical board(s) — the split unit.

Why: sessions contain re-shots of the same board. A split that puts re-shots
of one board into train AND test silently inflates every metric. Boards were
not reused across sessions (user-confirmed 2026-08-06), so every edge below is
within one session. Over-grouping is safe (the split just gets coarser);
under-grouping leaks.

Evidence available: capture time, and — the only board-UNIQUE signal — where
the labeled defects sit. Image similarity (pHash, NCC, DoG) was evaluated and
REJECTED in v1: every unit of a product has identical copper (23k false
"same board" edges). Edges (union-find), v2 (2026-09-28):

  (a) time edge — consecutive captures <= TIME_GAP_S apart (the v1 rule),
      EXCEPT between two labeled seated board-A/A2 photos whose defect sets
      clearly differ (< BOARD_CHANGE_MATCH of either photo's boxes has a
      same-class partner within PATTERN_TOL_PX): that is direct evidence the
      operator swapped boards. On Jul-20 the v1 rule chained 48 photos (10 min)
      into one "board"; the cut splits such runs at board changes. Handheld
      photos keep all time edges (hand pose moves the boxes — no evidence).
  (b) re-shot edge — two seated board-A/A2 photos of one session with the
      same defect set: >= 2 boxes each, >= PATTERN_MATCH of EACH photo's
      boxes matched, and at least one matched box at a NON-systematic
      position. Systematic positions (a box position present in >=
      SYSTEMATIC_FRAC of the session's positive photos — on Jul-20 the same
      two connector pads fail on most boards) cannot identify a board; a
      pattern built only from them links different boards. This edge catches
      re-shots after a pause (e.g. 95dd9cc6 13:50 <-> 32a04ceb 14:20, three
      rare defects in the same places), which the time rule misses and which
      split_v1 put into different buckets.
  (c) board B — all board-B photos of a session form ONE group: the defect
      constellations of 30 of the 33 Aug-17 photos chain into one panel under
      translation-tolerant matching; the rest are zoomed/empty shots of it.

Residual risk (documented, accepted): a re-shot whose labels differ by more
than half its boxes (annotator inconsistency) is treated as a board change.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from config.settings import settings

logger = logging.getLogger("ml.grouping")

TIME_GAP_S = 30.0
PATTERN_TOL_PX = 25.0
PATTERN_MATCH = 0.8
PATTERN_MIN_BOXES = 2
BOARD_CHANGE_MATCH = 0.5
SYSTEMATIC_FRAC = 0.25
GROUPS_FILE = "groups_v2.json"


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _ts(entry: dict) -> datetime:
    return datetime.fromisoformat(entry["captured_at"].replace("Z", "+00:00"))


def _centres(entry: dict) -> list[tuple[str, float, float]]:
    return [
        (name, (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for name, b in entry["boxes"]
    ]


def _near(p, q, tol: float = PATTERN_TOL_PX) -> bool:
    return p[0] == q[0] and abs(p[1] - q[1]) < tol and abs(p[2] - q[2]) < tol


def match_fractions(ca: list, cb: list) -> tuple[float, float]:
    """Fraction of a's boxes with a partner in b, and vice versa."""
    if not ca or not cb:
        return 0.0, 0.0
    fa = sum(1 for x in ca if any(_near(x, y) for y in cb)) / len(ca)
    fb = sum(1 for y in cb if any(_near(y, x) for x in ca)) / len(cb)
    return fa, fb


def _is_seated_a(e: dict) -> bool:
    return (
        e["board"] in ("A", "A2") and e["pose"] == "seated"
        and e["quarantine_reason"] is None and e["label_state"] == "annotated"
    )


def _systematic(positives: list[dict]) -> set[tuple[str, int]]:
    """(capture_id, box index) of boxes sitting at a systematic position."""
    out = set()
    cents = {e["capture_id"]: _centres(e) for e in positives}
    n = len(positives)
    for e in positives:
        for i, bx in enumerate(cents[e["capture_id"]]):
            freq = sum(
                1 for f in positives if any(_near(bx, q) for q in cents[f["capture_id"]])
            )
            if freq >= SYSTEMATIC_FRAC * n:
                out.add((e["capture_id"], i))
    return out


def is_reshot(a: dict, b: dict, systematic: set) -> bool:
    """Edge (b): same defect set incl. >= 1 matched box at a rare position."""
    ca, cb = _centres(a), _centres(b)
    if min(len(ca), len(cb)) < PATTERN_MIN_BOXES:
        return False
    fa, fb = match_fractions(ca, cb)
    if fa < PATTERN_MATCH or fb < PATTERN_MATCH:
        return False
    return any(
        (a["capture_id"], i) not in systematic and any(_near(x, y) for y in cb)
        for i, x in enumerate(ca)
    )


def is_board_change(a: dict, b: dict) -> bool:
    """Evidence two consecutive seated photos show different boards."""
    if not (_is_seated_a(a) and _is_seated_a(b)) or a["board"] != b["board"]:
        return False
    if not (a["boxes"] and b["boxes"]):
        return False  # an iO photo carries no board identity
    fa, fb = match_fractions(_centres(a), _centres(b))
    return min(fa, fb) < BOARD_CHANGE_MATCH


def build_groups(manifest: dict) -> dict:
    entries = sorted(manifest["entries"], key=lambda e: e["captured_at"])
    by_id = {e["capture_id"]: e for e in entries}
    uf = _UnionFind(list(by_id))
    edges = Counter()

    # (a) time edges, cut at evidenced board changes
    for prev, cur in zip(entries, entries[1:]):
        if prev["session"] != cur["session"]:
            continue
        if (_ts(cur) - _ts(prev)).total_seconds() > TIME_GAP_S:
            continue
        if is_board_change(prev, cur):
            edges["time_cut_board_change"] += 1
            continue
        edges["time"] += uf.union(prev["capture_id"], cur["capture_id"])

    by_session: dict[str, list[dict]] = {}
    for e in entries:
        by_session.setdefault(e["session"], []).append(e)
    for members in by_session.values():
        # (b) re-shot edges on seated board-A/A2 positives
        positives = [e for e in members if _is_seated_a(e) and e["boxes"]]
        systematic = _systematic(positives)
        for i, a in enumerate(positives):
            for b in positives[i + 1:]:
                if is_reshot(a, b, systematic):
                    edges["reshot"] += uf.union(a["capture_id"], b["capture_id"])
        # (c) board B: one panel per session
        board_b = [e for e in members if e["board"] == "B"]
        for e in board_b[1:]:
            edges["board_b"] += uf.union(board_b[0]["capture_id"], e["capture_id"])

    components: dict[str, list[str]] = {}
    for cid in by_id:
        components.setdefault(uf.find(cid), []).append(cid)

    groups = {}
    ordered = sorted(
        components.values(), key=lambda m: min(by_id[c]["captured_at"] for c in m)
    )
    for n, members in enumerate(ordered):
        members = sorted(members)
        es = [by_id[c] for c in members]
        sessions = {e["session"] for e in es}
        assert len(sessions) == 1, f"group spans sessions {sessions}"
        use = [e for e in es if e["label_state"] == "annotated"
               and e["quarantine_reason"] is None]
        box_counts = Counter(name for e in use for name, _ in e["boxes"])
        groups[f"g{n:04d}"] = {
            "captures": members,
            "session": es[0]["session"],
            "boards": sorted({e["board"] for e in es if e["board"]}),
            "poses": sorted({e["pose"] for e in es if e["pose"]}),
            "n_images": len(members),
            "n_usable": len(use),
            "n_nio": sum(1 for e in use if e["verdict"] == "NiO"),
            "n_io": sum(1 for e in use if e["verdict"] == "iO"),
            "n_io_seated": sum(
                1 for e in use if e["verdict"] == "iO" and e["pose"] == "seated"
            ),
            "n_nal_photos": sum(
                1 for e in use if "Nicht ausreichend Lot" in e["classes_present"]
            ),
            "box_counts": dict(sorted(box_counts.items())),
        }

    logger.info(f"{len(groups)} groups from {len(entries)} photos (edges: {dict(edges)})")
    return {
        "params": {
            "time_gap_s": TIME_GAP_S,
            "pattern_tol_px": PATTERN_TOL_PX,
            "pattern_match": PATTERN_MATCH,
            "pattern_min_boxes": PATTERN_MIN_BOXES,
            "board_change_match": BOARD_CHANGE_MATCH,
            "systematic_frac": SYSTEMATIC_FRAC,
            "board_b": "one group per session (single panel)",
            "phash": "rejected — see module docstring",
            "snapshot_id": manifest["snapshot_id"],
            "edges": dict(edges),
        },
        "groups": groups,
    }


def run() -> Path:
    from dataprep.ingest import load_manifest

    out = settings.datasets_dir / GROUPS_FILE
    split_path = settings.datasets_dir / f"{settings.split_id}.json"
    if split_path.exists() and out.exists():
        logger.info(f"{out.name} kept — {split_path.name} is frozen on it")
        return out
    groups = build_groups(load_manifest())
    out.write_text(json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"groups written: {out}")
    return out


def load_groups() -> dict:
    path = settings.datasets_dir / GROUPS_FILE
    if not path.exists():
        raise FileNotFoundError(f"run `python cli.py split` first — missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))
