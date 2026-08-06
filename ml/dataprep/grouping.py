"""
Group near-duplicate photos of the same physical board(s).

Why: sessions contain bursts of re-shots of the same panel (June-15 alone has
256 hand-held shots of a few boards). A random split would put re-shots of
one board in both train and test and silently inflate every metric. Boards
were NOT reused across sessions (user-confirmed 2026-08-06), so grouping is
within-session only.

Group edge: time-burst — same session-day AND inter-capture gap <= TIME_GAP_S.
Re-shots of one board are necessarily taken back-to-back; a >30 s pause in
these sessions means the operator swapped boards (or stacks). A burst may
contain SEVERAL distinct boards — that direction only over-groups, which
cannot leak, it just makes the split slightly conservative.

Perceptual hashing was evaluated and deliberately REJECTED: all photos show
the same product in the same fixture, so pHash cannot distinguish different
physical boards — measured on this dataset it produced 23k false "same board"
edges (Hamming<=10, hash 8) and collapsed whole sessions into single groups,
which made a stratified split impossible. Time is the only trustworthy
board-boundary signal available.

Measured structure at 30 s (2026-08-06): 55 groups; positive groups per
session: 06-12: 2, 06-15: 2, 06-24: 5, 07-20: 6.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import settings

logger = logging.getLogger("ml.grouping")

TIME_GAP_S = 30.0


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(manifest: dict) -> dict:
    entries = sorted(manifest["entries"], key=lambda e: e["captured_at"])
    ids = [e["capture_id"] for e in entries]
    uf = _UnionFind(ids)

    n_time_edges = 0
    for prev, cur in zip(entries, entries[1:]):
        if prev["session"] != cur["session"]:
            continue
        t0 = datetime.fromisoformat(prev["captured_at"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(cur["captured_at"].replace("Z", "+00:00"))
        if (t1 - t0).total_seconds() <= TIME_GAP_S:
            uf.union(prev["capture_id"], cur["capture_id"])
            n_time_edges += 1

    components: dict[str, list[str]] = {}
    for cid in ids:
        components.setdefault(uf.find(cid), []).append(cid)
    by_entry = {e["capture_id"]: e for e in entries}
    groups = {}
    for n, root in enumerate(
        sorted(components, key=lambda r: min(by_entry[c]["captured_at"] for c in components[r]))
    ):
        members = sorted(components[root])
        groups[f"g{n:04d}"] = {
            "captures": members,
            "session": by_entry[members[0]]["session"],
            "n_images": len(members),
            "n_positive": sum(
                1 for c in members if by_entry[c]["label_state"] == "annotated_positive"
            ),
            "n_nal_boxes": sum(len(by_entry[c]["nal_boxes"]) for c in members),
        }

    logger.info(
        f"{len(groups)} groups from {len(entries)} images "
        f"({n_time_edges} time edges, gap<={TIME_GAP_S:.0f}s); "
        f"positive groups: {sum(1 for g in groups.values() if g['n_positive'])}"
    )
    return {
        "params": {"time_gap_s": TIME_GAP_S, "phash": "rejected — see module docstring"},
        "groups": groups,
    }


def run() -> Path:
    from dataprep.ingest import load_manifest

    groups = build_groups(load_manifest())
    out = settings.datasets_dir / "groups.json"
    out.write_text(json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"groups written: {out}")
    return out
