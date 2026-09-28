"""
Windows cache: one 160-px window per candidate, cut ONCE from the decoded
full-resolution photo with infer.verifier.extract_window (the same function
the inspector-side Verifier uses), so training and inference crops can never
diverge.

    python verifier/cli.py crops --run-name v1

Writes ml_data/verifier/<vrun>/windows.npy (uint8 memmap [N,160,160,3], row i =
candidate id i) and windows_index.json, which pins the sha256 of the
candidates.json it was built from — every consumer checks it.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataprep.ingest import photo_path
from infer.pipeline import load_photo_rgb
from infer.verifier import WINDOW, extract_window
from verifier.common import WINDOWS, WINDOWS_INDEX, read_json, utc_now, vrun_dir, write_json_atomic
from verifier.mine import load_candidates

logger = logging.getLogger("ml.verifier.crops")


def build(vrun: str) -> Path:
    doc, cand_sha = load_candidates(vrun)
    cands = doc["candidates"]
    n = len(cands)
    d = vrun_dir(vrun)
    out, tmp = d / WINDOWS, d / f".{WINDOWS}.tmp"
    by_photo: dict[str, list[int]] = defaultdict(list)
    for c in cands:
        by_photo[c["capture_id"]].append(c["id"])
    if [c["id"] for c in cands] != list(range(n)):
        raise ValueError("candidate ids must be dense 0..n-1 (rerun mine)")

    t0 = time.perf_counter()
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, WINDOW, WINDOW, 3))
    try:
        for k, ph in enumerate(doc["photos"], 1):
            ids = by_photo.get(ph["capture_id"], [])
            if not ids:
                continue
            if not ph.get("image_path"):
                raise ValueError(f"photo {ph['capture_id']} has no image_path in the manifest")
            rgb = load_photo_rgb(photo_path({"image_path": ph["image_path"]}))
            if rgb.shape[:2] != (ph["height"], ph["width"]):
                raise ValueError(
                    f"{ph['capture_id']}: image is {rgb.shape[1]}x{rgb.shape[0]}, the eval run "
                    f"says {ph['width']}x{ph['height']}"
                )
            for i in ids:
                mm[i] = extract_window(rgb, cands[i]["box"])
            if k % 25 == 0 or k == len(doc["photos"]):
                logger.info(f"windows: {k}/{len(doc['photos'])} photos")
        mm.flush()
    finally:
        del mm
    os.replace(tmp, out)
    index = {
        "candidates_sha256": cand_sha, "n": n, "shape": [n, WINDOW, WINDOW, 3],
        "dtype": "uint8", "photos": len(doc["photos"]), "created": utc_now(),
        "seconds": round(time.perf_counter() - t0, 1), "smoke": doc["smoke"],
    }
    write_json_atomic(d / WINDOWS_INDEX, index)
    logger.info(f"{n} windows ({out.stat().st_size / 1e6:.0f} MB) in {index['seconds']} s -> {out}")
    return out


def load_windows(vrun: str, cand_sha: str) -> np.ndarray:
    """Read-only memmap of the windows, checked against the candidates sha."""
    d = vrun_dir(vrun)
    idx_path = d / WINDOWS_INDEX
    if not idx_path.exists():
        raise FileNotFoundError(f"run `verifier/cli.py crops --run-name {vrun}` first — missing {idx_path}")
    index = read_json(idx_path)
    if index["candidates_sha256"] != cand_sha:
        raise RuntimeError(
            "windows.npy was built from a different candidates.json — rerun `crops`"
        )
    mm = np.load(d / WINDOWS, mmap_mode="r")
    if tuple(mm.shape) != tuple(index["shape"]):
        raise RuntimeError(f"windows.npy shape {mm.shape} != index {index['shape']}")
    return mm
