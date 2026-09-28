"""
detector.json — the card that ships next to an exported detector ONNX.

ONE module for both sides of the contract (docs/CONTRACTS.md, "detector.json
v1"): `cli.py export` writes the card with `write_detector_card`, the
inspector reads it with `load_detector_card`. Both run the same validator, so
a card the writer accepts is exactly a card the reader accepts.

Stdlib only (plus infer.tiling, itself stdlib at import time): the inspector
image ships ml/infer without the training stack.

The card is authoritative for adapter, class_names, threshold and conf_floor.
The decision it encodes is infer/verdict.py's rule: NiO iff any detection
score >= threshold; classes = those with >= 1 such detection.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from infer.tiling import STRIDE, TILE_SIZE, TILER_VERSION

CONTRACT_VERSION = 1
KIND = "zander-detector"
# Mirror of infer.onnx_adapters.ADAPTERS keys (kept here so this module stays
# importable without onnxruntime; tests/test_detector_card.py checks parity).
KNOWN_ADAPTERS = ("ultralytics", "deim")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DetectorCardError(ValueError):
    """The card is missing, malformed, or does not match its model file."""


@dataclass(frozen=True)
class DetectorCard:
    model_path: Path
    model_sha256: str
    adapter: str
    schema: Optional[str]
    class_names: tuple[str, ...]
    threshold: float
    conf_floor: float
    tiler_version: str
    verifier: Optional[dict]
    raw: dict


def card_path_for(model_path: Path) -> Path:
    """detector.onnx -> detector.json (same folder, same stem)."""
    return Path(model_path).with_suffix(".json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_card(
    doc: Any,
    model_path: Optional[Path] = None,
    verify_sha: bool = True,
) -> None:
    """Raise DetectorCardError listing every problem found (none → returns).

    With `model_path`, also checks that the card names that file and — when
    `verify_sha` — that the file's sha256 matches `model_sha256`.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        raise DetectorCardError("card is not a JSON object")

    if doc.get("contract_version") != CONTRACT_VERSION:
        problems.append(
            f"contract_version must be {CONTRACT_VERSION}, got {doc.get('contract_version')!r}"
        )
    if doc.get("kind") != KIND:
        problems.append(f"kind must be {KIND!r}, got {doc.get('kind')!r}")

    model_file = doc.get("model_file")
    if not isinstance(model_file, str) or not model_file or "/" in model_file or "\\" in model_file:
        problems.append("model_file must be a bare file name")

    sha = doc.get("model_sha256")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        problems.append("model_sha256 must be 64 lowercase hex characters")

    if doc.get("adapter") not in KNOWN_ADAPTERS:
        problems.append(f"adapter must be one of {list(KNOWN_ADAPTERS)}, got {doc.get('adapter')!r}")

    names = doc.get("class_names")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        problems.append("class_names must be a non-empty list of unique, non-empty strings")

    threshold = doc.get("threshold")
    if not _is_number(threshold) or not (0.0 < float(threshold) < 1.0):
        problems.append(f"threshold must be a number in (0, 1), got {threshold!r}")

    floor = doc.get("conf_floor")
    if not _is_number(floor) or not (0.0 < float(floor) < 1.0):
        problems.append(f"conf_floor must be a number in (0, 1), got {floor!r}")

    verifier = doc.get("verifier")
    if verifier is not None and not isinstance(verifier, dict):
        problems.append("verifier must be null or an object")
    if _is_number(threshold) and _is_number(floor):
        if verifier is None and float(floor) > float(threshold):
            problems.append("conf_floor must be <= threshold when there is no verifier")
        if isinstance(verifier, dict):
            cand = verifier.get("candidate_floor")
            if not _is_number(cand) or float(cand) != float(floor):
                problems.append("with a verifier, conf_floor must equal verifier.candidate_floor")

    if doc.get("tiler_version") != TILER_VERSION:
        problems.append(
            f"tiler_version {doc.get('tiler_version')!r} does not match this code ({TILER_VERSION!r})"
        )
    for key, expected in (("tile_size", TILE_SIZE), ("stride", STRIDE)):
        if key in doc and doc[key] != expected:
            problems.append(f"{key} {doc[key]!r} does not match this code ({expected})")

    if model_path is not None:
        model_path = Path(model_path)
        if isinstance(model_file, str) and model_path.name != model_file:
            problems.append(f"card names {model_file!r} but the model is {model_path.name!r}")
        if not model_path.exists():
            problems.append(f"model file not found: {model_path}")
        elif verify_sha and isinstance(sha, str) and _SHA256_RE.match(sha):
            actual = sha256_file(model_path)
            if actual != sha:
                problems.append(f"model_sha256 mismatch: card {sha[:12]}…, file {actual[:12]}…")

    if problems:
        raise DetectorCardError("; ".join(problems))


def load_detector_card(
    model_path: Path,
    verify_sha: bool = True,
    card_path: Optional[Path] = None,
) -> DetectorCard:
    """Read and validate the card next to `model_path`. Raises DetectorCardError."""
    model_path = Path(model_path)
    path = Path(card_path) if card_path is not None else card_path_for(model_path)
    if not path.exists():
        raise DetectorCardError(f"card not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DetectorCardError(f"card unreadable: {path}: {e}") from e
    validate_card(doc, model_path=model_path, verify_sha=verify_sha)
    return DetectorCard(
        model_path=model_path,
        model_sha256=doc["model_sha256"],
        adapter=doc["adapter"],
        schema=doc.get("schema"),
        class_names=tuple(doc["class_names"]),
        threshold=float(doc["threshold"]),
        conf_floor=float(doc["conf_floor"]),
        tiler_version=doc["tiler_version"],
        verifier=doc.get("verifier"),
        raw=doc,
    )


def write_detector_card(path: Path, doc: dict) -> Path:
    """Validate against the model file beside `path`, then write atomically."""
    path = Path(path)
    validate_card(doc, model_path=path.parent / doc.get("model_file", ""), verify_sha=True)
    data = (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
