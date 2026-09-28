"""
Shared helpers for the verifier toolchain: working-dir layout, atomic JSON
writes, hashing, run-name validation, provenance.

Working files live in ml_data/verifier/<vrun>/ (CONTRACTS §8.4):
candidates.json, windows.npy (+ windows_index.json), oof_scores.json,
verifier_final.pt, stacker_fold<k>.json, verifier.onnx, stacker.json,
exit_report.{md,json}, train_log.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import REPO_ROOT, settings

FOLDS = ("fold0", "fold1", "fold2")
VRUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

CANDIDATES = "candidates.json"
WINDOWS = "windows.npy"
WINDOWS_INDEX = "windows_index.json"
OOF_SCORES = "oof_scores.json"
FINAL_WEIGHTS = "verifier_final.pt"
VERIFIER_ONNX = "verifier.onnx"
STACKER = "stacker.json"
EXIT_JSON = "exit_report.json"
EXIT_MD = "exit_report.md"
TRAIN_LOG = "train_log.json"


class Refusal(RuntimeError):
    """An honesty guard refused the request (the CLI prints it without a traceback)."""


def check_vrun(vrun: str) -> str:
    if not VRUN_RE.match(vrun or ""):
        raise ValueError(
            f"verifier run name {vrun!r} must match {VRUN_RE.pattern} (it becomes a "
            "folder name and part of the cascade eval run names)"
        )
    return vrun


def vrun_dir(vrun: str, create: bool = False) -> Path:
    d = settings.ml_data_root / "verifier" / check_vrun(vrun)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """temp file in the same dir -> fsync -> os.replace (never a torn file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_json_atomic(path: Path, obj: Any, indent: int | None = 1) -> str:
    """Write JSON atomically; returns the sha256 of the written bytes."""
    data = json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False).encode("utf-8")
    write_bytes_atomic(path, data)
    return sha256_bytes(data)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_state() -> dict:
    """{git_sha, git_dirty} of the repo, or nulls outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "ml"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def host() -> dict:
    return {"host": platform.node(), "machine": platform.machine(),
            "python": platform.python_version()}


def resolve_model_path(model: str | None) -> Path | None:
    """Eval runs store the model path as given on the command line (usually
    relative to ml/, e.g. ../ml_data/run_dirs/<run>/bundle/model.onnx)."""
    if not model:
        return None
    p = Path(model)
    if p.is_absolute():
        return p
    for base in (REPO_ROOT / "ml", Path.cwd()):
        q = (base / p).resolve()
        if q.exists():
            return q
    return (REPO_ROOT / "ml" / p).resolve()
