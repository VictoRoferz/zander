"""
Laptop receiver — mirrors camerapi artifacts into ~/zander-data/.

Endpoints:
  POST /api/v1/mirror/unlabeled  — image + metadata_json (multipart)
  POST /api/v1/mirror/labeled    — image + annotation JSON (multipart)
  GET  /api/v1/status            — paths + health

Runs alongside whatever else is on the laptop. The camerapi is the source
of truth; this service is a passive mirror.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

LOG = logging.getLogger("receiver")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "zander-data")).expanduser()
UNLABELED_DIR = DATA_ROOT / "unlabeled"
LABELED_DIR = DATA_ROOT / "labeled"
UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8002"))

app = FastAPI(title="zander-receiver", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def _atomic_write(target: Path, data: bytes) -> None:
    """Atomic write: temp file in target dir, fsync, then os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@app.post("/api/v1/mirror/unlabeled")
async def mirror_unlabeled(
    capture_id: str = Form(...),
    metadata_json: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Store an unlabeled image + its metadata sidecar."""
    image_bytes = await file.read()
    image_path = UNLABELED_DIR / f"{capture_id}.jpg"
    metadata_path = UNLABELED_DIR / f"{capture_id}.json"

    _atomic_write(image_path, image_bytes)

    # Validate the JSON (reject malformed payloads early) then pretty-print it.
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"metadata_json invalid: {e}")
    _atomic_write(metadata_path, json.dumps(parsed, indent=2).encode("utf-8"))

    LOG.info(f"Stored unlabeled {capture_id} ({len(image_bytes)} bytes)")
    return {
        "status": "ok",
        "capture_id": capture_id,
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
        "size_bytes": len(image_bytes),
    }


@app.post("/api/v1/mirror/labeled")
async def mirror_labeled(
    capture_id: str = Form(...),
    file: UploadFile = File(...),
    annotation: UploadFile = File(...),
) -> dict:
    """Store a labeled image + its annotation JSON. Idempotent — overwrites."""
    image_bytes = await file.read()
    annotation_bytes = await annotation.read()

    image_path = LABELED_DIR / f"{capture_id}.jpg"
    annotation_path = LABELED_DIR / f"{capture_id}.json"

    # Validate the annotation JSON but store the original bytes (preserve exact LS payload).
    try:
        json.loads(annotation_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"annotation JSON invalid: {e}")

    _atomic_write(image_path, image_bytes)
    _atomic_write(annotation_path, annotation_bytes)

    LOG.info(f"Stored labeled {capture_id} ({len(image_bytes)} bytes)")
    return {
        "status": "ok",
        "capture_id": capture_id,
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
    }


@app.get("/api/v1/status")
async def status() -> dict:
    return {
        "service": "zander-receiver",
        "data_root": str(DATA_ROOT),
        "unlabeled_dir": str(UNLABELED_DIR),
        "labeled_dir": str(LABELED_DIR),
        "unlabeled_count": _count_images(UNLABELED_DIR),
        "labeled_count": _count_images(LABELED_DIR),
    }


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "healthy"}


def _count_images(directory: Path) -> int:
    try:
        return sum(1 for p in directory.iterdir() if p.suffix.lower() == ".jpg")
    except FileNotFoundError:
        return 0


@app.get("/")
async def root() -> dict:
    return {
        "service": "zander-receiver",
        "endpoints": {
            "mirror_unlabeled": "POST /api/v1/mirror/unlabeled",
            "mirror_labeled": "POST /api/v1/mirror/labeled",
            "status": "GET /api/v1/status",
            "health": "GET /api/v1/health",
        },
    }


if __name__ == "__main__":
    LOG.info(f"Starting zander-receiver on {HOST}:{PORT}")
    LOG.info(f"Data root: {DATA_ROOT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
