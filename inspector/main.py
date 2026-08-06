"""
zander-inspector — solder-defect detection service (hub PC, :8004).

Serves the benchmark-winning ONNX model over HTTP with the same tiled
pipeline the evaluation used. Torch-free: onnxruntime + numpy only.

Status contract: /api/v1/health is liveness; /api/v1/status reports whether a
model is actually loaded. /inspect returns 503 while no model file exists, so
wiring the receiver/dashboard against it is safe before the first export.

Follow-ups (documented, not wired yet — see ml/README.md):
  - receiver: BackgroundTasks hook after ingest -> POST /inspect -> write
    unlabeled/<id>.pred.json sidecar (atomic write + chmod 0644)
  - dashboard: prediction badge/overlay from that sidecar
  - scripts/launch.py + deploy/docker-compose.yml entries for this service
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings
from services.detector_service import detector_service

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger("inspector")


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info(f"{settings.service_name} v{settings.service_version} starting")
    detector_service.load()
    yield
    LOG.info("inspector shutting down")


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/api/v1/status")
async def status() -> dict:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "model_loaded": detector_service.ready,
        "model_path": str(settings.model_path),
        "model_version": detector_service.model_sha8,
        "adapter": settings.adapter,
        "confidence_threshold": settings.confidence_threshold,
    }


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> dict:
    return {
        "service": settings.service_name,
        "endpoints": ["/api/v1/inspect", "/api/v1/status", "/api/v1/health"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
