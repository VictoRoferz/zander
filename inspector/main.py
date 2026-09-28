"""
zander-inspector: solder-defect detection service (hub PC, :8004).

Serves the exported ONNX detector over HTTP with the same tiled pipeline the
evaluation used. Torch-free: onnxruntime + numpy only. The model's card
(detector.json, next to the model) supplies adapter, class names and the
frozen threshold; see docs/CONTRACTS.md sections 1 and 3.

Status contract: /api/v1/health is liveness and answers {"status":"ok"} even
without a model; /api/v1/status reports whether a model is loaded and, if
not, why (`load_error`). /inspect returns 503 while no model is loaded, so
wiring the receiver against it is safe before the first export.

Who calls it: the receiver's prediction worker posts every ingested capture
to /inspect and writes the result to DATA_ROOT/predictions/<id>.pred.json
(docs/CONTRACTS.md section 2), which the dashboard renders. This service
never writes to DATA_ROOT (Docker mounts it read-only). A new model is picked
up on restart.
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
    detector_service.load()  # never raises; a refused model leaves load_error set
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
    return detector_service.status()


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
