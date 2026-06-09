"""
Receiver — the laptop ingestion hub.

Runs on the laptop next to Label Studio. Responsibilities:
  POST /api/v1/ingest                     — intake a capture from the Pi:
                                            store it + create the LS task
  POST /api/v1/webhook/annotation-created — LS callback: export labeled data
  GET  /api/v1/status                     — paths, counts, LS health
  GET  /api/v1/health                     — liveness

Label Studio reads images straight from data_root via local-files serving, so
the receiver owns LS project + local-storage setup. The dashboard reads LS
state through its own read-only client and never touches this service's LS init.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ingest import router as ingest_router
from api.webhooks import router as webhook_router
from config.settings import settings
from services.labelstudio_service import labelstudio_service
from services.storage_service import storage_service

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger("receiver")


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info(
        f"receiver starting: data_root={settings.data_root} ls={settings.labelstudio_url}"
    )
    # Bring Label Studio up if we can, but DON'T crash if it isn't ready yet —
    # /ingest lazily re-initializes and returns 503 until LS is reachable, so
    # the Pi's spool just retries. This is strictly safer than dropping tasks.
    try:
        labelstudio_service.initialize()
    except Exception as e:
        LOG.warning(
            f"Label Studio not ready at startup: {str(e)[:160]}. "
            "/ingest will retry lazily."
        )
    yield
    LOG.info("receiver shutting down")


app = FastAPI(title="zander-receiver", version=settings.service_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(webhook_router)


@app.get("/api/v1/status")
async def status() -> dict:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "data_root": str(settings.data_root),
        "unlabeled_dir": str(settings.unlabeled_dir),
        "labeled_dir": str(settings.labeled_dir),
        "unlabeled_count": storage_service.count_unlabeled(),
        "labeled_count": storage_service.count_labeled(),
        "label_studio": {
            "url": settings.labelstudio_url,
            "ready": labelstudio_service.is_healthy(),
        },
    }


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict:
    return {
        "service": settings.service_name,
        "endpoints": {
            "ingest": "POST /api/v1/ingest",
            "webhook": "POST /api/v1/webhook/annotation-created",
            "status": "GET /api/v1/status",
            "health": "GET /api/v1/health",
        },
    }


if __name__ == "__main__":
    LOG.info(f"Starting {settings.service_name} on {settings.host}:{settings.port}")
    LOG.info(f"Data root: {settings.data_root}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
