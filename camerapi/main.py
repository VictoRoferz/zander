"""
camerapi entry point (Raspberry Pi 3).

Thin capture node — grabs frames and spools them; a background uploader ships
them to the laptop ingestion hub (receiver). Label Studio and downstream
storage now live on the laptop. Exposes:
  POST /api/v1/capture       — capture → spool (uploader ships to the laptop)
  POST /api/v1/test-camera   — camera-only test (local save, not spooled)
  GET  /api/v1/status        — camera + spool depth + ingest target
  GET  /api/v1/health        — liveness
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as camera_router
from config.settings import settings
from services.spool_uploader import spool_uploader
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    logger.info(f"Ingest target:  {settings.ingest_endpoint}")
    logger.info(f"Spool dir:      {settings.spool_dir}")

    # Start the background uploader. Its first loop iteration re-scans the
    # spool, so anything left over from a previous run is shipped on startup.
    spool_uploader.start()

    yield

    spool_uploader.stop()
    logger.info(f"Shutting down {settings.service_name}")


app = FastAPI(
    title="camerapi",
    description="Raspberry Pi 3 capture node (spools to the laptop ingestion hub)",
    version=settings.service_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(camera_router)


@app.get("/")
async def root() -> dict:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "endpoints": {
            "capture": "POST /api/v1/capture",
            "test_camera": "POST /api/v1/test-camera",
            "status": "GET /api/v1/status",
            "health": "GET /api/v1/health",
            "docs": "GET /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
