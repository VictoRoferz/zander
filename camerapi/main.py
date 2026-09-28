"""
camerapi entry point (native capture service on the hub PC).

Thin capture node — grabs frames and spools them; a background uploader ships
them to the ingestion hub (receiver, normally the Docker containers on this
same machine). camerapi never talks to Label Studio. Exposes:
  POST /api/v1/capture              — capture -> spool (uploader ships to the receiver)
  POST /api/v1/test-camera          — camera-only test (local save, not spooled)
  GET  /api/v1/status               — camera + live preview + spool depth + ingest target
  GET  /api/v1/health               — liveness
  GET  /api/v1/preview/stream.mjpg  — live MJPEG preview (LIVE_PREVIEW_ENABLED=true)
  GET  /api/v1/preview/frame.jpg    — one fresh preview JPEG
  GET  /api/v1/preview/status       — live camera state
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.preview import close_streams
from api.preview import router as preview_router
from api.routes import router as camera_router
from config.settings import settings
from services.live_camera import live_camera
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

    if settings.live_preview_enabled:
        logger.info(
            "Live preview: ON (camera-owner thread; captures stop the stream "
            "and grab a fresh frame)"
        )
        if not settings.preview_token:
            logger.warning(
                "PREVIEW_TOKEN is empty: the preview only answers loopback "
                "requests to localhost (the Docker dashboard needs the token)"
            )
        live_camera.start()
    else:
        logger.info("Live preview: off (one-shot GrabOne per capture)")

    yield

    # Reverse order. (Open MJPEG streams were already ended by
    # _Server.shutdown below; they would otherwise hold the shutdown.)
    live_camera.stop()  # no-op when it was never started
    spool_uploader.stop()
    logger.info(f"Shutting down {settings.service_name}")


app = FastAPI(
    title="camerapi",
    description="Native capture service (spools to the ingestion hub)",
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
app.include_router(preview_router)


@app.get("/")
async def root() -> dict:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "live_preview_enabled": settings.live_preview_enabled,
        "endpoints": {
            "capture": "POST /api/v1/capture",
            "test_camera": "POST /api/v1/test-camera",
            "status": "GET /api/v1/status",
            "health": "GET /api/v1/health",
            "preview_stream": "GET /api/v1/preview/stream.mjpg",
            "preview_frame": "GET /api/v1/preview/frame.jpg",
            "preview_status": "GET /api/v1/preview/status",
            "docs": "GET /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    class _Server(uvicorn.Server):
        """uvicorn.Server that ends open MJPEG streams as shutdown begins
        (they never finish on their own), so Ctrl-C exits at once."""

        async def shutdown(self, *args, **kwargs):
            close_streams()
            await super().shutdown(*args, **kwargs)

    logger.info(f"Starting server on {settings.host}:{settings.port}")
    config = uvicorn.Config(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
        # Backstop: any request still open after 3 s is cancelled, so
        # Ctrl-C always exits promptly.
        timeout_graceful_shutdown=3,
    )
    try:
        _Server(config).run()
    except KeyboardInterrupt:
        pass  # uvicorn re-raises the Ctrl-C after its graceful shutdown (as uvicorn.run does)
