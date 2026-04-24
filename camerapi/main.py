"""
camerapi entry point.

Runs on the Raspberry Pi 3. Exposes:
  POST /api/v1/capture                    — capture → store → LS → mirror
  POST /api/v1/test-camera                — camera-only test (local save)
  GET  /api/v1/status                     — components + URLs
  GET  /api/v1/health                     — liveness
  POST /api/v1/webhook/annotation-created — Label Studio callback
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as camera_router
from api.webhooks import router as webhook_router
from config.settings import settings
from services.labelstudio_service import labelstudio_service
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    logger.info(f"Label Studio URL:   {settings.labelstudio_url}")
    logger.info(f"Laptop mirror:      enabled={settings.laptop_mirror_enabled} url={settings.laptop_mirror_url}")
    logger.info(f"Data root:          {settings.data_root}")

    # Connect to Label Studio and ensure the project exists. Failure here is
    # fatal by design — the whole point of camerapi v2 is LS integration.
    # Check /api/v1/status afterwards to verify.
    try:
        labelstudio_service.initialize()
    except Exception as e:
        logger.error(
            f"Label Studio initialization failed at startup: {e}. "
            "Capture will still work, but /capture will report label_studio.ok=false "
            "until this is resolved."
        )

    yield

    logger.info(f"Shutting down {settings.service_name}")


app = FastAPI(
    title="camerapi",
    description="Raspberry Pi 3 capture + Label Studio bridge",
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
app.include_router(webhook_router)


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
            "webhook": "POST /api/v1/webhook/annotation-created",
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
