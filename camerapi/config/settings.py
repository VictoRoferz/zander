"""
Configuration management for camerapi (Raspberry Pi 3).

camerapi is now a thin capture node: it grabs a frame, writes it to a local
spool, and a background uploader ships it to the laptop ingestion hub
(receiver, /api/v1/ingest). Label Studio and all downstream storage live on
the laptop now — camerapi no longer talks to Label Studio at all.

All values can be overridden via environment variables (see .env).
Settings are validated at startup via Pydantic.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Service identity ----
    service_name: str = "camera-service"
    service_version: str = "3.0.0"

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8001

    # ---- Camera ----
    use_camera: bool = True
    camera_index: int = 0  # OpenCV fallback index
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    fallback_image_path: str = "sample.jpg"
    # "auto" picks ethernet vs wifi by checking the route to the camera's IP.
    # Override to "ethernet" or "wifi" to force a profile (debugging / odd networks).
    camera_transport: str = "auto"

    # ---- Local spool (Pi side) ----
    # Captures land here first; the background uploader ships them to the
    # laptop and deletes each entry only once the laptop ACKs it. So a brief
    # laptop outage never loses a capture. Default suits the Pi; override
    # DATA_ROOT for dev on a laptop.
    data_root: Path = Path("/home/pi/zander-data")

    # ---- Laptop ingestion hub (receiver) ----
    # Single upload target now (was a comma-separated mirror fan-out). Each
    # capture is POSTed to {ingest_url}/api/v1/ingest with retry until ACK.
    ingest_url: str = "http://192.168.0.199:8002"
    upload_timeout: int = 30           # seconds per HTTP attempt
    upload_poll_interval: float = 2.0  # seconds between spool scans
    upload_max_backoff: float = 60.0   # cap for per-entry exponential backoff

    # ---- Dashboard (for GPIO-button user attribution) ----
    # When the GPIO button fires there is no X-Triggered-By header, so camerapi
    # asks the dashboard who is logged in via GET {dashboard_url}/api/current-user.
    # Best-effort — failure never blocks the capture (triggered_by becomes null).
    dashboard_url: str = "http://192.168.0.199:8003"
    dashboard_timeout: float = 1.5  # seconds; keep tight so the button stays snappy

    # ---- Logging ----
    log_level: str = "INFO"
    log_file: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure the spool folders exist on startup. Cheap and idempotent.
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.spool_failed_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derived paths ----

    @property
    def spool_dir(self) -> Path:
        return self.data_root / "spool"

    @property
    def spool_failed_dir(self) -> Path:
        """Poison entries (rejected by the laptop with a 4xx) are parked here."""
        return self.spool_dir / "failed"

    # ---- Derived URLs ----

    @property
    def ingest_endpoint(self) -> str:
        return f"{self.ingest_url.rstrip('/')}/api/v1/ingest"


# Global settings instance — imported everywhere else.
settings = Settings()
