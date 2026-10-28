"""
Configuration management for camerapi.

camerapi runs natively on the hub PC (Windows prod / macOS dev) with the
Basler GigE camera direct-attached. It is a thin capture node: it grabs a
frame, writes it to a local spool, and a background uploader ships it to the
ingestion hub (receiver, /api/v1/ingest) — normally the Docker containers on
this same machine. It never talks to Label Studio directly.

All values can be overridden via environment variables (see .env).
Settings are validated at startup via Pydantic.
"""
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Service identity ----
    service_name: str = "camera-service"
    service_version: str = "3.3.0"

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
    # "auto" = ethernet on Windows/macOS (direct-attach GigE is the norm);
    # on Linux it checks the route to the camera's IP (eth* vs wl*).
    # Override to "ethernet" or "wifi" to force a profile (debugging / odd networks).
    camera_transport: str = "auto"

    # ---- Live preview (docs/CONTRACTS.md section 4) ----
    # Off by default: every capture then uses the one-shot GrabOne path exactly
    # as before. On: one camera-owner thread holds the camera, streams a small
    # MJPEG preview while someone watches, and serves each capture by stopping
    # the stream and grabbing a fresh full-resolution frame.
    live_preview_enabled: bool = False
    # Shared secret the dashboard sends as X-Preview-Token (must equal the one
    # in deploy\.env). Empty = only loopback requests to localhost are allowed.
    preview_token: str = ""
    preview_fps: float = Field(default=5.0, gt=0, le=60)
    preview_max_width: int = Field(default=1280, ge=160, le=8192)
    preview_jpeg_quality: int = Field(default=70, ge=10, le=100)
    # The stream keeps running this long after the last viewer leaves, so a
    # page reload re-attaches instantly.
    preview_idle_stop_s: float = Field(default=10.0, ge=0)
    preview_max_clients: int = Field(default=8, ge=1, le=256)
    camera_reconnect_max_backoff_s: float = Field(default=30.0, gt=0)
    # Per-GrabOne-attempt timeout in live mode (3 attempts per capture). Not
    # used on the WiFi profile: a frame takes ~8 s there, so a capture makes
    # one 30 s attempt, as the one-shot path does.
    live_grab_timeout_ms: int = Field(default=5000, ge=100)
    # A capture request waits this long for the camera owner, then falls back
    # to sample.jpg (as a failed legacy grab does). At least 35 s on the WiFi
    # profile.
    live_capture_timeout_s: float = Field(default=20.0, gt=0)

    # ---- Local spool ----
    # Captures land here first; the background uploader ships them to the
    # receiver and deletes each entry only once it ACKs. So a capture is never
    # lost while the Docker side is down or still booting.
    # NOTE: this is NOT the canonical image store. The spool under DATA_ROOT
    # is transient; the canonical unlabeled/labeled store is the receiver's
    # DATA_ROOT (deploy\data under Docker). Files move by HTTP, never a
    # shared path — keep the two roots separate.
    data_root: Path = Path.home() / "zander-data"

    # ---- Ingestion hub (receiver) ----
    # Single upload target. Each capture is POSTed to
    # {ingest_url}/api/v1/ingest with retry until ACK. The receiver normally
    # runs in Docker on this same machine, port-mapped on localhost.
    ingest_url: str = "http://localhost:8002"
    upload_timeout: int = 30           # seconds per HTTP attempt
    upload_poll_interval: float = 2.0  # seconds between spool scans
    upload_max_backoff: float = 60.0   # cap for per-entry exponential backoff

    # ---- Dashboard (for USB-button user attribution) ----
    # When the USB button fires there is no X-Triggered-By header, so camerapi
    # asks the dashboard who is logged in via GET {dashboard_url}/api/current-user.
    # Best-effort — failure never blocks the capture (triggered_by becomes null).
    dashboard_url: str = "http://localhost:8003"
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
