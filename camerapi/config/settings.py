"""
Configuration management for camerapi (Raspberry Pi 3).

All values can be overridden via environment variables (see .env).
Settings are validated at startup via Pydantic.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Service identity ----
    service_name: str = "camera-service"
    service_version: str = "2.0.0"

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

    # ---- Persistent storage (Pi side) ----
    # Images land here on the Pi and never get auto-deleted.
    data_root: Path = Path("/home/pi/zander-data")

    # ---- Label Studio ----
    labelstudio_url: str = "http://localhost:8081"
    labelstudio_api_key: str = ""  # MUST be set via .env
    labelstudio_project_name: str = "PCB Defect Inspection"
    labelstudio_enable_webhooks: bool = True
    # Upload timeouts for the LS SDK (seconds)
    labelstudio_timeout: int = 30

    # ---- Laptop mirror (fixed-IP "Option 1") ----
    # If enabled, each capture is also sent to the laptop receiver.
    # Failure to reach the laptop does NOT fail the capture flow.
    laptop_mirror_enabled: bool = True
    laptop_mirror_url: str = "http://192.168.0.199:8002"
    laptop_mirror_timeout: int = 10  # seconds per HTTP call
    laptop_mirror_retries: int = 3

    # ---- Dashboard (for GPIO-button user attribution) ----
    # When the GPIO button fires, camerapi asks the dashboard who's logged
    # in via GET {dashboard_url}/api/current-user. Best-effort — failure
    # never blocks the capture (triggered_by just becomes null).
    dashboard_url: str = "http://192.168.0.199:8003"
    dashboard_timeout: float = 1.5  # seconds; keep tight so button stays snappy

    # ---- Logging ----
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ---- Health check ----
    health_check_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure persistent folders exist on startup. Cheap and idempotent.
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)
        self.labeled_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derived paths ----

    @property
    def unlabeled_dir(self) -> Path:
        return self.data_root / "unlabeled"

    @property
    def labeled_dir(self) -> Path:
        return self.data_root / "labeled"

    # ---- Derived URLs ----

    @property
    def laptop_unlabeled_url(self) -> str:
        return f"{self.laptop_mirror_url}/api/v1/mirror/unlabeled"

    @property
    def laptop_labeled_url(self) -> str:
        return f"{self.laptop_mirror_url}/api/v1/mirror/labeled"


# Global settings instance — imported everywhere else.
settings = Settings()
