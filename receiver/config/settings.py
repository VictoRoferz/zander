"""
Configuration for the receiver (laptop ingestion hub).

The receiver is the laptop's hub: it accepts captures uploaded by the Pi,
stores them on the laptop's disk, creates the Label Studio task, and handles
the LS annotation webhook. Label Studio runs on the same laptop and reads
images directly from `data_root` via local-files serving — so the receiver's
`data_root` MUST equal LS's LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.

Cross-platform: `data_root` defaults to ~/zander-data, which resolves on both
Windows (C:\\Users\\<u>\\zander-data) and macOS (/Users/<u>/zander-data).
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Service identity ----
    service_name: str = "zander-receiver"
    service_version: str = "2.0.0"

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8002

    # ---- Storage (laptop side) ----
    # Must match LS's LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.
    data_root: Path = Path.home() / "zander-data"

    # ---- Label Studio (running on the same laptop) ----
    labelstudio_url: str = "http://localhost:8081"
    labelstudio_api_key: str = ""  # REQUIRED — paste token from LS UI
    labelstudio_project_name: str = "PCB Defect Inspection"
    labelstudio_timeout: int = 30

    # ---- Logging ----
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Expand a leading ~ if DATA_ROOT was given as "~/zander-data".
        self.data_root = self.data_root.expanduser()
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)
        self.labeled_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derived paths ----

    @property
    def unlabeled_dir(self) -> Path:
        return self.data_root / "unlabeled"

    @property
    def labeled_dir(self) -> Path:
        return self.data_root / "labeled"


# Global settings instance.
settings = Settings()
