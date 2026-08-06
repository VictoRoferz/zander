"""
Configuration for the ML toolchain (dataset building, training, evaluation).

Not a service: `ml/` is run as a CLI from its own directory
(`cd ml && python cli.py <step>`), mirroring how the sibling services run
cwd-relative. Heavy artifacts (tiles, weights, runs) live under
`ml_data_root`, which is gitignored; everything under `ml/` itself
(manifests, split files, configs, reports) is small and committed.

`ml_data_root` defaults to <repo>/ml_data and can be moved via ML_DATA_ROOT
(e.g. to a Drive-mounted path on Colab).
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# zander repo root = parent of ml/ (this file lives at ml/config/settings.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ---- Identity ----
    toolchain_name: str = "zander-ml"
    toolchain_version: str = "0.1.0"

    # ---- Heavy-artifact root (gitignored) ----
    ml_data_root: Path = REPO_ROOT / "ml_data"

    # ---- Source data (raw photos + label snapshots) ----
    # The canonical capture dump: <id>.jpg + <id>.json sidecar + <id>.task.
    source_images_dir: Path = REPO_ROOT / "data_fotos" / "Bilder 27-07-26"
    # Label snapshot v1: the July-27 Label Studio YOLO export (lossy; a full
    # LS JSON export should replace it as snapshot v2 before test metrics).
    snapshot_dir: Path = (
        REPO_ROOT / "data_fotos" / "project-2-at-2026-07-27-06-28-47dc2d74"
    )
    snapshot_id: str = "v1_2026-07-27_yolo"

    # ---- Determinism ----
    seed: int = 20260806

    # ---- Quarantine thresholds ----
    dark_brightness_threshold: float = 35.0  # mean gray below this = junk frame

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
        self.ml_data_root = self.ml_data_root.expanduser()
        self.source_images_dir = self.source_images_dir.expanduser()
        self.snapshot_dir = self.snapshot_dir.expanduser()
        self.ml_data_root.mkdir(parents=True, exist_ok=True)

    # ---- Derived paths (committed side) ----

    @property
    def datasets_dir(self) -> Path:
        """Committed manifests / group maps / split files."""
        return REPO_ROOT / "ml" / "datasets"

    @property
    def reports_dir(self) -> Path:
        """Committed dataset card, eval reports, results.csv."""
        return REPO_ROOT / "ml" / "reports"

    # ---- Derived paths (gitignored side) ----

    @property
    def tiles_dir(self) -> Path:
        return self.ml_data_root / "tiles"

    @property
    def yolo_dir(self) -> Path:
        return self.ml_data_root / "yolo"

    @property
    def pretrained_dir(self) -> Path:
        return self.ml_data_root / "pretrained"

    @property
    def run_dirs_dir(self) -> Path:
        return self.ml_data_root / "run_dirs"

    @property
    def mlruns_dir(self) -> Path:
        return self.ml_data_root / "mlruns"

    @property
    def bench_dir(self) -> Path:
        return self.ml_data_root / "bench"

    @property
    def exports_dir(self) -> Path:
        """Deployable ONNX models (the winner lives here)."""
        return self.ml_data_root / "exports"


# Global settings instance.
settings = Settings()
