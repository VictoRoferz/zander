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
    # Capture dumps: <id>.jpg + <id>.json sidecar + <id>.task. Batch 1 = the
    # Pi-era captures (Jun 9 – Jul 20), batch 2 = the PC captures (Aug 7 + 17;
    # the folder is called "labeled 2" but holds unlabeled-format captures).
    source_images_dirs: list[Path] = [
        REPO_ROOT / "data_fotos" / "Bilder 27-07-26",
        REPO_ROOT / "data_fotos" / "labeled 2",
    ]
    # Label snapshot v2: the Aug-17 Label Studio YOLO export (superset of the
    # July-27 v1 export: identical 298 label files + 33 board-B photos).
    snapshot_dir: Path = (
        REPO_ROOT / "data_fotos" / "project-2-at-2026-08-17-13-14-5747402b"
    )
    snapshot_id: str = "v2_2026-08-17_yolo"
    # The gitignored photo audit the committed attributes table is built from.
    audit_csv: Path = REPO_ROOT / "ml_data" / "audit" / "photo_audit_2026-09-28.csv"

    # ---- Frozen split in use ----
    split_id: str = "split_v2"

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
        self.source_images_dirs = [p.expanduser() for p in self.source_images_dirs]
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

    @property
    def attributes_csv(self) -> Path:
        """Committed, anonymised per-photo attributes (board, pose, content)."""
        return self.datasets_dir / "photo_attributes_v1.csv"

    @property
    def test_ledger_file(self) -> Path:
        """Committed audit trail of every test-bucket evaluation (one-shot rule)."""
        return self.reports_dir / "test_ledger.csv"

    @property
    def eval_dir(self) -> Path:
        return self.ml_data_root / "eval"

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
    def bundles_dir(self) -> Path:
        """Imported bundles, extracted flat: bundles/<run_name>/{model.onnx, meta.json, ...}
        (`cli.py import-run`); eval needs model.onnx and meta.json side by side."""
        return self.ml_data_root / "bundles"

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
