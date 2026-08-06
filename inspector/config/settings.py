"""
Configuration for the inspector (defect-detection inference service).

Runs on the hub PC next to receiver/dashboard (:8004). Loads the winning
exported ONNX model and serves tiled full-photo detection over HTTP. CPU-only
by design — no torch/ultralytics anywhere in this service; the inference code
is shared with the ML toolchain (ml/infer, pure numpy + onnxruntime).

`confidence_threshold` must be set to the t* chosen on the dev folds by the
evaluation harness (ml/reports/results.csv) — never tuned on test-v1.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ---- Service identity ----
    service_name: str = "zander-inspector"
    service_version: str = "0.1.0"

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8004

    # ---- Model ----
    # Dev default: the ML toolchain's export dir. Production (Docker):
    # MODEL_PATH=/data/models/nal_detector.onnx on the shared DATA_ROOT.
    model_path: Path = REPO_ROOT / "ml_data" / "exports" / "nal_detector.onnx"
    adapter: str = "ultralytics"  # or "deim", matching the winning framework
    confidence_threshold: float = 0.25  # REPLACE with t* from the eval harness
    ort_threads: int = 0  # 0 = onnxruntime default

    # ---- Optional photo store (capture_id lookups) ----
    data_root: Path = Path.home() / "zander-data"

    # ---- Logging ----
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),  # allow the `model_path` field name
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = self.model_path.expanduser()
        self.data_root = self.data_root.expanduser()

    @property
    def unlabeled_dir(self) -> Path:
        return self.data_root / "unlabeled"


# Global settings instance.
settings = Settings()
