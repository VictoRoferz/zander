"""
Configuration for the inspector (defect-detection inference service).

Runs on the hub PC next to receiver/dashboard (:8004). Loads the exported
ONNX detector and serves tiled full-photo detection over HTTP. CPU-only by
design: no torch/ultralytics anywhere in this service; the inference code is
shared with the ML toolchain (ml/infer, pure numpy + onnxruntime).

The model ships with its card, `detector.json` (same folder, same stem as the
model; docs/CONTRACTS.md section 1). The card is AUTHORITATIVE: adapter, class
names, threshold and confidence floor come from it. The env vars ADAPTER /
CONFIDENCE_THRESHOLD / CLASS_NAMES are a legacy fallback, used only when no
card exists (and ignored with a WARNING when one does). REQUIRE_CARD=true
turns the fallback off: without a card the inspector refuses to load a model.

The threshold is the frozen t* from `cli.py choose-op`, carried into the card
by `cli.py export`; it is never tuned on the test split.

Several models (since v0.3.0): every <name>.onnx in the models folder (the
folder of MODEL_PATH unless MODELS_DIR says otherwise) is loaded at startup,
each with its own card. One of them is ACTIVE and serves /inspect; the
dashboard switches it (PUT /api/v1/models/active) and the choice is saved in
the models folder as active_model.json. Without a saved choice the active
model is MODEL_PATH (docs/CONTRACTS.md section 3).
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# parents[2] is the repo root in the repo layout; in the Docker image
# (/app/config/settings.py) it resolves to "/", which is why compose always
# sets MODEL_PATH explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Legacy fallbacks, used only when no detector.json sits next to the model.
DEFAULT_ADAPTER = "ultralytics"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
# Class order of the exported model (the training schema, ml/dataprep/remap.py).
DEFAULT_CLASS_NAMES = ("Nicht ausreichend Lot", "Brücke", "Fahne")

# The saved model choice, in the models folder. Not a card: a model named
# "active_model" is refused, because its card would be this file.
ACTIVE_MODEL_FILE = "active_model.json"


class Settings(BaseSettings):
    # ---- Service identity ----
    service_name: str = "zander-inspector"
    service_version: str = "0.3.0"

    # ---- Server ----
    # Loopback by default: nothing outside this PC needs the inspector (the
    # receiver calls it on localhost natively, over the compose network in
    # Docker, where the Dockerfile passes --host 0.0.0.0 itself).
    host: str = "127.0.0.1"
    port: int = 8004

    # ---- Model ----
    # Dev default: the ML toolchain's export dir. Production (Docker):
    # MODEL_PATH=/data/models/detector.onnx on the shared DATA_ROOT, with
    # detector.json next to it.
    model_path: Path = REPO_ROOT / "ml_data" / "exports" / "detector.onnx"
    # Every <name>.onnx in this folder is loaded too (None = MODEL_PATH's folder).
    # MODEL_PATH is the default active model.
    models_dir: Optional[Path] = None
    # true = refuse to load a model that has no detector.json card.
    require_card: bool = False
    # Legacy fallbacks (None = unset). Ignored whenever a card exists.
    adapter: Optional[str] = None  # DEFAULT_ADAPTER when unset
    confidence_threshold: Optional[float] = None  # DEFAULT_CONFIDENCE_THRESHOLD when unset
    class_names: Optional[list[str]] = None  # JSON list; DEFAULT_CLASS_NAMES when unset
    ort_threads: int = 0  # 0 = onnxruntime default

    # ---- Photo store (capture_id lookups) ----
    data_root: Path = Path.home() / "zander-data"

    # ---- Logging ----
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # "CONFIDENCE_THRESHOLD=" in a .env means "unset", not a parse error.
        env_ignore_empty=True,
        protected_namespaces=(),  # allow the `model_path` field name
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = self.model_path.expanduser()
        if self.models_dir is not None:
            self.models_dir = self.models_dir.expanduser()
        self.data_root = self.data_root.expanduser()

    @property
    def models_folder(self) -> Path:
        return self.models_dir if self.models_dir is not None else self.model_path.parent

    @property
    def active_model_file(self) -> Path:
        return self.models_folder / ACTIVE_MODEL_FILE

    @property
    def unlabeled_dir(self) -> Path:
        return self.data_root / "unlabeled"


# Global settings instance.
settings = Settings()
