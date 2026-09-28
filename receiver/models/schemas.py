"""
Typed payload models for the receiver.

`CaptureMetadata` is the exact shape camerapi serialises into each spool
sidecar and uploads as `metadata_json` — kept in sync with
camerapi/models/schemas.py.

`PredictionDoc` is pred.json v1 (docs/CONTRACTS.md section 2), the file the
receiver writes to DATA_ROOT/predictions/<capture_id>.pred.json and the
dashboard renders. Dump it with `by_alias=True` (the field `schema_` is
stored as "schema").
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ERROR_MAX_CHARS = 500


class CaptureMetadata(BaseModel):
    """Metadata that travels with every captured image (from the Pi)."""

    capture_id: str
    sha256: str
    captured_at: datetime
    source: str
    width: int
    height: int
    size_bytes: int
    camera_serial: Optional[str] = None
    camera_model: Optional[str] = None
    triggered_by: Optional[str] = None


class Detection(BaseModel):
    """One defect box in photo pixels (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str


class PredictionDoc(BaseModel):
    """pred.json v1. `pending` is never written: the dashboard derives it."""

    # model_* field names are part of the contract; `schema` would shadow
    # BaseModel.schema, hence the alias.
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    schema_version: Literal[1] = 1
    capture_id: str
    status: Literal["ok", "no_model", "error"]
    created_at: str  # ISO-8601 UTC with "Z": when the receiver wrote the file
    error: Optional[str] = None
    model_version: Optional[str] = None  # "<file>@<sha256[:8]>", the idempotency key
    model_sha256: Optional[str] = None
    schema_: Optional[str] = Field(default=None, alias="schema")
    tiler_version: Optional[str] = None
    class_names: Optional[list[str]] = None
    threshold: Optional[float] = None
    threshold_source: Optional[Literal["detector.json", "env", "default"]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    verdict: Optional[Literal["iO", "NiO"]] = None
    classes: list[str] = Field(default_factory=list)
    photo_score: Optional[float] = None
    detections: list[Detection] = Field(default_factory=list)
    latency_ms: Optional[float] = None

    @field_validator("error", mode="before")
    @classmethod
    def _truncate_error(cls, v):
        if isinstance(v, str) and len(v) > ERROR_MAX_CHARS:
            return v[: ERROR_MAX_CHARS - 3] + "..."
        return v

    @model_validator(mode="after")
    def _consistent_with_status(self):
        if self.status == "ok":
            if self.verdict is None:
                raise ValueError("an ok prediction needs a verdict")
            if self.error is not None:
                raise ValueError("an ok prediction carries no error")
        elif self.verdict is not None or self.classes or self.detections:
            raise ValueError("verdict, classes and detections are only set when status is ok")
        return self
