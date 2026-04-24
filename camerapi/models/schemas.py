"""
Typed payload models for camerapi.

Used for:
- Returning per-step capture results from /capture (fault-isolated)
- Parsing incoming Label Studio webhook payloads
- Passing structured metadata between services
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ----- Capture metadata ---------------------------------------------------

class CaptureMetadata(BaseModel):
    """Metadata that travels with every captured image."""

    capture_id: str  # primary stable ID (UUID, generated at capture time)
    sha256: str  # content hash (secondary; used for dedup, not identity)
    captured_at: datetime
    source: str  # "basler" | "opencv" | "fallback"
    width: int
    height: int
    size_bytes: int
    camera_serial: Optional[str] = None
    camera_model: Optional[str] = None


# ----- Per-step capture result (returned by POST /capture) ----------------

class StepResult(BaseModel):
    """Outcome of one step in the capture pipeline."""

    ok: bool
    detail: Optional[str] = None  # human-readable status / error reason


class CaptureResult(BaseModel):
    """
    Response body for POST /api/v1/capture.

    Each step is reported independently so callers can see partial success.
    """

    capture_id: str
    camera: StepResult
    label_studio: StepResult
    laptop_mirror: StepResult
    metadata: Optional[CaptureMetadata] = None


# ----- Label Studio webhook payload ---------------------------------------

class WebhookUser(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[int] = None
    email: Optional[str] = None


class WebhookAnnotation(BaseModel):
    """Partial shape of the `annotation` block from Label Studio."""

    model_config = ConfigDict(extra="allow")

    id: int
    task: int
    result: list[dict[str, Any]] = Field(default_factory=list)
    completed_by: Optional[WebhookUser] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WebhookTask(BaseModel):
    """Partial shape of the `task` block from Label Studio."""

    model_config = ConfigDict(extra="allow")

    id: int
    data: dict[str, Any] = Field(default_factory=dict)
    # `meta` is what we set via create_task_from_image — has capture_id + sha256
    meta: dict[str, Any] = Field(default_factory=dict)


class WebhookProject(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    title: Optional[str] = None


class AnnotationWebhookPayload(BaseModel):
    """
    Payload Label Studio sends to POST /api/v1/webhook/annotation-created.

    Unknown fields are tolerated (LS evolves). We only require what we use.
    """

    model_config = ConfigDict(extra="allow")

    action: str  # e.g. "ANNOTATION_CREATED" / "ANNOTATION_UPDATED"
    annotation: WebhookAnnotation
    task: WebhookTask
    project: Optional[WebhookProject] = None
