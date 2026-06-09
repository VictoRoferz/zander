"""
Typed payload models for camerapi.

camerapi only captures and spools now, so these models cover:
- Metadata that travels with every captured image (and into the spool sidecar).
- The per-step result returned by POST /api/v1/capture.

The Label Studio webhook models moved to the receiver along with the webhook
handler itself.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


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
    # Email of the dashboard user who triggered the capture, if any.
    # None = button press with no logged-in user, or curl without the header.
    triggered_by: Optional[str] = None


# ----- Per-step capture result (returned by POST /capture) ----------------

class StepResult(BaseModel):
    """Outcome of one step in the capture pipeline."""

    ok: bool
    detail: Optional[str] = None  # human-readable status / error reason


class CaptureResult(BaseModel):
    """Response body for POST /api/v1/capture."""

    capture_id: str
    camera: StepResult
    # True once the capture is written to the spool; the background uploader
    # ships it to the laptop afterwards (watch the logs for the outcome).
    spooled: bool = False
    metadata: Optional[CaptureMetadata] = None
