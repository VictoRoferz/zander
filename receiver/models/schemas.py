"""
Typed payload models for the receiver.

`CaptureMetadata` is the exact shape camerapi serialises into each spool
sidecar and uploads as `metadata_json` — kept in sync with
camerapi/models/schemas.py.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


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
