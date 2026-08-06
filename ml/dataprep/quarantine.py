"""
Quarantine rules: frames that must never enter training or evaluation.

Quarantine is orthogonal to label state — a quarantined frame keeps its label
info in the manifest (so the label-stat gates stay comparable to the raw
analysis) but every downstream step filters `quarantine_reason is None`.
"""
from __future__ import annotations

from config.settings import settings

# Manual review list: capture_ids with a known inconsistency worth a human
# look in Label Studio. They are NOT quarantined automatically.
#   0fef68e4…: image-level choice "iO" but has a drawn box (the box is Fahne,
#   so post-remap the image becomes a consistent negative — kept, flagged).
REVIEW_NOTES: dict[str, str] = {
    "0fef68e4e70449d787e01b8140034479": (
        "LS choice says iO but a (Fahne) box exists; consistent negative "
        "after single-class remap — review in Label Studio when convenient."
    ),
}


def quarantine_reason(source: str, brightness: float) -> str | None:
    """Return the reason a frame is junk, or None if it is usable."""
    if source != "basler":
        return f"non-camera source '{source}' (webcam/sample fallback frame)"
    if brightness < settings.dark_brightness_threshold:
        return (
            f"dark setup frame (mean brightness {brightness:.1f} < "
            f"{settings.dark_brightness_threshold:.0f})"
        )
    return None
