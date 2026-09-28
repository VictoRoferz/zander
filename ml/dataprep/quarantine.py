"""
Quarantine + review rules: frames that must never enter training or evaluation.

Quarantine is orthogonal to label state — a quarantined frame keeps its label
info in the manifest (so the label-stat gates stay comparable to the raw
export) but every downstream step filters `quarantine_reason is None`.

v2 (2026-09-28): the photo audit showed that the v1 brightness/source rule
caught only 17 of the 87 frames that show no usable PCB (empty fixture, hands,
test pin, setup shots, placeholders, black/white frames). The audit's
`content` column (committed in photo_attributes_v1.csv) is now the primary
rule; the old rule stays as a second net for future captures.

Review: photos whose labels contradict themselves. They are EXCLUDED from
training and evaluation (not quarantined — the photo itself is fine) until a
human resolves them in Label Studio. Box-level notes flag a doubtful box but
keep the photo.
"""
from __future__ import annotations

from config.settings import settings

# Photo-level contradictions -> label_state "review" (excluded everywhere).
REVIEW: dict[str, str] = {
    "0fef68e4e70449d787e01b8140034479": (
        "LS image choice says iO (June-24 export) but a Fahne box exists"
    ),
    "38d0b5c48e8146bcb9d967db4715bc20": (
        "LS image choice 'Weitere Überprüfung notwendig' but no box drawn"
    ),
}

# Box-level doubts -> kept, surfaced in review_note / dataset card.
BOX_NOTES: dict[str, str] = {
    "b6eeb74c0033498789957795df9e52fb": (
        "one Fahne box (75x65 px, board A) looks more like a scratch — verify in LS"
    ),
}


def quarantine_reason(source: str, brightness: float, content: str) -> str | None:
    """Return the reason a frame is junk, or None if it is usable."""
    if content != "pcb":
        return f"audit content '{content}' (no usable PCB photo)"
    if source != "basler":
        return f"non-camera source '{source}' (webcam/sample fallback frame)"
    if brightness < settings.dark_brightness_threshold:
        return (
            f"dark setup frame (mean brightness {brightness:.1f} < "
            f"{settings.dark_brightness_threshold:.0f})"
        )
    return None
