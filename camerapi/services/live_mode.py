"""
Live preview on/off while camerapi runs (docs/CONTRACTS.md section 4).

LIVE_PREVIEW_ENABLED is only the default for the first start. The dashboard's
live view switch (POST /api/v1/preview/mode) turns the preview on or off at
runtime, and the choice is saved in DATA_ROOT/live_preview.json, so a restart
keeps it. A missing or unreadable file means LIVE_PREVIEW_ENABLED again.

The camera has exactly one owner at any time, and `lock` keeps it that way:
switches and captures (api/routes.py) both hold it. Switching off stops the
camera-owner thread, which closes its handle, before the one-shot path may
open the camera; switching on releases the one-shot handle before the owner
thread starts. A switch therefore waits for a capture in progress.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from services.camera_service import camera_service
from services.live_camera import live_camera
from services.storage_service import _atomic_write_bytes
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

STATE_FILE_NAME = "live_preview.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LiveMode:
    def __init__(
        self,
        camera: Any = None,
        one_shot: Any = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._camera = camera if camera is not None else live_camera
        self._one_shot = one_shot if one_shot is not None else camera_service
        self._state_path = state_path
        # Reentrant: a capture holds it while the live owner may hand the
        # photo to the one-shot path (which never takes it again, but a
        # caller further up might).
        self.lock = threading.RLock()
        self._enabled = False
        self.origin = "env"  # "env" | "saved" | "switch": where `enabled` came from
        self.changed_by: Optional[str] = None
        self.changed_at: Optional[str] = None

    @property
    def state_path(self) -> Path:
        return self._state_path or settings.data_root / STATE_FILE_NAME

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ---- Lifecycle (main.lifespan) ------------------------------------------

    def start(self) -> None:
        """Startup: the saved choice, else LIVE_PREVIEW_ENABLED."""
        saved = self._read()
        with self.lock:
            if saved is not None:
                self._enabled = saved["enabled"]
                self.origin = "saved"
                self.changed_by = saved.get("changed_by") if isinstance(saved.get("changed_by"), str) else None
                self.changed_at = saved.get("changed_at") if isinstance(saved.get("changed_at"), str) else None
            else:
                self._enabled = bool(settings.live_preview_enabled)
                self.origin = "env"
            how = (
                f"saved choice in {self.state_path}" if self.origin == "saved"
                else "LIVE_PREVIEW_ENABLED"
            )
            if self._enabled:
                logger.info(
                    f"Live preview: ON ({how}; captures stop the stream and grab a "
                    "fresh frame, incomplete grabs are retaken with the one-shot path)"
                )
                if not settings.preview_token:
                    logger.warning(
                        "PREVIEW_TOKEN is empty: the preview only answers loopback "
                        "requests to localhost (the Docker dashboard needs the token)"
                    )
                self._camera.start()
            else:
                logger.info(f"Live preview: off ({how}; one-shot GrabOne per capture)")

    def stop(self) -> None:
        """Shutdown: waits for a capture in progress, then stops the owner."""
        with self.lock:
            self._camera.stop()

    # ---- Runtime switch -------------------------------------------------------

    def set(self, enabled: bool, changed_by: Optional[str] = None) -> bool:
        """
        Turn the preview on or off now and save the choice. Waits for a
        capture in progress. Returns False when the choice could not be
        saved: it still applies until camerapi restarts.
        """
        with self.lock:
            if enabled != self._enabled:
                if enabled:
                    self._one_shot.release()  # one owner: close the one-shot handle first
                    self._camera.start()
                else:
                    self._camera.stop()  # the owner thread closes its handle
                self._enabled = enabled
            self.origin = "switch"
            self.changed_by, self.changed_at = changed_by, _utc_now()
            logger.info(
                f"Live preview switched {'ON' if enabled else 'off'} "
                f"(by {changed_by!a})"  # ASCII: Windows cp1252 consoles
            )
            return self._save()

    def info(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "default": bool(settings.live_preview_enabled),
            "origin": self.origin,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at,
        }

    # ---- The saved choice -----------------------------------------------------

    def _save(self) -> bool:
        doc = {"enabled": self._enabled, "changed_by": self.changed_by, "changed_at": self.changed_at}
        try:
            _atomic_write_bytes(self.state_path, json.dumps(doc, indent=1).encode("utf-8"))
        except OSError as e:
            logger.warning(
                f"Live preview choice not saved to {self.state_path} ({type(e).__name__}: {e}); "
                "it applies until camerapi restarts"
            )
            return False
        return True

    def _read(self) -> Optional[dict]:
        path = self.state_path
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring the saved live preview choice {path}: {type(e).__name__}: {e}")
            return None
        if not isinstance(doc, dict) or not isinstance(doc.get("enabled"), bool):
            logger.warning(f"Ignoring the saved live preview choice {path}: no true/false 'enabled'")
            return None
        return doc


# Global instance (started in main.lifespan).
live_mode = LiveMode()
