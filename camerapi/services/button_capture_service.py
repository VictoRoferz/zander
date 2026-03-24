# services/button_capture_service.py

import threading
from typing import Dict, Any

from utils.logger import setup_logger
from config.settings import settings
from services.camera_service import camera_service
from services.upload_service import upload_service

logger = setup_logger(__name__, level=settings.log_level)


class ButtonCaptureService:
    """
    Triggers camera capture + upload.
    Wird von einer HTTP-Route oder extern (z.B. Button-Listener-Skript) genutzt.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def trigger_capture(self, source: str = "button") -> Dict[str, Any]:
        """
        Capture + Upload auslösen.
        source: nur fürs Logging ("button", "api", etc.)
        """
        # Nur ein Capture gleichzeitig
        if not self._lock.acquire(blocking=False):
            logger.warning("Capture already running, ignoring new trigger")
            return {"status": "busy", "reason": "capture_already_running"}

        try:
            logger.info(f"[{source}] Starting capture...")

            # 1) Bild aufnehmen
            image_path = camera_service.capture()

            if not image_path:
                logger.error(f"[{source}] capture returned no image_path")
                return {"status": "error", "reason": "no_image_path"}

            # 2) Upload zu Server 2
            logger.info(f"[{source}] Uploading {image_path.name} to Server 2...")
            upload_response = upload_service.upload_and_cleanup(image_path)

            status = upload_response.get("status", "unknown")
            logger.info(f"[{source}] Upload finished with status={status}")

            return {
                "status": "ok",
                "upload_status": status,
                "upload_response": upload_response,
            }

        except Exception as e:
            logger.error(f"[{source}] capture/upload failed: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
        finally:
            self._lock.release()


# Globale Instanz
button_capture_service = ButtonCaptureService()
